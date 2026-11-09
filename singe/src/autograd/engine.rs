use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use crate::{
    autograd::{AutogradMeta, Backward, Node, formulas, no_grad},
    error::{Error, Result},
    tensor::Tensor,
};

impl Tensor {
    /// Computes the gradient of this scalar with respect to every leaf it depends on, accumulating into their [`Tensor::grad`].
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor has more than one element or does not require gradients, a saved tensor was written after it was recorded, an operation has no backward formula, or a backward operation fails.
    pub fn backward(&self) -> Result<()> {
        if self.element_count() != 1 {
            return Err(Error::invalid_argument(
                "backward without a gradient needs a one-element tensor",
            ));
        }

        let seed = self.detach().ones_like()?;
        self.backward_with(&seed)
    }

    /// Computes gradients starting from `gradient`, the gradient of some scalar with respect to this tensor.
    ///
    /// # Errors
    ///
    /// Returns an error if `gradient` differs from this tensor in shape or data type, or for the reasons [`Tensor::backward`] lists.
    pub fn backward_with(&self, gradient: &Self) -> Result<()> {
        let root = self
            .autograd()
            .ok_or_else(|| Error::invalid_argument("backward of a tensor without gradients"))?;
        if gradient.value_spec() != self.value_spec() {
            return Err(Error::invalid_argument("backward gradient metadata"));
        }

        let _no_grad = no_grad();
        run(&root, gradient.detach())
    }
}

/// Propagates `gradient` from `root` through the recorded nodes to the leaves.
fn run(root: &Arc<AutogradMeta>, gradient: Tensor) -> Result<()> {
    let Some((root_node, root_index)) = root.grad_fn.clone() else {
        return accumulate_leaf(root, gradient);
    };
    let order = topological_order(&root_node);
    let mut pending = HashMap::<*const Node, Vec<Option<Tensor>>>::new();
    pending
        .entry(Arc::as_ptr(&root_node))
        .or_insert_with(|| vec![None; root_node.outputs.len()])[root_index] = Some(gradient);
    for node in order.iter().rev() {
        let Some(gradients) = pending.remove(&Arc::as_ptr(node)) else {
            continue;
        };
        let gradients = gradients
            .into_iter()
            .zip(&node.outputs)
            .map(|(gradient, spec)| {
                gradient.map_or_else(
                    || Tensor::zeroed_on(node.device, spec.data_type(), spec.shape()),
                    Ok,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let input_gradients = backward(node, &gradients)?;
        for (edge, gradient) in node.edges.iter().zip(input_gradients) {
            let (Some(meta), Some(gradient)) = (edge, gradient) else {
                continue;
            };
            match &meta.grad_fn {
                Some((producer, index)) => {
                    let slot = &mut pending
                        .entry(Arc::as_ptr(producer))
                        .or_insert_with(|| vec![None; producer.outputs.len()])[*index];
                    *slot = Some(match slot.take() {
                        Some(sum) => sum.add(&gradient)?,
                        None => gradient,
                    });
                }
                None => accumulate_leaf(meta, gradient)?,
            }
        }
    }
    Ok(())
}

/// Returns the nodes reachable from `root` with every node after the nodes that consume its outputs.
fn topological_order(root: &Arc<Node>) -> Vec<Arc<Node>> {
    let mut order = Vec::new();
    let mut visited = HashSet::new();
    // Depth-first post-order with an explicit stack, so deep graphs do not overflow the call stack.
    let mut stack = vec![(Arc::clone(root), false)];
    while let Some((node, expanded)) = stack.pop() {
        if expanded {
            order.push(node);
            continue;
        }

        if !visited.insert(Arc::as_ptr(&node)) {
            continue;
        }

        stack.push((Arc::clone(&node), true));
        for edge in node.edges.iter().flatten() {
            if let Some((producer, _)) = &edge.grad_fn
                && !visited.contains(&Arc::as_ptr(producer))
            {
                stack.push((Arc::clone(producer), false));
            }
        }
    }
    order
}

fn backward(node: &Node, gradients: &[Tensor]) -> Result<Vec<Option<Tensor>>> {
    match &node.backward {
        Backward::Operation {
            operation,
            attributes,
            inputs,
            versions,
            outputs,
        } => {
            for (input, version) in inputs.iter().zip(versions) {
                if input.version() != *version {
                    return Err(Error::invalid_argument(format!(
                        "saved tensor of `{}` backward",
                        operation.name()
                    )));
                }
            }
            formulas::backward(*operation, attributes, inputs, outputs, gradients)
        }
        Backward::Function { backward, .. } => backward(gradients),
    }
}

fn accumulate_leaf(meta: &AutogradMeta, gradient: Tensor) -> Result<()> {
    let sum = match meta.grad() {
        Some(existing) => existing.add(&gradient)?,
        None => gradient,
    };
    meta.set_grad(Some(sum));
    Ok(())
}
