use std::sync::Arc;

use crate::{
    autograd::{Backward, Node, edges, no_grad, records, set_grad_enabled},
    error::{Error, Result},
    tensor::Tensor,
};

/// Runs `function` without recording its intermediate values, and recomputes them during backward, trading compute for memory as `torch.utils.checkpoint` does.
///
/// The outputs require gradients when an input does; backward runs `function` again on the saved inputs with recording enabled and backpropagates through that recomputation.
/// `function` must be deterministic, since backward differentiates the recomputed values.
///
/// # Errors
///
/// Returns an error if `function` fails, or returns no outputs.
pub fn checkpoint<F>(inputs: &[Tensor], function: F) -> Result<Vec<Tensor>>
where
    F: Fn(&[Tensor]) -> Result<Vec<Tensor>> + Send + Sync + 'static,
{
    if !records(inputs) {
        return function(inputs);
    }

    let outputs = {
        let _no_grad = no_grad();
        function(inputs)?
    };
    let device = outputs
        .first()
        .ok_or_else(|| Error::missing_output("checkpoint"))?
        .device();
    let saved = inputs.iter().map(Tensor::detach).collect::<Vec<_>>();
    let versions = inputs.iter().map(Tensor::version).collect::<Vec<_>>();
    let function = Arc::new(function);
    let backward = move |gradients: &[Tensor]| -> Result<Vec<Option<Tensor>>> {
        if saved
            .iter()
            .zip(&versions)
            .any(|(input, version)| input.version() != *version)
        {
            return Err(Error::invalid_argument(
                "a checkpointed input was written after the forward pass",
            ));
        }

        // Fresh leaves collect the gradient of each input through the recomputation.
        let leaves = saved
            .iter()
            .map(|input| {
                let leaf = input.detach();
                if leaf.data_type().is_float() {
                    leaf.set_requires_grad(true)?;
                }
                Ok(leaf)
            })
            .collect::<Result<Vec<_>>>()?;
        let recomputed = {
            let _enabled = set_grad_enabled(true);
            function(&leaves)?
        };
        for (output, gradient) in recomputed.iter().zip(gradients) {
            if output.requires_grad() {
                output.backward_with(gradient)?;
            }
        }
        Ok(leaves.iter().map(Tensor::grad).collect())
    };
    let node = Arc::new(Node {
        backward: Backward::Function {
            name: "checkpoint",
            backward: Box::new(backward),
        },
        edges: edges(inputs),
        outputs: outputs.iter().map(Tensor::value_spec).collect(),
        device,
    });
    node.attach(&outputs);
    Ok(outputs)
}
