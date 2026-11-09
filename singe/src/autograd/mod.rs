//! Reverse-mode automatic differentiation.
//!
//! A tensor that requires gradients carries [`AutogradMeta`](crate::autograd::AutogradMeta).
//! While gradient recording is enabled, the [`DispatchKey::Autograd`](crate::dispatch::DispatchKey::Autograd) kernel runs each operation that reads such a tensor and records a [`Node`](crate::autograd::Node) holding the operation, its saved inputs and outputs, and edges to its inputs' gradient state.
//! [`Tensor::backward`](crate::tensor::Tensor::backward) walks the nodes in reverse topological order, computes input gradients with each operation's backward formula, and accumulates them into the leaves.

mod checkpoint;
mod engine;
mod formulas;
mod spellings;

use std::{
    cell::Cell,
    fmt::{self, Debug, Formatter},
    mem,
    sync::{Arc, Mutex},
};

pub use self::checkpoint::checkpoint;

use crate::{
    backend::registry::{AliasBehavior, Operation, OperationAttributes, operation_schema},
    device::Device,
    dispatch::{DispatchKey, Kernel, KernelCall, RegistryBuilder},
    error::{Error, Result},
    tensor::Tensor,
    value::ValueSpec,
};

thread_local! {
    static ENABLED: Cell<bool> = const { Cell::new(true) };
}

/// Returns whether operations on this thread record backward nodes.
#[must_use]
pub fn is_grad_enabled() -> bool {
    ENABLED.with(Cell::get)
}

/// Stops recording backward nodes on this thread until the guard drops, as `torch.no_grad` does.
pub fn no_grad() -> GradModeGuard {
    set_grad_enabled(false)
}

/// Sets whether operations on this thread record backward nodes until the guard drops.
pub fn set_grad_enabled(enabled: bool) -> GradModeGuard {
    let previous = ENABLED.with(|mode| mode.replace(enabled));
    GradModeGuard { previous }
}

/// Restores the previous gradient-recording mode when dropped.
#[derive(Debug)]
#[must_use = "the gradient mode applies only while the guard is alive"]
pub struct GradModeGuard {
    previous: bool,
}

impl Drop for GradModeGuard {
    fn drop(&mut self) {
        ENABLED.with(|mode| mode.set(self.previous));
    }
}

/// Gradient state of a tensor that requires gradients.
const NO_PANIC_UNDER_LOCK: &str = "these locks only clone or swap a value, so no holder can panic";

pub struct AutogradMeta {
    /// The node that produced the tensor and the tensor's result index, or `None` for a leaf.
    pub(crate) grad_fn: Option<(Arc<Node>, usize)>,
    /// The gradient `backward` accumulated into a leaf.
    pub(crate) grad: Mutex<Option<Tensor>>,
}

impl AutogradMeta {
    fn leaf() -> Self {
        Self {
            grad_fn: None,
            grad: Mutex::new(None),
        }
    }

    pub(crate) fn grad(&self) -> Option<Tensor> {
        self.grad.lock().expect(NO_PANIC_UNDER_LOCK).clone()
    }

    pub(crate) fn set_grad(&self, grad: Option<Tensor>) {
        // The old gradient's destructor runs after the lock is released.
        let previous = mem::replace(&mut *self.grad.lock().expect(NO_PANIC_UNDER_LOCK), grad);
        drop(previous);
    }
}

impl Debug for AutogradMeta {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AutogradMeta")
            .field(
                "grad_fn",
                &self
                    .grad_fn
                    .as_ref()
                    .map(|(node, index)| (node.name(), *index)),
            )
            .finish_non_exhaustive()
    }
}

/// Computes input gradients from output gradients for a node that is not a registry operation, such as a checkpointed function.
pub(crate) type BackwardFunction = dyn Fn(&[Tensor]) -> Result<Vec<Option<Tensor>>> + Send + Sync;

/// How a node computes its input gradients.
pub(crate) enum Backward {
    /// A registry operation, differentiated by its formula from the saved inputs and outputs.
    Operation {
        operation: Operation,
        attributes: OperationAttributes,
        inputs: Vec<Tensor>,
        versions: Vec<u64>,
        outputs: Vec<Tensor>,
    },
    /// A function with its own backward computation.
    Function {
        name: &'static str,
        backward: Box<BackwardFunction>,
    },
}

/// One recorded step of the backward graph.
pub struct Node {
    pub(crate) backward: Backward,
    /// Gradient state of each input, or `None` for inputs that do not require gradients.
    pub(crate) edges: Vec<Option<Arc<AutogradMeta>>>,
    /// The value of each output, which a missing output gradient fills with zeros.
    pub(crate) outputs: Vec<ValueSpec>,
    /// The device of the outputs.
    pub(crate) device: Device,
}

impl Node {
    fn name(&self) -> &'static str {
        match &self.backward {
            Backward::Operation { operation, .. } => operation.name(),
            Backward::Function { name, .. } => name,
        }
    }

    /// Attaches this node to the floating `outputs`, which become non-leaf tensors that require gradients.
    pub(crate) fn attach(self: &Arc<Self>, outputs: &[Tensor]) {
        for (index, output) in outputs.iter().enumerate() {
            if output.data_type().is_float() {
                output.set_autograd(Some(Arc::new(AutogradMeta {
                    grad_fn: Some((Arc::clone(self), index)),
                    grad: Mutex::new(None),
                })));
            }
        }
    }
}

/// Returns the gradient state of each of `inputs`, for the edges of a node.
pub(crate) fn edges(inputs: &[Tensor]) -> Vec<Option<Arc<AutogradMeta>>> {
    inputs.iter().map(Tensor::autograd).collect()
}

/// Returns whether an operation on `inputs` records a backward node.
pub(crate) fn records(inputs: &[Tensor]) -> bool {
    is_grad_enabled() && inputs.iter().any(Tensor::requires_grad)
}

pub(crate) fn register(builder: &mut RegistryBuilder) {
    builder.fallback(DispatchKey::Autograd, Kernel::new(record));
}

/// Runs the operation below the autograd key and records its backward node.
fn record(call: &KernelCall<'_>) -> Result<Vec<Tensor>> {
    let operation = call.operation();
    if operation_schema(operation).alias_behavior() == AliasBehavior::WritesInput {
        return Err(Error::unsupported_operation(
            "in-place writes to tensors that require gradients",
        ));
    }

    let outputs = call
        .redispatch()?
        .into_iter()
        .map(|output| {
            // An output that is one of the inputs gets its own gradient state.
            if call
                .inputs()
                .iter()
                .any(|input| Arc::ptr_eq(&input.inner, &output.inner))
            {
                output.alias_unrecorded()
            } else {
                output
            }
        })
        .collect::<Vec<_>>();
    if !outputs.iter().any(|output| output.data_type().is_float()) {
        return Ok(outputs);
    }

    let node = Arc::new(Node {
        backward: Backward::Operation {
            operation,
            attributes: call.attributes().clone(),
            inputs: call.inputs().iter().map(Tensor::alias_unrecorded).collect(),
            versions: call.inputs().iter().map(Tensor::version).collect(),
            outputs: outputs.iter().map(Tensor::alias_unrecorded).collect(),
        },
        edges: edges(call.inputs()),
        outputs: outputs.iter().map(Tensor::value_spec).collect(),
        device: outputs[0].device(),
    });
    node.attach(&outputs);
    Ok(outputs)
}

impl Tensor {
    pub(crate) fn autograd(&self) -> Option<Arc<AutogradMeta>> {
        self.inner
            .autograd
            .read()
            .expect(NO_PANIC_UNDER_LOCK)
            .clone()
    }

    fn set_autograd(&self, meta: Option<Arc<AutogradMeta>>) {
        // The old metadata's destructor runs after the lock is released.
        let previous = mem::replace(
            &mut *self.inner.autograd.write().expect(NO_PANIC_UNDER_LOCK),
            meta,
        );
        drop(previous);
    }

    /// Returns whether operations on this tensor record backward nodes.
    #[must_use]
    pub fn requires_grad(&self) -> bool {
        self.inner
            .autograd
            .read()
            .expect(NO_PANIC_UNDER_LOCK)
            .is_some()
    }

    /// Marks a leaf tensor, and the clones that share it, as requiring gradients or not.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor is not floating point, or is the result of a recorded operation.
    pub fn set_requires_grad(&self, requires_grad: bool) -> Result<()> {
        if !self.data_type().is_float() {
            return Err(Error::unsupported_data_type(
                "only floating tensors require gradients",
            ));
        }

        if self.grad_fn_name().is_some() {
            return Err(Error::invalid_argument(
                "only leaf tensors set requires_grad; use detach for a result",
            ));
        }

        self.set_autograd(requires_grad.then(|| Arc::new(AutogradMeta::leaf())));
        Ok(())
    }

    /// Returns the gradient that `backward` accumulated into this leaf tensor.
    #[must_use]
    pub fn grad(&self) -> Option<Self> {
        self.autograd().and_then(|meta| meta.grad())
    }

    /// Clears the accumulated gradient of this leaf tensor.
    pub fn zero_grad(&self) {
        if let Some(meta) = self.autograd() {
            meta.set_grad(None);
        }
    }

    /// Returns a tensor sharing this tensor's storage that does not require gradients.
    #[must_use]
    pub fn detach(&self) -> Self {
        self.alias_unrecorded()
    }

    /// Returns the name of the operation that produced this tensor, when it was recorded for gradients.
    #[must_use]
    pub fn grad_fn_name(&self) -> Option<&'static str> {
        self.autograd()
            .and_then(|meta| meta.grad_fn.as_ref().map(|(node, _)| node.name()))
    }
}

#[cfg(test)]
mod tests;
