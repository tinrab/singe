use crate::{
    backend::{
        cuda::{eager::storage::copy_tensor_to, kernel_plan},
        registry::{Operation, OperationAttributes},
    },
    device::Device,
    error::Result,
    tensor::Tensor,
};

/// Copies `src` to `device`.
///
/// # Errors
///
/// Returns an error if the source or destination device is unsupported or the copy fails.
pub fn copy_to(src: &Tensor, device: Device) -> Result<Tensor> {
    copy_tensor_to(src, device)
}

/// Copies `source` into the elements `destination` addresses and waits for the copy.
///
/// # Errors
///
/// Returns an error if either tensor is not CUDA storage in one context, kernel compilation fails, or the launch fails.
pub fn assign(destination: &Tensor, source: &Tensor) -> Result<()> {
    kernel_plan::assign(destination, source)
}

/// Runs a random operation with the shared Philox engine and waits for it.
///
/// # Errors
///
/// Returns an error if the attributes or inputs are invalid, kernel compilation fails, or the launch fails.
pub fn random(
    operation: Operation,
    attributes: &OperationAttributes,
    inputs: &[&Tensor],
) -> Result<Tensor> {
    kernel_plan::execute(operation, attributes, inputs)
}
