use crate::{
    backend::{
        cuda::kernel_plan,
        registry::{Operation, OperationAttributes, OperationRequest},
    },
    error::{Error, Result},
    tensor::Tensor,
};

/// Runs an element-wise operation on its generated kernel, the kernel IR expression the CPU interprets and compiled graphs fuse.
///
/// # Errors
///
/// Returns an error if the operands or attributes are invalid for the operation, or the kernel fails.
pub fn elementwise(
    operation: Operation,
    attributes: &OperationAttributes,
    inputs: &[&Tensor],
) -> Result<Tensor> {
    let inference = OperationRequest::new(
        operation,
        attributes.clone(),
        inputs.iter().map(|input| input.value_spec()).collect(),
    )
    .infer()?;
    let output = inference
        .outputs()
        .first()
        .ok_or_else(|| Error::missing_output("cuda elementwise"))?
        .data_type();
    let first = inputs
        .first()
        .ok_or_else(|| Error::missing_input("cuda elementwise"))?;
    if operation == Operation::Cast && first.data_type() == output {
        return Ok((*first).clone());
    }
    kernel_plan::execute(operation, attributes, inputs)
}
