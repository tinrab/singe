use crate::{
    backend::{
        cuda::kernel_plan,
        registry::{Operation, OperationAttributes, ReductionAttributes},
        spec,
    },
    error::Result,
    tensor::Tensor,
};

pub fn reduction(
    operation: Operation,
    input: &Tensor,
    dimensions: &[i64],
    keepdim: bool,
) -> Result<Tensor> {
    let problem = spec::reduction_problem(input.shape(), dimensions, keepdim)?;
    if let Some(fill) = spec::empty_reduction_fill(input.shape(), &problem, operation)? {
        return fill.full(input.device(), input.data_type(), &problem.output_shape);
    }
    kernel_plan::execute(
        operation,
        &OperationAttributes::Reduction(ReductionAttributes {
            dimensions: dimensions.to_vec(),
            keepdim,
        }),
        &[input],
    )
}
