use crate::{
    backend::{
        registry::{
            Operation, OperationAttributes, OperationInference,
            infer::{InferenceContext, elementwise::reject_bool},
        },
        spec,
    },
    data_type::DataType,
    error::{Error, Result},
    shape::normalize_dim_usize,
};

/// Reduces the selected dimensions.
pub fn reduction(context: &InferenceContext<'_>) -> Result<OperationInference> {
    let reduction = context.attributes(OperationAttributes::reduction)?;
    let problem =
        spec::reduction_problem(context.shape(), &reduction.dimensions, reduction.keepdim)?;
    spec::empty_reduction_fill(context.shape(), &problem, context.operation)?;
    // Sums and products of integers and Booleans accumulate in `I64`, as `PyTorch`'s do; a mean has no integer meaning.
    let input = context.first().data_type();
    let data_type = match context.operation {
        Operation::Sum | Operation::Product if !input.is_float() => DataType::I64,
        Operation::Mean | Operation::LogSumExp if !input.is_float() => {
            return Err(Error::invalid_argument(
                "mean or log-sum-exp of an integer or boolean tensor",
            ));
        }
        _ => input,
    };
    Ok(context.converted(&problem.output_shape, data_type))
}

/// Returns the indices of extreme values along one dimension.
pub fn arg_reduction(context: &InferenceContext<'_>) -> Result<OperationInference> {
    let dim = context.value(OperationAttributes::dimension)?;
    reject_bool(context, "boolean arg reduction")?;
    let inference =
        context.same_type(&spec::reduction_problem(context.shape(), &[dim], false)?.output_shape);
    Ok(order_in_f32(context, inference))
}

/// Returns the `I64` positions that sort each slice along one dimension.
pub fn argsort(context: &InferenceContext<'_>) -> Result<OperationInference> {
    let sort = context.attributes(OperationAttributes::sort)?;
    scan_dimension(context, sort.dim)?;
    Ok(order_in_f32(
        context,
        context.finish(context.shape(), DataType::I64),
    ))
}

/// Accumulates sums along one dimension, keeping the input's shape and data type.
pub fn cumulative(context: &InferenceContext<'_>) -> Result<OperationInference> {
    let dim = context.value(OperationAttributes::dimension)?;
    scan_dimension(context, dim)?;
    // Integer and Boolean sums accumulate in `I64`, as `PyTorch`'s do.
    let input = context.first().data_type();
    let data_type = if input.is_float() {
        input
    } else {
        DataType::I64
    };
    Ok(context.converted(context.shape(), data_type))
}

/// Checks a dimension that runs along slices, where a scalar is one slice of one element.
fn scan_dimension(context: &InferenceContext<'_>, dim: i64) -> Result<()> {
    let rank = context.shape().rank();
    if rank == 0 {
        return if matches!(dim, 0 | -1) {
            Ok(())
        } else {
            Err(Error::DimensionOutOfBounds)
        };
    }
    normalize_dim_usize(dim, rank).map(|_| ())
}

/// Selects the top `k` values along one dimension.
pub fn topk(context: &InferenceContext<'_>) -> Result<OperationInference> {
    let topk = context.attributes(OperationAttributes::topk)?;
    reject_bool(context, "boolean topk")?;
    let problem = spec::topk_problem(context.shape(), topk.k, topk.dim, "operation topk")?;
    Ok(context.same_type(&problem.output_shape))
}

// Sorts and arg reductions read half-precision and 8-bit floats as `F32`, whose conversion keeps their order exactly.
fn order_in_f32(
    context: &InferenceContext<'_>,
    mut inference: OperationInference,
) -> OperationInference {
    if matches!(
        context.first().data_type(),
        DataType::F16 | DataType::Bf16 | DataType::F8E4M3 | DataType::F8E5M2
    ) {
        inference.kernel_input_data_types = vec![DataType::F32];
    }
    inference
}
