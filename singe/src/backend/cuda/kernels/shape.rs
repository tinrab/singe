pub use crate::backend::cuda::eager::{
    indexing::{index, index_put, index_write},
    materialization::{concat, contiguous, repeat_kv},
};
use crate::{
    backend::{
        cuda::{eager::indexing::indexing, kernel_plan},
        registry::{Operation, OperationAttributes, QuantizationAttributes},
    },
    error::Result,
    tensor::Tensor,
};

pub fn index_select(input: &Tensor, dim: i64, indices: &Tensor) -> Result<Tensor> {
    indexing(
        Operation::IndexSelect,
        &OperationAttributes::IndexSelect(dim),
        input,
        indices,
    )
}

pub fn take(input: &Tensor, dim: i64, indices: &Tensor) -> Result<Tensor> {
    indexing(
        Operation::Take,
        &OperationAttributes::IndexSelect(dim),
        input,
        indices,
    )
}

pub fn quantize(
    input: &Tensor,
    scale: &Tensor,
    zero_point: &Tensor,
    attributes: QuantizationAttributes,
) -> Result<Tensor> {
    kernel_plan::execute(
        Operation::Quantize,
        &OperationAttributes::Quantization(attributes),
        &[input, scale, zero_point],
    )
}

pub fn dequantize(
    input: &Tensor,
    scale: &Tensor,
    zero_point: &Tensor,
    attributes: QuantizationAttributes,
) -> Result<Tensor> {
    kernel_plan::execute(
        Operation::Dequantize,
        &OperationAttributes::Quantization(attributes),
        &[input, scale, zero_point],
    )
}
