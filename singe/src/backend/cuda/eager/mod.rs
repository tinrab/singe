//! Eager CUDA execution of single operations, which the dispatcher's CUDA kernels call.

pub mod convolution;
pub mod elementwise;
pub mod fourier;
pub mod indexing;
pub mod linear_algebra;
pub mod materialization;
pub mod reduction;
pub mod shape;
pub mod storage;
pub mod tensor_algebra;

use crate::{
    backend::{
        cuda::{diagnostics::eager_unsupported_operation, utility::ensure_same_device},
        registry::Operation,
    },
    data_type::DataType,
    error::Result,
    shape::Shape,
    tensor::Tensor,
};

pub fn unsupported_eager_operation<T>(
    operation: Operation,
    output_data_type: DataType,
    output_shape: &Shape,
    input_data_types: &[DataType],
) -> Result<T> {
    Err(eager_unsupported_operation(
        operation,
        output_data_type,
        output_shape,
        input_data_types,
    ))
}

pub fn unsupported_eager_tensor_inputs<T>(
    operation: Operation,
    output_data_type: DataType,
    output_shape: &Shape,
    inputs: &[&Tensor],
) -> Result<T> {
    let input_data_types = inputs
        .iter()
        .map(|tensor| tensor.data_type())
        .collect::<Vec<_>>();
    unsupported_eager_operation(operation, output_data_type, output_shape, &input_data_types)
}

pub fn ensure_cuda_eager_same_device_inputs(reference: &Tensor, inputs: &[&Tensor]) -> Result<()> {
    for input in inputs {
        ensure_same_device(reference, *input)?;
    }
    Ok(())
}

#[cfg(all(test, feature = "testing"))]
mod tests;
