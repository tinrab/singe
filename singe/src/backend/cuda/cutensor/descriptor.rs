use singe_cuda::{data_type::DataType as CudaNativeDataType, error::Error as CudaError};
use singe_cutensor::{
    context::Context as CutensorContext, tensor::TensorDescriptor as CutensorTensorDescriptor,
    types::Mode as CutensorMode,
};

use crate::{
    backend::cuda::tensor_region::CudaTensorRegion,
    data_type::DataType,
    error::{Error, Result},
    layout::Layout,
    shape::Shape,
    utility::{to_i32, to_u64},
};

pub fn cutensor_tensor_descriptor(
    cutensor: &CutensorContext,
    tensor: CudaTensorRegion<'_>,
) -> Result<CutensorTensorDescriptor> {
    if tensor.region().context() != cutensor.cuda_context() {
        return Err(CudaError::MemoryContextMismatch.into());
    }

    if tensor.data_type() != DataType::F32 {
        return Err(Error::unsupported_data_type("cuTENSOR tensor"));
    }

    let alignment = cutensor_tensor_alignment(tensor);

    let shape = tensor_shape_u64(tensor.shape())?;
    let strides = tensor_strides_u64(tensor.layout())?;
    CutensorTensorDescriptor::create_with_strides(
        cutensor,
        &shape,
        Some(&strides),
        CudaNativeDataType::F32,
        alignment,
    )
    .map_err(Error::from)
}

pub fn cutensor_modes(modes: &[i64]) -> Result<Vec<CutensorMode>> {
    modes
        .iter()
        .copied()
        .map(|mode| to_i32(mode, "cuTENSOR mode").map(CutensorMode::from))
        .collect()
}

fn tensor_shape_u64(shape: &Shape) -> Result<Vec<u64>> {
    shape
        .dimensions()
        .iter()
        .copied()
        .map(|dimension| to_u64(dimension, "cuTENSOR tensor dimension"))
        .collect()
}

fn tensor_strides_u64(shape: &Layout) -> Result<Vec<u64>> {
    shape
        .strides()
        .iter()
        .copied()
        .map(|stride| to_u64(stride, "cuTENSOR tensor stride"))
        .collect()
}

pub fn cutensor_tensor_alignment(tensor: CudaTensorRegion<'_>) -> u32 {
    let address = tensor.region().as_ptr().addr();
    1u32 << address.trailing_zeros().min(7)
}
