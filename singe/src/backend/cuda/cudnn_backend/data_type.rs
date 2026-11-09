use singe_cudnn::data_type::DataType as CudnnDataType;

use crate::{
    data_type::DataType,
    error::{Error, Result},
};

/// Maps only encodings whose native element representation matches Singe storage.
pub fn storage_type(data_type: DataType) -> Result<CudnnDataType> {
    Ok(match data_type {
        DataType::Bool => CudnnDataType::ByteBoolean,
        DataType::U8 => CudnnDataType::U8,
        DataType::I8 => CudnnDataType::I8,
        DataType::U32 => CudnnDataType::U32,
        DataType::I32 => CudnnDataType::I32,
        DataType::I64 => CudnnDataType::I64,
        DataType::F16 => CudnnDataType::F16,
        DataType::Bf16 => CudnnDataType::BF16,
        DataType::F32 => CudnnDataType::F32,
        DataType::F64 => CudnnDataType::F64,
        DataType::F8E4M3 => CudnnDataType::F8E4M3,
        DataType::F8E5M2 => CudnnDataType::F8E5M2,
        DataType::U16 | DataType::I16 | DataType::U64 | DataType::F4E2M1 => {
            return Err(Error::unsupported_data_type("cuDNN graph binding encoding"));
        }
    })
}

/// Accumulate half and single-precision values in FP32, preserving FP64 arithmetic.
pub fn compute_type(data_type: DataType) -> Result<CudnnDataType> {
    match data_type {
        DataType::F16 | DataType::Bf16 | DataType::F32 => Ok(CudnnDataType::F32),
        DataType::F64 => Ok(CudnnDataType::F64),
        _ => Err(Error::unsupported_data_type(
            "cuDNN floating-point graph template",
        )),
    }
}
