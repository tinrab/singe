use singe_onnx::TensorDataType;

use crate::{
    data_type::DataType,
    onnx::error::{Error, Result},
};

pub fn data_type(name: &str, data_type: TensorDataType) -> Result<DataType> {
    match data_type {
        TensorDataType::Float => Ok(DataType::F32),
        TensorDataType::Uint8 => Ok(DataType::U8),
        TensorDataType::Int8 => Ok(DataType::I8),
        TensorDataType::Uint16 => Ok(DataType::U16),
        TensorDataType::Int16 => Ok(DataType::I16),
        TensorDataType::Int32 => Ok(DataType::I32),
        TensorDataType::Int64 => Ok(DataType::I64),
        TensorDataType::Bool => Ok(DataType::Bool),
        TensorDataType::Float16 => Ok(DataType::F16),
        TensorDataType::Double => Ok(DataType::F64),
        TensorDataType::Uint32 => Ok(DataType::U32),
        TensorDataType::Uint64 => Ok(DataType::U64),
        TensorDataType::Bfloat16 => Ok(DataType::Bf16),
        TensorDataType::Float8e4m3fn => Ok(DataType::F8E4M3),
        TensorDataType::Float8e5m2 => Ok(DataType::F8E5M2),
        TensorDataType::Undefined
        | TensorDataType::Float8e4m3fnuz
        | TensorDataType::Float8e5m2fnuz
        | TensorDataType::Float4e2m1
        | TensorDataType::String
        | TensorDataType::Complex64
        | TensorDataType::Complex128
        | TensorDataType::Uint4
        | TensorDataType::Int4
        | TensorDataType::Float8e8m0
        | TensorDataType::Uint2
        | TensorDataType::Int2 => Err(Error::UnsupportedType {
            name: name.into(),
            data_type,
            reason: "no compatible singe tensor dtype".into(),
        }),
    }
}
