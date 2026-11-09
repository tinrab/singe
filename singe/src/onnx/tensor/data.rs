use std::path::Path;

use singe_onnx::{Tensor, TensorDataLocation, TensorDataType, TensorElements};

use crate::{
    data_type::DataType,
    onnx::{
        error::{Error, Result},
        tensor::{external, tensor_name},
    },
    utility::checked_element_count,
};

pub fn tensor_bytes(
    tensor: &Tensor,
    data_type: DataType,
    model_dir: Option<&Path>,
) -> Result<Vec<u8>> {
    let name = tensor_name(tensor);
    if tensor.data_location == TensorDataLocation::External {
        return external::read_external_data(tensor, &name, model_dir);
    }
    match &tensor.data {
        Some(TensorElements::Raw(bytes)) => Ok(bytes.clone()),
        Some(TensorElements::Float(values)) if data_type == DataType::F32 => {
            Ok(encode(values, f32::to_le_bytes))
        }
        Some(TensorElements::Double(values)) if data_type == DataType::F64 => {
            Ok(encode(values, f64::to_le_bytes))
        }
        Some(TensorElements::Int32(values)) => {
            int32_bytes(&name, tensor.data_type, data_type, values)
        }
        Some(TensorElements::Int64(values)) if data_type == DataType::I64 => {
            Ok(encode(values, i64::to_le_bytes))
        }
        Some(TensorElements::Uint64(values)) if data_type == DataType::U64 => {
            Ok(encode(values, u64::to_le_bytes))
        }
        Some(TensorElements::Uint64(values)) if data_type == DataType::U32 => {
            let values = values
                .iter()
                .map(|value| {
                    u32::try_from(*value)
                        .map_err(|_| Error::invalid_initializer(&name, "uint32 value out of range"))
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(encode(&values, u32::to_le_bytes))
        }
        Some(TensorElements::String(_)) => Err(Error::UnsupportedType {
            name,
            data_type: tensor.data_type,
            reason: "string tensors are not supported".into(),
        }),
        None => {
            let element_count = checked_element_count(&tensor.dimensions, "onnx tensor")
                .map_err(|_| Error::invalid_initializer(&name, "invalid dimensions"))?;
            let byte_count = element_count
                .checked_mul(data_type.size_bytes())
                .ok_or_else(|| Error::invalid_initializer(&name, "byte count overflow"))?;
            if byte_count != 0 {
                return Err(Error::invalid_initializer(
                    &name,
                    "nonempty tensor has no data",
                ));
            }
            Ok(Vec::new())
        }
        _ => Err(Error::invalid_initializer(
            name,
            "tensor data field does not match tensor dtype",
        )),
    }
}

pub fn encode<T, const N: usize>(values: &[T], encode_one: impl Fn(T) -> [u8; N]) -> Vec<u8>
where
    T: Copy,
{
    let mut bytes = Vec::with_capacity(values.len() * N);
    for value in values {
        bytes.extend_from_slice(&encode_one(*value));
    }
    bytes
}

pub fn int32_bytes(
    name: &str,
    onnx_type: TensorDataType,
    data_type: DataType,
    values: &[i32],
) -> Result<Vec<u8>> {
    match (onnx_type, data_type) {
        (TensorDataType::Int32, DataType::I32) => Ok(encode(values, i32::to_le_bytes)),
        (TensorDataType::Int16, DataType::I16) => {
            let values = values
                .iter()
                .map(|value| {
                    i16::try_from(*value)
                        .map_err(|_| Error::invalid_initializer(name, "int16 value out of range"))
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(encode(&values, i16::to_le_bytes))
        }
        (TensorDataType::Uint16, DataType::U16) => {
            let values = values
                .iter()
                .map(|value| {
                    u16::try_from(*value)
                        .map_err(|_| Error::invalid_initializer(name, "uint16 value out of range"))
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(encode(&values, u16::to_le_bytes))
        }
        (TensorDataType::Int8, DataType::I8) => Ok(values
            .iter()
            .map(|value| {
                i8::try_from(*value)
                    .map(|value| value as u8)
                    .map_err(|_| Error::invalid_initializer(name, "int8 value out of range"))
            })
            .collect::<Result<Vec<_>>>()?),
        (TensorDataType::Uint8, DataType::U8) => Ok(values
            .iter()
            .map(|value| {
                u8::try_from(*value)
                    .map_err(|_| Error::invalid_initializer(name, "uint8 value out of range"))
            })
            .collect::<Result<Vec<_>>>()?),
        (TensorDataType::Bool, DataType::Bool) => values
            .iter()
            .map(|value| match value {
                0 => Ok(0),
                1 => Ok(1),
                _ => Err(Error::invalid_initializer(
                    name,
                    "boolean value must be 0 or 1",
                )),
            })
            .collect(),
        (TensorDataType::Float16, DataType::F16) | (TensorDataType::Bfloat16, DataType::Bf16) => {
            let bits = values
                .iter()
                .map(|value| {
                    u16::try_from(*value).map_err(|_| {
                        Error::invalid_initializer(name, "16-bit float encoding is out of range")
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(encode(&bits, u16::to_le_bytes))
        }
        _ => Err(Error::invalid_initializer(
            name,
            "int32_data cannot encode requested dtype",
        )),
    }
}
