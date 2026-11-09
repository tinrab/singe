use singe_onnx::proto::onnx::{
    AttributeProto, TensorProto, TensorShapeProto, TypeProto, ValueInfoProto, attribute_proto,
    tensor_proto, tensor_shape_proto, type_proto,
};

use crate::{
    data_type::{DataType, bf16, f16},
    onnx::error::{Error, Result},
    scalar::Scalar,
    tensor::Tensor,
};

pub fn unsupported(reason: impl Into<String>) -> Error {
    Error::UnsupportedExport {
        reason: reason.into(),
    }
}

pub fn dtype(dtype: DataType) -> Result<i32> {
    use tensor_proto::DataType as OnnxType;
    Ok(match dtype {
        DataType::Bool => OnnxType::Bool,
        DataType::U8 => OnnxType::Uint8,
        DataType::I8 => OnnxType::Int8,
        DataType::U16 => OnnxType::Uint16,
        DataType::I16 => OnnxType::Int16,
        DataType::U32 => OnnxType::Uint32,
        DataType::I32 => OnnxType::Int32,
        DataType::U64 => OnnxType::Uint64,
        DataType::I64 => OnnxType::Int64,
        DataType::F16 => OnnxType::Float16,
        DataType::Bf16 => OnnxType::Bfloat16,
        DataType::F32 => OnnxType::Float,
        DataType::F64 => OnnxType::Double,
        _ => return Err(unsupported(format!("storage dtype {dtype}"))),
    } as i32)
}

pub fn tensor(name: String, tensor: &Tensor) -> Result<TensorProto> {
    let mut bytes = tensor.to_bytes()?;
    if cfg!(target_endian = "big") {
        for element in bytes.chunks_exact_mut(tensor.data_type().size_bytes()) {
            element.reverse();
        }
    }
    raw_tensor(
        name,
        tensor.data_type(),
        tensor.shape().dimensions().to_vec(),
        bytes,
    )
}

pub fn raw_tensor(
    name: String,
    data_type: DataType,
    dims: Vec<i64>,
    raw_data: Vec<u8>,
) -> Result<TensorProto> {
    Ok(TensorProto {
        name,
        data_type: dtype(data_type)?,
        dims,
        raw_data,
        ..Default::default()
    })
}

pub fn scalar_bytes(value: Scalar, dtype: DataType) -> Result<Vec<u8>> {
    Ok(match dtype {
        DataType::F16 => f16::from_f64(value.float())
            .to_bits()
            .to_le_bytes()
            .to_vec(),
        DataType::Bf16 => bf16::from_f64(value.float())
            .to_bits()
            .to_le_bytes()
            .to_vec(),
        DataType::F32 => value.single().to_le_bytes().to_vec(),
        DataType::F64 => value.float().to_le_bytes().to_vec(),
        _ => {
            return Err(unsupported(
                "scalar export currently requires a floating result",
            ));
        }
    })
}

pub fn value_info(
    name: String,
    data_type: DataType,
    dimensions: Vec<Option<tensor_shape_proto::dimension::Value>>,
) -> Result<ValueInfoProto> {
    Ok(ValueInfoProto {
        name,
        r#type: Some(TypeProto {
            value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                elem_type: dtype(data_type)?,
                shape: Some(TensorShapeProto {
                    dim: dimensions
                        .into_iter()
                        .map(|value| tensor_shape_proto::Dimension {
                            value,
                            ..Default::default()
                        })
                        .collect(),
                }),
            })),
            ..Default::default()
        }),
        ..Default::default()
    })
}

pub fn integer(name: &str, value: i64) -> AttributeProto {
    AttributeProto {
        name: name.into(),
        r#type: attribute_proto::AttributeType::Int as i32,
        i: value,
        ..Default::default()
    }
}

pub fn integers(name: &str, values: Vec<i64>) -> AttributeProto {
    AttributeProto {
        name: name.into(),
        r#type: attribute_proto::AttributeType::Ints as i32,
        ints: values,
        ..Default::default()
    }
}

pub fn float(name: &str, value: f32) -> AttributeProto {
    AttributeProto {
        name: name.into(),
        r#type: attribute_proto::AttributeType::Float as i32,
        f: value,
        ..Default::default()
    }
}

pub fn string(name: &str, value: &str) -> AttributeProto {
    AttributeProto {
        name: name.into(),
        r#type: attribute_proto::AttributeType::String as i32,
        s: value.as_bytes().to_vec(),
        ..Default::default()
    }
}
