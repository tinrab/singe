//! The `KServe` v2 tensor endpoints of a model served without a task: `POST /v2/models/{model}/infer` runs it on named tensors, and `GET /v2/models/{model}` lists them.

use std::{collections::HashMap, sync::Arc};

use axum::{
    Json,
    extract::{Path, State},
};
use serde_json::{Number, Value};
use singe::{DataType, Device, Shape, Tensor};
use singe_model::function::{TensorFunction, TensorSpec};

use crate::{
    api::tensor::{InferRequest, InferResponse, InferTensor, ModelMetadata, TensorMetadata},
    error::{ApiError, ApiResult},
    handler::ApiJson,
    registry::{Engine, Served},
};

pub async fn infer(
    State(served): State<Served>,
    Path(name): Path<String>,
    ApiJson(request): ApiJson<InferRequest>,
) -> ApiResult<Json<InferResponse>> {
    let model = served.get(&name)?;
    let Engine::Tensors(worker) = &model.engine else {
        return Err(ApiError::wrong_task(
            &model.name,
            &model.task,
            "/v2/models/<model>/infer",
        ));
    };
    let InferRequest {
        id,
        inputs,
        outputs: requested,
    } = request;
    let outputs = worker
        .call(move |function| {
            let inputs = inputs
                .iter()
                .map(|input| Ok((input.name.clone(), decode_input(input)?)))
                .collect::<ApiResult<HashMap<_, _>>>()?;
            let outputs = function.run(&inputs, &[])?;
            let names = if requested.is_empty() {
                function.steps()[0]
                    .outputs
                    .iter()
                    .map(|spec| spec.name.clone())
                    .collect::<Vec<_>>()
            } else {
                requested.into_iter().map(|output| output.name).collect()
            };
            names
                .into_iter()
                .map(|name| {
                    let tensor = outputs.get(&name).ok_or_else(|| {
                        ApiError::invalid_field("outputs", format!("no output {name}"))
                    })?;
                    encode_output(name, tensor)
                })
                .collect::<ApiResult<Vec<_>>>()
        })
        .await?;
    Ok(Json(InferResponse {
        model_name: model.name.clone(),
        id,
        outputs,
    }))
}

pub async fn metadata(
    State(served): State<Served>,
    Path(name): Path<String>,
) -> ApiResult<Json<ModelMetadata>> {
    let model = served.get(&name)?;
    let Engine::Tensors(worker) = &model.engine else {
        return Err(ApiError::wrong_task(
            &model.name,
            &model.task,
            "/v2/models/<model>",
        ));
    };
    let (inputs, outputs) = worker
        .call(|function: &mut Arc<dyn TensorFunction>| {
            let step = &function.steps()[0];
            Ok((describe(&step.inputs), describe(&step.outputs)))
        })
        .await?;
    Ok(Json(ModelMetadata {
        name: model.name.clone(),
        platform: "singe",
        inputs,
        outputs,
    }))
}

fn describe(specs: &[TensorSpec]) -> Vec<TensorMetadata> {
    specs
        .iter()
        .map(|spec| TensorMetadata {
            name: spec.name.clone(),
            datatype: wire_type(spec.data_type).to_owned(),
            shape: spec
                .shape
                .iter()
                .zip(&spec.axes)
                .map(|(&size, axis)| if axis.is_some() { -1 } else { size })
                .collect(),
        })
        .collect()
}

const KSERVE_TYPES: &[(DataType, &str)] = &[
    (DataType::Bool, "BOOL"),
    (DataType::U8, "UINT8"),
    (DataType::I8, "INT8"),
    (DataType::U16, "UINT16"),
    (DataType::I16, "INT16"),
    (DataType::U32, "UINT32"),
    (DataType::I32, "INT32"),
    (DataType::U64, "UINT64"),
    (DataType::I64, "INT64"),
    (DataType::F16, "FP16"),
    (DataType::Bf16, "BF16"),
    (DataType::F32, "FP32"),
    (DataType::F64, "FP64"),
];

fn wire_type(data_type: DataType) -> &'static str {
    // A type `KServe` has no name for, such as an 8-bit float, travels as FP32 or INT64, which `encode_output` casts it to and `decode_input` casts back from.
    match KSERVE_TYPES.iter().find(|(known, _)| *known == data_type) {
        Some(&(_, name)) => name,
        None if data_type.is_float() => "FP32",
        None => "INT64",
    }
}

fn decode_input(input: &InferTensor) -> ApiResult<Tensor> {
    let invalid = |message: String| ApiError::invalid_field("inputs", message);
    let Some(&(data_type, _)) = KSERVE_TYPES
        .iter()
        .find(|(_, name)| *name == input.datatype)
    else {
        return Err(invalid(format!(
            "{}'s datatype {}",
            input.name, input.datatype
        )));
    };
    let shape = Shape::create(input.shape.clone())
        .map_err(|_| invalid(format!("{}'s shape {:?}", input.name, input.shape)))?;
    let mut data = Vec::new();
    flatten(&input.data, &mut data);
    if shape.element_count() != data.len() {
        return Err(invalid(format!(
            "{}'s {} values for shape {:?}, which holds {}",
            input.name,
            data.len(),
            input.shape,
            shape.element_count()
        )));
    }

    let tensor = match data_type {
        DataType::Bool => {
            let values = values(input, &data, "booleans", |value| {
                value.as_bool().map(u8::from)
            })?;
            Tensor::from_bytes(Device::Cpu, DataType::Bool, shape, &values)?
        }
        DataType::F64 => {
            Tensor::from_vec(Device::Cpu, shape, values(input, &data, "numbers", number)?)?
        }
        DataType::U64 => Tensor::from_vec(
            Device::Cpu,
            shape,
            values(input, &data, "unsigned integers", Value::as_u64)?,
        )?,
        data_type if data_type.is_float() => {
            let values = values(input, &data, "numbers", |value| {
                number(value).map(|value| value as f32)
            })?;
            Tensor::from_vec(Device::Cpu, shape, values)?.cast(data_type)?
        }
        data_type => {
            let values = values(input, &data, "integers", Value::as_i64)?;
            let (low, high) = integer_range(data_type);
            if let Some(value) = values.iter().find(|&&value| value < low || value > high) {
                return Err(invalid(format!(
                    "{}'s value {value}, outside {} from {low} to {high}",
                    input.name, input.datatype
                )));
            }
            Tensor::from_vec(Device::Cpu, shape, values)?.cast(data_type)?
        }
    };
    Ok(tensor)
}

fn flatten<'a>(values: &'a [Value], flat: &mut Vec<&'a Value>) {
    for value in values {
        match value {
            Value::Array(nested) => flatten(nested, flat),
            value => flat.push(value),
        }
    }
}

fn number(value: &Value) -> Option<f64> {
    // JSON has no non-finite numbers, so they travel as the strings protobuf's JSON mapping uses.
    match value.as_str() {
        Some("NaN") => Some(f64::NAN),
        Some("Infinity") => Some(f64::INFINITY),
        Some("-Infinity") => Some(f64::NEG_INFINITY),
        _ => value.as_f64(),
    }
}

fn values<T>(
    input: &InferTensor,
    data: &[&Value],
    kind: &str,
    read: fn(&Value) -> Option<T>,
) -> ApiResult<Vec<T>> {
    data.iter()
        .map(|value| read(value))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| {
            ApiError::invalid_field(
                "inputs",
                format!("{}'s data, which are not all {kind}", input.name),
            )
        })
}

const fn integer_range(data_type: DataType) -> (i64, i64) {
    match data_type {
        DataType::U8 => (0, u8::MAX as i64),
        DataType::I8 => (i8::MIN as i64, i8::MAX as i64),
        DataType::U16 => (0, u16::MAX as i64),
        DataType::I16 => (i16::MIN as i64, i16::MAX as i64),
        DataType::U32 => (0, u32::MAX as i64),
        DataType::I32 => (i32::MIN as i64, i32::MAX as i64),
        _ => (i64::MIN, i64::MAX),
    }
}

fn encode_output(name: String, tensor: &Tensor) -> ApiResult<InferTensor> {
    let data_type = tensor.data_type();
    let data = match data_type {
        DataType::Bool => tensor
            .to_bytes()?
            .into_iter()
            .map(|value| Value::Bool(value != 0))
            .collect(),
        DataType::U64 => tensor
            .to_vec::<u64>()?
            .into_iter()
            .map(Value::from)
            .collect(),
        data_type if data_type.is_float() => {
            let values = if data_type == DataType::F64 {
                tensor.to_vec::<f64>()?
            } else {
                tensor
                    .cast(DataType::F32)?
                    .to_vec::<f32>()?
                    .into_iter()
                    .map(f64::from)
                    .collect()
            };
            values
                .into_iter()
                .map(|value| {
                    Number::from_f64(value).map_or_else(
                        || {
                            Value::from(if value.is_nan() {
                                "NaN"
                            } else if value > 0.0 {
                                "Infinity"
                            } else {
                                "-Infinity"
                            })
                        },
                        Value::Number,
                    )
                })
                .collect()
        }
        _ => tensor
            .cast(DataType::I64)?
            .to_vec::<i64>()?
            .into_iter()
            .map(Value::from)
            .collect(),
    };
    Ok(InferTensor {
        name,
        shape: tensor.shape().dimensions().to_vec(),
        datatype: wire_type(data_type).to_owned(),
        data,
    })
}
