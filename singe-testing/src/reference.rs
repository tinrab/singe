use std::{collections::HashMap, fs, path::Path};

use safetensors::{Dtype, SafeTensors};
use serde_json::Value;
use singe::{DataType, Device, Shape, Tensor};

use crate::{HalfGap, host};

#[derive(Debug, Clone)]
pub struct Reference {
    tensors: HashMap<String, Tensor>,
    metadata: Value,
}

fn data_type(dtype: Dtype) -> DataType {
    match dtype {
        Dtype::BOOL => DataType::Bool,
        Dtype::U8 => DataType::U8,
        Dtype::I8 => DataType::I8,
        Dtype::I16 => DataType::I16,
        Dtype::I32 => DataType::I32,
        Dtype::I64 => DataType::I64,
        Dtype::F16 => DataType::F16,
        Dtype::BF16 => DataType::Bf16,
        Dtype::F32 => DataType::F32,
        Dtype::F64 => DataType::F64,
        other => panic!("a reference tensor of {other:?}"),
    }
}

impl Reference {
    #[must_use]
    pub fn load(directory: impl AsRef<Path>) -> Self {
        let directory = directory.as_ref();
        let read = |name: &str| {
            fs::read(directory.join(name)).unwrap_or_else(|error| {
                panic!(
                    "reading {}: {error}; `just prepare-tests` writes the references, and `just prepare-tests --real` the real ones",
                    directory.join(name).display()
                )
            })
        };
        let metadata = serde_json::from_slice(&read("reference.json")).unwrap();
        let mut tensors = HashMap::new();
        if directory.join("tensors.safetensors").exists() {
            let bytes = read("tensors.safetensors");
            let file = SafeTensors::deserialize(&bytes).unwrap();
            for (name, view) in file.tensors() {
                let dimensions = view
                    .shape()
                    .iter()
                    .map(|&extent| i64::try_from(extent).unwrap())
                    .collect::<Vec<_>>();
                let tensor = Tensor::from_bytes(
                    Device::Cpu,
                    data_type(view.dtype()),
                    Shape::create(dimensions).unwrap(),
                    view.data(),
                )
                .unwrap();
                tensors.insert(name, tensor);
            }
        }
        Self { tensors, metadata }
    }

    #[must_use]
    pub fn tensor(&self, name: &str, device: Device, data_type: DataType) -> Tensor {
        self.tensors
            .get(name)
            .unwrap_or_else(|| panic!("the reference has no tensor {name}"))
            .cast(data_type)
            .unwrap()
            .to_device(device)
            .unwrap()
    }

    #[must_use]
    pub fn values(&self, name: &str) -> Vec<f32> {
        host(&self.tensor(name, Device::Cpu, DataType::F32))
    }

    #[must_use]
    pub const fn metadata(&self) -> &Value {
        &self.metadata
    }

    #[must_use]
    pub fn half_gap(&self) -> HalfGap {
        let gap = &self.metadata["half_gap"];
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the gap is an F32 distance"
        )]
        let value = |key: &str| gap[key].as_f64().unwrap() as f32;
        HalfGap {
            worst: value("worst"),
            mean: value("mean"),
        }
    }

    #[must_use]
    pub fn stored(&self, name: &str) -> Tensor {
        self.tensors
            .get(name)
            .unwrap_or_else(|| panic!("the reference has no tensor {name}"))
            .clone()
    }

    #[must_use]
    pub fn resolved(&self) -> Value {
        self.resolve(&self.metadata)
    }

    fn resolve(&self, value: &Value) -> Value {
        match value {
            Value::Object(object) => {
                if let (1, Some(Value::String(name))) = (object.len(), object.get("tensor")) {
                    let tensor = self.stored(name);
                    let values = if tensor.data_type().is_float() {
                        host(&tensor)
                            .into_iter()
                            .map(|value| Value::from(f64::from(value)))
                            .collect()
                    } else {
                        tensor
                            .cast(DataType::I64)
                            .unwrap()
                            .to_vec::<i64>()
                            .unwrap()
                            .into_iter()
                            .map(Value::from)
                            .collect::<Vec<_>>()
                    };
                    return serde_json::json!({
                        "shape": tensor.shape().dimensions(),
                        "values": values,
                    });
                }
                Value::Object(
                    object
                        .iter()
                        .map(|(key, value)| (key.clone(), self.resolve(value)))
                        .collect(),
                )
            }
            Value::Array(items) => {
                Value::Array(items.iter().map(|item| self.resolve(item)).collect())
            }
            other => other.clone(),
        }
    }

    #[must_use]
    pub fn integers(&self, key: &str) -> Vec<i64> {
        integers(&self.metadata[key])
    }
}

#[must_use]
pub fn integers(value: &Value) -> Vec<i64> {
    value
        .as_array()
        .unwrap_or_else(|| panic!("{value} is not an array"))
        .iter()
        .map(|item| item.as_i64().unwrap())
        .collect()
}
