//! `POST /v2/models/{model}/infer` and `GET /v2/models/{model}`, in the `KServe` v2 inference protocol, for a model served without a task.

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct InferRequest {
    pub id: Option<String>,
    pub inputs: Vec<InferTensor>,
    #[serde(default)]
    pub outputs: Vec<RequestedOutput>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct RequestedOutput {
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InferTensor {
    pub name: String,
    pub shape: Vec<i64>,
    pub datatype: String,
    pub data: Vec<Value>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct InferResponse {
    pub model_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub outputs: Vec<InferTensor>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TensorMetadata {
    pub name: String,
    pub datatype: String,
    pub shape: Vec<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ModelMetadata {
    pub name: String,
    pub platform: &'static str,
    pub inputs: Vec<TensorMetadata>,
    pub outputs: Vec<TensorMetadata>,
}
