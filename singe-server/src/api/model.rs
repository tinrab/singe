//! `GET /v1/models` and `GET /v1/models/{model}`, with what Singe adds: each model's task, endpoints, and readiness.

use serde::Serialize;

/// One served model, as OpenAI's model object describes it with Singe's fields added.
#[derive(Debug, Clone, Serialize)]
pub struct ModelObject {
    /// The model's name, which requests give as their `model`.
    pub id: String,
    /// Always `model`.
    pub object: &'static str,
    /// When the server registered the model, in seconds since the Unix epoch.
    pub created: u64,
    /// Always `singe`.
    pub owned_by: &'static str,
    /// The Hugging Face task id the model serves, such as `text-generation`.
    pub task: String,
    /// The paths of the endpoints that serve the model's task.
    pub endpoints: Vec<&'static str>,
    /// The most tokens a sequence holds, prompt and generated together; only a text-generation model has it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_model_len: Option<usize>,
    /// Whether the model has loaded and takes requests.
    pub ready: bool,
}

/// The response to `GET /v1/models`.
#[derive(Debug, Clone, Serialize)]
pub struct ModelList {
    /// Always `list`.
    pub object: &'static str,
    /// Every served model.
    pub data: Vec<ModelObject>,
}
