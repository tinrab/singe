//! `POST /tokenize` and `POST /detokenize`, as vLLM serves them.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::api::chat::ChatMessage;

/// A request to `POST /tokenize`, which sets exactly one of `prompt` and `messages`.
#[derive(Debug, Clone, Deserialize)]
pub struct TokenizeRequest {
    /// The served model's name, as `/v1/models` lists it.
    pub model: String,
    /// A text to tokenize as it is.
    pub prompt: Option<String>,
    /// A conversation to render with the model's chat template and then tokenize; only a text-generation model takes it.
    pub messages: Option<Vec<ChatMessage>>,
    /// Whether a `prompt` gets the tokenizer's special tokens, `true` by default.
    pub add_special_tokens: Option<bool>,
    /// Whether rendered `messages` end with the template's assistant turn opening, `true` by default.
    pub add_generation_prompt: Option<bool>,
    /// Extra variables the chat template reads when it renders `messages`.
    #[serde(default)]
    pub chat_template_kwargs: Map<String, Value>,
}

/// The response to a tokenize request.
#[derive(Debug, Clone, Serialize)]
pub struct TokenizeResponse {
    /// How many tokens there are.
    pub count: usize,
    /// The most tokens a sequence holds, prompt and generated together; only a text-generation model has it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_model_len: Option<usize>,
    /// The token ids.
    pub tokens: Vec<i64>,
}

/// A request to `POST /detokenize`.
#[derive(Debug, Clone, Deserialize)]
pub struct DetokenizeRequest {
    /// The served model's name, as `/v1/models` lists it.
    pub model: String,
    /// The token ids to decode.
    pub tokens: Vec<i64>,
}

/// The response to a detokenize request.
#[derive(Debug, Clone, Serialize)]
pub struct DetokenizeResponse {
    /// The decoded text, special tokens included.
    pub prompt: String,
}
