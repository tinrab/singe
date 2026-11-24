//! `POST /v1/embeddings`.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// A request to `POST /v1/embeddings`.
#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingRequest {
    /// The served model's name, as `/v1/models` lists it.
    pub model: String,
    /// The texts to embed, none of them empty.
    pub input: EmbeddingInput,
    /// How each vector is written, [`EncodingFormat::Float`] by default.
    pub encoding_format: Option<EncodingFormat>,
    /// Keeps the first this many dimensions of each vector, normalized again; it must not be zero.
    pub dimensions: Option<usize>,
    /// Accepted and ignored.
    pub user: Option<String>,
    /// Fields the server does not read, which it logs and ignores.
    #[serde(flatten)]
    pub unknown: Map<String, Value>,
}

/// The input an embedding request carries, in any of OpenAI's shapes.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    /// One text.
    Text(String),
    /// Several texts, embedded in order.
    Texts(Vec<String>),
    /// One text's token ids, which the server refuses.
    Tokens(Vec<i64>),
    /// Several texts' token ids, which the server refuses.
    TokenLists(Vec<Vec<i64>>),
}

/// How an embedding's vector is written in the response.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EncodingFormat {
    /// An array of numbers.
    #[default]
    Float,
    /// Each vector's little-endian `f32`s, in base64.
    Base64,
}

/// The response to an embedding request.
#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingList {
    /// Always `list`.
    pub object: &'static str,
    /// One embedding per input text, in the inputs' order.
    pub data: Vec<Embedding>,
    /// The served model's name.
    pub model: String,
    /// The tokens the inputs took.
    pub usage: EmbeddingUsage,
}

/// One input text's embedding.
#[derive(Debug, Clone, Serialize)]
pub struct Embedding {
    /// Always `embedding`.
    pub object: &'static str,
    /// The input's position in the request, from 0.
    pub index: usize,
    /// The vector, written as the request's [`EncodingFormat`] asks.
    pub embedding: EmbeddingVector,
}

/// An embedding's vector, as [`EncodingFormat`] writes it.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum EmbeddingVector {
    /// The vector's values.
    Float(Vec<f32>),
    /// The vector's little-endian `f32`s, in base64.
    Base64(String),
}

/// The tokens an embedding request took.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct EmbeddingUsage {
    /// The tokens of all inputs, special tokens included, or 0 for a model without a tokenizer.
    pub prompt_tokens: usize,
    /// The same count as `prompt_tokens`, since embedding generates nothing.
    pub total_tokens: usize,
}
