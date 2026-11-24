//! `POST /v1/rerank`, in the shape Cohere, Jina, vLLM, and Text Embeddings Inference share.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// A request to `POST /v1/rerank`.
#[derive(Debug, Clone, Deserialize)]
pub struct RerankRequest {
    /// The served model's name, as `/v1/models` lists it.
    pub model: String,
    /// The text the documents are ranked against.
    pub query: String,
    /// The documents to rank, at least one.
    pub documents: Vec<RerankDocument>,
    /// Returns only this many of the best documents, or all of them when unset; it must not be zero.
    pub top_n: Option<usize>,
    /// Whether each result repeats its document's text, `false` by default.
    #[serde(default)]
    pub return_documents: bool,
    /// Fields the server does not read, which it logs and ignores.
    #[serde(flatten)]
    pub unknown: Map<String, Value>,
}

/// A document to rank, as a bare string or an object with a `text` field.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(untagged)]
pub enum RerankDocument {
    /// The document's text.
    Text(String),
    /// An object holding the document's text.
    Object {
        /// The document's text.
        text: String,
    },
}

impl RerankDocument {
    /// Returns the document's text, whichever shape it came in.
    #[must_use]
    pub fn text(&self) -> &str {
        match self {
            Self::Text(text) | Self::Object { text } => text,
        }
    }
}

/// The response to a rerank request.
#[derive(Debug, Clone, Serialize)]
pub struct RerankResponse {
    /// `rerank-` followed by the request's `X-Request-Id`, or by a new id when it has none.
    pub id: String,
    /// The served model's name.
    pub model: String,
    /// The ranked documents, most relevant first.
    pub results: Vec<RerankResult>,
    /// The tokens the ranking took.
    pub usage: RerankUsage,
}

/// One ranked document.
#[derive(Debug, Clone, Serialize)]
pub struct RerankResult {
    /// The document's position in the request's `documents`, from 0.
    pub index: usize,
    /// The cross-encoder's score for the document against the query; a higher score ranks first.
    pub relevance_score: f32,
    /// The document's text, present only when the request sets `return_documents`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub document: Option<RerankedText>,
}

/// A ranked document's text, as a [`RerankResult`] repeats it.
#[derive(Debug, Clone, Serialize)]
pub struct RerankedText {
    /// The document's text.
    pub text: String,
}

/// The tokens a rerank request took.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct RerankUsage {
    /// The query's tokens once per document plus every document's tokens, or 0 for a model without a tokenizer.
    pub total_tokens: usize,
}
