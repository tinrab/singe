//! The wire types: OpenAI's requests and responses where OpenAI defines an endpoint, the reranking shape Cohere, Jina, and vLLM share, and Singe's own for the task API.
//! Requests read leniently: optional fields default, and fields the server does not know are collected so a handler can log them by name.

pub mod audio;
pub mod chat;
pub mod completion;
pub mod embedding;
pub mod image;
pub mod model;
pub mod rerank;
pub mod task;
pub mod tokenize;

use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// A response's token counts, as chat and text completions report them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
pub struct Usage {
    /// Tokens in the prompts, summed over the choices.
    pub prompt_tokens: usize,
    /// Tokens generated, summed over the choices.
    pub completion_tokens: usize,
    /// The sum of `prompt_tokens` and `completion_tokens`.
    pub total_tokens: usize,
}

impl Usage {
    /// Creates the counts, with `total_tokens` as their sum.
    #[must_use]
    pub const fn new(prompt_tokens: usize, completion_tokens: usize) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        }
    }
}

/// The `stop` field: text that ends generation where it appears, as one string or several.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(untagged)]
pub enum Stop {
    /// One stop string, given as a JSON string.
    One(String),
    /// Several stop strings, given as a JSON array.
    Many(Vec<String>),
}

impl Stop {
    /// Returns the stop strings as a list, empty when the request gives none.
    #[must_use]
    pub fn strings(stop: Option<&Self>) -> Vec<String> {
        match stop {
            None => Vec::new(),
            Some(Self::One(stop)) => vec![stop.clone()],
            Some(Self::Many(stops)) => stops.clone(),
        }
    }
}

/// Returns the seconds since the Unix epoch, as responses' `created` gives them.
/// Returns 0 when the system clock reads before the epoch.
#[must_use]
pub fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs())
}

/// Logs a warning naming `endpoint` and the fields of a request the server does not read.
/// Logs nothing when `fields` is empty.
pub fn warn_unknown(endpoint: &str, fields: &Map<String, Value>) {
    if !fields.is_empty() {
        let names = fields.keys().map(String::as_str).collect::<Vec<_>>();
        tracing::warn!(endpoint, fields = ?names, "ignored unknown request fields");
    }
}
