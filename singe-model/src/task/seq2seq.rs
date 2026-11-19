use serde::Deserialize;
use serde_json::Value;

use crate::{error::Result, source::Source};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(default)]
pub struct Seq2SeqParameters {
    // The most tokens to generate, or as many as the snapshot's generation config allows.
    pub max_new_tokens: Option<usize>,
    // Runs compiled steps rather than eager ones.
    pub compile: bool,
}

impl Default for Seq2SeqParameters {
    fn default() -> Self {
        Self {
            max_new_tokens: None,
            compile: true,
        }
    }
}

// `config.json`'s `task_specific_params`, which give each task's settings, such as T5's prefix `summarize: `.
#[derive(Debug, Clone)]
pub(crate) struct TaskPrefixes(Value);

impl TaskPrefixes {
    pub(crate) fn load(source: &Source) -> Result<Self> {
        let config: Value = source.json("config.json")?;
        Ok(Self(config["task_specific_params"].clone()))
    }

    pub(crate) fn prefix(&self, task: &str) -> &str {
        self.0[task]["prefix"].as_str().unwrap_or_default()
    }
}
