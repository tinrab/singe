use serde::Deserialize;

use crate::{
    config::{DecoderConfig, unsupported},
    error::Result,
};

// The fields of a Hugging Face `config.json` this definition reads.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Qwen3Config {
    #[serde(flatten)]
    pub decoder: DecoderConfig,
    #[serde(default)]
    pub use_sliding_window: bool,
}

impl Qwen3Config {
    // A checkpoint using one of these would load and then compute something else.
    pub fn validate(&self) -> Result<()> {
        self.decoder.rotary.frequencies()?;
        self.decoder.refuse_sliding_layers()?;
        if self.use_sliding_window {
            return Err(unsupported("a sliding window"));
        }
        Ok(())
    }
}
