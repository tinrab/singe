//! Reads a `transformers` Qwen3 config.

use serde::Deserialize;

use crate::{
    assembly::DecoderConfig,
    error::{Error, Result},
    format::transformers::{DecoderDefaults, DecoderFields},
    source::Source,
};

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Fields {
    #[serde(flatten)]
    decoder: DecoderFields,
    #[serde(default)]
    use_sliding_window: bool,
}

pub(crate) fn config(source: &Source) -> Result<DecoderConfig> {
    let fields: Fields = source.json("config.json")?;
    fields.decoder.refuse_sliding_layers()?;
    if fields.use_sliding_window {
        return Err(Error::unsupported("a sliding window"));
    }
    Ok(DecoderConfig {
        head_norms: true,
        ..fields.decoder.config(DecoderDefaults {
            tied: false,
            head_dim: Some(128),
        })?
    })
}
