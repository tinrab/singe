// Reads a `transformers` Llama config.

use crate::{
    assembly::DecoderConfig,
    error::Result,
    format::transformers::{DecoderDefaults, DecoderFields},
    source::Source,
};

pub(crate) fn config(source: &Source) -> Result<DecoderConfig> {
    let fields: DecoderFields = source.json("config.json")?;
    fields.refuse_sliding_layers()?;
    fields.config(DecoderDefaults {
        tied: false,
        head_dim: None,
    })
}
