use std::path::Path;

use singe::{DataType, Device, nn::Model};

use crate::{
    assembly::{GroupedDecoder, causal, grouped_decoder, load_weights},
    config::{DecoderConfig, Snapshot},
    error::Result,
};

// Llama: multi-head attention over rotary positions and a gated feed-forward layer.
pub type LlamaLayers = GroupedDecoder;

pub type Llama = Model<LlamaLayers>;

// Loads a Hugging Face snapshot directory, converting the weights to `data_type`.
pub fn load(snapshot: impl AsRef<Path>, device: Device, data_type: DataType) -> Result<Llama> {
    let config: DecoderConfig = Snapshot::new(&snapshot).config()?;
    config.refuse_sliding_layers()?;
    let schema = grouped_decoder(&config.spec(data_type, false)?, causal())?;
    load_weights(&schema, snapshot, device, None)
}
