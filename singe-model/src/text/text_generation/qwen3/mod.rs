mod config;

use std::path::Path;

use singe::{DataType, Device, nn::Model};

pub use self::config::Qwen3Config;
use crate::{
    assembly::{DecoderSpec, GroupedDecoder, causal, grouped_decoder, load_weights},
    config::Snapshot,
    error::Result,
};

// Qwen3: grouped-query attention with normed queries and keys, and a gated feed-forward layer.
pub type Qwen3Layers = GroupedDecoder;

pub type Qwen3 = Model<Qwen3Layers>;

// Loads a Hugging Face snapshot directory, converting the weights to `data_type`.
//
// The GPU runs half precision on tensor cores, and every other float on a slower general kernel.
pub fn load(snapshot: impl AsRef<Path>, device: Device, data_type: DataType) -> Result<Qwen3> {
    let config: Qwen3Config = Snapshot::new(&snapshot).config()?;
    config.validate()?;
    let spec = DecoderSpec {
        head_norms: true,
        ..config.decoder.spec(data_type, false)?
    };
    load_weights(&grouped_decoder(&spec, causal())?, snapshot, device, None)
}
