// Models that generate text one token at a time, Hugging Face's `text-generation`, each under its `model_type`.

pub mod deepseek_v3;
pub mod gemma2;
pub mod gpt2;
pub mod llama;
pub mod qwen3;

use std::path::Path;

use singe::{
    DataType, Device,
    generation::{Decoder, DecoderOptions},
};

use crate::{Error, Result, assembly, config::Snapshot};

// Loads the model a snapshot's `model_type` names, and returns a decoder over it, compiled if asked.
pub fn load_decoder(
    snapshot: impl AsRef<Path>,
    device: Device,
    data_type: DataType,
    options: DecoderOptions,
    compile: bool,
) -> Result<Box<dyn Decoder>> {
    let snapshot = snapshot.as_ref();
    let model_type = Snapshot::new(snapshot).model_type()?;
    match model_type.as_str() {
        "deepseek_v3" => assembly::decoder(
            &deepseek_v3::load(snapshot, device, data_type)?,
            options,
            compile,
        ),
        "gemma2" => assembly::decoder(
            &gemma2::load(snapshot, device, data_type)?,
            options,
            compile,
        ),
        "gpt2" => assembly::decoder(&gpt2::load(snapshot, device, data_type)?, options, compile),
        "llama" => assembly::decoder(&llama::load(snapshot, device, data_type)?, options, compile),
        "qwen3" => assembly::decoder(&qwen3::load(snapshot, device, data_type)?, options, compile),
        other => Err(Error::UnsupportedConfig(format!("model type {other}"))),
    }
}
