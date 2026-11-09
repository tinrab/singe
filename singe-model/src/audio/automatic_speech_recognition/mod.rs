// Models that turn speech into text, Hugging Face's `automatic-speech-recognition`, each under its `model_type`.

pub mod whisper;

use std::path::Path;

use singe::{DataType, Device};

use crate::{Error, Result, config::Snapshot};

// Loads the transcriber a snapshot's `model_type` names.
pub fn load_transcriber(
    snapshot: impl AsRef<Path>,
    device: Device,
    data_type: DataType,
) -> Result<whisper::Whisper> {
    let snapshot = snapshot.as_ref();
    match Snapshot::new(snapshot).model_type()?.as_str() {
        "whisper" => whisper::load(snapshot, device, data_type),
        other => Err(Error::UnsupportedConfig(format!("model type {other}"))),
    }
}
