// Models that turn speech into text, Hugging Face's `automatic-speech-recognition`, each under its `model_type`.

pub mod qwen3_asr;
pub mod whisper;

use std::path::Path;

use singe::{DataType, Device, Tensor};

pub use self::qwen3_asr::Transcription;
use self::{
    qwen3_asr::{Qwen3Asr, parse_transcription},
    whisper::{TranscribeOptions, Whisper, WhisperTask, generation_options, prompt},
};
use crate::{Error, Result, config::Snapshot, tokenization::TextTokenizer};

// The model a [`Transcriber`] runs.
#[derive(Debug, Clone)]
pub enum TranscriberModel {
    Whisper(Box<Whisper>),
    Qwen3Asr(Box<Qwen3Asr>),
}

// A speech model with its snapshot's tokenizer, which transcribes a clip to text.
#[derive(Debug, Clone)]
pub struct Transcriber {
    model: TranscriberModel,
    tokenizer: TextTokenizer,
    snapshot: Snapshot,
}

impl Transcriber {
    #[must_use]
    pub const fn model(&self) -> &TranscriberModel {
        &self.model
    }

    // Transcribes one clip of 16 kHz samples, `[N]`, generating greedily up to `max_new_tokens` tokens, compiled if asked.
    //
    // Whisper transcribes in `language`, English if it names none; Qwen3-ASR names the language itself unless `language`, its full English name, forces one.
    pub fn transcribe(
        &self,
        samples: &Tensor,
        language: Option<&str>,
        max_new_tokens: usize,
        compile: bool,
    ) -> Result<Transcription> {
        let generation = generation_options(self.snapshot.directory(), max_new_tokens)?;
        let options = TranscribeOptions::new(generation).with_compile(compile);
        match &self.model {
            TranscriberModel::Whisper(model) => {
                let language = language.unwrap_or("en");
                let prefix = prompt(&self.tokenizer, language, WhisperTask::Transcribe)?;
                let tokens = model.transcribe(samples, &prefix, &options)?;
                Ok(Transcription {
                    language: Some(language.to_owned()),
                    text: self.tokenizer.decode(&tokens, true)?.trim().to_owned(),
                })
            }
            TranscriberModel::Qwen3Asr(model) => {
                let tokens = model.transcribe(&self.tokenizer, samples, language, &options)?;
                let output = self.tokenizer.decode(&tokens, true)?;
                // A forced language is part of the prompt, so the output is the text alone.
                Ok(language.map_or_else(
                    || parse_transcription(&output),
                    |language| Transcription {
                        language: Some(language.to_owned()),
                        text: output.trim().to_owned(),
                    },
                ))
            }
        }
    }
}

// Loads the transcriber a snapshot's `model_type` names, with its tokenizer.
pub fn load_transcriber(
    snapshot: impl AsRef<Path>,
    device: Device,
    data_type: DataType,
) -> Result<Transcriber> {
    let snapshot = Snapshot::new(snapshot);
    let model = match snapshot.model_type()?.as_str() {
        "whisper" => TranscriberModel::Whisper(Box::new(whisper::load(
            snapshot.directory(),
            device,
            data_type,
        )?)),
        "qwen3_asr" => TranscriberModel::Qwen3Asr(Box::new(qwen3_asr::load(
            snapshot.directory(),
            device,
            data_type,
        )?)),
        other => return Err(Error::UnsupportedConfig(format!("model type {other}"))),
    };
    Ok(Transcriber {
        model,
        tokenizer: TextTokenizer::load(snapshot.directory())?,
        snapshot,
    })
}
