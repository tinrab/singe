use std::fmt::Debug;

use singe::{Tensor, generation::GenerationOptions};

use super::{LoopLimits, LoopSteps};
use crate::{BatchOptions, error::Result, processing::text::TextTokenizer};

// What a transcription says: the language the model names, if any, and the text.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct Transcription {
    pub language: Option<String>,
    pub text: String,
}

// A model that writes the text a clip of speech says.
pub trait SpeechToText: Debug {
    // The samples per second the model hears.
    fn sample_rate(&self) -> i64;

    // Transcribes one clip of samples, `[N]`, generating as `options` asks; `language` names the clip's language when known.
    fn transcribe(
        &self,
        tokenizer: &TextTokenizer,
        samples: &Tensor,
        language: Option<&str>,
        options: &BatchOptions,
    ) -> Result<Transcription>;

    // Writes what the model says of one clip, `[N]`, under `instruction`, as the snapshot's chat template places it, such as Qwen3-ASR's context; a model that takes no instruction refuses.
    fn respond(
        &self,
        _tokenizer: &TextTokenizer,
        _samples: &Tensor,
        _instruction: &str,
        _options: &BatchOptions,
    ) -> Result<String> {
        Err(crate::Error::unsupported("an instruction beside the clip"))
    }

    // The steps an exported transcription loop runs within `limits`: the encode step, which reads samples, and the decoder's steps, choosing as `generation` asks beside the model's own suppressed tokens.
    fn loop_steps(&self, limits: &LoopLimits, generation: &GenerationOptions) -> Result<LoopSteps>;
}
