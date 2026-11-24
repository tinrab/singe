use std::fmt::Debug;

use serde::Serialize;
use singe::{Tensor, generation::GenerationOptions};

use crate::{
    BatchOptions,
    error::{Error, Result},
    head::{LoopLimits, LoopSteps},
    processor::text::TextTokenizer,
};

/// What a transcription says: the language the model names, if any, and the text.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Transcription {
    pub language: Option<String>,
    pub text: String,
}

/// A model that writes the text a clip of speech says.
pub trait SpeechToText: Debug {
    /// Returns the samples per second the model hears.
    fn sample_rate(&self) -> i64;

    /// Returns the most tokens a transcription generates when its caller sets no limit, or `None` where decoding ends with the clip.
    /// The default is 128.
    fn default_max_new_tokens(&self) -> Option<usize> {
        Some(128)
    }

    /// Transcribes one clip of samples, `[N]`, generating as `options` asks.
    /// `language` names the clip's language when known.
    fn transcribe(
        &self,
        tokenizer: &TextTokenizer,
        samples: &Tensor,
        language: Option<&str>,
        options: &BatchOptions,
    ) -> Result<Transcription>;

    /// Returns what the model says of one clip, `[N]`, under `instruction`, placed as the snapshot's chat template places it, such as Qwen3-ASR's context.
    /// The default returns an error, for a model that takes no instruction.
    fn respond(
        &self,
        _tokenizer: &TextTokenizer,
        _samples: &Tensor,
        _instruction: &str,
        _options: &BatchOptions,
    ) -> Result<String> {
        Err(Error::unsupported("an instruction beside the clip"))
    }

    /// Returns the steps an exported transcription loop runs within `limits`: an encode step that reads the clip, and the decoder's steps, choosing tokens as `generation` asks.
    /// Whisper keeps only `generation`'s sampling and otherwise chooses as its own generation config says; a transducer returns an error.
    fn loop_steps(&self, limits: &LoopLimits, generation: &GenerationOptions) -> Result<LoopSteps>;
}
