//! `automatic-speech-recognition`: a clip of speech to the text it says.

use serde::Deserialize;

use crate::{
    BatchOptions,
    error::Result,
    head::{CtcRecognizer, Head, SpeechToText, Transcription},
    model,
    processor::{
        audio::Audio,
        text::{CtcTokenizer, TextTokenizer},
    },
    source::{LoadOptions, Source},
    task::Pipeline,
};

/// A model that writes text, or a CTC recognizer whose frames' best tokens CTC decoding reads.
#[derive(Debug)]
enum Recognizer {
    Generative {
        model: Box<dyn SpeechToText>,
        tokenizer: Box<TextTokenizer>,
    },
    Ctc {
        model: Box<CtcRecognizer>,
        vocabulary: CtcTokenizer,
    },
}

#[derive(Debug)]
pub struct AutomaticSpeechRecognitionPipeline {
    model: Recognizer,
    options: LoadOptions,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(default)]
pub struct AutomaticSpeechRecognitionParameters {
    /// The tokens a transcription generates at most, or the model's own limit, if it has one, when `None`.
    pub max_new_tokens: Option<usize>,
    /// The clip's language, or `None` to leave it to the model: Qwen3-ASR detects it, Whisper assumes English, and transducers and CTC models ignore it.
    pub language: Option<String>,
    pub compile: bool,
}

impl Default for AutomaticSpeechRecognitionParameters {
    fn default() -> Self {
        Self {
            max_new_tokens: None,
            language: None,
            compile: true,
        }
    }
}

impl Pipeline for AutomaticSpeechRecognitionPipeline {
    type Input = Audio;

    type Parameters = AutomaticSpeechRecognitionParameters;

    type Output = Transcription;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        // An exported program or an ONNX file is a single pass, which only a CTC recognizer is.
        let model = if source.function().is_some() || model::default_head(source)? == Head::Ctc {
            Recognizer::Ctc {
                model: Box::new(model::load_served(source, options)?),
                vocabulary: CtcTokenizer::load(source)?,
            }
        } else {
            Recognizer::Generative {
                model: model::load(source, options)?,
                tokenizer: Box::new(TextTokenizer::load(source)?),
            }
        };
        Ok(Self { model, options })
    }

    fn run_one(&self, clip: &Self::Input, parameters: &Self::Parameters) -> Result<Self::Output> {
        match &self.model {
            Recognizer::Generative { model, tokenizer } => {
                let clip = clip.at_rate(model.sample_rate())?;
                let max_new_tokens = parameters
                    .max_new_tokens
                    .or_else(|| model.default_max_new_tokens())
                    .unwrap_or(usize::MAX);
                model.transcribe(
                    tokenizer,
                    &clip.to_tensor(self.options.device)?,
                    parameters.language.as_deref(),
                    &BatchOptions::new(max_new_tokens, parameters.compile),
                )
            }
            Recognizer::Ctc { model, vocabulary } => {
                let clip = clip.at_rate(model.sample_rate())?;
                Ok(Transcription {
                    language: None,
                    text: vocabulary.decode(&model.best_tokens(&clip.samples)?)?,
                })
            }
        }
    }
}
