// `automatic-speech-recognition`: a clip of speech to the text it says.

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

// A model that writes text, or one whose frames' characters CTC decoding reads.
#[derive(Debug)]
enum Recognizer {
    Generative {
        model: Box<dyn SpeechToText>,
        tokenizer: Box<TextTokenizer>,
    },
    Ctc {
        model: CtcRecognizer,
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
    pub max_new_tokens: usize,
    // The clip's language, which a model that names it itself otherwise detects.
    pub language: Option<String>,
    pub compile: bool,
}

impl Default for AutomaticSpeechRecognitionParameters {
    fn default() -> Self {
        Self {
            max_new_tokens: 128,
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
        let model = if model::default_head(source)? == Head::Ctc {
            Recognizer::Ctc {
                model: model::load(source, options)?,
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
                clip.expect_rate(model.sample_rate())?;
                model.transcribe(
                    tokenizer,
                    &clip.to_tensor(self.options.device)?,
                    parameters.language.as_deref(),
                    &BatchOptions::new(parameters.max_new_tokens, parameters.compile),
                )
            }
            Recognizer::Ctc { model, vocabulary } => {
                clip.expect_rate(model.sample_rate())?;
                Ok(Transcription {
                    language: None,
                    text: vocabulary.decode(&model.best_tokens(&clip.samples)?)?,
                })
            }
        }
    }
}
