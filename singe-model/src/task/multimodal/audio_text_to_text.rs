// `audio-text-to-text`: a clip and an instruction to the text a speech model writes, here Qwen3-ASR's transcription with the instruction as its context, as its chat template places it in the system's turn.

use serde::Deserialize;

use crate::{
    BatchOptions,
    error::Result,
    head::SpeechToText,
    model,
    processor::{audio::Audio, text::TextTokenizer},
    source::{LoadOptions, Source},
    task::{Pipeline, outputs::GeneratedText},
};

#[derive(Debug)]
pub struct AudioTextToTextPipeline {
    model: Box<dyn SpeechToText>,
    tokenizer: TextTokenizer,
    options: LoadOptions,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AudioInstruction {
    pub audio: Audio,
    pub text: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(default)]
pub struct AudioTextToTextParameters {
    pub max_new_tokens: usize,
    pub compile: bool,
}

impl Default for AudioTextToTextParameters {
    fn default() -> Self {
        Self {
            max_new_tokens: 128,
            compile: true,
        }
    }
}

impl Pipeline for AudioTextToTextPipeline {
    type Input = AudioInstruction;

    type Parameters = AudioTextToTextParameters;

    type Output = GeneratedText;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self {
            model: model::load(source, options)?,
            tokenizer: TextTokenizer::load(source)?,
            options,
        })
    }

    fn run_one(&self, input: &Self::Input, parameters: &Self::Parameters) -> Result<Self::Output> {
        input.audio.expect_rate(self.model.sample_rate())?;
        Ok(GeneratedText {
            generated_text: self.model.respond(
                &self.tokenizer,
                &input.audio.to_tensor(self.options.device)?,
                &input.text,
                &BatchOptions::new(parameters.max_new_tokens, parameters.compile),
            )?,
        })
    }
}
