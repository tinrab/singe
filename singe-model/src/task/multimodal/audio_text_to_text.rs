// `audio-text-to-text`: a clip and an instruction to the text a speech model writes, here Qwen3-ASR's transcription with the instruction as its context, as its chat template places it in the system's turn.

use serde::{Deserialize, Serialize};
use singe::{Device, Shape, Tensor, generation::GenerationOptions};

use crate::{
    BatchOptions,
    error::Result,
    head::SpeechToText,
    model,
    processor::{audio::Audio, text::TextTokenizer},
    source::{LoadOptions, Source},
    task::{Pipeline, Task},
};

#[derive(Debug)]
pub struct AudioTextToTextPipeline {
    model: Box<dyn SpeechToText>,
    tokenizer: TextTokenizer,
    device: Device,
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AudioTextToTextOutput {
    pub generated_text: String,
}

impl Pipeline for AudioTextToTextPipeline {
    const TASK: Task = Task::AudioTextToText;

    type Input = AudioInstruction;

    type Parameters = AudioTextToTextParameters;

    type Output = AudioTextToTextOutput;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self {
            model: model::load(source, options)?,
            tokenizer: TextTokenizer::load(source.directory())?,
            device: options.device,
        })
    }

    fn run(
        &self,
        inputs: &[Self::Input],
        parameters: &Self::Parameters,
    ) -> Result<Vec<Self::Output>> {
        let options = BatchOptions::new(GenerationOptions {
            max_new_tokens: parameters.max_new_tokens,
            ..GenerationOptions::default()
        })
        .with_compile(parameters.compile);
        inputs
            .iter()
            .map(|input| {
                input.audio.expect_rate(self.model.sample_rate())?;
                let count = i64::try_from(input.audio.samples.len())
                    .map_err(|_| crate::Error::unsupported("a clip this long"))?;
                let samples = Tensor::from_vec(
                    self.device,
                    Shape::create([count])?,
                    input.audio.samples.clone(),
                )?;
                Ok(AudioTextToTextOutput {
                    generated_text: self.model.respond(
                        &self.tokenizer,
                        &samples,
                        &input.text,
                        &options,
                    )?,
                })
            })
            .collect()
    }
}
