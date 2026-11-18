// `text-to-audio`: text to speech, here from VITS, as `transformers`' text-to-speech pipeline gives it; its noise is Singe's own, seeded, so it matches `transformers`' only with the noise scales at zero.

use serde::Deserialize;

use crate::{
    error::Result,
    head::TextToSpeech,
    model,
    processing::text::CharacterTokenizer,
    source::{LoadOptions, Source},
    task::{Pipeline, Task, audio::audio_to_audio::GeneratedAudio},
};

#[derive(Debug)]
pub struct TextToAudioPipeline {
    model: TextToSpeech,
    tokenizer: CharacterTokenizer,
}

#[derive(Debug, Clone, Copy, PartialEq, Default, Deserialize)]
#[serde(default)]
pub struct TextToAudioParameters {
    // How much noise varies the voice and the durations, or the model's own amounts.
    pub noise_scale: Option<f32>,
    pub noise_scale_duration: Option<f32>,
    pub seed: u64,
}

impl Pipeline for TextToAudioPipeline {
    const TASK: Task = Task::TextToAudio;

    type Input = String;

    type Parameters = TextToAudioParameters;

    type Output = GeneratedAudio;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self {
            model: model::load(source, options)?,
            tokenizer: CharacterTokenizer::load(source.directory())?,
        })
    }

    fn run(
        &self,
        inputs: &[Self::Input],
        parameters: &Self::Parameters,
    ) -> Result<Vec<Self::Output>> {
        inputs
            .iter()
            .map(|text| {
                let tokens = self.tokenizer.encode(text);
                Ok(GeneratedAudio {
                    audio: self.model.speak(
                        &tokens,
                        (parameters.noise_scale, parameters.noise_scale_duration),
                        parameters.seed,
                    )?,
                    sampling_rate: self.model.speech().sample_rate,
                })
            })
            .collect()
    }
}
