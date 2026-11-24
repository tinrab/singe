//! `text-to-audio`: text to speech, here from VITS, as `transformers`' text-to-speech pipeline gives it; its noise is Singe's own, seeded, so it matches `transformers`' only with the noise scales at zero.

use serde::Deserialize;

use crate::{
    error::Result,
    head::TextToSpeech,
    model,
    processor::text::CharacterTokenizer,
    source::{LoadOptions, Source},
    task::{Pipeline, outputs::GeneratedAudio},
};

#[derive(Debug)]
pub struct TextToAudioPipeline {
    model: TextToSpeech,
    tokenizer: CharacterTokenizer,
}

#[derive(Debug, Clone, Copy, PartialEq, Default, Deserialize)]
#[serde(default)]
pub struct TextToAudioParameters {
    /// How much noise varies the voice, or the model's own amount when `None`.
    pub noise_scale: Option<f32>,
    /// How much noise varies the durations, or the model's own amount when `None`.
    pub noise_scale_duration: Option<f32>,
    pub seed: u64,
}

impl Pipeline for TextToAudioPipeline {
    type Input = String;

    type Parameters = TextToAudioParameters;

    type Output = GeneratedAudio;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self {
            model: model::load(source, options)?,
            tokenizer: CharacterTokenizer::load(source)?,
        })
    }

    fn run_one(&self, text: &Self::Input, parameters: &Self::Parameters) -> Result<Self::Output> {
        let tokens = self.tokenizer.encode(text);
        Ok(GeneratedAudio {
            audio: self.model.speak(
                &tokens,
                (parameters.noise_scale, parameters.noise_scale_duration),
                parameters.seed,
            )?,
            sampling_rate: self.model.sample_rate(),
        })
    }
}
