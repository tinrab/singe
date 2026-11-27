//! `audio-to-audio`: a clip to another clip: a codec's reconstruction of it from its codes, or a GAN vocoder's resynthesis of it from its spectrogram and seeded noise.

use serde::Deserialize;

use crate::{
    error::Result,
    head::{AudioCodec, Head, Vocoder},
    model,
    processor::audio::Audio,
    source::{LoadOptions, Source},
    task::{Pipeline, outputs::GeneratedAudio},
};

#[derive(Debug)]
enum AudioModel {
    Codec(AudioCodec),
    Vocoder(Vocoder),
}

#[derive(Debug)]
pub struct AudioToAudioPipeline {
    model: AudioModel,
}

#[derive(Debug, Clone, Copy, PartialEq, Default, Deserialize)]
#[serde(default)]
pub struct AudioToAudioParameters {
    /// The seed of a vocoder's noise.
    pub seed: u64,
    /// The kilobits a second an `EnCodec` codes at, one its config lists, or its smallest when `None`.
    pub bandwidth: Option<f64>,
}

impl Pipeline for AudioToAudioPipeline {
    type Input = Audio;

    type Parameters = AudioToAudioParameters;

    type Output = GeneratedAudio;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        let model = if model::default_head(source)? == Head::Vocoder {
            AudioModel::Vocoder(model::load(source, options)?)
        } else {
            AudioModel::Codec(model::load(source, options)?)
        };
        Ok(Self { model })
    }

    fn run_one(&self, clip: &Self::Input, parameters: &Self::Parameters) -> Result<Self::Output> {
        let (audio, sampling_rate) = match &self.model {
            AudioModel::Codec(model) => {
                let clip = clip.at_rate(model.sample_rate())?;
                (
                    model.reconstruct(&clip.samples, parameters.bandwidth)?,
                    model.sample_rate(),
                )
            }
            AudioModel::Vocoder(model) => {
                let clip = clip.at_rate(model.sample_rate())?;
                (
                    model.resynthesize(&clip.samples, parameters.seed)?,
                    model.sample_rate(),
                )
            }
        };
        Ok(GeneratedAudio {
            audio,
            sampling_rate,
        })
    }
}
