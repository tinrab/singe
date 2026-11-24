//! `audio-to-audio`: a clip to another clip, here a codec's reconstruction of it from its codes.

use crate::{
    error::Result,
    head::AudioCodec,
    model,
    processor::audio::Audio,
    source::{LoadOptions, Source},
    task::{Pipeline, outputs::GeneratedAudio},
};

#[derive(Debug)]
pub struct AudioToAudioPipeline {
    model: AudioCodec,
}

impl Pipeline for AudioToAudioPipeline {
    type Input = Audio;

    type Parameters = ();

    type Output = GeneratedAudio;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self {
            model: model::load(source, options)?,
        })
    }

    fn run_one(&self, clip: &Self::Input, (): &Self::Parameters) -> Result<Self::Output> {
        let clip = clip.at_rate(self.model.sample_rate())?;
        Ok(GeneratedAudio {
            audio: self.model.reconstruct(&clip.samples)?,
            sampling_rate: self.model.sample_rate(),
        })
    }
}
