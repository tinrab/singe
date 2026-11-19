// `audio-to-audio`: a clip to another clip, here a codec's reconstruction of it from its codes.

use serde::Serialize;

use crate::{
    error::Result,
    head::AudioCodec,
    model,
    processor::audio::Audio,
    source::{LoadOptions, Source},
    task::{Pipeline, Task},
};

#[derive(Debug)]
pub struct AudioToAudioPipeline {
    model: AudioCodec,
}

// A clip's samples and their rate, as `transformers`' audio pipelines give them.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GeneratedAudio {
    pub audio: Vec<f32>,
    pub sampling_rate: i64,
}

impl Pipeline for AudioToAudioPipeline {
    const TASK: Task = Task::AudioToAudio;

    type Input = Audio;

    type Parameters = ();

    type Output = GeneratedAudio;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self {
            model: model::load(source, options)?,
        })
    }

    fn run(&self, inputs: &[Self::Input], (): &Self::Parameters) -> Result<Vec<Self::Output>> {
        inputs
            .iter()
            .map(|clip| {
                clip.expect_rate(self.model.sample_rate())?;
                Ok(GeneratedAudio {
                    audio: self.model.reconstruct(&clip.samples)?,
                    sampling_rate: self.model.sample_rate(),
                })
            })
            .collect()
    }
}
