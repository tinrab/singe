//! `audio-classification`: a clip to its labels' scores, best first, as `transformers`' pipeline ranks them.

use crate::{
    error::Result,
    head::AudioClassifier,
    model,
    processor::audio::Audio,
    source::{LoadOptions, Source},
    task::{
        Pipeline,
        ranking::{ClassificationParameters, LabelScore, ranked},
    },
};

#[derive(Debug)]
pub struct AudioClassificationPipeline {
    model: AudioClassifier,
}

impl Pipeline for AudioClassificationPipeline {
    type Input = Audio;

    type Parameters = ClassificationParameters;

    type Output = Vec<LabelScore>;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self {
            model: model::load_served(source, options)?,
        })
    }

    fn run_one(&self, clip: &Self::Input, parameters: &Self::Parameters) -> Result<Self::Output> {
        let config = self.model.classifier.config();
        let clip = clip.at_rate(self.model.sample_rate())?;
        let scores = self.model.scores(&clip.samples)?;
        Ok(ranked(
            &scores,
            config.labels(),
            parameters
                .function_to_apply
                .unwrap_or_else(|| config.function()),
            parameters.top_k,
        ))
    }
}
