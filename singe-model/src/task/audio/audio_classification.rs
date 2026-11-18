// `audio-classification`: a clip to its labels' scores, best first, as `transformers`' pipeline ranks them.

use crate::{
    error::Result,
    head::AudioClassifier,
    model,
    processing::audio::Audio,
    source::{LoadOptions, Source},
    task::{
        Pipeline, Task,
        vision::image_classification::{ClassificationParameters, LabelScore, ranked},
    },
};

#[derive(Debug)]
pub struct AudioClassificationPipeline {
    model: AudioClassifier,
}

impl Pipeline for AudioClassificationPipeline {
    const TASK: Task = Task::AudioClassification;

    type Input = Audio;

    type Parameters = ClassificationParameters;

    type Output = Vec<LabelScore>;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self {
            model: model::load(source, options)?,
        })
    }

    fn run(
        &self,
        inputs: &[Self::Input],
        parameters: &Self::Parameters,
    ) -> Result<Vec<Self::Output>> {
        let config = self.model.classifier.config();
        let function = parameters.function_to_apply.unwrap_or(config.function);
        inputs
            .iter()
            .map(|clip| {
                clip.expect_rate(self.model.features.sample_rate)?;
                let scores = self.model.scores(&clip.samples)?;
                Ok(ranked(&scores, &config.labels, function, parameters.top_k))
            })
            .collect()
    }
}
