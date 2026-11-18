// `image-classification`: an image to its labels' scores, best first, as `transformers`' pipeline ranks them.

use std::slice;

use serde::{Deserialize, Serialize};
use singe::DataType;

use crate::{
    error::Result,
    head::{ImageClassifier, ScoreFunction},
    model,
    processing::image::{Image, ImageTransforms},
    source::{LoadOptions, Source},
    task::{Pipeline, Task},
};

#[derive(Debug)]
pub struct ImageClassificationPipeline {
    model: ImageClassifier,
    processor: ImageTransforms,
    data_type: DataType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(default)]
pub struct ClassificationParameters {
    // The labels to keep from the best, at most every one.
    pub top_k: usize,
    // How scores become what the output reports, or the model's own choice when absent.
    pub function_to_apply: Option<ScoreFunction>,
}

impl Default for ClassificationParameters {
    fn default() -> Self {
        Self {
            top_k: 5,
            function_to_apply: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LabelScore {
    pub score: f32,
    pub label: String,
}

// Each label's score after `function`, best first, the first `top_k`.
#[must_use]
pub fn ranked(
    scores: &[f32],
    labels: &[String],
    function: ScoreFunction,
    top_k: usize,
) -> Vec<LabelScore> {
    let mut ranked = function
        .apply(scores)
        .into_iter()
        .zip(labels)
        .map(|(score, label)| LabelScore {
            score,
            label: label.clone(),
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| right.score.total_cmp(&left.score));
    ranked.truncate(top_k);
    ranked
}

impl Pipeline for ImageClassificationPipeline {
    const TASK: Task = Task::ImageClassification;

    type Input = Image;

    type Parameters = ClassificationParameters;

    type Output = Vec<LabelScore>;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self {
            model: model::load(source, options)?,
            processor: ImageTransforms::load(source)?,
            data_type: options.data_type,
        })
    }

    fn run(
        &self,
        inputs: &[Self::Input],
        parameters: &Self::Parameters,
    ) -> Result<Vec<Self::Output>> {
        let classifier = &self.model.0;
        let device = classifier.model().state().device()?;
        let config = classifier.config();
        let function = parameters.function_to_apply.unwrap_or(config.function);
        inputs
            .iter()
            .map(|image| {
                let pixels =
                    self.processor
                        .preprocess(slice::from_ref(image), device, self.data_type)?;
                let scores = classifier.scores(&pixels)?;
                Ok(ranked(
                    &scores[0],
                    &config.labels,
                    function,
                    parameters.top_k,
                ))
            })
            .collect()
    }
}
