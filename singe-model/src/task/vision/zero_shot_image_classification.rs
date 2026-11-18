// `zero-shot-image-classification`: an image and candidate labels to each label's score, best first: the softmax of the image's scaled cosines with each label's hypothesis, as `transformers`' pipeline scores them with a CLIP-style model.

use std::slice;

use serde::Deserialize;
use singe::DataType;

use crate::{
    error::Result,
    head::{ContrastiveEncoder, ScoreFunction},
    model,
    processing::{
        image::{Image, ImageTransforms},
        text::TextTokenizer,
    },
    source::{LoadOptions, Source},
    task::{
        Pipeline, Task, hypotheses,
        vision::image_classification::{LabelScore, ranked},
    },
};

#[derive(Debug)]
pub struct ZeroShotImageClassificationPipeline {
    model: ContrastiveEncoder,
    processor: ImageTransforms,
    tokenizer: TextTokenizer,
    data_type: DataType,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ZeroShotImageInput {
    pub image: Image,
    pub candidate_labels: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(default)]
pub struct ZeroShotImageClassificationParameters {
    // Where a label goes in its hypothesis.
    pub hypothesis_template: String,
}

impl Default for ZeroShotImageClassificationParameters {
    fn default() -> Self {
        Self {
            hypothesis_template: "This is a photo of {}.".to_owned(),
        }
    }
}

impl Pipeline for ZeroShotImageClassificationPipeline {
    const TASK: Task = Task::ZeroShotImageClassification;

    type Input = ZeroShotImageInput;

    type Parameters = ZeroShotImageClassificationParameters;

    type Output = Vec<LabelScore>;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self {
            model: model::load(source, options)?,
            processor: ImageTransforms::load(source)?,
            tokenizer: TextTokenizer::load(source.directory())?,
            data_type: options.data_type,
        })
    }

    fn run(
        &self,
        inputs: &[Self::Input],
        parameters: &Self::Parameters,
    ) -> Result<Vec<Self::Output>> {
        let device = self.model.inputs().state().device()?;
        inputs
            .iter()
            .map(|input| {
                let hypotheses =
                    hypotheses(&parameters.hypothesis_template, &input.candidate_labels)?;
                let texts = hypotheses.iter().map(String::as_str).collect::<Vec<_>>();
                let tokens = self.tokenizer.encode_batch(&texts, None, device)?;
                let pixels = self.processor.preprocess(
                    slice::from_ref(&input.image),
                    device,
                    self.data_type,
                )?;
                let logits = self.model.logits(&pixels, &tokens)?;
                Ok(ranked(
                    &logits[0],
                    &input.candidate_labels,
                    ScoreFunction::Softmax,
                    usize::MAX,
                ))
            })
            .collect()
    }
}
