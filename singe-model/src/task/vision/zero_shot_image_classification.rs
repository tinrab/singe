//! `zero-shot-image-classification`: an image and candidate labels to each label's score, best first: the softmax of the image's scaled cosines with each label's hypothesis, as `transformers`' pipeline scores them with a CLIP-style model, or each one's sigmoid with `SigLIP`'s bias added.

use serde::Deserialize;

use crate::{
    error::Result,
    head::ContrastiveEncoder,
    model,
    processor::{image::ImageTransforms, text::TextTokenizer},
    source::{LoadOptions, Source},
    task::{
        Pipeline, hypotheses,
        inputs::ImageCandidates,
        ranking::{LabelScore, ranked},
    },
};

#[derive(Debug)]
pub struct ZeroShotImageClassificationPipeline {
    model: ContrastiveEncoder,
    processor: ImageTransforms,
    tokenizer: TextTokenizer,
    options: LoadOptions,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(default)]
pub struct ZeroShotImageClassificationParameters {
    /// The hypothesis a label goes in, in place of its `{}`; `This is a photo of {}.` by default.
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
    type Input = ImageCandidates;

    type Parameters = ZeroShotImageClassificationParameters;

    type Output = Vec<LabelScore>;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self {
            model: model::load(source, options)?,
            processor: ImageTransforms::load(source)?,
            tokenizer: TextTokenizer::load(source)?,
            options,
        })
    }

    fn run_one(&self, input: &Self::Input, parameters: &Self::Parameters) -> Result<Self::Output> {
        let LoadOptions {
            device, data_type, ..
        } = self.options;
        let hypotheses = hypotheses(&parameters.hypothesis_template, &input.candidate_labels)?;
        let texts = hypotheses.iter().map(String::as_str).collect::<Vec<_>>();
        let tokens = self.tokenizer.encode_batch(&texts, None, device)?;
        let pixels = self.processor.preprocess(&input.image, device, data_type)?;
        let logits = self.model.logits(&pixels, &tokens)?;
        Ok(ranked(
            &logits[0],
            &input.candidate_labels,
            self.model.score_function(),
            usize::MAX,
        ))
    }
}
