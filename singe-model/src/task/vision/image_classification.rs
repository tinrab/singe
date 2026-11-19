// `image-classification`: an image to its labels' scores, best first, as `transformers`' pipeline ranks them.

use crate::{
    error::Result,
    head::ImageClassifier,
    model,
    processor::image::{Image, ImageTransforms},
    source::{LoadOptions, Source},
    task::{
        Pipeline,
        ranking::{ClassificationParameters, LabelScore, ranked},
    },
};

#[derive(Debug)]
pub struct ImageClassificationPipeline {
    model: ImageClassifier,
    processor: ImageTransforms,
    options: LoadOptions,
}

impl Pipeline for ImageClassificationPipeline {
    type Input = Image;

    type Parameters = ClassificationParameters;

    type Output = Vec<LabelScore>;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self {
            model: model::load(source, options)?,
            processor: ImageTransforms::load(source)?,
            options,
        })
    }

    fn run_one(&self, image: &Self::Input, parameters: &Self::Parameters) -> Result<Self::Output> {
        let config = self.model.config();
        let pixels =
            self.processor
                .preprocess(image, self.options.device, self.options.data_type)?;
        let scores = self.model.scores(&pixels)?;
        Ok(ranked(
            &scores[0],
            config.labels(),
            parameters
                .function_to_apply
                .unwrap_or_else(|| config.function()),
            parameters.top_k,
        ))
    }
}
