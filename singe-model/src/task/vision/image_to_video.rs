//! `image-to-video`: an image and a seed to a clip that animates it, as `diffusers`' `PIAPipeline` generates one from an empty prompt.

use crate::{
    error::Result,
    processor::image::Image,
    source::{LoadOptions, Source},
    task::{
        Pipeline,
        inputs::{SeededImage, SeededImagePrompt},
        multimodal::image_text_to_video::{ImageTextToVideoParameters, ImageTextToVideoPipeline},
    },
};

#[derive(Debug)]
pub struct ImageToVideoPipeline(ImageTextToVideoPipeline);

impl Pipeline for ImageToVideoPipeline {
    type Input = SeededImage;

    type Parameters = ImageTextToVideoParameters;

    type Output = Vec<Image>;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self(ImageTextToVideoPipeline::load(source, options)?))
    }

    fn run_one(&self, input: &Self::Input, parameters: &Self::Parameters) -> Result<Self::Output> {
        self.0.run_one(
            &SeededImagePrompt {
                image: input.image.clone(),
                prompt: String::new(),
                seed: input.seed,
            },
            parameters,
        )
    }
}
