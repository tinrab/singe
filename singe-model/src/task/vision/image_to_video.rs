// `image-to-video`: an image and a seed to a clip that animates it, as `diffusers`' `PIAPipeline` generates one from an empty prompt.

use serde::Deserialize;

use crate::{
    error::Result,
    processing::image::Image,
    source::{LoadOptions, Source},
    task::{
        Pipeline, Task,
        multimodal::image_text_to_video::{
            ImageTextToVideoInput, ImageTextToVideoParameters, ImageTextToVideoPipeline,
        },
    },
};

#[derive(Debug)]
pub struct ImageToVideoPipeline(ImageTextToVideoPipeline);

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ImageToVideoInput {
    pub image: Image,
    pub seed: u64,
}

impl Pipeline for ImageToVideoPipeline {
    const TASK: Task = Task::ImageToVideo;

    type Input = ImageToVideoInput;

    type Parameters = ImageTextToVideoParameters;

    type Output = Vec<Image>;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self(ImageTextToVideoPipeline::load(source, options)?))
    }

    fn run(
        &self,
        inputs: &[Self::Input],
        parameters: &Self::Parameters,
    ) -> Result<Vec<Self::Output>> {
        inputs
            .iter()
            .map(|input| {
                self.0.generate(
                    &ImageTextToVideoInput {
                        image: input.image.clone(),
                        prompt: String::new(),
                        seed: input.seed,
                    },
                    parameters,
                )
            })
            .collect()
    }
}
