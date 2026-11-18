// `image-to-3d`: an image and a seed to a colored mesh, as `diffusers`' `ShapEImg2ImgPipeline` generates one: the image's CLIP patch states condition the prior, with classifier-free guidance against zeros.

use std::slice;

use serde::Deserialize;
use singe::{DataType, nn::Model};

use crate::{
    error::Result,
    head::PixelLayers,
    model::{diffusion::shap_e::Mesh, multimodal::clip::load_vision_states},
    processing::image::{Image, ImageTransforms},
    source::{LoadOptions, Source},
    task::{
        Pipeline, Task,
        vision::text_to_3d::{ShapE, ShapEParameters},
    },
};

#[derive(Debug)]
pub struct ImageTo3dPipeline {
    shap_e: ShapE,
    image_encoder: Model<PixelLayers>,
    processor: ImageTransforms,
    options: LoadOptions,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ImageTo3dInput {
    pub image: Image,
    pub seed: u64,
}

impl Pipeline for ImageTo3dPipeline {
    const TASK: Task = Task::ImageTo3d;

    type Input = ImageTo3dInput;

    type Parameters = ShapEParameters;

    type Output = Mesh;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self {
            shap_e: ShapE::load(source, options)?,
            image_encoder: load_vision_states(&source.component("image_encoder")?, options)?,
            processor: ImageTransforms::load(&source.subdirectory("image_processor"))?,
            options,
        })
    }

    fn run(
        &self,
        inputs: &[Self::Input],
        parameters: &Self::Parameters,
    ) -> Result<Vec<Self::Output>> {
        inputs
            .iter()
            .map(|input| {
                let pixels = self.processor.preprocess(
                    slice::from_ref(&input.image),
                    self.options.device,
                    self.options.data_type,
                )?;
                let condition = self.image_encoder.forward(&pixels)?.cast(DataType::F32)?;
                self.shap_e.generate(&condition, input.seed, parameters)
            })
            .collect()
    }
}
