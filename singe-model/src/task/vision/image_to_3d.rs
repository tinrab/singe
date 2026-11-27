//! `image-to-3d`: an image and a seed to a colored mesh, as `diffusers`' `ShapEImg2ImgPipeline` generates one: the image's CLIP patch states condition the prior, with classifier-free guidance against zeros.

use singe::{
    DataType,
    nn::{Layer, Model},
};

use crate::{
    error::Result,
    head::Boxed,
    model::{diffusion::shap_e::Mesh, multimodal::clip::load_vision_states},
    processor::image::ImagePreprocessor,
    source::{LoadOptions, Source},
    task::{
        Pipeline,
        diffusion::{ShapE, ShapEParameters},
        inputs::SeededImage,
    },
};

#[derive(Debug)]
pub struct ImageTo3dPipeline {
    shap_e: ShapE,
    image_encoder: Model<Boxed<dyn Layer>>,
    processor: ImagePreprocessor,
    options: LoadOptions,
}

impl Pipeline for ImageTo3dPipeline {
    type Input = SeededImage;

    type Parameters = ShapEParameters;

    type Output = Mesh;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self {
            shap_e: ShapE::load(source, options)?,
            image_encoder: load_vision_states(&source.component("image_encoder")?, options)?,
            processor: ImagePreprocessor::load(&source.subdirectory("image_processor"))?,
            options,
        })
    }

    fn run_one(&self, input: &Self::Input, parameters: &Self::Parameters) -> Result<Self::Output> {
        let pixels =
            self.processor
                .preprocess(&input.image, self.options.device, self.options.data_type)?;
        let condition = self.image_encoder.forward(&pixels)?.cast(DataType::F32)?;
        self.shap_e.generate(&condition, input.seed, parameters)
    }
}
