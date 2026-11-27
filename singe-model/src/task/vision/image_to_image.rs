//! `image-to-image`: an image to another image: a super-resolution model's larger image, as `transformers`' processor and model give it, or an autoencoder's reconstruction, as `diffusers`' autoencoders give it from images in `[-1, 1]`.

use singe::DataType;

use crate::{
    error::Result,
    head::{Autoencoder, Head, ImageUpscaler},
    model,
    processor::image::{Image, ImagePreprocessor, PixelRange},
    source::{LoadOptions, Source},
    task::Pipeline,
};

#[derive(Debug)]
enum ImageModel {
    Autoencoder(Autoencoder),
    Upscaler {
        model: ImageUpscaler,
        processor: ImagePreprocessor,
    },
}

#[derive(Debug)]
pub struct ImageToImagePipeline {
    model: ImageModel,
    options: LoadOptions,
}

impl Pipeline for ImageToImagePipeline {
    type Input = Image;

    type Parameters = ();

    type Output = Image;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        let model = if model::default_head(source)? == Head::SuperResolution {
            ImageModel::Upscaler {
                model: model::load(source, options)?,
                processor: ImagePreprocessor::load(source)?,
            }
        } else {
            ImageModel::Autoencoder(model::load(source, options)?)
        };
        Ok(Self { model, options })
    }

    fn run_one(&self, image: &Self::Input, (): &Self::Parameters) -> Result<Self::Output> {
        let LoadOptions {
            device, data_type, ..
        } = self.options;
        let output = match &self.model {
            ImageModel::Autoencoder(model) => {
                let pixels = image
                    .to_tensor(device)?
                    .mul_scalar(2.0 / 255.0)?
                    .add_scalar(-1.0)?
                    .cast(data_type)?;
                model
                    .decode(&model.encode(&pixels)?)?
                    .add_scalar(1.0)?
                    .mul_scalar(0.5)?
            }
            ImageModel::Upscaler { model, processor } => {
                model.upscale(&processor.preprocess(image, device, data_type)?)?
            }
        };
        Ok(Image::from_tensor(&output.cast(DataType::F32)?, PixelRange::Unit)?.remove(0))
    }
}
