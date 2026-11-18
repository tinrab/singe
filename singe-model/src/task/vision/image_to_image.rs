// `image-to-image`: an image to another image: a super-resolution model's larger image, as `transformers`' processor and model give it, or an autoencoder's reconstruction, as `diffusers`' autoencoders give it from images in [-1, 1].

use std::slice;

use singe::{DataType, Tensor};

use crate::{
    error::{Error, Result},
    head::{Autoencoder, Head, ImageUpscaler},
    model,
    processing::image::{Image, ImageTransforms},
    source::{LoadOptions, Source},
    task::{Pipeline, Task},
};

#[derive(Debug)]
enum ImageModel {
    Autoencoder(Autoencoder),
    Upscaler {
        model: ImageUpscaler,
        processor: ImageTransforms,
    },
}

#[derive(Debug)]
pub struct ImageToImagePipeline {
    model: ImageModel,
    data_type: DataType,
}

impl Pipeline for ImageToImagePipeline {
    const TASK: Task = Task::ImageToImage;

    type Input = Image;

    type Parameters = ();

    type Output = Image;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        let model = if model::head(source)? == Head::SuperResolution {
            ImageModel::Upscaler {
                model: model::load(source, options)?,
                processor: ImageTransforms::load(source)?,
            }
        } else {
            ImageModel::Autoencoder(model::load(source, options)?)
        };
        Ok(Self {
            model,
            data_type: options.data_type,
        })
    }

    fn run(&self, inputs: &[Self::Input], (): &Self::Parameters) -> Result<Vec<Self::Output>> {
        inputs
            .iter()
            .map(|image| match &self.model {
                ImageModel::Autoencoder(model) => {
                    let device = model.encoder().state().device()?;
                    let pixels = image
                        .to_tensor(device)?
                        .mul_scalar(2.0 / 255.0)?
                        .add_scalar(-1.0)?
                        .cast(self.data_type)?;
                    to_image(
                        &model
                            .decode(&model.encode(&pixels)?)?
                            .add_scalar(1.0)?
                            .mul_scalar(0.5)?,
                    )
                }
                ImageModel::Upscaler { model, processor } => {
                    let device = model.model().state().device()?;
                    let pixels =
                        processor.preprocess(slice::from_ref(image), device, self.data_type)?;
                    to_image(&model.upscale(&pixels)?)
                }
            })
            .collect()
    }
}

// Turns `[1, 3, H, W]` in [0, 1] into an image, rounding to bytes.
fn to_image(tensor: &Tensor) -> Result<Image> {
    let dimensions = tensor.shape().dimensions();
    let (height, width) = (dimensions[2], dimensions[3]);
    let values = tensor
        .squeeze(0)?
        .cast(DataType::F32)?
        .clamp(0.0, 1.0)?
        .permute(&[1, 2, 0])?
        .mul_scalar(255.0)?
        .round()?
        .contiguous()?
        .to_vec::<f32>()?;
    let side =
        |extent: i64| u32::try_from(extent).map_err(|_| Error::unsupported("this image size"));
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "values are rounded and clamped to a byte's range"
    )]
    let pixels = values.iter().map(|&value| value as u8).collect();
    Ok(Image {
        width: side(width)?,
        height: side(height)?,
        pixels,
    })
}
