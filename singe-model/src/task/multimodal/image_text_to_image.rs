// `image-text-to-image`: an image, an instruction, and a seed to the edited image, as `diffusers`' `StableDiffusionInstructPix2PixPipeline` edits one: the image's latents read beside the noisy latents, and guidance toward the instruction and toward the image.

use serde::Deserialize;
use singe::{DataType, Tensor};

use crate::{
    error::Result,
    processor::{image::Image, random::TorchGenerator, scheduler::Sampler},
    source::{LoadOptions, Source},
    task::{Pipeline, Task, vision::text_to_image::StableDiffusion},
};

#[derive(Debug)]
pub struct ImageTextToImagePipeline(StableDiffusion);

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ImageEdit {
    pub image: Image,
    pub prompt: String,
    pub seed: u64,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default)]
pub struct ImageTextToImageParameters {
    pub num_inference_steps: usize,
    pub guidance_scale: f32,
    pub image_guidance_scale: f32,
}

impl Default for ImageTextToImageParameters {
    fn default() -> Self {
        Self {
            num_inference_steps: 100,
            guidance_scale: 7.5,
            image_guidance_scale: 1.5,
        }
    }
}

impl Pipeline for ImageTextToImagePipeline {
    const TASK: Task = Task::ImageTextToImage;

    type Input = ImageEdit;

    type Parameters = ImageTextToImageParameters;

    type Output = Image;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self(StableDiffusion::load(source, options)?))
    }

    fn run(
        &self,
        inputs: &[Self::Input],
        parameters: &Self::Parameters,
    ) -> Result<Vec<Self::Output>> {
        let parts = &self.0;
        let guided = parameters.guidance_scale > 1.0 && parameters.image_guidance_scale >= 1.0;
        inputs
            .iter()
            .map(|input| {
                let mut scheduler =
                    Sampler::create(&parts.scheduler, parameters.num_inference_steps)?;
                // The image's latents' mean, unscaled.
                let image_latents = parts
                    .vae
                    .encode(&parts.image_pixels(&input.image, None)?)?
                    .cast(DataType::F32)?;
                let dimensions = image_latents.shape().dimensions();
                let shape = [1, dimensions[1], dimensions[2], dimensions[3]];
                let prompt = parts.encode_prompt(&input.prompt)?;
                // The instruction's, then the image's alone, then neither, as the pipeline orders its batch.
                let (contexts, images) = if guided {
                    let empty = parts.encode_prompt("")?;
                    (
                        Tensor::concat(&[prompt, empty.clone(), empty], 0)?,
                        Tensor::concat(
                            &[
                                image_latents.clone(),
                                image_latents.clone(),
                                image_latents.zeros_like()?,
                            ],
                            0,
                        )?,
                    )
                } else {
                    (prompt, image_latents)
                };
                let copies = if guided { 3 } else { 1 };
                let device = parts.options.device;
                let mut generator = TorchGenerator::seeded(input.seed);
                let mut latents = generator
                    .normal_tensor(device, &shape)?
                    .mul_scalar(f64::from(scheduler.initial_scale()))?;
                for index in 0..scheduler.timesteps().len() {
                    let timestep = scheduler.timesteps()[index];
                    let repeated = Tensor::concat(&vec![latents.clone(); copies], 0)?;
                    let input = Tensor::concat(
                        &[scheduler.scale_input(index, &repeated)?, images.clone()],
                        1,
                    )?;
                    let predicted = parts.unet.predict(&input, timestep, &contexts)?;
                    let predicted = if guided {
                        let (text, image, neither) = (
                            predicted.narrow(0, 0, 1)?,
                            predicted.narrow(0, 1, 1)?,
                            predicted.narrow(0, 2, 1)?,
                        );
                        neither
                            .add(
                                &text
                                    .sub(&image)?
                                    .mul_scalar(f64::from(parameters.guidance_scale))?,
                            )?
                            .add(
                                &image
                                    .sub(&neither)?
                                    .mul_scalar(f64::from(parameters.image_guidance_scale))?,
                            )?
                    } else {
                        predicted
                    };
                    latents = scheduler.step(index, &predicted, &latents, || {
                        generator.normal_tensor(device, &shape)
                    })?;
                }
                Ok(parts.decode(&latents)?.remove(0))
            })
            .collect()
    }
}
