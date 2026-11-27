//! `image-text-to-image`: an image, an instruction, and a seed to the edited image, as `diffusers`' `StableDiffusionInstructPix2PixPipeline` edits one: the image's latents read beside the noisy latents, and guidance toward the instruction and toward the image.

use serde::Deserialize;
use singe::{DataType, Error as SingeError, Tensor};

use crate::{
    error::Result,
    processor::{image::Image, random::TorchGenerator, scheduler::Sampler},
    source::{LoadOptions, Source},
    task::{
        Pipeline,
        diffusion::{Denoising, InstructPix2PixDefaults, StableDiffusion, denoise},
        inputs::SeededImagePrompt,
    },
};

#[derive(Debug)]
pub struct ImageTextToImagePipeline(StableDiffusion);

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default)]
pub struct ImageTextToImageParameters {
    #[serde(flatten)]
    pub denoising: Denoising<InstructPix2PixDefaults>,
    pub image_guidance_scale: f32,
}

impl Default for ImageTextToImageParameters {
    fn default() -> Self {
        Self {
            denoising: Denoising::default(),
            image_guidance_scale: 1.5,
        }
    }
}

impl ImageTextToImagePipeline {
    fn conditions(
        &self,
        image_latents: Tensor,
        prompt: &str,
        guided: bool,
    ) -> Result<(Tensor, Tensor)> {
        let parts = &self.0;
        let prompt = parts.encode_prompt(prompt)?;
        if !guided {
            return Ok((prompt, image_latents));
        }

        // The instruction's, then the image's alone, then neither, as the pipeline orders its batch.
        let empty = parts.encode_prompt("")?;
        Ok((
            Tensor::concat(&[prompt, empty.clone(), empty], 0)?,
            Tensor::concat(
                &[
                    image_latents.clone(),
                    image_latents.clone(),
                    image_latents.zeros_like()?,
                ],
                0,
            )?,
        ))
    }
}

fn guide_edit(predicted: &Tensor, parameters: &ImageTextToImageParameters) -> Result<Tensor> {
    let (text, image, neither) = (
        predicted.narrow(0, 0, 1)?,
        predicted.narrow(0, 1, 1)?,
        predicted.narrow(0, 2, 1)?,
    );
    Ok(neither
        .add(
            &text
                .sub(&image)?
                .mul_scalar(f64::from(parameters.denoising.scale()))?,
        )?
        .add(
            &image
                .sub(&neither)?
                .mul_scalar(f64::from(parameters.image_guidance_scale))?,
        )?)
}

impl Pipeline for ImageTextToImagePipeline {
    type Input = SeededImagePrompt;

    type Parameters = ImageTextToImageParameters;

    type Output = Image;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self(StableDiffusion::load(source, options)?))
    }

    fn run_one(&self, input: &Self::Input, parameters: &Self::Parameters) -> Result<Self::Output> {
        let parts = &self.0;
        let denoising = &parameters.denoising;
        let guided = denoising.guidance().is_some() && parameters.image_guidance_scale >= 1.0;
        let mut scheduler = Sampler::create(&parts.scheduler, denoising.num_inference_steps)?;
        // The image's latents' mean, unscaled.
        let image_latents = parts
            .vae
            .encode(&parts.image_pixels(&input.image, None)?)?
            .cast(DataType::F32)?;
        let &[_, channels, height, width] = image_latents.shape().dimensions() else {
            return Err(SingeError::invalid_argument("image latents, [1, C, h, w]").into());
        };
        let shape = [1, channels, height, width];
        let (contexts, images) = self.conditions(image_latents, &input.prompt, guided)?;
        let copies = if guided { 3 } else { 1 };
        let device = parts.options.device;
        let mut generator = TorchGenerator::seeded(input.seed);
        let latents = generator
            .normal_tensor(device, &shape)?
            .mul_scalar(f64::from(scheduler.initial_scale()))?;
        let latents = denoise(
            &mut scheduler,
            latents,
            |scheduler, index, latents| {
                let timestep = scheduler.timesteps()[index];
                let repeated = Tensor::concat(&vec![latents.clone(); copies], 0)?;
                let input = Tensor::concat(
                    &[scheduler.scale_input(index, &repeated)?, images.clone()],
                    1,
                )?;
                let predicted = parts.unet.predict(&input, timestep, &contexts)?;
                if guided {
                    guide_edit(&predicted, parameters)
                } else {
                    Ok(predicted)
                }
            },
            || generator.normal_tensor(device, &shape),
        )?;
        Ok(parts.decode(&latents)?.remove(0))
    }
}
