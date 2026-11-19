// `text-to-image`: a prompt and a seed to an image, as `diffusers`' `StableDiffusionPipeline` generates one: the prompt's text states, latents denoised from the seed's noise under the scheduler, with classifier-free guidance where its scale exceeds one, and decoded by the VAE.

use serde::Deserialize;

use crate::{
    error::Result,
    processor::{image::Image, random::TorchGenerator, scheduler::Sampler},
    source::{LoadOptions, Source},
    task::{
        Pipeline,
        diffusion::{Denoising, StableDiffusion, StableDiffusionDefaults, denoise},
        inputs::SeededPrompt,
    },
};

#[derive(Debug)]
pub struct TextToImagePipeline(StableDiffusion);

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default)]
pub struct TextToImageParameters {
    #[serde(flatten)]
    pub denoising: Denoising<StableDiffusionDefaults>,
    pub height: i64,
    pub width: i64,
}

impl Default for TextToImageParameters {
    fn default() -> Self {
        Self {
            denoising: Denoising::default(),
            height: 512,
            width: 512,
        }
    }
}

impl Pipeline for TextToImagePipeline {
    type Input = SeededPrompt;

    type Parameters = TextToImageParameters;

    type Output = Image;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self(StableDiffusion::load(source, options)?))
    }

    fn run_one(&self, input: &Self::Input, parameters: &Self::Parameters) -> Result<Self::Output> {
        let parts = &self.0;
        let (height, width) = parts.latent_size(parameters.height, parameters.width)?;
        let shape = [1, parts.unet.in_channels(), height, width];
        let device = parts.options.device;
        let denoising = &parameters.denoising;
        let guidance = denoising.guidance();
        let mut scheduler = Sampler::create(&parts.scheduler, denoising.num_inference_steps)?;
        let contexts = parts.contexts(&input.prompt, guidance)?;
        let mut generator = TorchGenerator::seeded(input.seed);
        let latents = generator
            .normal_tensor(device, &shape)?
            .mul_scalar(f64::from(scheduler.initial_scale()))?;
        let latents = denoise(
            &mut scheduler,
            latents,
            |scheduler, index, latents| {
                parts.predict(scheduler, index, latents, &contexts, guidance)
            },
            || generator.normal_tensor(device, &shape),
        )?;
        Ok(parts.decode(&latents)?.remove(0))
    }
}
