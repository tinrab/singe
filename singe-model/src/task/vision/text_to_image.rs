//! `text-to-image`: a prompt and a seed to an image, as `diffusers`' `StableDiffusionPipeline` generates one: the prompt's text states, latents denoised from the seed's noise under the scheduler, with classifier-free guidance where its scale exceeds one, and decoded by the VAE.
//! A model that generates image tokens, as Janus does, generates them under the guidance scale, its own where none is given, greedily or, where its generation config or the parameters ask, sampled from the seed, and ignores the denoising steps.

use serde::Deserialize;
use singe::shape::spatial::Spatial2d;

use crate::{
    error::Result,
    head::{ImageGenerator, SamplingRequest},
    model,
    processor::{image::Image, random::TorchGenerator, scheduler::Sampler},
    source::{Format, LoadOptions, Source},
    task::{
        Pipeline,
        diffusion::{Denoising, StableDiffusion, StableDiffusionDefaults, denoise},
        inputs::SeededPrompt,
    },
};

#[derive(Debug)]
pub enum TextToImagePipeline {
    Diffusion(Box<StableDiffusion>),
    Tokens(Box<dyn ImageGenerator>),
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default)]
pub struct TextToImageParameters {
    #[serde(flatten)]
    pub denoising: Denoising<StableDiffusionDefaults>,
    pub height: i64,
    pub width: i64,
    /// Whether a model of image tokens samples them, or its generation config's choice when `None`.
    pub do_sample: Option<bool>,
    pub temperature: Option<f32>,
    pub top_k: Option<usize>,
}

impl Default for TextToImageParameters {
    fn default() -> Self {
        Self {
            denoising: Denoising::default(),
            height: 512,
            width: 512,
            do_sample: None,
            temperature: None,
            top_k: None,
        }
    }
}

impl Pipeline for TextToImagePipeline {
    type Input = SeededPrompt;

    type Parameters = TextToImageParameters;

    type Output = Image;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(if source.format() == Format::Diffusers {
            Self::Diffusion(Box::new(StableDiffusion::load(source, options)?))
        } else {
            Self::Tokens(model::load(source, options)?)
        })
    }

    fn run_one(&self, input: &Self::Input, parameters: &Self::Parameters) -> Result<Self::Output> {
        let parts = match self {
            Self::Diffusion(parts) => parts,
            Self::Tokens(model) => {
                let sampling = SamplingRequest {
                    do_sample: parameters.do_sample,
                    temperature: parameters.temperature,
                    top_k: parameters.top_k,
                    seed: input.seed,
                };
                return model.generate_image(
                    &input.prompt,
                    parameters.denoising.guidance_scale,
                    &sampling,
                );
            }
        };
        let Spatial2d { height, width } =
            parts.latent_size(Spatial2d::new(parameters.height, parameters.width))?;
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
