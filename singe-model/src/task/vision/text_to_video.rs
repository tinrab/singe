//! `text-to-video`: a prompt and a seed to a clip's frames, as `diffusers`' `AnimateDiffPipeline` generates them: a Stable Diffusion `UNet` with a motion adapter denoising every frame's latents together, with classifier-free guidance where its scale exceeds one, and each frame decoded by the VAE.

use serde::Deserialize;
use singe::shape::spatial::Spatial2d;

use crate::{
    error::Result,
    processor::{image::Image, random::TorchGenerator, scheduler::Sampler},
    source::{LoadOptions, Source},
    task::{
        Pipeline,
        diffusion::{AnimateDiff, Denoising, StableDiffusionDefaults, denoise, load_animatediff},
        inputs::SeededPrompt,
    },
};

#[derive(Debug)]
pub struct TextToVideoPipeline(AnimateDiff);

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default)]
pub struct TextToVideoParameters {
    pub num_frames: i64,
    #[serde(flatten)]
    pub denoising: Denoising<StableDiffusionDefaults>,
    pub height: i64,
    pub width: i64,
}

impl Default for TextToVideoParameters {
    fn default() -> Self {
        Self {
            num_frames: 16,
            denoising: Denoising::default(),
            height: 512,
            width: 512,
        }
    }
}

impl Pipeline for TextToVideoPipeline {
    type Input = SeededPrompt;

    type Parameters = TextToVideoParameters;

    type Output = Vec<Image>;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self(load_animatediff(source, options)?))
    }

    fn run_one(&self, input: &Self::Input, parameters: &Self::Parameters) -> Result<Self::Output> {
        let parts = &self.0;
        let Spatial2d { height, width } =
            parts.latent_size(Spatial2d::new(parameters.height, parameters.width))?;
        let shape = [
            1,
            parts.latent_channels(),
            parameters.num_frames,
            height,
            width,
        ];
        let denoising = &parameters.denoising;
        let guidance = denoising.guidance();
        let mut scheduler = Sampler::create(&parts.scheduler, denoising.num_inference_steps)?;
        let contexts = parts.contexts(&input.prompt, guidance)?;
        let mut generator = TorchGenerator::seeded(input.seed);
        let latents = parts
            .clip_noise(&mut generator, &shape)?
            .mul_scalar(f64::from(scheduler.initial_scale()))?;
        let latents = denoise(
            &mut scheduler,
            latents,
            |scheduler, index, latents| {
                parts.predict_clip(scheduler, index, latents, &contexts, guidance)
            },
            || parts.clip_noise(&mut generator, &shape),
        )?;
        parts.decode_clip(&latents)
    }
}
