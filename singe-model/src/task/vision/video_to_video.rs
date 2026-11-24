//! `video-to-video`: a clip, a prompt, and a seed to a clip redrawn after the prompt, as `diffusers`' `AnimateDiffVideoToVideoPipeline` redraws it: the clip's latents sampled from the VAE, noised to the step `strength` starts from, and denoised from there.

use serde::Deserialize;
use singe::Tensor;

use crate::{
    error::{Error, Result},
    processor::{image::Image, random::TorchGenerator, scheduler::Sampler, video::Video},
    source::{LoadOptions, Source},
    task::{
        Pipeline,
        diffusion::{AnimateDiff, Denoising, StableDiffusionDefaults, denoise, load_animatediff},
    },
};

#[derive(Debug)]
pub struct VideoToVideoPipeline(AnimateDiff);

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct VideoToVideoInput {
    pub video: Video,
    pub prompt: String,
    pub seed: u64,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default)]
pub struct VideoToVideoParameters {
    #[serde(flatten)]
    pub denoising: Denoising<StableDiffusionDefaults>,
    pub strength: f64,
    /// The height the clip's frames are resized to, 512 by default.
    pub height: i64,
    /// The width the clip's frames are resized to, 512 by default.
    pub width: i64,
}

impl Default for VideoToVideoParameters {
    fn default() -> Self {
        Self {
            denoising: Denoising::default(),
            strength: 0.8,
            height: 512,
            width: 512,
        }
    }
}

impl VideoToVideoPipeline {
    fn scheduler(&self, parameters: &VideoToVideoParameters) -> Result<Sampler> {
        let steps = parameters.denoising.num_inference_steps;
        let mut scheduler = Sampler::create(&self.0.scheduler, steps)?;
        // `int(num_inference_steps * strength)`, in `F64`.
        let kept = ((steps as f64 * parameters.strength) as usize).min(steps);
        scheduler.skip(steps - kept);
        if scheduler.timesteps().is_empty() {
            return Err(Error::invalid("a strength that keeps no step"));
        }

        Ok(scheduler)
    }

    fn clean_latents(
        &self,
        video: &Video,
        parameters: &VideoToVideoParameters,
        generator: &mut TorchGenerator,
    ) -> Result<Tensor> {
        let parts = &self.0;
        let pixels = video
            .frames
            .iter()
            .map(|frame| parts.image_pixels(frame, Some((parameters.height, parameters.width))))
            .collect::<Result<Vec<_>>>()?;
        let mut sampled = Vec::new();
        for chunk in pixels.chunks(16) {
            sampled.push(parts.sampled_latents(&Tensor::concat(chunk, 0)?, generator)?);
        }
        Ok(Tensor::concat(&sampled, 0)?
            .mul_scalar(f64::from(parts.scaling))?
            .unsqueeze(0)?)
    }
}

impl Pipeline for VideoToVideoPipeline {
    type Input = VideoToVideoInput;

    type Parameters = VideoToVideoParameters;

    type Output = Vec<Image>;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self(load_animatediff(source, options)?))
    }

    fn run_one(&self, input: &Self::Input, parameters: &Self::Parameters) -> Result<Self::Output> {
        let parts = &self.0;
        let mut scheduler = self.scheduler(parameters)?;
        let mut generator = TorchGenerator::seeded(input.seed);
        let clean = self.clean_latents(&input.video, parameters, &mut generator)?;
        let noise = parts.clip_noise(&mut generator, clean.shape().dimensions())?;
        let latents = scheduler
            .add_noise(&clean.cast(parts.options.data_type)?, &noise)?
            .permute(&[0, 2, 1, 3, 4])?
            .contiguous()?;
        let shape = latents.shape().dimensions().to_vec();
        let guidance = parameters.denoising.guidance();
        let contexts = parts.contexts(&input.prompt, guidance)?;
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
