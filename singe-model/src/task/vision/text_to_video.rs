// `text-to-video`: a prompt and a seed to a clip's frames, as `diffusers`' `AnimateDiffPipeline` generates them: a Stable Diffusion UNet with a motion adapter denoising every frame's latents together, with classifier-free guidance where its scale exceeds one, and each frame decoded by the VAE.

use serde::Deserialize;
use singe::Tensor;

use crate::{
    error::Result,
    model::diffusion::unet_2d_condition::{UnetMotion, load_motion},
    processing::{image::Image, random::TorchGenerator, scheduler::Sampler},
    source::{LoadOptions, Source},
    task::{
        Pipeline, Task,
        vision::text_to_image::{StableDiffusion, guide_pair},
    },
};

// An AnimateDiff pipeline: Stable Diffusion's components, its UNet joined by the `motion_adapter` component.
pub type AnimateDiff = StableDiffusion<UnetMotion>;

// Loads an AnimateDiff pipeline whose `unet` component the `motion_adapter` component extends.
pub fn load_animatediff(source: &Source, options: LoadOptions) -> Result<AnimateDiff> {
    let unet = load_motion(
        &source.component("unet")?,
        &source.component("motion_adapter")?,
        options,
    )?;
    StableDiffusion::load_with(source, options, unet)
}

impl AnimateDiff {
    // Noise of `shape` drawn from `generator` in the model's type, as `diffusers` keeps a clip's latents in it.
    pub fn clip_noise(&self, generator: &mut TorchGenerator, shape: &[i64]) -> Result<Tensor> {
        Ok(generator
            .normal_tensor(self.options.device, shape)?
            .cast(self.options.data_type)?)
    }

    // The noise predicted for clips' latents, `[1, C, F, h, w]`, at step `index`, guided toward `context` and away from `unconditioned` where given, both in one batch as the pipeline batches them.
    pub fn guided_clip(
        &self,
        scheduler: &Sampler,
        index: usize,
        latents: &Tensor,
        context: &Tensor,
        unconditioned: Option<(&Tensor, f32)>,
    ) -> Result<Tensor> {
        let timestep = scheduler.timesteps()[index];
        match unconditioned {
            Some((unconditioned, guidance)) => {
                let input = scheduler.scale_input(
                    index,
                    &Tensor::concat(&[latents.clone(), latents.clone()], 0)?,
                )?;
                let contexts = Tensor::concat(&[unconditioned.clone(), context.clone()], 0)?;
                guide_pair(&self.unet.predict(&input, timestep, &contexts)?, guidance)
            }
            None => self
                .unet
                .predict(&scheduler.scale_input(index, latents)?, timestep, context),
        }
    }

    // A clip's latents, `[1, C, F, h, w]`, decoded frame by frame, as `decode_latents` decodes them in chunks of 16.
    pub fn decode_clip(&self, latents: &Tensor) -> Result<Vec<Image>> {
        let dimensions = latents.shape().dimensions();
        let (channels, frames, height, width) =
            (dimensions[1], dimensions[2], dimensions[3], dimensions[4]);
        let frames_first = latents
            .permute(&[0, 2, 1, 3, 4])?
            .contiguous()?
            .reshape([frames, channels, height, width])?;
        let mut images = Vec::new();
        for start in (0..frames).step_by(16) {
            images.extend(self.decode(&frames_first.narrow(
                0,
                start,
                (frames - start).min(16),
            )?)?);
        }
        Ok(images)
    }
}

#[derive(Debug)]
pub struct TextToVideoPipeline(AnimateDiff);

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct TextToVideoInput {
    pub prompt: String,
    pub seed: u64,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default)]
pub struct TextToVideoParameters {
    pub num_frames: i64,
    pub num_inference_steps: usize,
    pub guidance_scale: f32,
    pub height: i64,
    pub width: i64,
}

impl Default for TextToVideoParameters {
    fn default() -> Self {
        Self {
            num_frames: 16,
            num_inference_steps: 50,
            guidance_scale: 7.5,
            height: 512,
            width: 512,
        }
    }
}

impl Pipeline for TextToVideoPipeline {
    const TASK: Task = Task::TextToVideo;

    type Input = TextToVideoInput;

    type Parameters = TextToVideoParameters;

    type Output = Vec<Image>;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self(load_animatediff(source, options)?))
    }

    fn run(
        &self,
        inputs: &[Self::Input],
        parameters: &Self::Parameters,
    ) -> Result<Vec<Self::Output>> {
        let parts = &self.0;
        let (height, width) = parts.latent_size(parameters.height, parameters.width)?;
        let shape = [
            1,
            parts.unet.unet.layer().config().in_channels,
            parameters.num_frames,
            height,
            width,
        ];
        let guided = parameters.guidance_scale > 1.0;
        inputs
            .iter()
            .map(|input| {
                let mut scheduler =
                    Sampler::create(&parts.scheduler, parameters.num_inference_steps)?;
                let context = parts.encode_prompt(&input.prompt)?;
                let unconditioned = guided.then(|| parts.encode_prompt("")).transpose()?;
                let mut generator = TorchGenerator::seeded(input.seed);
                let mut latents = parts
                    .clip_noise(&mut generator, &shape)?
                    .mul_scalar(f64::from(scheduler.initial_scale()))?;
                for index in 0..scheduler.timesteps().len() {
                    let predicted = parts.guided_clip(
                        &scheduler,
                        index,
                        &latents,
                        &context,
                        unconditioned
                            .as_ref()
                            .map(|states| (states, parameters.guidance_scale)),
                    )?;
                    latents = scheduler.step(index, &predicted, &latents, || {
                        parts.clip_noise(&mut generator, &shape)
                    })?;
                }
                parts.decode_clip(&latents)
            })
            .collect()
    }
}
