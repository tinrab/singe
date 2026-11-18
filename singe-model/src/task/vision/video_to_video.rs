// `video-to-video`: a clip, a prompt, and a seed to a clip redrawn after the prompt, as `diffusers`' `AnimateDiffVideoToVideoPipeline` redraws it: the clip's latents sampled from the VAE, noised to the step `strength` starts from, and denoised from there.

use serde::Deserialize;
use singe::Tensor;

use crate::{
    error::{Error, Result},
    processing::{image::Image, random::TorchGenerator, scheduler::Sampler, video::Video},
    source::{LoadOptions, Source},
    task::{
        Pipeline, Task,
        multimodal::image_text_to_video::sampled_latents,
        vision::text_to_video::{AnimateDiff, load_animatediff},
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
    pub num_inference_steps: usize,
    pub guidance_scale: f32,
    pub strength: f64,
    // The frames' size, which the clip's frames must have, as the pipeline would resize them to it.
    pub height: i64,
    pub width: i64,
}

impl Default for VideoToVideoParameters {
    fn default() -> Self {
        Self {
            num_inference_steps: 50,
            guidance_scale: 7.5,
            strength: 0.8,
            height: 512,
            width: 512,
        }
    }
}

impl Pipeline for VideoToVideoPipeline {
    const TASK: Task = Task::VideoToVideo;

    type Input = VideoToVideoInput;

    type Parameters = VideoToVideoParameters;

    type Output = Vec<Image>;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self(load_animatediff(source, options)?))
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "step counts are small, and `int` truncates as `as` does"
    )]
    fn run(
        &self,
        inputs: &[Self::Input],
        parameters: &Self::Parameters,
    ) -> Result<Vec<Self::Output>> {
        let parts = &self.0;
        let steps = parameters.num_inference_steps;
        inputs
            .iter()
            .map(|input| {
                let mut scheduler = Sampler::create(&parts.scheduler, steps)?;
                // `int(num_inference_steps * strength)`, in `F64`.
                let kept = ((steps as f64 * parameters.strength) as usize).min(steps);
                scheduler.skip(steps - kept);
                if scheduler.timesteps().is_empty() {
                    return Err(Error::unsupported("a strength that keeps no step"));
                }

                if input.video.frames.iter().any(|frame| {
                    i64::from(frame.height) != parameters.height
                        || i64::from(frame.width) != parameters.width
                }) {
                    return Err(Error::unsupported(
                        "frames other than the size the parameters give",
                    ));
                }

                let mut generator = TorchGenerator::seeded(input.seed);
                // Each frame's latents sampled from the VAE, 16 frames at a time, scaled and noised to the first kept step.
                let pixels = input
                    .video
                    .frames
                    .iter()
                    .map(|frame| parts.image_pixels(frame))
                    .collect::<Result<Vec<_>>>()?;
                let mut sampled = Vec::new();
                for chunk in pixels.chunks(16) {
                    sampled.push(sampled_latents(
                        parts,
                        &Tensor::concat(chunk, 0)?,
                        &mut generator,
                    )?);
                }

                let clean = Tensor::concat(&sampled, 0)?
                    .mul_scalar(f64::from(parts.scaling))?
                    .unsqueeze(0)?;
                let noise = parts.clip_noise(&mut generator, clean.shape().dimensions())?;
                let mut latents = scheduler
                    .add_noise(&clean.cast(parts.options.data_type)?, &noise)?
                    .permute(&[0, 2, 1, 3, 4])?
                    .contiguous()?;
                let shape = latents.shape().dimensions().to_vec();
                let context = parts.encode_prompt(&input.prompt)?;
                let guided = parameters.guidance_scale > 1.0;
                let unconditioned = guided.then(|| parts.encode_prompt("")).transpose()?;
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
