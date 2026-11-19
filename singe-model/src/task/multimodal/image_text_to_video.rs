// `image-text-to-video`: an image, a prompt, and a seed to a clip that animates the image, as `diffusers`' `PIAPipeline` generates it: AnimateDiff whose UNet also reads, beside each frame's latents, a mask of how much each frame keeps the image and the image's latents.

use std::iter;

use serde::Deserialize;
use singe::{DataType, Shape, Tensor, tensor::InterpolateMode};

use crate::{
    error::{Error, Result},
    processing::{image::Image, random::TorchGenerator, scheduler::Sampler},
    source::{LoadOptions, Source},
    task::{
        Pipeline, Task,
        vision::{
            text_to_image::guide_pair,
            text_to_video::{AnimateDiff, load_animatediff},
        },
    },
};

// `RANGE_list`: how much each frame keeps the image, by motion scale, from the image's frame on; the last value repeats.
const MOTION_SCALES: &[&[f32]] = &[
    &[1.0, 0.9, 0.85, 0.85, 0.85, 0.8],
    &[1.0, 0.8, 0.8, 0.8, 0.79, 0.78, 0.75],
    &[
        1.0, 0.8, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.6, 0.5, 0.5,
    ],
    &[
        1.0, 0.9, 0.85, 0.85, 0.85, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.85, 0.85, 0.9, 1.0,
    ],
    &[
        1.0, 0.8, 0.8, 0.8, 0.79, 0.78, 0.75, 0.75, 0.75, 0.75, 0.75, 0.78, 0.79, 0.8, 0.8, 1.0,
    ],
    &[
        1.0, 0.8, 0.7, 0.7, 0.7, 0.7, 0.6, 0.5, 0.5, 0.6, 0.7, 0.7, 0.7, 0.7, 0.8, 1.0,
    ],
    &[0.5, 0.4, 0.4, 0.4, 0.35, 0.3],
    &[0.5, 0.4, 0.4, 0.4, 0.35, 0.35, 0.3, 0.25, 0.2],
    &[0.5, 0.2],
];

#[derive(Debug)]
pub struct ImageTextToVideoPipeline(AnimateDiff);

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ImageTextToVideoInput {
    pub image: Image,
    pub prompt: String,
    pub seed: u64,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default)]
pub struct ImageTextToVideoParameters {
    pub num_frames: i64,
    pub num_inference_steps: usize,
    pub guidance_scale: f32,
    pub height: i64,
    pub width: i64,
    pub motion_scale: usize,
}

impl Default for ImageTextToVideoParameters {
    fn default() -> Self {
        Self {
            num_frames: 16,
            num_inference_steps: 50,
            guidance_scale: 7.5,
            height: 512,
            width: 512,
            motion_scale: 0,
        }
    }
}

// Latents sampled from the VAE's distribution for `pixels`, drawing their noise from `generator`, as `latent_dist.sample(generator)` does.
pub(crate) fn sampled_latents(
    parts: &AnimateDiff,
    pixels: &Tensor,
    generator: &mut TorchGenerator,
) -> Result<Tensor> {
    let (mean, deviation) = parts.vae.encode_distribution(pixels)?;
    let (mean, deviation) = (
        mean.cast(DataType::F32)?,
        deviation
            .ok_or_else(|| Error::unsupported("a VAE without a latent distribution"))?
            .cast(DataType::F32)?,
    );
    let noise = generator.normal_tensor(parts.options.device, mean.shape().dimensions())?;
    Ok(mean.add(&deviation.mul(&noise)?)?)
}

impl ImageTextToVideoPipeline {
    pub(crate) fn generate(
        &self,
        input: &ImageTextToVideoInput,
        parameters: &ImageTextToVideoParameters,
    ) -> Result<Vec<Image>> {
        let parts = &self.0;
        let frames = parameters.num_frames;
        let (height, width) = parts.latent_size(parameters.height, parameters.width)?;
        let coefficients = MOTION_SCALES
            .get(parameters.motion_scale)
            .ok_or_else(|| Error::unsupported("a motion scale beyond PIA's"))?;
        let mut scheduler = Sampler::create(&parts.scheduler, parameters.num_inference_steps)?;
        let context = parts.encode_prompt(&input.prompt)?;
        let guided = parameters.guidance_scale > 1.0;
        let unconditioned = guided.then(|| parts.encode_prompt("")).transpose()?;
        let mut generator = TorchGenerator::seeded(input.seed);
        let shape = [
            1,
            parts.unet.unet.layer().config().in_channels,
            frames,
            height,
            width,
        ];
        let mut latents = parts
            .clip_noise(&mut generator, &shape)?
            .mul_scalar(f64::from(scheduler.initial_scale()))?;
        // The image's latents at the latents' size, scaled, as each frame's condition, and each frame's mask of how much it keeps the image.
        let image = sampled_latents(
            parts,
            &parts.image_pixels(&input.image, None)?,
            &mut generator,
        )?
        .interpolate((height, width), InterpolateMode::Nearest)?
        .mul_scalar(f64::from(parts.scaling))?;
        let image_channels = image.shape().dimensions()[1];
        let masked = image
            .unsqueeze(2)?
            .expand(Shape::create([1, image_channels, frames, height, width])?)?
            .contiguous()?;
        let (frame_count, positions) = (
            usize::try_from(frames).map_err(|_| singe::Error::out_of_range("frames"))?,
            usize::try_from(height * width)
                .map_err(|_| singe::Error::out_of_range("latent size"))?,
        );
        // The coefficients from the image's frame on, the last repeated.
        let mask_values = (0..frame_count)
            .flat_map(|frame| {
                let value = coefficients[frame.min(coefficients.len() - 1)];
                iter::repeat_n(value, positions)
            })
            .collect::<Vec<f32>>();
        let mask = Tensor::from_vec(
            parts.options.device,
            Shape::create([1, 1, frames, height, width])?,
            mask_values,
        )?;
        let copies = if guided { 2 } else { 1 };
        let repeat = |tensor: &Tensor| Tensor::concat(&vec![tensor.clone(); copies], 0);
        let data_type = parts.options.data_type;
        let (mask, masked) = (
            repeat(&mask.cast(data_type)?)?,
            repeat(&masked.cast(data_type)?)?,
        );
        let contexts = match &unconditioned {
            Some(unconditioned) => Tensor::concat(&[unconditioned.clone(), context], 0)?,
            None => context,
        };
        for index in 0..scheduler.timesteps().len() {
            let input = Tensor::concat(
                &[
                    scheduler.scale_input(index, &repeat(&latents)?)?,
                    mask.clone(),
                    masked.clone(),
                ],
                1,
            )?;
            let timestep = scheduler.timesteps()[index];
            let predicted = parts.unet.predict(&input, timestep, &contexts)?;
            let predicted = if guided {
                guide_pair(&predicted, parameters.guidance_scale)?
            } else {
                predicted
            };
            latents = scheduler.step(index, &predicted, &latents, || {
                parts.clip_noise(&mut generator, &shape)
            })?;
        }
        parts.decode_clip(&latents)
    }
}

impl Pipeline for ImageTextToVideoPipeline {
    const TASK: Task = Task::ImageTextToVideo;

    type Input = ImageTextToVideoInput;

    type Parameters = ImageTextToVideoParameters;

    type Output = Vec<Image>;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self(load_animatediff(source, options)?))
    }

    fn run(
        &self,
        inputs: &[Self::Input],
        parameters: &Self::Parameters,
    ) -> Result<Vec<Self::Output>> {
        inputs
            .iter()
            .map(|input| self.generate(input, parameters))
            .collect()
    }
}
