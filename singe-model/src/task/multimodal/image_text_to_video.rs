//! `image-text-to-video`: an image, a prompt, and a seed to a clip that animates the image, as `diffusers`' `PIAPipeline` generates it: `AnimateDiff` whose `UNet` also reads, beside each frame's latents, a mask of how much each frame keeps the image and the image's latents.

use std::iter;

use serde::Deserialize;
use singe::{
    Error as SingeError, Shape, Tensor,
    shape::spatial::{Spatial2d, Spatial2dI64},
    tensor::InterpolateMode,
};

use crate::{
    error::{Error, Result},
    processor::{image::Image, random::TorchGenerator, scheduler::Sampler},
    source::{LoadOptions, Source},
    task::{
        Pipeline,
        diffusion::{AnimateDiff, Denoising, StableDiffusionDefaults, denoise, load_animatediff},
        inputs::SeededImagePrompt,
    },
};

/// How much each frame keeps the image, by motion scale, from the image's frame on, as `diffusers`' `RANGE_list` gives it; the last value repeats.
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

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default)]
pub struct ImageTextToVideoParameters {
    pub num_frames: i64,
    #[serde(flatten)]
    pub denoising: Denoising<StableDiffusionDefaults>,
    pub height: i64,
    pub width: i64,
    pub motion_scale: usize,
}

impl Default for ImageTextToVideoParameters {
    fn default() -> Self {
        Self {
            num_frames: 16,
            denoising: Denoising::default(),
            height: 512,
            width: 512,
            motion_scale: 0,
        }
    }
}

impl ImageTextToVideoPipeline {
    fn condition(
        &self,
        image: &Image,
        parameters: &ImageTextToVideoParameters,
        Spatial2d { height, width }: Spatial2dI64,
        generator: &mut TorchGenerator,
        copies: usize,
    ) -> Result<Tensor> {
        let parts = &self.0;
        let frames = parameters.num_frames;
        let coefficients = MOTION_SCALES
            .get(parameters.motion_scale)
            .ok_or_else(|| Error::invalid("a motion scale beyond PIA's"))?;
        // The image's latents at the latents' size, scaled, as each frame's condition, and each frame's mask of how much it keeps the image.
        let image = parts
            .sampled_latents(&parts.image_pixels(image, None)?, generator)?
            .interpolate(Spatial2d::new(height, width), InterpolateMode::Nearest)?
            .mul_scalar(f64::from(parts.scaling))?;
        let &[_, image_channels, ..] = image.shape().dimensions() else {
            return Err(SingeError::invalid_argument("image latents, [1, C, h, w]").into());
        };
        let masked = image
            .unsqueeze(2)?
            .expand(Shape::create([1, image_channels, frames, height, width])?)?
            .contiguous()?;
        let (frame_count, positions) = (
            usize::try_from(frames).map_err(|_| SingeError::out_of_range("frames"))?,
            usize::try_from(height * width).map_err(|_| SingeError::out_of_range("latent size"))?,
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
        let data_type = parts.options.data_type;
        let repeat = |tensor: &Tensor| Tensor::concat(&vec![tensor.clone(); copies], 0);
        Ok(Tensor::concat(
            &[
                repeat(&mask.cast(data_type)?)?,
                repeat(&masked.cast(data_type)?)?,
            ],
            1,
        )?)
    }
}

impl Pipeline for ImageTextToVideoPipeline {
    type Input = SeededImagePrompt;

    type Parameters = ImageTextToVideoParameters;

    type Output = Vec<Image>;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self(load_animatediff(source, options)?))
    }

    fn run_one(&self, input: &Self::Input, parameters: &Self::Parameters) -> Result<Self::Output> {
        let parts = &self.0;
        let Spatial2d { height, width } =
            parts.latent_size(Spatial2d::new(parameters.height, parameters.width))?;
        let denoising = &parameters.denoising;
        let guidance = denoising.guidance();
        let mut scheduler = Sampler::create(&parts.scheduler, denoising.num_inference_steps)?;
        let contexts = parts.contexts(&input.prompt, guidance)?;
        let mut generator = TorchGenerator::seeded(input.seed);
        let shape = [
            1,
            parts.latent_channels(),
            parameters.num_frames,
            height,
            width,
        ];
        let latents = parts
            .clip_noise(&mut generator, &shape)?
            .mul_scalar(f64::from(scheduler.initial_scale()))?;
        let copies = if guidance.is_some() { 2 } else { 1 };
        let condition = self.condition(
            &input.image,
            parameters,
            Spatial2d::new(height, width),
            &mut generator,
            copies,
        )?;
        let latents = denoise(
            &mut scheduler,
            latents,
            |scheduler, index, latents| {
                parts
                    .predict_masked_clip(scheduler, index, latents, &condition, &contexts, guidance)
            },
            || parts.clip_noise(&mut generator, &shape),
        )?;
        parts.decode_clip(&latents)
    }
}
