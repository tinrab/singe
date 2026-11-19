// `unconditional-image-generation`: a seed to an image, denoised from the seed's noise by a diffusion model under its scheduler, as `diffusers`' `DDPMPipeline` generates one with `torch.Generator("cpu").manual_seed(seed)`.

use serde::Deserialize;
use singe::{DataType, Tensor};

use crate::{
    error::{Error, Result},
    head::Denoiser,
    model,
    processor::{
        image::Image,
        random::TorchGenerator,
        scheduler::{DdpmScheduler, SchedulerConfig},
    },
    source::{LoadOptions, Source},
    task::{Pipeline, Task},
};

#[derive(Debug)]
pub struct UnconditionalImageGenerationPipeline {
    model: Box<dyn Denoiser>,
    scheduler: SchedulerConfig,
    options: LoadOptions,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct GenerationSeed {
    pub seed: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(default)]
pub struct UnconditionalImageGenerationParameters {
    pub num_inference_steps: usize,
}

impl Default for UnconditionalImageGenerationParameters {
    fn default() -> Self {
        Self {
            num_inference_steps: 1000,
        }
    }
}

// Each sample's pixels, `[B, C, H, W]` in `[-1, 1]`, as images, rounded as `numpy_to_pil` rounds them.
pub(crate) fn images(samples: &Tensor) -> Result<Vec<Image>> {
    let dimensions = samples.shape().dimensions();
    let (height, width) = (dimensions[2], dimensions[3]);
    let bytes = samples
        .div_scalar(2.0)?
        .add_scalar(0.5)?
        .clamp(0.0, 1.0)?
        .mul_scalar(255.0)?
        .round()?
        .permute(&[0, 2, 3, 1])?
        .contiguous()?
        .cast(DataType::F32)?
        .to_vec::<f32>()?;
    let size =
        usize::try_from(height * width * 3).map_err(|_| Error::unsupported("an image size"))?;
    bytes
        .chunks(size)
        .map(|pixels| {
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "rounded within a byte"
            )]
            let pixels = pixels.iter().map(|&value| value as u8).collect();
            Ok(Image {
                width: u32::try_from(width).map_err(|_| Error::unsupported("an image width"))?,
                height: u32::try_from(height).map_err(|_| Error::unsupported("an image height"))?,
                pixels,
            })
        })
        .collect()
}

impl Pipeline for UnconditionalImageGenerationPipeline {
    const TASK: Task = Task::UnconditionalImageGeneration;

    type Input = GenerationSeed;

    type Parameters = UnconditionalImageGenerationParameters;

    type Output = Image;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self {
            model: model::load(source, options)?,
            scheduler: SchedulerConfig::load(source)?,
            options,
        })
    }

    fn run(
        &self,
        inputs: &[Self::Input],
        parameters: &Self::Parameters,
    ) -> Result<Vec<Self::Output>> {
        let scheduler = DdpmScheduler::create(&self.scheduler, parameters.num_inference_steps)?;
        let [channels, height, width] = self.model.sample_shape();
        let shape = [1, channels, height, width];
        let device = self.options.device;
        inputs
            .iter()
            .map(|input| {
                let mut generator = TorchGenerator::seeded(input.seed);
                let mut sample = generator.normal_tensor(device, &shape)?;
                for (index, &timestep) in scheduler.timesteps().iter().enumerate() {
                    let predicted = self.model.predict(&sample, timestep)?;
                    sample = scheduler.step(index, &predicted, &sample, || {
                        generator.normal_tensor(device, &shape)
                    })?;
                }
                Ok(images(&sample)?.remove(0))
            })
            .collect()
    }
}
