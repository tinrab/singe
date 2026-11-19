// `unconditional-image-generation`: a seed to an image, denoised from the seed's noise by a diffusion model under its scheduler, as `diffusers`' `DDPMPipeline` generates one with `torch.Generator("cpu").manual_seed(seed)`.

use serde::Deserialize;

use crate::{
    error::Result,
    head::Denoiser,
    model,
    processor::{
        image::{Image, PixelRange},
        random::TorchGenerator,
        scheduler::{Sampler, SchedulerConfig},
    },
    source::{LoadOptions, Source},
    task::{Pipeline, diffusion::denoise},
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

impl Pipeline for UnconditionalImageGenerationPipeline {
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

    fn run_one(&self, input: &Self::Input, parameters: &Self::Parameters) -> Result<Self::Output> {
        let [channels, height, width] = self.model.sample_shape();
        let shape = [1, channels, height, width];
        let device = self.options.device;
        let mut scheduler = Sampler::create(&self.scheduler, parameters.num_inference_steps)?;
        let mut generator = TorchGenerator::seeded(input.seed);
        let sample = generator.normal_tensor(device, &shape)?;
        let sample = denoise(
            &mut scheduler,
            sample,
            |scheduler, index, sample| self.model.predict(sample, scheduler.timesteps()[index]),
            || generator.normal_tensor(device, &shape),
        )?;
        Ok(Image::from_tensor(&sample, PixelRange::Signed)?.remove(0))
    }
}
