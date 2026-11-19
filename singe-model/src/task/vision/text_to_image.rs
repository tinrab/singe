// `text-to-image`: a prompt and a seed to an image, as `diffusers`' `StableDiffusionPipeline` generates one: the prompt's text states, latents denoised from the seed's noise under the scheduler, with classifier-free guidance where its scale exceeds one, and decoded by the VAE.

use serde::Deserialize;
use singe::{DataType, Tensor, nn::TokenBatch};

use crate::{
    error::{Error, Result},
    head::{Autoencoder, ConditionalDenoiser, TextEncoder},
    model,
    processing::{
        image::{Image, Resample},
        random::TorchGenerator,
        scheduler::{Sampler, SchedulerConfig},
        text::TextTokenizer,
    },
    source::{LoadOptions, Source},
    task::{Pipeline, Task, vision::unconditional_image_generation::images},
};

// A Stable Diffusion pipeline's components: its text encoder and tokenizer, its UNet, by default an image UNet, its VAE, and its scheduler's config.
#[derive(Debug)]
pub struct StableDiffusion<U = Box<dyn ConditionalDenoiser>> {
    pub text_encoder: TextEncoder,
    pub tokenizer: TextTokenizer,
    pub unet: U,
    pub vae: Autoencoder,
    pub scheduler: SchedulerConfig,
    // What the VAE's latents are multiplied by before denoising.
    pub scaling: f32,
    // The pixels a latent spans along each side.
    pub vae_scale: i64,
    pub options: LoadOptions,
}

#[derive(Debug, Deserialize)]
struct VaeConfig {
    #[serde(default = "stable_diffusion_scaling")]
    scaling_factor: f32,
    // `AutoencoderTiny` saves these as its decoder's widths, `decoder_block_out_channels`, which `diffusers` registers as `block_out_channels`.
    #[serde(alias = "decoder_block_out_channels")]
    block_out_channels: Vec<i64>,
}

// `AutoencoderKL`'s default, which older configs leave out.
const fn stable_diffusion_scaling() -> f32 {
    0.18215
}

impl VaeConfig {
    // `2^(blocks − 1)`, as `diffusers`' pipelines derive the VAE's scale factor.
    fn scale_factor(&self) -> Result<i64> {
        self.block_out_channels
            .len()
            .checked_sub(1)
            .and_then(|halvings| u32::try_from(halvings).ok())
            .and_then(|halvings| 2_i64.checked_pow(halvings))
            .ok_or_else(|| Error::unsupported("a VAE whose blocks give no scale factor"))
    }
}

// The tokens a prompt takes, as CLIP's tokenizer pads and truncates a prompt for Stable Diffusion.
const PROMPT_TOKENS: usize = 77;

// A prompt's tokens, truncated to 77 with the end token kept, as CLIP's tokenizer truncates them.
pub(crate) fn prompt_tokens(tokenizer: &TextTokenizer, prompt: &str) -> Result<Vec<i64>> {
    let mut tokens = tokenizer.encode(prompt, true)?;
    if tokens.len() > PROMPT_TOKENS {
        let end = tokens[tokens.len() - 1];
        tokens.truncate(PROMPT_TOKENS - 1);
        tokens.push(end);
    }
    Ok(tokens)
}

// Classifier-free guidance: the unconditioned prediction moved toward the conditioned one by `scale` times their difference.
pub(crate) fn guide(unconditioned: &Tensor, conditioned: &Tensor, scale: f32) -> Result<Tensor> {
    Ok(unconditioned.add(
        &conditioned
            .sub(unconditioned)?
            .mul_scalar(f64::from(scale))?,
    )?)
}

// `guide` over a batch of the unconditioned prediction and then the conditioned one, as pipelines batch them.
pub(crate) fn guide_pair(pair: &Tensor, scale: f32) -> Result<Tensor> {
    guide(&pair.narrow(0, 0, 1)?, &pair.narrow(0, 1, 1)?, scale)
}

impl StableDiffusion {
    // Loads the components `model_index.json` names from their directories.
    pub fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Self::load_with(
            source,
            options,
            model::load(&source.component("unet")?, options)?,
        )
    }
}

impl<U> StableDiffusion<U> {
    // Loads the components but the UNet, which `unet` is.
    pub fn load_with(source: &Source, options: LoadOptions, unet: U) -> Result<Self> {
        let vae = source.component("vae")?;
        let vae_config = vae.json::<VaeConfig>("config.json")?;
        Ok(Self {
            text_encoder: model::load(&source.component("text_encoder")?, options)?,
            tokenizer: TextTokenizer::load(source.directory().join("tokenizer"))?,
            unet,
            scaling: vae_config.scaling_factor,
            vae_scale: vae_config.scale_factor()?,
            vae: model::load(&vae, options)?,
            scheduler: SchedulerConfig::load(source)?,
            options,
        })
    }

    // A prompt's text states, `[1, 77, D]`: its tokens truncated, padded with the tokenizer's pad token, and encoded whole.
    pub fn encode_prompt(&self, prompt: &str) -> Result<Tensor> {
        let mut tokens = prompt_tokens(&self.tokenizer, prompt)?;
        let pad = self
            .tokenizer
            .special_id("pad_token")
            .ok_or_else(|| Error::unsupported("a tokenizer without a pad token"))?;
        tokens.resize(PROMPT_TOKENS, pad);
        let batch = TokenBatch::from_sequences(self.options.device, &[&tokens])?;
        Ok(self
            .text_encoder
            .forward_group(&batch)?
            .cast(DataType::F32)?)
    }

    // The latents' height and width for images of `height × width`, which `diffusers` refuses unless both are multiples of 8.
    pub fn latent_size(&self, height: i64, width: i64) -> Result<(i64, i64)> {
        if height % 8 != 0 || width % 8 != 0 {
            return Err(Error::unsupported(
                "an image size whose sides are not multiples of 8",
            ));
        }
        Ok((height / self.vae_scale, width / self.vae_scale))
    }

    // An image in [-1, 1], `[1, 3, H, W]` in the model's type, as `VaeImageProcessor.preprocess` gives it: resized with PIL's Lanczos filter to `size`, `(height, width)`, or without one to its sides floored to multiples of the VAE's scale factor, where it differs.
    pub fn image_pixels(&self, image: &Image, size: Option<(i64, i64)>) -> Result<Tensor> {
        let (height, width) = size.unwrap_or_else(|| {
            let floored = |length: u32| {
                let length = i64::from(length);
                length - length % self.vae_scale
            };
            (floored(image.height), floored(image.width))
        });
        let pixels =
            |length: i64| u32::try_from(length).map_err(|_| Error::unsupported("an image side"));
        let (height, width) = (pixels(height)?, pixels(width)?);
        if height == 0 || width == 0 {
            return Err(Error::unsupported(
                "an image smaller than the VAE's scale factor",
            ));
        }

        let resized;
        let image = if (image.height, image.width) == (height, width) {
            image
        } else {
            resized = image.resized(height, width, Resample::Lanczos);
            &resized
        };
        Ok(image
            .to_tensor(self.options.device)?
            .div_scalar(255.0)?
            .mul_scalar(2.0)?
            .sub_scalar(1.0)?
            .cast(self.options.data_type)?)
    }

    // Latents, `[B, 4, h, w]`, decoded to images.
    pub fn decode(&self, latents: &Tensor) -> Result<Vec<Image>> {
        // The CUDA kernel divides by a host scalar through its reciprocal.
        let decoded = self
            .vae
            .decode(
                &latents
                    .mul_scalar(f64::from(1.0 / self.scaling))?
                    .cast(self.options.data_type)?,
            )?
            .cast(DataType::F32)?;
        images(&decoded)
    }
}

impl StableDiffusion {
    // The noise the UNet predicts for `latents` at step `index`, guided toward `context` and away from `unconditioned` by `guidance` where that is given.
    pub fn guided(
        &self,
        scheduler: &Sampler,
        index: usize,
        latents: &Tensor,
        context: &Tensor,
        unconditioned: Option<(&Tensor, f32)>,
    ) -> Result<Tensor> {
        let input = scheduler.scale_input(index, latents)?;
        let timestep = scheduler.timesteps()[index];
        let conditioned = self.unet.predict(&input, timestep, context)?;
        match unconditioned {
            Some((unconditioned, guidance)) => guide(
                &self.unet.predict(&input, timestep, unconditioned)?,
                &conditioned,
                guidance,
            ),
            None => Ok(conditioned),
        }
    }
}

#[derive(Debug)]
pub struct TextToImagePipeline(StableDiffusion);

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct TextToImageInput {
    pub prompt: String,
    pub seed: u64,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default)]
pub struct TextToImageParameters {
    pub num_inference_steps: usize,
    pub guidance_scale: f32,
    pub height: i64,
    pub width: i64,
}

impl Default for TextToImageParameters {
    fn default() -> Self {
        Self {
            num_inference_steps: 50,
            guidance_scale: 7.5,
            height: 512,
            width: 512,
        }
    }
}

impl Pipeline for TextToImagePipeline {
    const TASK: Task = Task::TextToImage;

    type Input = TextToImageInput;

    type Parameters = TextToImageParameters;

    type Output = Image;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self(StableDiffusion::load(source, options)?))
    }

    fn run(
        &self,
        inputs: &[Self::Input],
        parameters: &Self::Parameters,
    ) -> Result<Vec<Self::Output>> {
        let parts = &self.0;
        let (height, width) = parts.latent_size(parameters.height, parameters.width)?;
        let shape = [1, parts.unet.in_channels(), height, width];
        let device = parts.options.device;
        let guided = parameters.guidance_scale > 1.0;
        inputs
            .iter()
            .map(|input| {
                let mut scheduler =
                    Sampler::create(&parts.scheduler, parameters.num_inference_steps)?;
                let context = parts.encode_prompt(&input.prompt)?;
                let unconditioned = guided.then(|| parts.encode_prompt("")).transpose()?;
                let mut generator = TorchGenerator::seeded(input.seed);
                let mut latents = generator
                    .normal_tensor(device, &shape)?
                    .mul_scalar(f64::from(scheduler.initial_scale()))?;
                for index in 0..scheduler.timesteps().len() {
                    let predicted = parts.guided(
                        &scheduler,
                        index,
                        &latents,
                        &context,
                        unconditioned
                            .as_ref()
                            .map(|states| (states, parameters.guidance_scale)),
                    )?;
                    latents = scheduler.step(index, &predicted, &latents, || {
                        generator.normal_tensor(device, &shape)
                    })?;
                }
                Ok(parts.decode(&latents)?.remove(0))
            })
            .collect()
    }
}
