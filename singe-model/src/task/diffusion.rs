use std::marker::PhantomData;

use serde::Deserialize;
use singe::{DataType, Error as SingeError, Tensor, nn::TokenBatch};

use crate::{
    error::{Error, Result},
    head::{Autoencoder, ConditionalDenoiser, TextEncoder},
    model::{
        self,
        diffusion::{
            shap_e::{Mesh, Prior, Renderer, load_prior, load_renderer},
            unet_2d_condition::{UnetMotion, load_motion},
        },
    },
    processor::{
        image::{Image, PixelRange, Resample, extent},
        random::TorchGenerator,
        scheduler::{Sampler, SchedulerConfig},
        text::TextTokenizer,
    },
    source::{LoadOptions, Source},
};

pub trait DenoisingDefaults {
    const STEPS: usize;
    const GUIDANCE: f32;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StableDiffusionDefaults;

impl DenoisingDefaults for StableDiffusionDefaults {
    const STEPS: usize = 50;
    const GUIDANCE: f32 = 7.5;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InstructPix2PixDefaults;

impl DenoisingDefaults for InstructPix2PixDefaults {
    const STEPS: usize = 100;
    const GUIDANCE: f32 = 7.5;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShapEDefaults;

impl DenoisingDefaults for ShapEDefaults {
    const STEPS: usize = 25;
    const GUIDANCE: f32 = 4.0;
}

#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(default, bound = "D: DenoisingDefaults")]
pub struct Denoising<D> {
    pub num_inference_steps: usize,
    pub guidance_scale: f32,
    #[serde(skip)]
    defaults: PhantomData<D>,
}

impl<D: DenoisingDefaults> Default for Denoising<D> {
    fn default() -> Self {
        Self {
            num_inference_steps: D::STEPS,
            guidance_scale: D::GUIDANCE,
            defaults: PhantomData,
        }
    }
}

impl<D> Denoising<D> {
    #[must_use]
    pub fn guidance(&self) -> Option<f32> {
        (self.guidance_scale > 1.0).then_some(self.guidance_scale)
    }
}

pub type ShapEParameters = Denoising<ShapEDefaults>;

fn guide_pair(pair: &Tensor, scale: f32) -> Result<Tensor> {
    let (unconditioned, conditioned) = (pair.narrow(0, 0, 1)?, pair.narrow(0, 1, 1)?);
    Ok(unconditioned.add(
        &conditioned
            .sub(&unconditioned)?
            .mul_scalar(f64::from(scale))?,
    )?)
}

pub(crate) fn denoise(
    scheduler: &mut Sampler,
    mut latents: Tensor,
    mut predict: impl FnMut(&Sampler, usize, &Tensor) -> Result<Tensor>,
    mut noise: impl FnMut() -> Result<Tensor>,
) -> Result<Tensor> {
    for index in 0..scheduler.timesteps().len() {
        let predicted = predict(scheduler, index, &latents)?;
        latents = scheduler.step(index, &predicted, &latents, &mut noise)?;
    }
    Ok(latents)
}

fn guided(
    latents: &Tensor,
    guidance: Option<f32>,
    predict: impl FnOnce(&Tensor) -> Result<Tensor>,
) -> Result<Tensor> {
    match guidance {
        Some(scale) => guide_pair(
            &predict(&Tensor::concat(&[latents.clone(), latents.clone()], 0)?)?,
            scale,
        ),
        None => predict(latents),
    }
}

/// A Stable Diffusion pipeline's components: its text encoder and tokenizer, its `UNet`, by default an image `UNet`, its VAE, and its scheduler's config.
#[derive(Debug)]
pub struct StableDiffusion<U = Box<dyn ConditionalDenoiser>> {
    pub text_encoder: TextEncoder,
    pub tokenizer: TextTokenizer,
    pub unet: U,
    pub vae: Autoencoder,
    pub scheduler: SchedulerConfig,
    /// The VAE config's `scaling_factor`: encoded latents are multiplied by it before denoising, and denoised latents divided by it before decoding.
    pub scaling: f32,
    /// The pixels a latent spans along each side.
    pub vae_scale: i64,
    pub options: LoadOptions,
}

#[derive(Debug, Deserialize)]
struct VaeConfig {
    #[serde(default = "stable_diffusion_scaling")]
    scaling_factor: f32,
    /// The widths of the VAE's blocks; `AutoencoderTiny` saves these as its decoder's widths, `decoder_block_out_channels`, which `diffusers` registers as `block_out_channels`.
    #[serde(alias = "decoder_block_out_channels")]
    block_out_channels: Vec<i64>,
}

/// Returns `AutoencoderKL`'s default scaling factor, which older configs leave out.
const fn stable_diffusion_scaling() -> f32 {
    0.18215
}

impl VaeConfig {
    /// Returns `2^(blocks − 1)`, as `diffusers`' pipelines derive the VAE's scale factor.
    fn scale_factor(&self) -> Result<i64> {
        self.block_out_channels
            .len()
            .checked_sub(1)
            .and_then(|halvings| u32::try_from(halvings).ok())
            .and_then(|halvings| 2_i64.checked_pow(halvings))
            .ok_or_else(|| Error::unsupported("a VAE whose blocks give no scale factor"))
    }
}

/// The tokens a prompt takes, as CLIP's tokenizer pads and truncates a prompt for Stable Diffusion.
const PROMPT_TOKENS: usize = 77;

/// Returns a prompt's tokens, truncated to 77 with the end token kept, as CLIP's tokenizer truncates them.
pub(crate) fn prompt_tokens(tokenizer: &TextTokenizer, prompt: &str) -> Result<Vec<i64>> {
    let mut tokens = tokenizer.encode(prompt, true)?;
    if tokens.len() > PROMPT_TOKENS {
        let end = tokens[tokens.len() - 1];
        tokens.truncate(PROMPT_TOKENS - 1);
        tokens.push(end);
    }
    Ok(tokens)
}

impl StableDiffusion {
    /// Loads a Stable Diffusion pipeline from the components `model_index.json` names, each from its directory.
    pub fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Self::load_with(
            source,
            options,
            model::load(&source.component("unet")?, options)?,
        )
    }

    pub fn predict(
        &self,
        scheduler: &Sampler,
        index: usize,
        latents: &Tensor,
        contexts: &Tensor,
        guidance: Option<f32>,
    ) -> Result<Tensor> {
        let timestep = scheduler.timesteps()[index];
        guided(latents, guidance, |batch| {
            self.unet
                .predict(&scheduler.scale_input(index, batch)?, timestep, contexts)
        })
    }
}

impl<U> StableDiffusion<U> {
    /// Loads the components other than the `UNet`, and takes `unet` in its place.
    pub fn load_with(source: &Source, options: LoadOptions, unet: U) -> Result<Self> {
        let vae = source.component("vae")?;
        let vae_config = vae.json::<VaeConfig>("config.json")?;
        Ok(Self {
            text_encoder: model::load(&source.component("text_encoder")?, options)?,
            tokenizer: TextTokenizer::load(&source.subdirectory("tokenizer"))?,
            unet,
            scaling: vae_config.scaling_factor,
            vae_scale: vae_config.scale_factor()?,
            vae: model::load(&vae, options)?,
            scheduler: SchedulerConfig::load(source)?,
            options,
        })
    }

    /// Encodes a prompt to its text states, `[1, 77, D]` in `F32`: its tokens truncated, padded with the tokenizer's pad token, and encoded whole.
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

    pub fn contexts(&self, prompt: &str, guidance: Option<f32>) -> Result<Tensor> {
        let context = self.encode_prompt(prompt)?;
        Ok(match guidance {
            Some(_) => Tensor::concat(&[self.encode_prompt("")?, context], 0)?,
            None => context,
        })
    }

    /// Returns the latents' height and width for images of `height × width`, which `diffusers` refuses unless both are multiples of 8.
    ///
    /// # Errors
    ///
    /// Returns an error if `height` or `width` is not a multiple of 8.
    pub fn latent_size(&self, height: i64, width: i64) -> Result<(i64, i64)> {
        if height % 8 != 0 || width % 8 != 0 {
            return Err(Error::invalid(
                "an image size whose sides are not multiples of 8",
            ));
        }
        Ok((height / self.vae_scale, width / self.vae_scale))
    }

    /// Returns an image in `[-1, 1]`, `[1, 3, H, W]` in the model's type, as `VaeImageProcessor.preprocess` gives it: resized with PIL's Lanczos filter to `size`, `(height, width)`, or without one to its sides floored to multiples of the VAE's scale factor, where it differs.
    pub fn image_pixels(&self, image: &Image, size: Option<(i64, i64)>) -> Result<Tensor> {
        let (height, width) = size.unwrap_or_else(|| {
            let floored = |length: u32| {
                let length = i64::from(length);
                length - length % self.vae_scale
            };
            (floored(image.height), floored(image.width))
        });
        let (height, width) = (extent(height)?, extent(width)?);
        if height == 0 || width == 0 {
            return Err(Error::invalid(
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

    /// Returns latents sampled from the VAE's distribution for `pixels`, drawing their noise from `generator`, as `latent_dist.sample(generator)` does.
    pub fn sampled_latents(
        &self,
        pixels: &Tensor,
        generator: &mut TorchGenerator,
    ) -> Result<Tensor> {
        let (mean, deviation) = self.vae.encode_distribution(pixels)?;
        let (mean, deviation) = (
            mean.cast(DataType::F32)?,
            deviation
                .ok_or_else(|| Error::unsupported("a VAE without a latent distribution"))?
                .cast(DataType::F32)?,
        );
        let noise = generator.normal_tensor(self.options.device, mean.shape().dimensions())?;
        Ok(mean.add(&deviation.mul(&noise)?)?)
    }

    /// Decodes latents, `[B, C, h, w]`, to images, dividing them by `scaling` first.
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
        Image::from_tensor(&decoded, PixelRange::Signed)
    }
}

/// An `AnimateDiff` pipeline: Stable Diffusion's components, its `UNet` joined by the `motion_adapter` component.
pub type AnimateDiff = StableDiffusion<UnetMotion>;

/// Loads an `AnimateDiff` pipeline whose `unet` component the `motion_adapter` component extends.
pub fn load_animatediff(source: &Source, options: LoadOptions) -> Result<AnimateDiff> {
    let unet = load_motion(
        &source.component("unet")?,
        &source.component("motion_adapter")?,
        options,
    )?;
    StableDiffusion::load_with(source, options, unet)
}

impl AnimateDiff {
    #[must_use]
    pub fn latent_channels(&self) -> i64 {
        self.unet.unet.layer().config().unet.in_channels
    }

    /// Returns noise of `shape` drawn from `generator` in the model's type, as `diffusers` keeps a clip's latents in it.
    pub fn clip_noise(&self, generator: &mut TorchGenerator, shape: &[i64]) -> Result<Tensor> {
        Ok(generator
            .normal_tensor(self.options.device, shape)?
            .cast(self.options.data_type)?)
    }

    pub fn predict_clip(
        &self,
        scheduler: &Sampler,
        index: usize,
        latents: &Tensor,
        contexts: &Tensor,
        guidance: Option<f32>,
    ) -> Result<Tensor> {
        let timestep = scheduler.timesteps()[index];
        guided(latents, guidance, |batch| {
            self.unet
                .predict(&scheduler.scale_input(index, batch)?, timestep, contexts)
        })
    }

    pub fn predict_masked_clip(
        &self,
        scheduler: &Sampler,
        index: usize,
        latents: &Tensor,
        condition: &Tensor,
        contexts: &Tensor,
        guidance: Option<f32>,
    ) -> Result<Tensor> {
        let timestep = scheduler.timesteps()[index];
        guided(latents, guidance, |batch| {
            let input = Tensor::concat(
                &[scheduler.scale_input(index, batch)?, condition.clone()],
                1,
            )?;
            self.unet.predict(&input, timestep, contexts)
        })
    }

    /// Decodes a clip's latents, `[1, C, F, h, w]`, to its frames, in chunks of 16 frames as `decode_latents` decodes them.
    pub fn decode_clip(&self, latents: &Tensor) -> Result<Vec<Image>> {
        let &[_, channels, frames, height, width] = latents.shape().dimensions() else {
            return Err(SingeError::invalid_argument("clip latents, [1, C, F, h, w]").into());
        };
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

/// Shap-E's prior, renderer, and scheduler, which its text and image pipelines share.
#[derive(Debug)]
pub struct ShapE {
    prior: Prior,
    renderer: Renderer,
    scheduler: SchedulerConfig,
    options: LoadOptions,
}

impl ShapE {
    pub fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self {
            prior: load_prior(&source.component("prior")?, options)?,
            renderer: load_renderer(&source.component("shap_e_renderer")?, options)?,
            scheduler: SchedulerConfig::load(source)?,
            options,
        })
    }

    #[must_use]
    pub const fn options(&self) -> LoadOptions {
        self.options
    }

    /// Generates a mesh from the latent the prior denoises from `seed`'s noise, conditioned on `condition`, `[1, C]` or `[1, K, C]`, and guided away from a zero condition.
    pub fn generate(
        &self,
        condition: &Tensor,
        seed: u64,
        parameters: &ShapEParameters,
    ) -> Result<Mesh> {
        let mut scheduler = Sampler::create(&self.scheduler, parameters.num_inference_steps)?;
        let config = self.prior.config();
        let (count, width) = (config.embeddings, config.embedding_width);
        let mut generator = TorchGenerator::seeded(seed);
        let shape = [1, count, width];
        let latent = generator
            .normal_tensor(self.options.device, &shape)?
            .mul_scalar(f64::from(scheduler.initial_scale()))?;
        let guidance = parameters.guidance();
        let conditions = match guidance {
            Some(_) => Tensor::concat(&[condition.zeros_like()?, condition.clone()], 0)?,
            None => condition.clone(),
        };
        let latent = denoise(
            &mut scheduler,
            latent,
            |scheduler, index, latent| {
                let timestep = scheduler.timesteps()[index];
                guided(latent, guidance, |batch| {
                    self.prior
                        .predict(&scheduler.scale_input(index, batch)?, timestep, &conditions)
                })
            },
            || generator.normal_tensor(self.options.device, &shape),
        )?;
        self.renderer.decode_mesh(&latent.squeeze(0)?)
    }
}
