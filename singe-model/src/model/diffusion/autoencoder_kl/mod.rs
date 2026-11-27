//! `diffusers`' `AutoencoderKL`, Stable Diffusion's VAE: an encoder of residual blocks down to a latent distribution, its means and log-variances, and a decoder of residual blocks back up to an image, each with a self-attending middle block.
//!
//! Its encoder and decoder load as two parts, `encoder` and `decoder`.
//! A part's state is `input`, its stages `stages.N` (`resnets.M`, and `downsample` or `upsample` but in the last), `middle` (`resnets.0`, `attentions.0`, `resnets.1`), `norm`, `output`, and `quant`, the 1 × 1 convolution after the encoder or before the decoder.

mod diffusers;

use serde::{Deserialize, Serialize};
use singe::{
    Result as SingeResult, Tensor,
    nn::{Activation, Conv2d, GroupNorm, Layer, ModelState, Sequential},
};

use crate::{
    assembly::{
        GroupNormSpec, MiddleBlock, ResnetBlock, SpatialAttention, UpsampleBlock, downsample,
        resnet_stack, same_2d,
    },
    error::Result,
    format::Loader,
    head::{Autoencoder, HeadLoader},
    model::ModelEntry,
    schema::Schema,
    source::Format,
};

const NORM_EPS: f32 = 1e-6;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AutoencoderKlConfig {
    pub in_channels: i64,
    pub out_channels: i64,
    pub latent_channels: i64,
    pub widths: Vec<i64>,
    pub layers_per_block: usize,
    pub groups: i64,
}

impl AutoencoderKlConfig {
    const fn norms(&self) -> GroupNormSpec {
        GroupNormSpec {
            groups: self.groups,
            eps: NORM_EPS,
        }
    }
}

fn conv(
    schema: &mut Schema<'_>,
    name: &str,
    [input, output]: [i64; 2],
    kernel: i64,
) -> SingeResult<Conv2d> {
    schema.conv2d(name, input, output, kernel, same_2d(kernel, 1), true)
}

/// Creates `UNetMidBlock2D` without a timestep: a residual block, self-attention with one head, and another residual block.
fn middle(
    schema: &mut Schema<'_>,
    width: i64,
    norms: GroupNormSpec,
) -> SingeResult<MiddleBlock<SpatialAttention>> {
    MiddleBlock::create(
        &mut schema.scope("middle"),
        width,
        None,
        norms,
        1.0,
        |schema| SpatialAttention::create(schema, width, 1, norms, 1.0),
    )
}

#[derive(Debug, Clone)]
struct Stage<R> {
    resnets: Vec<ResnetBlock>,
    resample: Option<R>,
}

impl<R: Layer> Layer for Stage<R> {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let mut hidden = input.clone();
        for resnet in &self.resnets {
            hidden = resnet.forward(&hidden, None, state)?;
        }

        match &self.resample {
            Some(resample) => hidden.apply(resample, state),
            None => Ok(hidden),
        }
    }
}

/// The encoder: its input convolution, stages, middle, and output, then `quant`, from images to the latent distribution's means and log-variances.
#[derive(Debug, Clone)]
struct Encoder {
    input: Conv2d,
    stages: Vec<Stage<Conv2d>>,
    middle: MiddleBlock<SpatialAttention>,
    norm: GroupNorm,
    output: Conv2d,
    quant: Conv2d,
}

impl Encoder {
    fn create(schema: &mut Schema<'_>, config: &AutoencoderKlConfig) -> SingeResult<Self> {
        let (widths, norms) = (&config.widths, config.norms());
        let last = widths.len() - 1;
        let stages = widths
            .iter()
            .enumerate()
            .map(|(index, &output)| {
                let mut stage = schema.scope(format!("stages.{index}"));
                Ok(Stage {
                    resnets: resnet_stack(
                        &mut stage,
                        config.layers_per_block,
                        [widths[index.saturating_sub(1)], output],
                        None,
                        norms,
                    )?,
                    resample: (index < last)
                        .then(|| downsample(&mut stage, "downsample", output, 0))
                        .transpose()?,
                })
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let latent = config.latent_channels;
        Ok(Self {
            input: conv(schema, "input", [config.in_channels, widths[0]], 3)?,
            stages,
            middle: middle(schema, widths[last], norms)?,
            norm: schema.group_norm("norm", widths[last], norms.groups, norms.eps)?,
            output: conv(schema, "output", [widths[last], 2 * latent], 3)?,
            quant: conv(schema, "quant", [2 * latent, 2 * latent], 1)?,
        })
    }
}

impl Layer for Encoder {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let mut hidden = input.apply(&self.input, state)?;
        for stage in &self.stages {
            hidden = hidden.apply(stage, state)?;
        }
        hidden
            .apply(&self.middle, state)?
            .apply(&self.norm, state)?
            .apply(&Activation::Silu, state)?
            .apply(&self.output, state)?
            .apply(&self.quant, state)
    }
}

/// The decoder: `quant`, then its input convolution, the middle, its stages, and the output.
#[derive(Debug, Clone)]
struct Decoder {
    quant: Conv2d,
    input: Conv2d,
    middle: MiddleBlock<SpatialAttention>,
    stages: Vec<Stage<UpsampleBlock>>,
    norm: GroupNorm,
    output: Conv2d,
}

impl Decoder {
    fn create(schema: &mut Schema<'_>, config: &AutoencoderKlConfig) -> SingeResult<Self> {
        let norms = config.norms();
        let reversed = config.widths.iter().rev().copied().collect::<Vec<_>>();
        let last = reversed.len() - 1;
        let stages = reversed
            .iter()
            .enumerate()
            .map(|(index, &output)| {
                let mut stage = schema.scope(format!("stages.{index}"));
                Ok(Stage {
                    resnets: resnet_stack(
                        &mut stage,
                        config.layers_per_block + 1,
                        [reversed[index.saturating_sub(1)], output],
                        None,
                        norms,
                    )?,
                    resample: (index < last)
                        .then(|| UpsampleBlock::create(&mut stage, "upsample", output))
                        .transpose()?,
                })
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let latent = config.latent_channels;
        Ok(Self {
            quant: conv(schema, "quant", [latent, latent], 1)?,
            input: conv(schema, "input", [latent, reversed[0]], 3)?,
            middle: middle(schema, reversed[0], norms)?,
            stages,
            norm: schema.group_norm("norm", reversed[last], norms.groups, norms.eps)?,
            output: conv(schema, "output", [reversed[last], config.out_channels], 3)?,
        })
    }
}

impl Layer for Decoder {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let mut hidden = input
            .apply(&self.quant, state)?
            .apply(&self.input, state)?
            .apply(&self.middle, state)?;
        for stage in &self.stages {
            hidden = hidden.apply(stage, state)?;
        }
        hidden
            .apply(&self.norm, state)?
            .apply(&Activation::Silu, state)?
            .apply(&self.output, state)
    }
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "autoencoder_kl",
    formats: &[(Format::Diffusers, &["AutoencoderKL"])],
    loaders: &[HeadLoader::Autoencoder(load)],
};

pub fn load(loader: &mut Loader<'_>) -> Result<Autoencoder> {
    let config: AutoencoderKlConfig = loader.config(diffusers::config)?;
    let data_type = loader.options().data_type;
    let encoder = Schema::model(data_type, |schema| {
        Encoder::create(schema, &config).map(|encoder| Sequential::new().with(encoder))
    })?;
    let decoder = Schema::model(data_type, |schema| {
        Decoder::create(schema, &config).map(|decoder| Sequential::new().with(decoder))
    })?;
    let encoder = loader.load("encoder", &encoder, &diffusers::encoder_rules())?;
    let decoder = loader.load("decoder", &decoder, &diffusers::decoder_rules())?;
    loader.finish()?;
    Ok(Autoencoder::new(encoder, decoder).with_distribution(config.latent_channels))
}
