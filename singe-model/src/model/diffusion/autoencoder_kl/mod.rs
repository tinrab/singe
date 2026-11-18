// `diffusers`' `AutoencoderKL`, Stable Diffusion's VAE: an encoder of residual blocks down to a latent distribution, its means and log-variances, and a decoder of residual blocks back up to an image, each with a self-attending middle block.
// Its encoder and decoder load as two parts, `encoder` and `decoder`.
// A part's state is `input`, its stages `stages.N` (`resnets.M`, and `downsample` or `upsample` but in the last), `middle` (`resnets.0`, `attentions.0`, `resnets.1`), `norm`, `output`, and `quant`, the 1 × 1 convolution after the encoder or before the decoder.

mod diffusers;

use serde::{Deserialize, Serialize};
use singe::{
    Result as SingeResult, Tensor,
    nn::{
        Activation, Conv2d, Conv2dConfig, GroupNorm, Layer, Model, ModelState, Sequential,
        StateBuilder,
    },
};

use crate::{
    assembly::{Norms, ResnetBlock, SpatialAttention, UpsampleBlock, downsample},
    error::Result,
    format::Loader,
    head::{Autoencoder, HeadLoader},
    model::ModelEntry,
    schema::Schema,
    source::Format,
};

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
    const fn norms(&self) -> Norms {
        Norms {
            groups: self.groups,
            eps: 1e-6,
        }
    }
}

fn conv(
    schema: &mut Schema<'_>,
    name: &str,
    input: i64,
    output: i64,
    kernel: i64,
) -> SingeResult<Conv2d> {
    let padding = kernel / 2;
    schema.conv2d(
        name,
        input,
        output,
        kernel,
        Conv2dConfig {
            padding: (padding, padding, padding, padding),
            ..Conv2dConfig::default()
        },
        true,
    )
}

// `UNetMidBlock2D` without a timestep: a residual block, self-attention with one head, and another residual block.
#[derive(Debug, Clone)]
struct Middle {
    first: ResnetBlock,
    attention: SpatialAttention,
    second: ResnetBlock,
}

impl Middle {
    fn create(schema: &mut Schema<'_>, width: i64, norms: Norms) -> SingeResult<Self> {
        let mut schema = schema.scope("middle");
        Ok(Self {
            first: ResnetBlock::create(
                &mut schema.scope("resnets.0"),
                [width, width],
                None,
                norms,
                1.0,
            )?,
            attention: SpatialAttention::create(
                &mut schema.scope("attentions.0"),
                width,
                1,
                norms,
                1.0,
            )?,
            second: ResnetBlock::create(
                &mut schema.scope("resnets.1"),
                [width, width],
                None,
                norms,
                1.0,
            )?,
        })
    }
}

impl Layer for Middle {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let hidden = self
            .first
            .forward(input, None, state)?
            .apply(&self.attention, state)?;
        self.second.forward(&hidden, None, state)
    }
}

#[derive(Debug, Clone)]
struct Stage {
    resnets: Vec<ResnetBlock>,
    downsample: Option<Conv2d>,
    upsample: Option<UpsampleBlock>,
}

impl Layer for Stage {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let mut hidden = input.clone();
        for resnet in &self.resnets {
            hidden = resnet.forward(&hidden, None, state)?;
        }

        if let Some(downsample) = &self.downsample {
            hidden = hidden.apply(downsample, state)?;
        }

        if let Some(upsample) = &self.upsample {
            hidden = hidden.apply(upsample, state)?;
        }
        Ok(hidden)
    }
}

// Either half: for the encoder, its stages, the middle, and the output, then `quant`, to the latent distribution's means and log-variances; for the decoder, `quant`, then the middle, its stages, and the output.
#[derive(Debug, Clone)]
struct Half {
    quant: Conv2d,
    input: Conv2d,
    stages: Vec<Stage>,
    middle: Middle,
    norm: GroupNorm,
    output: Conv2d,
    // Whether this half encodes, to the latent distribution's means and log-variances.
    encodes: bool,
}

impl Layer for Half {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let finish = |hidden: Tensor| -> SingeResult<Tensor> {
            hidden
                .apply(&self.norm, state)?
                .apply(&Activation::Silu, state)?
                .apply(&self.output, state)
        };
        if self.encodes {
            let mut hidden = input.apply(&self.input, state)?;
            for stage in &self.stages {
                hidden = hidden.apply(stage, state)?;
            }
            finish(hidden.apply(&self.middle, state)?)?.apply(&self.quant, state)
        } else {
            let mut hidden = input
                .apply(&self.quant, state)?
                .apply(&self.input, state)?
                .apply(&self.middle, state)?;
            for stage in &self.stages {
                hidden = hidden.apply(stage, state)?;
            }
            finish(hidden)
        }
    }
}

fn encoder(schema: &mut Schema<'_>, config: &AutoencoderKlConfig) -> SingeResult<Half> {
    let (widths, norms) = (&config.widths, config.norms());
    let last = widths.len() - 1;
    let stages = widths
        .iter()
        .enumerate()
        .map(|(index, &output)| {
            let mut stage = schema.scope(format!("stages.{index}"));
            let input = if index == 0 {
                widths[0]
            } else {
                widths[index - 1]
            };
            Ok(Stage {
                resnets: (0..config.layers_per_block)
                    .map(|layer| {
                        ResnetBlock::create(
                            &mut stage.scope(format!("resnets.{layer}")),
                            [if layer == 0 { input } else { output }, output],
                            None,
                            norms,
                            1.0,
                        )
                    })
                    .collect::<SingeResult<Vec<_>>>()?,
                downsample: (index < last)
                    .then(|| downsample(&mut stage, "downsample", output, 0))
                    .transpose()?,
                upsample: None,
            })
        })
        .collect::<SingeResult<Vec<_>>>()?;
    let latent = config.latent_channels;
    Ok(Half {
        input: conv(schema, "input", config.in_channels, widths[0], 3)?,
        stages,
        middle: Middle::create(schema, widths[last], norms)?,
        norm: schema.group_norm("norm", widths[last], config.groups, 1e-6)?,
        output: conv(schema, "output", widths[last], 2 * latent, 3)?,
        quant: conv(schema, "quant", 2 * latent, 2 * latent, 1)?,
        encodes: true,
    })
}

fn decoder(schema: &mut Schema<'_>, config: &AutoencoderKlConfig) -> SingeResult<Half> {
    let norms = config.norms();
    let reversed = config.widths.iter().rev().copied().collect::<Vec<_>>();
    let last = reversed.len() - 1;
    let stages = reversed
        .iter()
        .enumerate()
        .map(|(index, &output)| {
            let mut stage = schema.scope(format!("stages.{index}"));
            let input = reversed[index.saturating_sub(1)];
            Ok(Stage {
                resnets: (0..=config.layers_per_block)
                    .map(|layer| {
                        ResnetBlock::create(
                            &mut stage.scope(format!("resnets.{layer}")),
                            [if layer == 0 { input } else { output }, output],
                            None,
                            norms,
                            1.0,
                        )
                    })
                    .collect::<SingeResult<Vec<_>>>()?,
                downsample: None,
                upsample: (index < last)
                    .then(|| UpsampleBlock::create(&mut stage, "upsample", output))
                    .transpose()?,
            })
        })
        .collect::<SingeResult<Vec<_>>>()?;
    let latent = config.latent_channels;
    Ok(Half {
        quant: conv(schema, "quant", latent, latent, 1)?,
        input: conv(schema, "input", latent, reversed[0], 3)?,
        stages,
        middle: Middle::create(schema, reversed[0], norms)?,
        norm: schema.group_norm("norm", reversed[last], config.groups, 1e-6)?,
        output: conv(schema, "output", reversed[last], config.out_channels, 3)?,
        encodes: false,
    })
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "autoencoder_kl",
    formats: &[(Format::Diffusers, &["AutoencoderKL"])],
    heads: &[HeadLoader::Autoencoder(load)],
};

pub fn load(loader: &mut Loader<'_>) -> Result<Autoencoder> {
    let config: AutoencoderKlConfig = loader.config(diffusers::config)?;
    let data_type = loader.options().data_type;
    let part = |build: fn(&mut Schema<'_>, &AutoencoderKlConfig) -> SingeResult<Half>| -> Result<Model<Sequential>> {
        let mut builder = StateBuilder::new();
        let half = {
            let mut schema = Schema::root(&mut builder, data_type);
            build(&mut schema, &config)?
        };
        Ok(Model::new(Sequential::new().with(half), builder.finish()?))
    };
    let encoder = loader.load("encoder", &part(encoder)?, &diffusers::encoder_rules())?;
    let decoder = loader.load("decoder", &part(decoder)?, &diffusers::decoder_rules())?;
    loader.finish()?;
    Ok(Autoencoder::new(encoder, decoder).with_distribution(config.latent_channels))
}
