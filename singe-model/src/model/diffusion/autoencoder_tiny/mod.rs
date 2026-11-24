//! TAESD, `diffusers`' `AutoencoderTiny`: a tiny autoencoder of plain convolutions for Stable Diffusion's latents.
//!
//! Its encoder and decoder load as two parts, `encoder` and `decoder`, each with its own state.
//! A part's state is its `input` convolution, its stages under `stages.S` (blocks `blocks.B`, and the encoder's strided `downsample` or the decoder's `upsample` convolution), and its `output` convolution.

mod diffusers;

use serde::{Deserialize, Serialize};
use singe::{
    Result as SingeResult, Tensor,
    nn::{Activation, Conv2d, Layer, ModelState, Sequential, Upsample},
};

use crate::{
    assembly::same_2d,
    error::Result,
    format::Loader,
    head::{Autoencoder, HeadLoader},
    model::ModelEntry,
    schema::Schema,
    source::Format,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Interpolation {
    Nearest,
    Bilinear,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AutoencoderTinyConfig {
    pub input_channels: i64,
    pub output_channels: i64,
    pub latent_channels: i64,
    /// Each encoder stage's width, one for each of `encoder_blocks`, with at least one stage.
    pub encoder_widths: Vec<i64>,
    /// Each encoder stage's block count.
    pub encoder_blocks: Vec<usize>,
    /// Each decoder stage's width, one for each of `decoder_blocks`, with at least one stage.
    pub decoder_widths: Vec<i64>,
    /// Each decoder stage's block count.
    pub decoder_blocks: Vec<usize>,
    pub activation: Activation,
    pub upsampling: Interpolation,
    pub upsampling_factor: i64,
}

impl AutoencoderTinyConfig {
    const fn upsample(&self) -> Upsample {
        match self.upsampling {
            Interpolation::Nearest => Upsample::nearest(self.upsampling_factor),
            Interpolation::Bilinear => Upsample::bilinear(self.upsampling_factor),
        }
    }
}

/// `AutoencoderTinyBlock`: three padded 3 by 3 convolutions with the activation between them, plus the input, through a `ReLU`.
#[derive(Debug, Clone)]
struct TinyBlock {
    convolutions: [Conv2d; 3],
    activation: Activation,
}

impl TinyBlock {
    fn create(schema: &mut Schema<'_>, width: i64, activation: Activation) -> SingeResult<Self> {
        let mut convolution =
            |name: &str| schema.conv2d(name, width, width, 3, same_2d(3, 1), true);
        Ok(Self {
            convolutions: [
                convolution("convolutions.0")?,
                convolution("convolutions.1")?,
                convolution("convolutions.2")?,
            ],
            activation,
        })
    }

    fn stack(
        schema: &mut Schema<'_>,
        stage: usize,
        count: usize,
        width: i64,
        activation: Activation,
    ) -> SingeResult<Vec<Self>> {
        (0..count)
            .map(|index| {
                Self::create(
                    &mut schema.scope(format!("stages.{stage}.blocks.{index}")),
                    width,
                    activation,
                )
            })
            .collect()
    }
}

impl Layer for TinyBlock {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let [first, second, third] = &self.convolutions;
        input
            .apply(first, state)?
            .apply(&self.activation, state)?
            .apply(second, state)?
            .apply(&self.activation, state)?
            .apply(third, state)?
            .add(input)?
            .apply(&Activation::Relu, state)
    }
}

#[derive(Debug, Clone)]
struct EncoderStage {
    downsample: Option<Conv2d>,
    blocks: Vec<TinyBlock>,
}

/// `EncoderTiny`: a convolution and blocks per stage, each stage after the first halving the image with a strided convolution, from images in [-1, 1], which TAESD reads in [0, 1].
#[derive(Debug, Clone)]
struct TinyEncoder {
    input: Conv2d,
    stages: Vec<EncoderStage>,
    output: Conv2d,
}

impl TinyEncoder {
    fn create(schema: &mut Schema<'_>, config: &AutoencoderTinyConfig) -> SingeResult<Self> {
        let widths = &config.encoder_widths;
        let input = schema.conv2d(
            "input",
            config.input_channels,
            widths[0],
            3,
            same_2d(3, 1),
            true,
        )?;
        let stages = widths
            .iter()
            .zip(&config.encoder_blocks)
            .enumerate()
            .map(|(stage, (&width, &count))| {
                // As `diffusers` builds it, a later stage's convolution reads its own width, so widths of stages after the first agree.
                let downsample = (stage > 0)
                    .then(|| {
                        let name = format!("stages.{stage}.downsample");
                        schema.conv2d(&name, width, width, 3, same_2d(3, 2), false)
                    })
                    .transpose()?;
                Ok(EncoderStage {
                    downsample,
                    blocks: TinyBlock::stack(schema, stage, count, width, config.activation)?,
                })
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let last = widths[widths.len() - 1];
        Ok(Self {
            input,
            stages,
            output: schema.conv2d(
                "output",
                last,
                config.latent_channels,
                3,
                same_2d(3, 1),
                true,
            )?,
        })
    }
}

impl Layer for TinyEncoder {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let mut hidden = input
            .add_scalar(1.0)?
            .div_scalar(2.0)?
            .apply(&self.input, state)?;
        for stage in &self.stages {
            if let Some(downsample) = &stage.downsample {
                hidden = hidden.apply(downsample, state)?;
            }
            for block in &stage.blocks {
                hidden = hidden.apply(block, state)?;
            }
        }
        hidden.apply(&self.output, state)
    }
}

#[derive(Debug, Clone)]
struct DecoderStage {
    blocks: Vec<TinyBlock>,
    upsample: Option<Conv2d>,
}

/// `DecoderTiny`: latents kept within (-3, 3) smoothly, a convolution and the activation, then blocks per stage, each stage before the last upsampling the image by `upsampling_factor` before its convolution, to images in [-1, 1].
#[derive(Debug, Clone)]
struct TinyDecoder {
    input: Conv2d,
    activation: Activation,
    stages: Vec<DecoderStage>,
    upsampling: Upsample,
    output: Conv2d,
}

impl TinyDecoder {
    fn create(schema: &mut Schema<'_>, config: &AutoencoderTinyConfig) -> SingeResult<Self> {
        let widths = &config.decoder_widths;
        let last = widths.len() - 1;
        let input = schema.conv2d(
            "input",
            config.latent_channels,
            widths[0],
            3,
            same_2d(3, 1),
            true,
        )?;
        let stages = widths
            .iter()
            .zip(&config.decoder_blocks)
            .enumerate()
            .map(|(stage, (&width, &count))| {
                let blocks = TinyBlock::stack(schema, stage, count, width, config.activation)?;
                let upsample = (stage < last)
                    .then(|| {
                        let name = format!("stages.{stage}.upsample");
                        schema.conv2d(&name, width, width, 3, same_2d(3, 1), false)
                    })
                    .transpose()?;
                Ok(DecoderStage { blocks, upsample })
            })
            .collect::<SingeResult<Vec<_>>>()?;
        Ok(Self {
            input,
            activation: config.activation,
            stages,
            upsampling: config.upsample(),
            output: schema.conv2d(
                "output",
                widths[last],
                config.output_channels,
                3,
                same_2d(3, 1),
                true,
            )?,
        })
    }
}

impl Layer for TinyDecoder {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let mut hidden = input
            .div_scalar(3.0)?
            .tanh()?
            .mul_scalar(3.0)?
            .apply(&self.input, state)?
            .apply(&self.activation, state)?;
        for stage in &self.stages {
            for block in &stage.blocks {
                hidden = hidden.apply(block, state)?;
            }
            if let Some(upsample) = &stage.upsample {
                hidden = hidden
                    .apply(&self.upsampling, state)?
                    .apply(upsample, state)?;
            }
        }
        hidden
            .apply(&self.output, state)?
            .mul_scalar(2.0)?
            .sub_scalar(1.0)
    }
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "autoencoder_tiny",
    formats: &[(Format::Diffusers, &["AutoencoderTiny"])],
    loaders: &[HeadLoader::Autoencoder(load)],
};

pub fn load(loader: &mut Loader<'_>) -> Result<Autoencoder> {
    let config: AutoencoderTinyConfig = loader.config(diffusers::config)?;
    let data_type = loader.options().data_type;
    let encoder = Schema::model(data_type, |schema| {
        TinyEncoder::create(schema, &config).map(|encoder| Sequential::new().with(encoder))
    })?;
    let decoder = Schema::model(data_type, |schema| {
        TinyDecoder::create(schema, &config).map(|decoder| Sequential::new().with(decoder))
    })?;
    let encoder = loader.load("encoder", &encoder, &diffusers::encoder_rules(&config))?;
    let decoder = loader.load("decoder", &decoder, &diffusers::decoder_rules(&config))?;
    loader.finish()?;
    Ok(Autoencoder::new(encoder, decoder))
}
