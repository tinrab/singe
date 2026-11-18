// TAESD, `diffusers`' `AutoencoderTiny`: a tiny autoencoder of plain convolutions for Stable Diffusion's latents.
// Its encoder and decoder load as two parts, `encoder` and `decoder`, each with its own state.
// A part's state is its `input` convolution, its stages under `stages.S` (blocks `blocks.B`, and the encoder's strided `downsample` or the decoder's `upsample` convolution), and its `output` convolution.

mod diffusers;

use serde::{Deserialize, Serialize};
use singe::{
    DataType, Result as SingeResult, Tensor,
    nn::{
        Activation, Conv2d, Conv2dConfig, Layer, Model, ModelState, Sequential, StateBuilder,
        Upsample,
    },
};

use crate::{
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
    // Each stage's width and block count, as many of one as of the other, and at least one stage.
    pub encoder_widths: Vec<i64>,
    pub encoder_blocks: Vec<usize>,
    pub decoder_widths: Vec<i64>,
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

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "autoencoder_tiny",
    formats: &[(Format::Diffusers, &["AutoencoderTiny"])],
    heads: &[HeadLoader::Autoencoder(load)],
};

// A step of no state between layers, such as rescaling an image's range.
#[derive(Debug, Clone, Copy)]
struct Map(fn(&Tensor) -> SingeResult<Tensor>);

impl Layer for Map {
    fn forward(&self, input: &Tensor, _state: &ModelState) -> SingeResult<Tensor> {
        (self.0)(input)
    }
}

// TAESD works on images in [0, 1], and `diffusers` passes them in [-1, 1].
fn to_unit_range(images: &Tensor) -> SingeResult<Tensor> {
    images.add_scalar(1.0)?.div_scalar(2.0)
}

fn to_signed_range(images: &Tensor) -> SingeResult<Tensor> {
    images.mul_scalar(2.0)?.sub_scalar(1.0)
}

// Keeps latents within (-3, 3) smoothly before decoding them.
fn soft_clamp(latents: &Tensor) -> SingeResult<Tensor> {
    latents.div_scalar(3.0)?.tanh()?.mul_scalar(3.0)
}

// `AutoencoderTinyBlock`: three padded 3 by 3 convolutions with the activation between them, plus the input, through a ReLU.
#[derive(Debug, Clone)]
struct TinyBlock {
    convolutions: [Conv2d; 3],
    skip: Option<Conv2d>,
    activation: Activation,
}

impl Layer for TinyBlock {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let [first, second, third] = &self.convolutions;
        let branch = input
            .apply(first, state)?
            .apply(&self.activation, state)?
            .apply(second, state)?
            .apply(&self.activation, state)?
            .apply(third, state)?;
        let skip = match &self.skip {
            Some(skip) => input.apply(skip, state)?,
            None => input.clone(),
        };
        branch.add(&skip)?.apply(&Activation::Relu, state)
    }
}

const fn padded(stride: i64) -> Conv2dConfig {
    Conv2dConfig {
        stride: (stride, stride),
        padding: (1, 1, 1, 1),
        dilation: (1, 1),
        groups: 1,
    }
}

// Registers `convolutions.0` to `convolutions.2`, and `skip` when the block changes the width.
fn block(
    schema: &mut Schema<'_>,
    input: i64,
    out: i64,
    activation: Activation,
) -> SingeResult<TinyBlock> {
    let convolutions = [
        schema.conv2d("convolutions.0", input, out, 3, padded(1), true)?,
        schema.conv2d("convolutions.1", out, out, 3, padded(1), true)?,
        schema.conv2d("convolutions.2", out, out, 3, padded(1), true)?,
    ];
    let skip = (input != out)
        .then(|| schema.conv2d("skip", input, out, 1, Conv2dConfig::default(), false))
        .transpose()?;
    Ok(TinyBlock {
        convolutions,
        skip,
        activation,
    })
}

// Adds stage `stage`'s `count` blocks of `width` to `layers`.
fn blocks(
    schema: &mut Schema<'_>,
    mut layers: Sequential,
    stage: usize,
    count: usize,
    width: i64,
    activation: Activation,
) -> SingeResult<Sequential> {
    for index in 0..count {
        let mut schema = schema.scope(format!("stages.{stage}.blocks.{index}"));
        layers = layers.with(block(&mut schema, width, width, activation)?);
    }
    Ok(layers)
}

// `EncoderTiny`: a convolution and blocks per stage, each stage after the first halving the image with a strided convolution.
fn encoder(schema: &mut Schema<'_>, config: &AutoencoderTinyConfig) -> SingeResult<Sequential> {
    let mut layers = Sequential::new().with(Map(to_unit_range));
    let stages = config.encoder_widths.iter().zip(&config.encoder_blocks);
    for (stage, (&width, &count)) in stages.enumerate() {
        // As `diffusers` builds it, a later stage's convolution reads its own width, so widths of stages after the first agree.
        layers = layers.with(if stage == 0 {
            schema.conv2d("input", config.input_channels, width, 3, padded(1), true)?
        } else {
            let name = format!("stages.{stage}.downsample");
            schema.conv2d(&name, width, width, 3, padded(2), false)?
        });
        layers = blocks(schema, layers, stage, count, width, config.activation)?;
    }

    let last = config.encoder_widths[config.encoder_widths.len() - 1];
    Ok(layers.with(schema.conv2d("output", last, config.latent_channels, 3, padded(1), true)?))
}

// `DecoderTiny`: a convolution and the activation, then blocks per stage, each stage before the last doubling the image before its convolution.
fn decoder(schema: &mut Schema<'_>, config: &AutoencoderTinyConfig) -> SingeResult<Sequential> {
    let widths = &config.decoder_widths;
    let mut layers = Sequential::new()
        .with(Map(soft_clamp))
        .with(schema.conv2d(
            "input",
            config.latent_channels,
            widths[0],
            3,
            padded(1),
            true,
        )?)
        .with(config.activation);
    let stages = widths.iter().zip(&config.decoder_blocks);
    for (stage, (&width, &count)) in stages.enumerate() {
        layers = blocks(schema, layers, stage, count, width, config.activation)?;
        layers = if stage + 1 == widths.len() {
            let output = config.output_channels;
            layers.with(schema.conv2d("output", width, output, 3, padded(1), true)?)
        } else {
            let name = format!("stages.{stage}.upsample");
            layers.with(config.upsample()).with(schema.conv2d(
                &name,
                width,
                width,
                3,
                padded(1),
                false,
            )?)
        };
    }
    Ok(layers.with(Map(to_signed_range)))
}

// Builds one half on `Device::Meta`.
fn half(
    config: &AutoencoderTinyConfig,
    data_type: DataType,
    build: fn(&mut Schema<'_>, &AutoencoderTinyConfig) -> SingeResult<Sequential>,
) -> Result<Model<Sequential>> {
    let mut builder = StateBuilder::new();
    let layers = build(&mut Schema::root(&mut builder, data_type), config)?;
    Ok(Model::new(layers, builder.finish()?))
}

pub fn load(loader: &mut Loader<'_>) -> Result<Autoencoder> {
    let config: AutoencoderTinyConfig = loader.config(diffusers::config)?;
    let data_type = loader.options().data_type;
    let encoder = loader.load(
        "encoder",
        &half(&config, data_type, encoder)?,
        &diffusers::encoder_rules(&config),
    )?;
    let decoder = loader.load(
        "decoder",
        &half(&config, data_type, decoder)?,
        &diffusers::decoder_rules(&config),
    )?;
    loader.finish()?;
    Ok(Autoencoder::new(encoder, decoder))
}
