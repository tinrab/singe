// TAESD, `diffusers`' `AutoencoderTiny`: a tiny autoencoder of plain convolutions for Stable Diffusion's latents.
// Its snapshots carry no `pipeline_tag`, and it sits under `image_to_image` with the other autoencoders.

use std::path::Path;

use serde::Deserialize;
use singe::{
    DataType, Device, Result as SingeResult, Tensor,
    nn::{
        Activation, Conv2d, Conv2dConfig, Layer, Model, ModelState, Sequential, StateBuilder,
        Upsample,
    },
};

use crate::{
    assembly::load_weights,
    config::{Snapshot, unsupported},
    error::Result,
    schema::Schema,
    vision::image_to_image::Autoencoder,
};

// The fields of an `AutoencoderTiny` `config.json` this definition reads, all of which `diffusers` writes.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct TaesdConfig {
    pub in_channels: i64,
    pub out_channels: i64,
    pub latent_channels: i64,
    pub encoder_block_out_channels: Vec<i64>,
    pub decoder_block_out_channels: Vec<i64>,
    pub num_encoder_blocks: Vec<usize>,
    pub num_decoder_blocks: Vec<usize>,
    pub act_fn: Activation,
    pub upsample_fn: String,
    pub upsampling_scaling_factor: i64,
}

impl TaesdConfig {
    fn validate(&self) -> Result<()> {
        if self.encoder_block_out_channels.len() != self.num_encoder_blocks.len()
            || self.decoder_block_out_channels.len() != self.num_decoder_blocks.len()
            || self.encoder_block_out_channels.is_empty()
            || self.decoder_block_out_channels.is_empty()
        {
            return Err(unsupported("stage counts that differ from their widths"));
        }

        if !matches!(self.upsample_fn.as_str(), "nearest" | "bilinear") {
            return Err(unsupported("upsampling other than nearest or bilinear"));
        }
        Ok(())
    }

    fn upsample(&self) -> Upsample {
        match self.upsample_fn.as_str() {
            "bilinear" => Upsample::bilinear(self.upsampling_scaling_factor),
            _ => Upsample::nearest(self.upsampling_scaling_factor),
        }
    }
}

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

fn block(
    schema: &mut Schema<'_>,
    input: i64,
    out: i64,
    activation: Activation,
) -> SingeResult<TinyBlock> {
    let convolutions = [
        schema.conv2d("conv.0", input, out, 3, padded(1), true)?,
        schema.conv2d("conv.2", out, out, 3, padded(1), true)?,
        schema.conv2d("conv.4", out, out, 3, padded(1), true)?,
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

// Numbers the layers of a `torch.nn.Sequential`, whose stateless layers take an index too.
#[derive(Debug, Default)]
struct Indices(usize);

impl Indices {
    fn next(&mut self) -> String {
        let name = format!("layers.{}", self.0);
        self.0 += 1;
        name
    }
}

// `EncoderTiny`: a convolution and blocks per stage, each stage after the first halving the image with a strided convolution.
fn encoder(schema: &mut Schema<'_>, config: &TaesdConfig) -> SingeResult<Sequential> {
    let mut layers = Sequential::new().with(Map(to_unit_range));
    let mut index = Indices::default();
    let stages = config
        .encoder_block_out_channels
        .iter()
        .zip(&config.num_encoder_blocks);
    for (stage, (&width, &blocks)) in stages.enumerate() {
        // As `diffusers` builds it, a later stage's convolution reads its own width, so widths of stages after the first agree.
        layers = layers.with(if stage == 0 {
            schema.conv2d(&index.next(), config.in_channels, width, 3, padded(1), true)?
        } else {
            schema.conv2d(&index.next(), width, width, 3, padded(2), false)?
        });
        for _ in 0..blocks {
            layers = layers.with(block(
                &mut schema.scope(index.next()),
                width,
                width,
                config.act_fn,
            )?);
        }
    }

    let last = config.encoder_block_out_channels[config.encoder_block_out_channels.len() - 1];
    Ok(layers.with(schema.conv2d(
        &index.next(),
        last,
        config.latent_channels,
        3,
        padded(1),
        true,
    )?))
}

// `DecoderTiny`: a convolution and the activation, then blocks per stage, each stage before the last doubling the image before its convolution.
fn decoder(schema: &mut Schema<'_>, config: &TaesdConfig) -> SingeResult<Sequential> {
    let widths = &config.decoder_block_out_channels;
    let mut index = Indices::default();
    let mut layers = Sequential::new().with(Map(soft_clamp)).with(schema.conv2d(
        &index.next(),
        config.latent_channels,
        widths[0],
        3,
        padded(1),
        true,
    )?);
    index.next();
    layers = layers.with(config.act_fn);
    let stages = widths.iter().zip(&config.num_decoder_blocks);
    for (stage, (&width, &blocks)) in stages.enumerate() {
        let last = stage + 1 == widths.len();
        for _ in 0..blocks {
            layers = layers.with(block(
                &mut schema.scope(index.next()),
                width,
                width,
                config.act_fn,
            )?);
        }

        if !last {
            index.next();
            layers = layers.with(config.upsample());
        }

        let out = if last { config.out_channels } else { width };
        layers = layers.with(schema.conv2d(&index.next(), width, out, 3, padded(1), last)?);
    }
    Ok(layers.with(Map(to_signed_range)))
}

// Loads one half, whose checkpoint names start with `prefix`, from the snapshot's whole checkpoint.
fn half(
    snapshot: &Path,
    device: Device,
    data_type: DataType,
    config: &TaesdConfig,
    prefix: &'static str,
    build: fn(&mut Schema<'_>, &TaesdConfig) -> SingeResult<Sequential>,
) -> Result<Model<Sequential>> {
    let mut builder = StateBuilder::new();
    let layers = build(&mut Schema::new(builder.scope(prefix), data_type), config)?;
    let schema = Model::new(layers, builder.finish()?);
    let names = move |name: &str| {
        name.strip_prefix(prefix)
            .is_some_and(|rest| rest.starts_with('.'))
            .then(|| name.to_owned())
    };
    load_weights(&schema, snapshot, device, Some(&names))
}

// Loads a `diffusers` snapshot directory, converting the weights to `data_type`.
pub fn load(
    snapshot: impl AsRef<Path>,
    device: Device,
    data_type: DataType,
) -> Result<Autoencoder> {
    let snapshot = snapshot.as_ref();
    let config: TaesdConfig = Snapshot::new(snapshot).config()?;
    config.validate()?;
    Ok(Autoencoder::new(
        half(snapshot, device, data_type, &config, "encoder", encoder)?,
        half(snapshot, device, data_type, &config, "decoder", decoder)?,
    ))
}
