// Reads a `diffusers` `AutoencoderTiny` config and imports its checkpoint.
//
// `diffusers` stores each half as a `torch.nn.Sequential`, `encoder.layers.N` and `decoder.layers.N`, whose stateless layers take an index too.

use serde::Deserialize;
use singe::nn::Activation;

use super::{AutoencoderTinyConfig, Interpolation};
use crate::{
    error::{Error, Result},
    format::{Rules, transformers::activation_field},
    source::Source,
};

// The fields of an `AutoencoderTiny` `config.json` this reads, all of which `diffusers` writes.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct Fields {
    in_channels: i64,
    out_channels: i64,
    latent_channels: i64,
    encoder_block_out_channels: Vec<i64>,
    decoder_block_out_channels: Vec<i64>,
    num_encoder_blocks: Vec<usize>,
    num_decoder_blocks: Vec<usize>,
    // `diffusers` names activations as `transformers` does.
    #[serde(deserialize_with = "activation_field")]
    act_fn: Activation,
    upsample_fn: String,
    upsampling_scaling_factor: i64,
}

pub(crate) fn config(source: &Source) -> Result<AutoencoderTinyConfig> {
    let fields: Fields = source.json("config.json")?;
    if fields.encoder_block_out_channels.len() != fields.num_encoder_blocks.len()
        || fields.decoder_block_out_channels.len() != fields.num_decoder_blocks.len()
        || fields.encoder_block_out_channels.is_empty()
        || fields.decoder_block_out_channels.is_empty()
    {
        return Err(Error::unsupported(
            "stage counts that differ from their widths",
        ));
    }

    let upsampling = match fields.upsample_fn.as_str() {
        "nearest" => Interpolation::Nearest,
        "bilinear" => Interpolation::Bilinear,
        _ => {
            return Err(Error::unsupported(
                "upsampling other than nearest or bilinear",
            ));
        }
    };
    Ok(AutoencoderTinyConfig {
        input_channels: fields.in_channels,
        output_channels: fields.out_channels,
        latent_channels: fields.latent_channels,
        encoder_widths: fields.encoder_block_out_channels,
        encoder_blocks: fields.num_encoder_blocks,
        decoder_widths: fields.decoder_block_out_channels,
        decoder_blocks: fields.num_decoder_blocks,
        activation: fields.act_fn,
        upsampling,
        upsampling_factor: fields.upsampling_scaling_factor,
    })
}

// Numbers the layers of one half's `torch.nn.Sequential` in order, renaming each stateful one to its Singe name.
#[derive(Debug)]
struct Layers {
    half: &'static str,
    next: usize,
    renames: Vec<(String, String)>,
}

impl Layers {
    const fn new(half: &'static str) -> Self {
        Self {
            half,
            next: 0,
            renames: Vec::new(),
        }
    }

    fn take(&mut self) -> String {
        let name = format!("{}.layers.{}", self.half, self.next);
        self.next += 1;
        name
    }

    // Passes over a stateless layer, such as an activation or an upsampling.
    fn stateless(&mut self) {
        self.take();
    }

    fn convolution(&mut self, to: String) {
        let from = self.take();
        self.renames.push((from, to));
    }

    // Stage `stage`'s `AutoencoderTinyBlock`s, whose `conv` is a `torch.nn.Sequential` of convolutions and activations.
    fn blocks(&mut self, stage: usize, count: usize) {
        for index in 0..count {
            let from = self.take();
            let to = format!("stages.{stage}.blocks.{index}");
            for (name, target) in [
                ("conv.0", "convolutions.0"),
                ("conv.2", "convolutions.1"),
                ("conv.4", "convolutions.2"),
                ("skip", "skip"),
            ] {
                self.renames
                    .push((format!("{from}.{name}"), format!("{to}.{target}")));
            }
        }
    }

    fn rules(&self) -> Rules {
        self.renames
            .iter()
            .fold(Rules::new(), |rules, (from, to)| rules.rename(from, to))
    }
}

// Imports `EncoderTiny`: per stage, a convolution and blocks, then the convolution to latents.
pub(crate) fn encoder_rules(config: &AutoencoderTinyConfig) -> Rules {
    let mut layers = Layers::new("encoder");
    for (stage, &count) in config.encoder_blocks.iter().enumerate() {
        layers.convolution(if stage == 0 {
            "input".to_owned()
        } else {
            format!("stages.{stage}.downsample")
        });
        layers.blocks(stage, count);
    }

    layers.convolution("output".to_owned());
    layers.rules()
}

// Imports `DecoderTiny`: a convolution and the activation, then each stage's blocks, followed by an upsampling and a convolution in every stage but the last, which ends in the convolution to images.
pub(crate) fn decoder_rules(config: &AutoencoderTinyConfig) -> Rules {
    let mut layers = Layers::new("decoder");
    layers.convolution("input".to_owned());
    layers.stateless();
    let stages = config.decoder_blocks.len();
    for (stage, &count) in config.decoder_blocks.iter().enumerate() {
        layers.blocks(stage, count);
        if stage + 1 == stages {
            layers.convolution("output".to_owned());
        } else {
            layers.stateless();
            layers.convolution(format!("stages.{stage}.upsample"));
        }
    }
    layers.rules()
}
