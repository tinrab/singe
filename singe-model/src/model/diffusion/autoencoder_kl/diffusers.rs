// Reads a `diffusers` `AutoencoderKL`: its config and its tensors' names.

use serde::Deserialize;

use super::AutoencoderKlConfig;
use crate::{
    assembly::unet_block_rules,
    error::{Error, Result},
    format::Rules,
    source::Source,
};

fn silu() -> String {
    "silu".to_owned()
}

fn down_blocks() -> Vec<String> {
    vec!["DownEncoderBlock2D".to_owned()]
}

fn up_blocks() -> Vec<String> {
    vec!["UpDecoderBlock2D".to_owned()]
}

const fn yes() -> bool {
    true
}

#[derive(Debug, Deserialize)]
struct Fields {
    in_channels: i64,
    out_channels: i64,
    latent_channels: i64,
    block_out_channels: Vec<i64>,
    #[serde(default = "down_blocks")]
    down_block_types: Vec<String>,
    #[serde(default = "up_blocks")]
    up_block_types: Vec<String>,
    layers_per_block: usize,
    norm_num_groups: i64,
    #[serde(default = "silu")]
    act_fn: String,
    #[serde(default = "yes")]
    mid_block_add_attention: bool,
    #[serde(default = "yes")]
    use_quant_conv: bool,
    #[serde(default = "yes")]
    use_post_quant_conv: bool,
}

impl Fields {
    // Refuses the options `diffusers` offers that the model does not build.
    fn validate(&self) -> Result<()> {
        if self.act_fn != "silu" || !self.mid_block_add_attention {
            return Err(Error::unsupported(
                "an AutoencoderKL other than one with SiLU and attending middle blocks",
            ));
        }

        if !self.use_quant_conv || !self.use_post_quant_conv {
            return Err(Error::unsupported(
                "an AutoencoderKL without its quantization convolutions",
            ));
        }

        let blocks = self.block_out_channels.len();
        let plain = |types: &[String], kind: &str| {
            types.len() == blocks && types.iter().all(|found| found == kind)
        };
        if blocks == 0
            || !plain(&self.down_block_types, "DownEncoderBlock2D")
            || !plain(&self.up_block_types, "UpDecoderBlock2D")
        {
            return Err(Error::unsupported(
                "an AutoencoderKL other than one of a DownEncoderBlock2D and an UpDecoderBlock2D per width",
            ));
        }
        Ok(())
    }
}

pub(crate) fn config(source: &Source) -> Result<AutoencoderKlConfig> {
    let fields: Fields = source.json("config.json")?;
    fields.validate()?;
    Ok(AutoencoderKlConfig {
        in_channels: fields.in_channels,
        out_channels: fields.out_channels,
        latent_channels: fields.latent_channels,
        widths: fields.block_out_channels,
        layers_per_block: fields.layers_per_block,
        groups: fields.norm_num_groups,
    })
}

// Imports one half, stored under `half` (`encoder` or `decoder`), its blocks under `blocks` (`down_blocks` or `up_blocks`), and the 1 × 1 convolution `quant` beside it.
fn half_rules(half: &str, blocks: &str, quant: &str) -> Rules {
    Rules::new()
        .rename(&format!("{half}.conv_in"), "input")
        .then(unet_block_rules(&format!("{half}.{blocks}.*"), "stages.*"))
        .then(unet_block_rules(&format!("{half}.mid_block"), "middle"))
        .rename(&format!("{half}.conv_norm_out"), "norm")
        .rename(&format!("{half}.conv_out"), "output")
        .rename(quant, "quant")
}

pub(crate) fn encoder_rules() -> Rules {
    half_rules("encoder", "down_blocks", "quant_conv")
}

pub(crate) fn decoder_rules() -> Rules {
    half_rules("decoder", "up_blocks", "post_quant_conv")
}
