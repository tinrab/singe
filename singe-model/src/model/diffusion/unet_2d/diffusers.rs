// Reads a `diffusers` `UNet2DModel`: its config and its tensors' names.

use serde::Deserialize;

use super::Unet2dConfig;
use crate::{
    assembly::{attends, unet_block_rules},
    error::{Error, Result},
    format::Rules,
    source::Source,
};

fn silu() -> String {
    "silu".to_owned()
}

fn positional() -> String {
    "positional".to_owned()
}

fn default_kind() -> String {
    "default".to_owned()
}

fn conv() -> String {
    "conv".to_owned()
}

#[expect(
    clippy::unnecessary_wraps,
    reason = "an explicit null in the config means no head width"
)]
const fn eight() -> Option<i64> {
    Some(8)
}

#[expect(
    clippy::unnecessary_wraps,
    reason = "an explicit null in the config means no middle block"
)]
fn middle_block() -> Option<String> {
    Some("UNetMidBlock2D".to_owned())
}

const fn one() -> f64 {
    1.0
}

const fn padding() -> i64 {
    1
}

const fn yes() -> bool {
    true
}

#[derive(Debug, Deserialize)]
struct Fields {
    in_channels: i64,
    out_channels: i64,
    sample_size: i64,
    block_out_channels: Vec<i64>,
    down_block_types: Vec<String>,
    up_block_types: Vec<String>,
    layers_per_block: usize,
    norm_num_groups: i64,
    norm_eps: f32,
    // A head width absent from the config is `diffusers`' 8; a null one makes each attention one head.
    #[serde(default = "eight")]
    attention_head_dim: Option<i64>,
    #[serde(default = "padding")]
    downsample_padding: i64,
    #[serde(default = "yes")]
    flip_sin_to_cos: bool,
    #[serde(default)]
    freq_shift: f32,
    #[serde(default = "one")]
    mid_block_scale_factor: f64,
    #[serde(default = "silu")]
    act_fn: String,
    #[serde(default = "positional")]
    time_embedding_type: String,
    #[serde(default)]
    time_embedding_dim: Option<i64>,
    #[serde(default)]
    center_input_sample: bool,
    #[serde(default = "middle_block")]
    mid_block_type: Option<String>,
    #[serde(default = "conv")]
    downsample_type: String,
    #[serde(default = "conv")]
    upsample_type: String,
    #[serde(default)]
    attn_norm_num_groups: Option<i64>,
    #[serde(default = "default_kind")]
    resnet_time_scale_shift: String,
    #[serde(default = "yes")]
    add_attention: bool,
    #[serde(default)]
    class_embed_type: Option<String>,
    #[serde(default)]
    num_class_embeds: Option<i64>,
}

impl Fields {
    // Refuses the options `diffusers` offers that the model does not build.
    fn validate(&self) -> Result<()> {
        if self.act_fn != "silu"
            || self.time_embedding_type != "positional"
            || self.time_embedding_dim.is_some()
            || self.center_input_sample
        {
            return Err(Error::unsupported(
                "a UNet2DModel other than one with SiLU and positional timesteps of the default width",
            ));
        }

        if self.resnet_time_scale_shift != "default"
            || self.downsample_type != "conv"
            || self.upsample_type != "conv"
        {
            return Err(Error::unsupported(
                "a UNet2DModel other than one of convolutional resamplers and the default time scale shift",
            ));
        }

        if self.mid_block_type.as_deref() != Some("UNetMidBlock2D")
            || !self.add_attention
            || self.attn_norm_num_groups.is_some()
        {
            return Err(Error::unsupported(
                "a UNet2DModel other than one whose middle block attends with the residual blocks' norms",
            ));
        }

        if self.class_embed_type.is_some() || self.num_class_embeds.is_some() {
            return Err(Error::unsupported("a UNet2DModel with class embeddings"));
        }

        let blocks = self.block_out_channels.len();
        if blocks == 0
            || self.down_block_types.len() != blocks
            || self.up_block_types.len() != blocks
        {
            return Err(Error::unsupported(
                "a UNet2DModel without one down and one up block per width",
            ));
        }
        Ok(())
    }
}

pub(crate) fn config(source: &Source) -> Result<Unet2dConfig> {
    let fields: Fields = source.json("config.json")?;
    fields.validate()?;
    Ok(Unet2dConfig {
        in_channels: fields.in_channels,
        out_channels: fields.out_channels,
        sample_size: fields.sample_size,
        down_attention: attends(&fields.down_block_types, "DownBlock2D", "AttnDownBlock2D")?,
        up_attention: attends(&fields.up_block_types, "UpBlock2D", "AttnUpBlock2D")?,
        widths: fields.block_out_channels,
        layers_per_block: fields.layers_per_block,
        groups: fields.norm_num_groups,
        eps: fields.norm_eps,
        head_width: fields.attention_head_dim,
        downsample_padding: fields.downsample_padding,
        flip_sin_to_cos: fields.flip_sin_to_cos,
        freq_shift: fields.freq_shift,
        middle_scale: fields.mid_block_scale_factor,
    })
}

pub(crate) fn rules() -> Rules {
    Rules::new()
        .rename("conv_in", "input")
        .rename("time_embedding.linear_1", "time.input")
        .rename("time_embedding.linear_2", "time.output")
        .then(unet_block_rules("down_blocks.*", "down.*"))
        .then(unet_block_rules("mid_block", "middle"))
        .then(unet_block_rules("up_blocks.*", "up.*"))
        .rename("conv_norm_out", "norm")
        .rename("conv_out", "output")
}
