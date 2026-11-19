use std::result::Result as StdResult;

use serde::{Deserialize, Deserializer};
use singe::nn::Activation;

pub use crate::format::transformers::{activation_field, default_true};
use crate::{
    assembly::UnetConfig,
    error::{Error, Result},
};

#[must_use]
pub const fn silu() -> Activation {
    Activation::Silu
}

#[must_use]
pub fn positional() -> String {
    "positional".to_owned()
}

#[must_use]
pub fn default_kind() -> String {
    "default".to_owned()
}

#[must_use]
pub const fn padding() -> i64 {
    1
}

#[must_use]
pub const fn unit() -> f64 {
    1.0
}

#[must_use]
pub const fn one() -> usize {
    1
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(untagged)]
pub enum AttentionHeadDim {
    One(i64),
    Each(Vec<i64>),
}

#[expect(
    clippy::unnecessary_wraps,
    reason = "an explicit null in the config differs from the default"
)]
const fn default_attention_head_dim() -> Option<AttentionHeadDim> {
    Some(AttentionHeadDim::One(8))
}

fn present<'de, D: Deserializer<'de>, T: Deserialize<'de>>(
    deserializer: D,
) -> StdResult<Option<T>, D::Error> {
    T::deserialize(deserializer).map(Some)
}

fn attends(types: &[String], plain: &str, attending: &str) -> Result<Vec<bool>> {
    types
        .iter()
        .map(|kind| match kind.as_str() {
            kind if kind == plain => Ok(false),
            kind if kind == attending => Ok(true),
            other => Err(Error::unsupported(format!("UNet block {other}"))),
        })
        .collect()
}

// The fields `UNet2DModel`'s and `UNet2DConditionModel`'s `config.json` share.
#[derive(Debug, Deserialize)]
pub struct UnetFields {
    pub in_channels: i64,
    pub out_channels: i64,
    pub sample_size: i64,
    pub block_out_channels: Vec<i64>,
    pub down_block_types: Vec<String>,
    pub up_block_types: Vec<String>,
    pub layers_per_block: usize,
    pub norm_num_groups: i64,
    pub norm_eps: f32,
    #[serde(default = "default_attention_head_dim")]
    pub attention_head_dim: Option<AttentionHeadDim>,
    #[serde(default = "padding")]
    pub downsample_padding: i64,
    #[serde(default = "default_true")]
    pub flip_sin_to_cos: bool,
    #[serde(default)]
    pub freq_shift: f32,
    #[serde(default = "unit")]
    pub mid_block_scale_factor: f64,
    #[serde(default = "silu", deserialize_with = "activation_field")]
    pub act_fn: Activation,
    #[serde(default = "positional")]
    pub time_embedding_type: String,
    #[serde(default)]
    pub time_embedding_dim: Option<i64>,
    #[serde(default)]
    pub center_input_sample: bool,
    #[serde(default = "default_kind")]
    pub resnet_time_scale_shift: String,
    #[serde(default)]
    pub class_embed_type: Option<String>,
    #[serde(default)]
    pub num_class_embeds: Option<i64>,
    #[serde(default, deserialize_with = "present")]
    pub mid_block_type: Option<Option<String>>,
}

impl UnetFields {
    // Refuses the options `diffusers` offers that `class`, whose middle block is `middle`, does not build.
    pub fn validate(&self, class: &str, middle: &str) -> Result<()> {
        if self.act_fn != Activation::Silu {
            return Err(Error::unsupported(format!(
                "{class} act_fn other than silu"
            )));
        }

        if self.time_embedding_type != "positional"
            || self.time_embedding_dim.is_some()
            || self.center_input_sample
        {
            return Err(Error::unsupported(format!(
                "{class} time_embedding_type, time_embedding_dim, or center_input_sample other than the defaults"
            )));
        }

        if self.resnet_time_scale_shift != "default" {
            return Err(Error::unsupported(format!(
                "{class} resnet_time_scale_shift {}",
                self.resnet_time_scale_shift
            )));
        }

        if self
            .mid_block_type
            .as_ref()
            .is_some_and(|kind| kind.as_deref() != Some(middle))
        {
            return Err(Error::unsupported(format!(
                "{class} mid_block_type other than {middle}"
            )));
        }

        if self.class_embed_type.is_some() || self.num_class_embeds.is_some() {
            return Err(Error::unsupported(format!(
                "{class} class_embed_type or num_class_embeds"
            )));
        }

        let blocks = self.block_out_channels.len();
        if blocks == 0
            || self.down_block_types.len() != blocks
            || self.up_block_types.len() != blocks
        {
            return Err(Error::unsupported(format!(
                "{class} down_block_types and up_block_types other than one per block_out_channels"
            )));
        }
        Ok(())
    }

    // The shared config, each down block `DownBlock2D` or the attending `down`, and each up block `UpBlock2D` or the attending `up`.
    pub fn config(&self, [down, up]: [&str; 2]) -> Result<UnetConfig> {
        Ok(UnetConfig {
            in_channels: self.in_channels,
            out_channels: self.out_channels,
            sample_size: self.sample_size,
            widths: self.block_out_channels.clone(),
            down_attention: attends(&self.down_block_types, "DownBlock2D", down)?,
            up_attention: attends(&self.up_block_types, "UpBlock2D", up)?,
            layers_per_block: self.layers_per_block,
            groups: self.norm_num_groups,
            eps: self.norm_eps,
            downsample_padding: self.downsample_padding,
            flip_sin_to_cos: self.flip_sin_to_cos,
            freq_shift: self.freq_shift,
            middle_scale: self.mid_block_scale_factor,
        })
    }
}
