//! Reads a `diffusers` `UNet2DConditionModel`: its config and its tensors' names.

use serde::Deserialize;
use serde_json::Value;
use singe::Error as SingeError;

use crate::{
    assembly::unet_block_rules,
    error::{Error, Result},
    format::{
        Rules,
        diffusers::{AttentionHeadDim, UnetFields, default_kind, default_true, one, unit},
        join,
    },
    model::diffusion::unet_2d_condition::{MotionConfig, Unet2dConditionConfig},
    source::Source,
};

const CLASS: &str = "UNet2DConditionModel";

const fn kernel() -> i64 {
    3
}

#[derive(Debug, Deserialize)]
struct Fields {
    #[serde(flatten)]
    unet: UnetFields,
    cross_attention_dim: i64,
    #[serde(default)]
    num_attention_heads: Option<Value>,
    #[serde(default)]
    use_linear_projection: bool,
    #[serde(default = "one")]
    transformer_layers_per_block: usize,
    #[serde(default)]
    addition_embed_type: Option<String>,
    /// Whether blocks skip self-attention, one flag for all or one per block.
    #[serde(default)]
    only_cross_attention: Value,
    #[serde(default)]
    dual_cross_attention: bool,
    #[serde(default)]
    resnet_skip_time_act: bool,
    #[serde(default = "unit")]
    resnet_out_scale_factor: f64,
    #[serde(default)]
    time_embedding_act_fn: Option<String>,
    #[serde(default)]
    timestep_post_act: Option<String>,
    #[serde(default)]
    time_cond_proj_dim: Option<i64>,
    #[serde(default = "kernel")]
    conv_in_kernel: i64,
    #[serde(default = "kernel")]
    conv_out_kernel: i64,
    #[serde(default)]
    encoder_hid_dim: Option<i64>,
    #[serde(default)]
    encoder_hid_dim_type: Option<String>,
    #[serde(default = "default_kind")]
    attention_type: String,
    #[serde(default)]
    cross_attention_norm: Option<String>,
}

/// Returns whether a flag, or any of a list of flags, is set.
fn any_set(flags: &Value) -> bool {
    match flags {
        Value::Null | Value::Bool(false) => false,
        Value::Array(flags) => flags.iter().any(any_set),
        _ => true,
    }
}

impl Fields {
    /// Refuses the options `diffusers` offers that the model does not build.
    ///
    /// It leaves `upcast_attention` unread, which changes nothing in `F32`.
    fn validate(&self) -> Result<()> {
        self.unet.validate(CLASS, "UNetMidBlock2DCrossAttn")?;
        if self.time_embedding_act_fn.is_some()
            || self.timestep_post_act.is_some()
            || self.time_cond_proj_dim.is_some()
        {
            return Err(Error::unsupported(format!(
                "{CLASS} time_embedding_act_fn, timestep_post_act, or time_cond_proj_dim"
            )));
        }

        if self.resnet_skip_time_act || (self.resnet_out_scale_factor - 1.0).abs() > f64::EPSILON {
            return Err(Error::unsupported(format!(
                "{CLASS} resnet_skip_time_act or resnet_out_scale_factor other than the defaults"
            )));
        }

        if self.conv_in_kernel != 3 || self.conv_out_kernel != 3 {
            return Err(Error::unsupported(format!(
                "{CLASS} conv_in_kernel or conv_out_kernel other than 3"
            )));
        }

        if self.transformer_layers_per_block != 1
            || any_set(&self.only_cross_attention)
            || self.dual_cross_attention
            || self.attention_type != "default"
            || self.cross_attention_norm.is_some()
        {
            return Err(Error::unsupported(format!(
                "{CLASS} transformer_layers_per_block, only_cross_attention, dual_cross_attention, attention_type, or cross_attention_norm other than the defaults"
            )));
        }

        if self.addition_embed_type.is_some()
            || self.encoder_hid_dim.is_some()
            || self.encoder_hid_dim_type.is_some()
        {
            return Err(Error::unsupported(format!(
                "{CLASS} addition_embed_type, encoder_hid_dim, or encoder_hid_dim_type"
            )));
        }

        if self.num_attention_heads.is_some() {
            return Err(Error::unsupported(format!(
                "{CLASS} num_attention_heads, which diffusers refuses"
            )));
        }
        Ok(())
    }

    /// Returns each down block's head count, which `diffusers` reads from `attention_head_dim`, as Stable Diffusion's configs were written, from one count for all or one per block.
    fn heads(&self) -> Result<Vec<i64>> {
        let blocks = self.unet.block_out_channels.len();
        let heads = match &self.unet.attention_head_dim {
            Some(AttentionHeadDim::One(heads)) => vec![*heads; blocks],
            Some(AttentionHeadDim::Each(heads)) => heads.clone(),
            None => {
                return Err(Error::unsupported(format!(
                    "{CLASS} attention_head_dim null"
                )));
            }
        };
        if heads.len() != blocks {
            return Err(Error::unsupported(format!(
                "{CLASS} attention_head_dim other than one per block"
            )));
        }
        Ok(heads)
    }
}

pub(crate) fn config(source: &Source) -> Result<Unet2dConditionConfig> {
    let fields: Fields = source.json("config.json")?;
    fields.validate()?;
    Ok(Unet2dConditionConfig {
        unet: fields
            .unet
            .config(["CrossAttnDownBlock2D", "CrossAttnUpBlock2D"])?,
        heads: fields.heads()?,
        context_width: fields.cross_attention_dim,
        linear_projection: fields.use_linear_projection,
    })
}

/// Returns the rules that import the transformers stored under `from` (such as `down_blocks.*.attentions.*`) into `to` (such as `down.*.attentions.*`).
fn transformer_rules(from: &str, to: &str) -> Rules {
    let (block, target) = (join(from, "transformer_blocks.*"), join(to, "blocks.*"));
    Rules::new()
        .rename_each(
            from,
            to,
            &[
                ("norm", "norm"),
                ("proj_in", "proj_in"),
                ("proj_out", "proj_out"),
            ],
        )
        .rename_each(
            &block,
            &target,
            &[
                ("norm1", "attention_norm"),
                ("attn1.to_q", "attention.query"),
                ("attn1.to_k", "attention.key"),
                ("attn1.to_v", "attention.value"),
                ("attn1.to_out.0", "attention.output"),
                ("norm2", "cross_attention_norm"),
                ("attn2.to_q", "cross_attention.query"),
                ("attn2.to_k", "cross_attention.key"),
                ("attn2.to_v", "cross_attention.value"),
                ("attn2.to_out.0", "cross_attention.output"),
                ("norm3", "feed_forward_norm"),
                ("ff.net.2", "feed_forward.down"),
                ("pos_embed.pe", "positions"),
            ],
        )
        // GEGLU's projection holds the values and then the gate.
        .split(
            &join(&block, "ff.net.0.proj"),
            0,
            &[
                &join(&target, "feed_forward.up"),
                &join(&target, "feed_forward.gate"),
            ],
        )
}

pub(crate) fn rules() -> Rules {
    Rules::new()
        .rename("conv_in", "input")
        .rename("time_embedding.linear_1", "time.up")
        .rename("time_embedding.linear_2", "time.down")
        .then(transformer_rules(
            "down_blocks.*.attentions.*",
            "down.*.attentions.*",
        ))
        .then(transformer_rules(
            "mid_block.attentions.*",
            "middle.attentions.*",
        ))
        .then(transformer_rules(
            "up_blocks.*.attentions.*",
            "up.*.attentions.*",
        ))
        .then(unet_block_rules("down_blocks.*", "down.*"))
        .then(unet_block_rules("mid_block", "middle"))
        .then(unet_block_rules("up_blocks.*", "up.*"))
        .rename("conv_norm_out", "norm")
        .rename("conv_out", "output")
}

/// Returns the rules that import a `MotionAdapter`'s temporal transformers into the motion modules' `down.N.motions.M`, `middle.motions.0`, and `up.N.motions.M`, and, for an adapter with its own input convolution, its weights for the channels past the `latents` channels that the U-Net's input convolution reads.
pub(crate) fn motion_rules(latents: i64) -> Rules {
    Rules::new()
        // PIA's input convolution: only its channels past the latents' are its own.
        .derive("conv_in.weight", "input.weight", move |weight| {
            let &[_, channels, ..] = weight.shape().dimensions() else {
                return Err(SingeError::invalid_argument("conv_in.weight"));
            };
            weight.narrow(1, latents, channels - latents)?.contiguous()
        })
        .skip("conv_in.bias")
        .then(transformer_rules(
            "down_blocks.*.motion_modules.*",
            "down.*.motions.*",
        ))
        .then(transformer_rules(
            "mid_block.motion_modules.*",
            "middle.motions.*",
        ))
        .then(transformer_rules(
            "up_blocks.*.motion_modules.*",
            "up.*.motions.*",
        ))
}

const fn motion_layers() -> usize {
    2
}

const fn motion_heads() -> i64 {
    8
}

const fn motion_groups() -> i64 {
    32
}

const fn motion_positions() -> i64 {
    32
}

#[derive(Debug, Deserialize)]
struct MotionFields {
    block_out_channels: Vec<i64>,
    #[serde(default = "motion_layers")]
    motion_layers_per_block: usize,
    #[serde(default = "one")]
    motion_transformer_layers_per_block: usize,
    #[serde(default = "one")]
    motion_mid_block_layers_per_block: usize,
    #[serde(default = "one")]
    motion_transformer_layers_per_mid_block: usize,
    #[serde(default = "motion_heads")]
    motion_num_attention_heads: i64,
    #[serde(default = "motion_positions")]
    motion_max_seq_length: i64,
    #[serde(default = "default_true")]
    use_motion_mid_block: bool,
    #[serde(default = "motion_groups")]
    motion_norm_num_groups: i64,
    #[serde(default)]
    conv_in_channels: Option<i64>,
}

pub(crate) fn motion_config(source: &Source) -> Result<MotionConfig> {
    let fields: MotionFields = source.json("config.json")?;
    if fields.motion_transformer_layers_per_block != 1
        || fields.motion_mid_block_layers_per_block != 1
        || fields.motion_transformer_layers_per_mid_block != 1
    {
        return Err(Error::unsupported(
            "MotionAdapter motion_transformer_layers_per_block, motion_mid_block_layers_per_block, or motion_transformer_layers_per_mid_block other than 1",
        ));
    }
    Ok(MotionConfig {
        widths: fields.block_out_channels,
        layers_per_block: fields.motion_layers_per_block,
        heads: fields.motion_num_attention_heads,
        max_positions: fields.motion_max_seq_length,
        middle: fields.use_motion_mid_block,
        groups: fields.motion_norm_num_groups,
        in_channels: fields.conv_in_channels,
    })
}
