// Reads a `diffusers` `UNet2DConditionModel`: its config and its tensors' names.

use serde::Deserialize;
use serde_json::Value;

use super::{MotionConfig, Unet2dConditionConfig};
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

#[expect(
    clippy::unnecessary_wraps,
    reason = "an explicit null in the config means no middle block"
)]
fn middle_block() -> Option<String> {
    Some("UNetMidBlock2DCrossAttn".to_owned())
}

fn eight() -> Value {
    Value::from(8)
}

const fn one() -> usize {
    1
}

const fn unit() -> f64 {
    1.0
}

const fn padding() -> i64 {
    1
}

const fn kernel() -> i64 {
    3
}

const fn yes() -> bool {
    true
}

#[derive(Debug, Deserialize)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "the fields are `config.json`'s"
)]
struct Fields {
    in_channels: i64,
    out_channels: i64,
    sample_size: i64,
    block_out_channels: Vec<i64>,
    down_block_types: Vec<String>,
    up_block_types: Vec<String>,
    layers_per_block: usize,
    cross_attention_dim: i64,
    // The head count, which `diffusers` reads from `attention_head_dim` and refuses in `num_attention_heads`, as Stable Diffusion's configs were written.
    #[serde(default = "eight")]
    attention_head_dim: Value,
    #[serde(default)]
    num_attention_heads: Option<Value>,
    #[serde(default)]
    use_linear_projection: bool,
    norm_num_groups: i64,
    norm_eps: f32,
    #[serde(default = "padding")]
    downsample_padding: i64,
    #[serde(default = "yes")]
    flip_sin_to_cos: bool,
    #[serde(default)]
    freq_shift: f32,
    #[serde(default = "unit")]
    mid_block_scale_factor: f64,
    #[serde(default = "one")]
    transformer_layers_per_block: usize,
    #[serde(default)]
    addition_embed_type: Option<String>,
    #[serde(default)]
    class_embed_type: Option<String>,
    #[serde(default)]
    num_class_embeds: Option<i64>,
    #[serde(default = "middle_block")]
    mid_block_type: Option<String>,
    // Whether blocks skip self-attention, one flag for all or one per block.
    #[serde(default)]
    only_cross_attention: Value,
    #[serde(default)]
    dual_cross_attention: bool,
    #[serde(default = "silu")]
    act_fn: String,
    #[serde(default = "default_kind")]
    resnet_time_scale_shift: String,
    #[serde(default)]
    resnet_skip_time_act: bool,
    #[serde(default = "unit")]
    resnet_out_scale_factor: f64,
    #[serde(default = "positional")]
    time_embedding_type: String,
    #[serde(default)]
    time_embedding_dim: Option<i64>,
    #[serde(default)]
    time_embedding_act_fn: Option<String>,
    #[serde(default)]
    timestep_post_act: Option<String>,
    #[serde(default)]
    time_cond_proj_dim: Option<i64>,
    #[serde(default)]
    center_input_sample: bool,
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

// Whether a flag, or any of a list of flags, is set.
fn any_set(flags: &Value) -> bool {
    match flags {
        Value::Null | Value::Bool(false) => false,
        Value::Array(flags) => flags.iter().any(any_set),
        _ => true,
    }
}

impl Fields {
    // Refuses the options `diffusers` offers that the model does not build.
    // It leaves `upcast_attention` unread, which changes nothing in `F32`.
    fn validate(&self) -> Result<()> {
        if self.act_fn != "silu"
            || self.time_embedding_type != "positional"
            || self.time_embedding_dim.is_some()
            || self.time_embedding_act_fn.is_some()
            || self.timestep_post_act.is_some()
            || self.time_cond_proj_dim.is_some()
            || self.center_input_sample
        {
            return Err(Error::unsupported(
                "a UNet2DConditionModel other than one with SiLU and positional timesteps of the default embedding",
            ));
        }

        if self.resnet_time_scale_shift != "default"
            || self.resnet_skip_time_act
            || (self.resnet_out_scale_factor - 1.0).abs() > f64::EPSILON
            || self.conv_in_kernel != 3
            || self.conv_out_kernel != 3
        {
            return Err(Error::unsupported(
                "a UNet2DConditionModel other than one of default residual blocks and 3 × 3 input and output convolutions",
            ));
        }

        if self.transformer_layers_per_block != 1
            || any_set(&self.only_cross_attention)
            || self.dual_cross_attention
            || self.attention_type != "default"
            || self.cross_attention_norm.is_some()
            || self.mid_block_type.as_deref() != Some("UNetMidBlock2DCrossAttn")
        {
            return Err(Error::unsupported(
                "a UNet2DConditionModel other than one of single default transformers in its blocks and middle",
            ));
        }

        if self.addition_embed_type.is_some()
            || self.class_embed_type.is_some()
            || self.num_class_embeds.is_some()
            || self.encoder_hid_dim.is_some()
            || self.encoder_hid_dim_type.is_some()
        {
            return Err(Error::unsupported(
                "a UNet2DConditionModel with added, class, or projected context embeddings",
            ));
        }

        if self.num_attention_heads.is_some() {
            return Err(Error::unsupported(
                "a UNet2DConditionModel setting num_attention_heads, which diffusers refuses",
            ));
        }

        let blocks = self.block_out_channels.len();
        if blocks == 0
            || self.down_block_types.len() != blocks
            || self.up_block_types.len() != blocks
        {
            return Err(Error::unsupported(
                "a UNet2DConditionModel without one down and one up block per width",
            ));
        }
        Ok(())
    }

    // Each down block's head count, from one count for all or one per block.
    fn heads(&self) -> Result<Vec<i64>> {
        let count = |heads: &Value| {
            heads
                .as_i64()
                .ok_or_else(|| Error::unsupported("an attention head count other than an integer"))
        };
        let blocks = self.block_out_channels.len();
        let heads = match &self.attention_head_dim {
            Value::Array(heads) => heads.iter().map(count).collect::<Result<Vec<_>>>()?,
            heads => vec![count(heads)?; blocks],
        };
        if heads.len() != blocks {
            return Err(Error::unsupported(
                "attention head counts other than one per block",
            ));
        }
        Ok(heads)
    }
}

pub(crate) fn config(source: &Source) -> Result<Unet2dConditionConfig> {
    let fields: Fields = source.json("config.json")?;
    fields.validate()?;
    Ok(Unet2dConditionConfig {
        in_channels: fields.in_channels,
        out_channels: fields.out_channels,
        sample_size: fields.sample_size,
        down_attention: attends(
            &fields.down_block_types,
            "DownBlock2D",
            "CrossAttnDownBlock2D",
        )?,
        up_attention: attends(&fields.up_block_types, "UpBlock2D", "CrossAttnUpBlock2D")?,
        heads: fields.heads()?,
        widths: fields.block_out_channels,
        layers_per_block: fields.layers_per_block,
        context_width: fields.cross_attention_dim,
        linear_projection: fields.use_linear_projection,
        groups: fields.norm_num_groups,
        eps: fields.norm_eps,
        downsample_padding: fields.downsample_padding,
        flip_sin_to_cos: fields.flip_sin_to_cos,
        freq_shift: fields.freq_shift,
        middle_scale: fields.mid_block_scale_factor,
    })
}

// Imports the transformers stored under `from` (such as `down_blocks.*.attentions.*`) into `to` (such as `down.*.attentions.*`).
pub(crate) fn transformer_rules(from: &str, to: &str) -> Rules {
    let pair = |stored: &str, target: &str| (format!("{from}.{stored}"), format!("{to}.{target}"));
    let block = |name: &str| format!("transformer_blocks.*.{name}");
    let target = |name: &str| format!("blocks.*.{name}");
    let mut renames = vec![
        pair("norm", "norm"),
        pair("proj_in", "proj_in"),
        pair("proj_out", "proj_out"),
    ];
    for name in ["norm1", "norm2", "norm3"] {
        renames.push(pair(&block(name), &target(name)));
    }

    for attention in ["attn1", "attn2"] {
        for (stored, name) in [
            ("to_q", "query"),
            ("to_k", "key"),
            ("to_v", "value"),
            ("to_out.0", "output"),
        ] {
            renames.push(pair(
                &block(&format!("{attention}.{stored}")),
                &target(&format!("{attention}.{name}")),
            ));
        }
    }

    renames.push(pair(&block("ff.net.2"), &target("feed_forward.down")));
    // A temporal block's sinusoidal positions, which `diffusers` stores.
    renames.push(pair(&block("pos_embed.pe"), &target("positions")));
    let (projection, _) = pair(&block("ff.net.0.proj"), "");
    renames
        .into_iter()
        .fold(Rules::new(), |rules, (stored, target)| {
            rules.rename(&stored, &target)
        })
        // GEGLU's projection holds the values and then the gate.
        .split(
            &projection,
            0,
            &[
                &format!("{to}.blocks.*.feed_forward.up"),
                &format!("{to}.blocks.*.feed_forward.gate"),
            ],
        )
}

pub(crate) fn rules() -> Rules {
    Rules::new()
        .rename("conv_in", "input")
        .rename("time_embedding.linear_1", "time.input")
        .rename("time_embedding.linear_2", "time.output")
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

// Imports a `MotionAdapter`'s temporal transformers into the motion modules' `down.N.motions.M`, `middle.motions.0`, and `up.N.motions.M`, and, for an adapter with its own input convolution, the weights past the `latents` channels the UNet's reads.
pub(crate) fn motion_rules(latents: i64) -> Rules {
    Rules::new()
        // PIA's input convolution: only its channels past the latents' are its own.
        .derive("conv_in.weight", "input.weight", move |weight| {
            let channels = weight.shape().dimensions()[1];
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
    #[serde(default = "yes")]
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
            "a MotionAdapter other than one of a single transformer per motion layer and one motion layer in the middle",
        ));
    }
    Ok(MotionConfig {
        widths: fields.block_out_channels,
        layers_per_block: fields.motion_layers_per_block,
        heads: fields.motion_num_attention_heads,
        positions: fields.motion_max_seq_length,
        middle: fields.use_motion_mid_block,
        groups: fields.motion_norm_num_groups,
        in_channels: fields.conv_in_channels,
    })
}
