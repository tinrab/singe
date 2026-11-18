// Reads a `transformers` DINOv2 config and imports its checkpoint.

use serde::Deserialize;
use singe::nn::Activation;

use super::Dinov2Config;
use crate::{
    error::{Error, Result},
    format::{Rules, transformers::activation_field},
    source::Source,
};

// `Dinov2Config`'s defaults, which a backbone's partial config leaves to them.
const fn enabled() -> bool {
    true
}

const fn width() -> i64 {
    768
}

const fn twelve() -> i64 {
    12
}

const fn four() -> f64 {
    4.0
}

const fn gelu() -> Activation {
    Activation::GeluExact
}

const fn eps() -> f32 {
    1e-6
}

const fn image_size() -> i64 {
    224
}

const fn patch_size() -> i64 {
    14
}

const fn three() -> i64 {
    3
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub(crate) struct Fields {
    #[serde(default = "width")]
    hidden_size: i64,
    #[serde(default = "twelve")]
    num_hidden_layers: i64,
    #[serde(default = "twelve")]
    num_attention_heads: i64,
    #[serde(default = "four")]
    mlp_ratio: f64,
    #[serde(default = "gelu", deserialize_with = "activation_field")]
    hidden_act: Activation,
    #[serde(default = "eps")]
    layer_norm_eps: f32,
    #[serde(default = "image_size")]
    image_size: i64,
    #[serde(default = "patch_size")]
    patch_size: i64,
    #[serde(default = "three")]
    num_channels: i64,
    #[serde(default = "enabled")]
    qkv_bias: bool,
    #[serde(default)]
    use_swiglu_ffn: bool,
}

impl Fields {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        reason = "`transformers` truncates the product to an integer the same way"
    )]
    fn inner(&self) -> i64 {
        (self.hidden_size as f64 * self.mlp_ratio) as i64
    }
}

pub(crate) fn config(source: &Source) -> Result<Dinov2Config> {
    fields_config(&source.json("config.json")?)
}

pub(crate) fn fields_config(fields: &Fields) -> Result<Dinov2Config> {
    if fields.use_swiglu_ffn {
        return Err(Error::unsupported("a SwiGLU feed-forward"));
    }

    if fields.patch_size <= 0 || fields.image_size % fields.patch_size != 0 {
        return Err(Error::unsupported(
            "an image size that is not whole patches",
        ));
    }
    Ok(Dinov2Config {
        width: fields.hidden_size,
        layers: fields.num_hidden_layers,
        heads: fields.num_attention_heads,
        inner: fields.inner(),
        activation: fields.hidden_act,
        eps: fields.layer_norm_eps,
        image_size: fields.image_size,
        patch_size: fields.patch_size,
        channels: fields.num_channels,
        attention_bias: fields.qkv_bias,
    })
}

// Imports the encoder stored under `prefix` to the schema's names under `to`.
pub(crate) fn encoder_rules(prefix: &str, to: &str) -> Rules {
    let embeddings = format!("{prefix}embeddings");
    let layer = |name: &str| format!("{prefix}encoder.layer.*.{name}");
    let target = |name: &str| format!("{to}{name}");
    Rules::new()
        .rename(
            &format!("{embeddings}.patch_embeddings.projection"),
            &target("embeddings.patches"),
        )
        .rename(
            &format!("{embeddings}.cls_token"),
            &target("embeddings.class"),
        )
        .rename(
            &format!("{embeddings}.position_embeddings"),
            &target("embeddings.positions"),
        )
        // The mask token serves only masked pretraining.
        .skip(&format!("{embeddings}.mask_token"))
        .rename(&layer("norm1"), &target("blocks.*.attention_norm"))
        .rename(
            &layer("attention.attention.query"),
            &target("blocks.*.attention.query"),
        )
        .rename(
            &layer("attention.attention.key"),
            &target("blocks.*.attention.key"),
        )
        .rename(
            &layer("attention.attention.value"),
            &target("blocks.*.attention.value"),
        )
        .rename(
            &layer("attention.output.dense"),
            &target("blocks.*.attention.output"),
        )
        .rename(
            &layer("layer_scale1.lambda1"),
            &target("blocks.*.attention_scale.scale"),
        )
        .rename(&layer("norm2"), &target("blocks.*.feed_forward_norm"))
        .rename(&layer("mlp.fc1"), &target("blocks.*.feed_forward.up"))
        .rename(&layer("mlp.fc2"), &target("blocks.*.feed_forward.down"))
        .rename(
            &layer("layer_scale2.lambda1"),
            &target("blocks.*.feed_forward_scale.scale"),
        )
        .rename(&format!("{prefix}layernorm"), &target("norm"))
}

// Checkpoints of a classifier store the encoder under `dinov2.`.
pub(crate) fn rules() -> Rules {
    encoder_rules("", "")
        .then(encoder_rules("dinov2.", ""))
        // The classifier serves image classification, which no head loads yet.
        .skip("classifier")
}
