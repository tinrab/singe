//! Reads a `transformers` `DINOv2` config and imports its checkpoint.

use serde::Deserialize;
use singe::nn::Activation;

use crate::{
    assembly::{PreNormEncoderConfig, VisionTowerConfig},
    error::{Error, Result},
    format::{
        Rules, join,
        transformers::{activation_field, default_channels, default_true},
    },
    source::Source,
};

const fn default_width() -> i64 {
    768
}

const fn default_layers() -> i64 {
    12
}

const fn default_heads() -> i64 {
    12
}

const fn default_mlp_ratio() -> f64 {
    4.0
}

const fn default_activation() -> Activation {
    Activation::GeluExact
}

const fn default_eps() -> f32 {
    1e-6
}

const fn default_image_size() -> i64 {
    224
}

const fn default_patch_size() -> i64 {
    14
}

/// A `transformers` `DINOv2` config, whose fields a backbone's partial config may leave out to take `Dinov2Config`'s defaults.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub(crate) struct Fields {
    #[serde(default = "default_width")]
    hidden_size: i64,
    #[serde(default = "default_layers")]
    num_hidden_layers: i64,
    #[serde(default = "default_heads")]
    num_attention_heads: i64,
    #[serde(default = "default_mlp_ratio")]
    mlp_ratio: f64,
    #[serde(default = "default_activation", deserialize_with = "activation_field")]
    hidden_act: Activation,
    #[serde(default = "default_eps")]
    layer_norm_eps: f32,
    #[serde(default = "default_image_size")]
    image_size: i64,
    #[serde(default = "default_patch_size")]
    patch_size: i64,
    #[serde(default = "default_channels")]
    num_channels: i64,
    #[serde(default = "default_true")]
    qkv_bias: bool,
    #[serde(default)]
    use_swiglu_ffn: bool,
}

impl Fields {
    fn inner(&self) -> i64 {
        (self.hidden_size as f64 * self.mlp_ratio) as i64
    }
}

pub(crate) fn config(source: &Source) -> Result<VisionTowerConfig> {
    fields_config(&source.json("config.json")?)
}

pub(crate) fn fields_config(fields: &Fields) -> Result<VisionTowerConfig> {
    if fields.use_swiglu_ffn {
        return Err(Error::unsupported("a SwiGLU feed-forward"));
    }

    if fields.patch_size <= 0 || fields.image_size % fields.patch_size != 0 {
        return Err(Error::unsupported(format!(
            "an image size of {} that is not whole patches of {}",
            fields.image_size, fields.patch_size
        )));
    }

    let bias = fields.qkv_bias;
    Ok(VisionTowerConfig {
        encoder: PreNormEncoderConfig {
            width: fields.hidden_size,
            heads: fields.num_attention_heads,
            inner: fields.inner(),
            activation: fields.hidden_act,
            eps: fields.layer_norm_eps,
            projection_biases: [bias, bias, bias, true],
            causal: false,
            layer_scale: true,
        },
        layers: fields.num_hidden_layers,
        image_size: fields.image_size,
        patch_size: fields.patch_size,
        channels: fields.num_channels,
    })
}

/// Returns the rules that import the encoder stored under `from` to the schema's names under `to`.
pub(crate) fn encoder_rules(from: &str, to: &str) -> Rules {
    let embeddings = join(from, "embeddings");
    Rules::new()
        .rename_each(
            &embeddings,
            &join(to, "embeddings"),
            &[
                ("patch_embeddings.projection", "patches"),
                ("cls_token", "class"),
                ("position_embeddings", "positions"),
            ],
        )
        // The mask token serves only masked pretraining.
        .skip(&join(&embeddings, "mask_token"))
        .rename_each(
            &join(from, "encoder.layer.*"),
            &join(to, "blocks.*"),
            &[
                ("norm1", "attention_norm"),
                ("attention.attention.query", "attention.query"),
                ("attention.attention.key", "attention.key"),
                ("attention.attention.value", "attention.value"),
                ("attention.output.dense", "attention.output"),
                ("layer_scale1.lambda1", "attention_scale.scale"),
                ("norm2", "feed_forward_norm"),
                ("mlp.fc1", "feed_forward.up"),
                ("mlp.fc2", "feed_forward.down"),
                ("layer_scale2.lambda1", "feed_forward_scale.scale"),
            ],
        )
        .rename(&join(from, "layernorm"), &join(to, "norm"))
}

/// Returns the rules that import a checkpoint's encoder, stored at its root or, in a classifier's checkpoint, under `dinov2.`.
pub(crate) fn rules() -> Rules {
    encoder_rules("", "")
        .then(encoder_rules("dinov2", ""))
        // The classifier serves image classification, which no head loads yet.
        .skip("classifier")
}
