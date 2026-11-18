// Reads a `transformers` Swin2SR config and imports its checkpoint.

use serde::Deserialize;
use singe::nn::Activation;

use super::Swin2SrConfig;
use crate::{
    error::{Error, Result},
    format::{Rules, transformers::activation_field},
    source::Source,
};

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Fields {
    embed_dim: i64,
    depths: Vec<i64>,
    num_heads: Vec<i64>,
    mlp_ratio: f64,
    #[serde(deserialize_with = "activation_field")]
    hidden_act: Activation,
    layer_norm_eps: f32,
    window_size: i64,
    num_channels: i64,
    upscale: i64,
    upsampler: String,
    resi_connection: String,
    img_range: f64,
    patch_size: i64,
    #[serde(default)]
    use_absolute_embeddings: bool,
    image_size: i64,
}

pub(crate) fn config(source: &Source) -> Result<Swin2SrConfig> {
    let fields: Fields = source.json("config.json")?;
    // The classical models' upsampler, one convolution closing each stage, patches of one point, and windows the configured image size does not shrink.
    if fields.upsampler != "pixelshuffle"
        || fields.resi_connection != "1conv"
        || fields.patch_size != 1
        || fields.use_absolute_embeddings
        || (fields.img_range - 1.0).abs() > f64::EPSILON
        || fields.depths.len() != fields.num_heads.len()
        || fields.upscale <= 0
        || fields.upscale & (fields.upscale - 1) != 0
        || fields.image_size <= fields.window_size
    {
        return Err(Error::unsupported("this Swin2SR config"));
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        reason = "`transformers` truncates the product to an integer the same way"
    )]
    let inner = (fields.mlp_ratio * fields.embed_dim as f64) as i64;
    Ok(Swin2SrConfig {
        width: fields.embed_dim,
        stages: fields.depths.into_iter().zip(fields.num_heads).collect(),
        inner,
        activation: fields.hidden_act,
        eps: fields.layer_norm_eps,
        window: fields.window_size,
        channels: fields.num_channels,
        upscale: fields.upscale,
        upsampler: 64,
    })
}

pub(crate) fn rules() -> Rules {
    let mut rules = Rules::new();
    // One convolution before each doubling, numbered in its name.
    for index in 0..4 {
        rules = rules.rename(
            &format!("upsample.upsample.convolution_{index}"),
            &format!("upsample.{index}"),
        );
    }

    let layer = |name: &str| format!("swin2sr.encoder.stages.*.layers.*.{name}");
    let block = |name: &str| format!("stages.*.blocks.*.{name}");
    rules
        .rename("swin2sr.first_convolution", "first")
        .rename(
            "swin2sr.embeddings.patch_embeddings.projection",
            "projection",
        )
        .rename(
            "swin2sr.embeddings.patch_embeddings.layernorm",
            "projection_norm",
        )
        .rename(&layer("attention.self.query"), &block("attention.query"))
        .rename(&layer("attention.self.key"), &block("attention.key"))
        .rename(&layer("attention.self.value"), &block("attention.value"))
        .rename(
            &layer("attention.self.logit_scale"),
            &block("attention.logit_scale"),
        )
        .rename(
            &layer("attention.self.continuous_position_bias_mlp.0"),
            &block("attention.bias_hidden"),
        )
        .rename(
            &layer("attention.self.continuous_position_bias_mlp.2"),
            &block("attention.bias_out"),
        )
        .rename(&layer("attention.output.dense"), &block("attention.output"))
        .rename(&layer("layernorm_before"), &block("attention_norm"))
        .rename(&layer("intermediate.dense"), &block("up"))
        .rename(&layer("output.dense"), &block("down"))
        .rename(&layer("layernorm_after"), &block("feed_forward_norm"))
        .rename("swin2sr.encoder.stages.*.conv", "stages.*.convolution")
        .rename(
            "swin2sr.encoder.stages.*.patch_embed.projection",
            "stages.*.projection",
        )
        .rename("swin2sr.layernorm", "norm")
        .rename("swin2sr.conv_after_body", "after_body")
        .rename("upsample.conv_before_upsample", "before_upsample")
        .rename("upsample.final_convolution", "last")
}
