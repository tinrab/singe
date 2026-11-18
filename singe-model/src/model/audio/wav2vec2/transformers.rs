// Reads a `transformers` Wav2Vec2 config and its feature extractor's, and imports its checkpoint.

use serde::Deserialize;
use singe::nn::Activation;

use super::Wav2Vec2Config;
use crate::{
    error::{Error, Result},
    format::{Rules, transformers::activation_field},
    source::Source,
};

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Fields {
    conv_dim: Vec<i64>,
    conv_kernel: Vec<i64>,
    conv_stride: Vec<i64>,
    #[serde(default)]
    conv_bias: bool,
    feat_extract_norm: String,
    #[serde(deserialize_with = "activation_field")]
    feat_extract_activation: Activation,
    hidden_size: i64,
    num_hidden_layers: i64,
    num_attention_heads: i64,
    intermediate_size: i64,
    #[serde(deserialize_with = "activation_field")]
    hidden_act: Activation,
    layer_norm_eps: f32,
    num_conv_pos_embeddings: i64,
    num_conv_pos_embedding_groups: i64,
    #[serde(default)]
    do_stable_layer_norm: bool,
    vocab_size: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct Extractor {
    sampling_rate: i64,
    #[serde(default = "enabled")]
    do_normalize: bool,
}

const fn enabled() -> bool {
    true
}

pub(crate) fn config(source: &Source) -> Result<Wav2Vec2Config> {
    let fields: Fields = source.json("config.json")?;
    // The base models' layout: a group norm after only the first convolution, and blocks that norm after each residual.
    if fields.feat_extract_norm != "group" || fields.do_stable_layer_norm {
        return Err(Error::unsupported("this Wav2Vec2 layout"));
    }

    let extractor: Extractor = source.json("preprocessor_config.json")?;
    Ok(Wav2Vec2Config {
        convolutions: fields
            .conv_dim
            .iter()
            .zip(&fields.conv_kernel)
            .zip(&fields.conv_stride)
            .map(|((&channels, &kernel), &stride)| (channels, kernel, stride))
            .collect(),
        convolution_bias: fields.conv_bias,
        feature_activation: fields.feat_extract_activation,
        width: fields.hidden_size,
        layers: fields.num_hidden_layers,
        heads: fields.num_attention_heads,
        inner: fields.intermediate_size,
        activation: fields.hidden_act,
        eps: fields.layer_norm_eps,
        position_kernel: fields.num_conv_pos_embeddings,
        position_groups: fields.num_conv_pos_embedding_groups,
        vocab_size: fields.vocab_size,
        sample_rate: extractor.sampling_rate,
        normalize: extractor.do_normalize,
    })
}

pub(crate) fn rules() -> Rules {
    let layer = |name: &str| format!("wav2vec2.encoder.layers.*.{name}");
    Rules::new()
        .rename(
            "wav2vec2.feature_extractor.conv_layers.*.conv",
            "convolutions.*",
        )
        .rename(
            "wav2vec2.feature_extractor.conv_layers.0.layer_norm",
            "first_norm",
        )
        .rename("wav2vec2.feature_projection.layer_norm", "projection_norm")
        .rename("wav2vec2.feature_projection.projection", "projection")
        .weight_norm(
            "wav2vec2.encoder.pos_conv_embed.conv",
            "positions.weight",
            2,
        )
        .rename(
            "wav2vec2.encoder.pos_conv_embed.conv.bias",
            "positions.bias",
        )
        .rename("wav2vec2.encoder.layer_norm", "norm")
        .rename(&layer("attention.q_proj"), "blocks.*.attention.query")
        .rename(&layer("attention.k_proj"), "blocks.*.attention.key")
        .rename(&layer("attention.v_proj"), "blocks.*.attention.value")
        .rename(&layer("attention.out_proj"), "blocks.*.attention.output")
        .rename(&layer("layer_norm"), "blocks.*.attention_norm")
        .rename(
            &layer("feed_forward.intermediate_dense"),
            "blocks.*.feed_forward.up",
        )
        .rename(
            &layer("feed_forward.output_dense"),
            "blocks.*.feed_forward.down",
        )
        .rename(&layer("final_layer_norm"), "blocks.*.feed_forward_norm")
        .rename("lm_head", "head")
        // The embedding of masked frames, which only pretraining reads.
        .skip("wav2vec2.masked_spec_embed")
}
