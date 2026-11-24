//! Reads a `transformers` `Wav2Vec2` config and its feature extractor's, and imports its checkpoint.

use serde::Deserialize;
use singe::nn::Activation;

use crate::{
    assembly::attention_rules,
    error::{Error, Result},
    format::{
        Rules,
        transformers::{activation_field, default_true},
    },
    model::audio::wav2vec2::{FeatureConvolution, Wav2Vec2Config},
    source::Source,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
enum FeatureNorm {
    Group,
    Layer,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Fields {
    conv_dim: Vec<i64>,
    conv_kernel: Vec<i64>,
    conv_stride: Vec<i64>,
    #[serde(default)]
    conv_bias: bool,
    feat_extract_norm: FeatureNorm,
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
    #[serde(default = "default_true")]
    do_normalize: bool,
}

pub(crate) fn config(source: &Source) -> Result<Wav2Vec2Config> {
    let fields: Fields = source.json("config.json")?;
    // The base models' layout: a group norm after only the first convolution, and blocks that norm after each residual.
    if fields.feat_extract_norm != FeatureNorm::Group {
        return Err(Error::unsupported(
            "a Wav2Vec2 `feat_extract_norm` other than `group`",
        ));
    }

    if fields.do_stable_layer_norm {
        return Err(Error::unsupported("a Wav2Vec2 `do_stable_layer_norm`"));
    }

    let count = fields.conv_dim.len();
    if count == 0 || fields.conv_kernel.len() != count || fields.conv_stride.len() != count {
        return Err(Error::unsupported(
            "Wav2Vec2 `conv_dim`, `conv_kernel`, and `conv_stride` of other than one nonzero length",
        ));
    }

    let extractor: Extractor = source.json("preprocessor_config.json")?;
    Ok(Wav2Vec2Config {
        convolutions: fields
            .conv_dim
            .iter()
            .zip(&fields.conv_kernel)
            .zip(&fields.conv_stride)
            .map(|((&channels, &kernel), &stride)| FeatureConvolution {
                channels,
                kernel,
                stride,
            })
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
    let layers = "wav2vec2.encoder.layers.*";
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
        .then(attention_rules(
            &format!("{layers}.attention"),
            "blocks.*.attention",
        ))
        .rename_each(
            layers,
            "blocks.*",
            &[
                ("layer_norm", "attention_norm"),
                ("feed_forward.intermediate_dense", "feed_forward.up"),
                ("feed_forward.output_dense", "feed_forward.down"),
                ("final_layer_norm", "feed_forward_norm"),
            ],
        )
        .rename("lm_head", "head")
        // The embedding of masked frames, which only pretraining reads.
        .skip("wav2vec2.masked_spec_embed")
}
