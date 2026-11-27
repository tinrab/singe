//! Reads a `transformers` `TimeSformer` config with divided space-time attention and imports its checkpoint, splitting each fused query, key, and value projection.

use serde::Deserialize;
use singe::nn::Activation;

use crate::{
    error::{Error, Result},
    format::{
        Rules,
        transformers::{activation_field, classifier_config},
    },
    model::video::timesformer::TimesformerConfig,
    source::Source,
};

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Fields {
    hidden_size: i64,
    num_hidden_layers: i64,
    num_attention_heads: i64,
    intermediate_size: i64,
    #[serde(deserialize_with = "activation_field")]
    hidden_act: Activation,
    layer_norm_eps: f32,
    patch_size: i64,
    image_size: i64,
    num_frames: i64,
    num_channels: i64,
    attention_type: String,
    qkv_bias: bool,
}

pub(crate) fn config(source: &Source) -> Result<TimesformerConfig> {
    let fields: Fields = source.json("config.json")?;
    if fields.attention_type != "divided_space_time" || !fields.qkv_bias {
        return Err(Error::unsupported(
            "a TimeSformer other than one of divided space-time attention with query, key, and value biases",
        ));
    }

    Ok(TimesformerConfig {
        width: fields.hidden_size,
        layers: fields.num_hidden_layers,
        heads: fields.num_attention_heads,
        inner: fields.intermediate_size,
        activation: fields.hidden_act,
        eps: fields.layer_norm_eps,
        patch_size: fields.patch_size,
        image_size: fields.image_size,
        frames: fields.num_frames,
        channels: fields.num_channels,
        classifier: classifier_config(source)?,
    })
}

pub(crate) fn rules() -> Rules {
    let layers = "timesformer.encoder.layer.*";
    let mut rules = Rules::new()
        .rename("timesformer.embeddings.cls_token", "class_token")
        .rename("timesformer.embeddings.position_embeddings", "positions")
        .rename("timesformer.embeddings.time_embeddings", "times")
        .rename(
            "timesformer.embeddings.patch_embeddings.projection",
            "patches",
        )
        .rename("timesformer.layernorm", "norm")
        .rename("classifier", "classifier");
    for (from, to) in [
        ("attention", "spatial_attention"),
        ("temporal_attention", "temporal_attention"),
    ] {
        rules = rules
            .split(
                &format!("{layers}.{from}.attention.qkv"),
                0,
                &[
                    &format!("layers.*.{to}.query"),
                    &format!("layers.*.{to}.key"),
                    &format!("layers.*.{to}.value"),
                ],
            )
            .rename(
                &format!("{layers}.{from}.output.dense"),
                &format!("layers.*.{to}.output"),
            );
    }
    rules.rename_each(
        layers,
        "layers.*",
        &[
            ("temporal_dense", "temporal_output"),
            ("temporal_layernorm", "temporal_norm"),
            ("layernorm_before", "spatial_norm"),
            ("layernorm_after", "feed_forward_norm"),
            ("intermediate.dense", "feed_forward.up"),
            ("output.dense", "feed_forward.down"),
        ],
    )
}
