//! Reads a `transformers` V-JEPA 2 config and imports its encoder, alone or under a classifier, leaving out the predictor.

use serde::Deserialize;
use singe::nn::Activation;

use crate::{
    assembly::clip_layer_rules,
    error::Result,
    format::{
        Rules, join,
        transformers::{activation_field, classifier_config},
    },
    model::video::vjepa2::Vjepa2Config,
    source::Source,
};

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Fields {
    hidden_size: i64,
    num_hidden_layers: i64,
    num_attention_heads: i64,
    mlp_ratio: f64,
    #[serde(deserialize_with = "activation_field")]
    hidden_act: Activation,
    layer_norm_eps: f32,
    in_chans: i64,
    patch_size: i64,
    tubelet_size: i64,
    crop_size: i64,
    #[serde(default = "pooler_layers")]
    num_pooler_layers: i64,
}

/// `VJEPA2Config`'s default pooler layers, which a base checkpoint's config leaves out.
const fn pooler_layers() -> i64 {
    3
}

pub(crate) fn config(source: &Source) -> Result<Vjepa2Config> {
    let fields: Fields = source.json("config.json")?;
    Ok(Vjepa2Config {
        width: fields.hidden_size,
        layers: fields.num_hidden_layers,
        heads: fields.num_attention_heads,
        inner: (fields.hidden_size as f64 * fields.mlp_ratio) as i64,
        activation: fields.hidden_act,
        eps: fields.layer_norm_eps,
        channels: fields.in_chans,
        patch_size: fields.patch_size,
        tubelet: fields.tubelet_size,
        grid: fields.crop_size / fields.patch_size,
        pooler_layers: fields.num_pooler_layers,
        classifier: classifier_config(source)?,
    })
}

/// Imports the encoder stored under `prefix`, `vjepa2` in a classifier and nothing in a `VJEPA2Model`, skipping the predictor and any classifier.
pub(crate) fn encoder_rules(prefix: &str) -> Rules {
    let encoder = |name: &str| join(prefix, &format!("encoder.{name}"));
    Rules::new()
        .rename(&encoder("embeddings.patch_embeddings.proj"), "tubelets")
        .rename_each(
            &encoder("layer.*"),
            "blocks.*",
            &[
                ("norm1", "attention_norm"),
                ("attention.query", "attention.query"),
                ("attention.key", "attention.key"),
                ("attention.value", "attention.value"),
                ("attention.proj", "attention.output"),
                ("norm2", "feed_forward_norm"),
                ("mlp.fc1", "feed_forward.up"),
                ("mlp.fc2", "feed_forward.down"),
            ],
        )
        .rename(&encoder("layernorm"), "norm")
        .skip(&join(prefix, "predictor"))
        .skip("pooler")
        .skip("classifier")
}

/// Imports a classifier: its pooler and classifier, then its encoder under `vjepa2`.
pub(crate) fn rules() -> Rules {
    clip_layer_rules("pooler.self_attention_layers", "pooler.blocks")
        .rename("pooler.query_tokens", "pooler.query")
        .rename_each(
            "pooler.cross_attention_layer",
            "pooler",
            &[
                ("layer_norm1", "cross_norm"),
                ("cross_attn.q_proj", "cross_attention.query"),
                ("cross_attn.k_proj", "cross_attention.key"),
                ("cross_attn.v_proj", "cross_attention.value"),
                ("layer_norm2", "cross_feed_forward_norm"),
                ("mlp.fc1", "cross_feed_forward.up"),
                ("mlp.fc2", "cross_feed_forward.down"),
            ],
        )
        .rename("classifier", "classifier")
        .then(encoder_rules("vjepa2"))
}
