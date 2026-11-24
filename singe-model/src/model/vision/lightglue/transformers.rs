//! Reads a `transformers` `LightGlue` config, which nests its keypoint detector's, and imports its checkpoint, which stores the detector under `keypoint_detector.`.

use serde::Deserialize;
use singe::nn::Activation;

use crate::{
    error::{Error, Result},
    format::{Rules, transformers::activation_field},
    model::vision::{
        lightglue::LightGlueConfig,
        superpoint::transformers::{Fields as SuperPointFields, detector_rules, fields_config},
    },
    source::Source,
};

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Fields {
    keypoint_detector_config: SuperPointFields,
    descriptor_dim: i64,
    hidden_size: i64,
    intermediate_size: i64,
    num_hidden_layers: i64,
    num_attention_heads: i64,
    num_key_value_heads: i64,
    #[serde(deserialize_with = "activation_field")]
    hidden_act: Activation,
    attention_bias: bool,
    filter_threshold: f32,
    depth_confidence: f32,
    width_confidence: f32,
}

pub(crate) fn config(source: &Source) -> Result<LightGlueConfig> {
    let fields: Fields = source.json("config.json")?;
    if fields.num_key_value_heads != fields.num_attention_heads {
        return Err(Error::unsupported(format!(
            "attention of {} key and value heads for {} query heads",
            fields.num_key_value_heads, fields.num_attention_heads
        )));
    }

    if !fields.attention_bias {
        return Err(Error::unsupported("attention without biases in LightGlue"));
    }

    if fields.hidden_size != fields.descriptor_dim {
        return Err(Error::unsupported(format!(
            "a LightGlue hidden size of {} for descriptors of {}",
            fields.hidden_size, fields.descriptor_dim
        )));
    }

    if fields.intermediate_size != 2 * fields.descriptor_dim {
        return Err(Error::unsupported(format!(
            "a LightGlue intermediate size of {} for descriptors of {}",
            fields.intermediate_size, fields.descriptor_dim
        )));
    }

    // Matching that never stops early or prunes takes another path through `transformers`' forward.
    if fields.depth_confidence <= 0.0 || fields.width_confidence <= 0.0 {
        return Err(Error::unsupported(format!(
            "matching without early stopping or pruning (depth confidence {}, width confidence {})",
            fields.depth_confidence, fields.width_confidence
        )));
    }

    if fields.num_hidden_layers < 1 {
        return Err(Error::unsupported(format!(
            "a LightGlue of {} layers",
            fields.num_hidden_layers
        )));
    }

    let detector = fields_config(&fields.keypoint_detector_config)?;
    if detector.descriptor != fields.descriptor_dim {
        return Err(Error::unsupported("descriptors projected to another width"));
    }

    Ok(LightGlueConfig {
        detector,
        width: fields.descriptor_dim,
        heads: fields.num_attention_heads,
        inner: fields.intermediate_size,
        layers: fields.num_hidden_layers,
        activation: fields.hidden_act,
        filter_threshold: fields.filter_threshold,
        depth_confidence: fields.depth_confidence,
        width_confidence: fields.width_confidence,
    })
}

pub(crate) fn rules() -> Rules {
    const ATTENTION: &[(&str, &str)] = &[
        ("q_proj", "query"),
        ("k_proj", "key"),
        ("v_proj", "value"),
        ("o_proj", "output"),
    ];

    const UPDATE: &[(&str, &str)] = &[("fc1", "up"), ("layer_norm", "norm"), ("fc2", "down")];
    Rules::new()
        .rename("positional_encoder.projector", "positions")
        .rename(
            "match_assignment_layers.*.final_projection",
            "assignments.*.projection",
        )
        .rename(
            "match_assignment_layers.*.matchability",
            "assignments.*.matchability",
        )
        .rename("token_confidence.*.token", "confidences.*")
        .rename_each(
            "transformer_layers.*.self_attention",
            "blocks.*.self_attention",
            ATTENTION,
        )
        .rename_each(
            "transformer_layers.*.cross_attention",
            "blocks.*.cross_attention",
            ATTENTION,
        )
        .rename_each(
            "transformer_layers.*.self_mlp",
            "blocks.*.self_update",
            UPDATE,
        )
        .rename_each(
            "transformer_layers.*.cross_mlp",
            "blocks.*.cross_update",
            UPDATE,
        )
        .then(detector_rules("keypoint_detector", "detector"))
}
