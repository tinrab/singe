// Reads a `transformers` LightGlue config, which nests its keypoint detector's, and imports its checkpoint, which stores the detector under `keypoint_detector.`.

use serde::Deserialize;
use singe::nn::Activation;

use super::LightGlueConfig;
use crate::{
    error::{Error, Result},
    format::{Rules, transformers::activation_field},
    model::vision::superpoint::transformers::{
        Fields as SuperPointFields, detector_rules, fields_config,
    },
    source::Source,
};

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Fields {
    keypoint_detector_config: SuperPointFields,
    descriptor_dim: i64,
    hidden_size: i64,
    intermediate_size: i64,
    num_hidden_layers: usize,
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
    // Matching that never stops early or prunes takes another path through `transformers`' forward.
    if fields.num_key_value_heads != fields.num_attention_heads
        || !fields.attention_bias
        || fields.hidden_size != fields.descriptor_dim
        || fields.intermediate_size != 2 * fields.descriptor_dim
        || fields.depth_confidence <= 0.0
        || fields.width_confidence <= 0.0
        || fields.num_hidden_layers == 0
    {
        return Err(Error::unsupported("this LightGlue config"));
    }

    let detector = fields_config(fields.keypoint_detector_config)?;
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
    let block = |name: &str| format!("transformer_layers.*.{name}");
    let mut rules = Rules::new()
        .rename("positional_encoder.projector", "positions")
        .rename(
            "match_assignment_layers.*.final_projection",
            "assignments.*.projection",
        )
        .rename(
            "match_assignment_layers.*.matchability",
            "assignments.*.matchability",
        )
        .rename("token_confidence.*.token", "confidences.*");
    for attention in ["self_attention", "cross_attention"] {
        for (from, to) in [
            ("q_proj", "query"),
            ("k_proj", "key"),
            ("v_proj", "value"),
            ("o_proj", "output"),
        ] {
            rules = rules.rename(
                &block(&format!("{attention}.{from}")),
                &format!("blocks.*.{attention}.{to}"),
            );
        }
    }

    for (from, to) in [("self_mlp", "self_update"), ("cross_mlp", "cross_update")] {
        for (layer, target) in [("fc1", "up"), ("layer_norm", "norm"), ("fc2", "down")] {
            rules = rules.rename(
                &block(&format!("{from}.{layer}")),
                &format!("blocks.*.{to}.{target}"),
            );
        }
    }
    rules.then(detector_rules("keypoint_detector.", "detector."))
}
