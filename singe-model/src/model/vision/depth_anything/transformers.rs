// Reads a `transformers` Depth Anything config, whose backbone is a partial DINOv2 config, and imports its checkpoint.

use serde::Deserialize;

use super::DepthAnythingConfig;
use crate::{
    error::{Error, Result},
    format::Rules,
    model::vision::dinov2::transformers::{Fields as Dinov2Fields, encoder_rules, fields_config},
    source::Source,
};

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Backbone {
    #[serde(flatten)]
    fields: Dinov2Fields,
    out_indices: Vec<usize>,
    #[serde(default)]
    reshape_hidden_states: bool,
    #[serde(default = "enabled")]
    apply_layernorm: bool,
}

const fn enabled() -> bool {
    true
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Fields {
    backbone_config: Backbone,
    neck_hidden_sizes: Vec<i64>,
    reassemble_factors: Vec<f64>,
    reassemble_hidden_size: i64,
    fusion_hidden_size: i64,
    head_hidden_size: i64,
    #[serde(default = "last")]
    head_in_index: i64,
    depth_estimation_type: Option<String>,
    #[serde(default = "unit")]
    max_depth: f64,
}

const fn last() -> i64 {
    -1
}

const fn unit() -> f64 {
    1.0
}

pub(crate) fn config(source: &Source) -> Result<DepthAnythingConfig> {
    let fields: Fields = source.json("config.json")?;
    let backbone = fields_config(&fields.backbone_config.fields)?;
    // The neck reads the normed states as tokens, and the head the finest fused map.
    if fields.backbone_config.reshape_hidden_states
        || !fields.backbone_config.apply_layernorm
        || fields.head_in_index != -1
        || fields.reassemble_hidden_size != backbone.width
        || fields.neck_hidden_sizes.len() != fields.reassemble_factors.len()
        || fields.neck_hidden_sizes.len() != fields.backbone_config.out_indices.len()
    {
        return Err(Error::unsupported("this Depth Anything neck"));
    }

    let metric = match fields.depth_estimation_type.as_deref() {
        None | Some("relative") => false,
        Some("metric") => true,
        Some(other) => {
            return Err(Error::UnsupportedConfig(format!(
                "depth estimation of type {other}"
            )));
        }
    };
    Ok(DepthAnythingConfig {
        backbone,
        stages: fields.backbone_config.out_indices,
        neck: fields
            .neck_hidden_sizes
            .into_iter()
            .zip(fields.reassemble_factors)
            .collect(),
        fusion: fields.fusion_hidden_size,
        head: fields.head_hidden_size,
        metric,
        max_depth: fields.max_depth,
    })
}

pub(crate) fn rules() -> Rules {
    let fusion = |name: &str| format!("neck.fusion_stage.layers.*.{name}");
    encoder_rules("backbone.", "backbone.")
        .rename("neck.reassemble_stage.layers", "reassemble")
        .rename("neck.convs", "convolutions")
        .rename(
            &fusion("residual_layer1.convolution1"),
            "fusions.*.residual.first",
        )
        .rename(
            &fusion("residual_layer1.convolution2"),
            "fusions.*.residual.second",
        )
        .rename(
            &fusion("residual_layer2.convolution1"),
            "fusions.*.refine.first",
        )
        .rename(
            &fusion("residual_layer2.convolution2"),
            "fusions.*.refine.second",
        )
        .rename(&fusion("projection"), "fusions.*.projection")
        .rename("head.conv1", "head.0")
        .rename("head.conv2", "head.1")
        .rename("head.conv3", "head.2")
}
