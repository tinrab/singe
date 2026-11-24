//! Reads a `transformers` Depth Anything config, whose backbone is a partial `DINOv2` config, and imports its checkpoint.

use serde::Deserialize;

use crate::{
    error::{Error, Result},
    format::{Rules, transformers::default_true},
    model::vision::{
        depth_anything::DepthAnythingConfig,
        dinov2::transformers::{Fields as Dinov2Fields, encoder_rules, fields_config},
    },
    source::Source,
};

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Backbone {
    #[serde(flatten)]
    fields: Dinov2Fields,
    out_indices: Vec<usize>,
    #[serde(default)]
    reshape_hidden_states: bool,
    #[serde(default = "default_true")]
    apply_layernorm: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
enum DepthEstimationType {
    #[default]
    Relative,
    Metric,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Fields {
    backbone_config: Backbone,
    neck_hidden_sizes: Vec<i64>,
    reassemble_factors: Vec<f64>,
    reassemble_hidden_size: i64,
    fusion_hidden_size: i64,
    head_hidden_size: i64,
    #[serde(default = "default_head_in_index")]
    head_in_index: i64,
    #[serde(default)]
    depth_estimation_type: Option<DepthEstimationType>,
    #[serde(default = "default_max_depth")]
    max_depth: f64,
}

const fn default_head_in_index() -> i64 {
    -1
}

const fn default_max_depth() -> f64 {
    1.0
}

pub(crate) fn config(source: &Source) -> Result<DepthAnythingConfig> {
    let fields: Fields = source.json("config.json")?;
    let backbone = fields_config(&fields.backbone_config.fields)?;
    // The neck reads the normed states as tokens, and the head the finest fused map.
    if fields.backbone_config.reshape_hidden_states {
        return Err(Error::unsupported(
            "a Depth Anything backbone with `reshape_hidden_states`",
        ));
    }

    if !fields.backbone_config.apply_layernorm {
        return Err(Error::unsupported(
            "a Depth Anything backbone without `apply_layernorm`",
        ));
    }

    if fields.head_in_index != -1 {
        return Err(Error::unsupported(format!(
            "a Depth Anything head reading stage {}",
            fields.head_in_index
        )));
    }

    if fields.reassemble_hidden_size != backbone.encoder.width {
        return Err(Error::unsupported(format!(
            "a Depth Anything `reassemble_hidden_size` of {} beside a backbone width of {}",
            fields.reassemble_hidden_size, backbone.encoder.width
        )));
    }

    let stages = fields.neck_hidden_sizes.len();
    if fields.reassemble_factors.len() != stages
        || fields.backbone_config.out_indices.len() != stages
    {
        return Err(Error::unsupported(format!(
            "Depth Anything `reassemble_factors` or `out_indices` of other than {stages} entries"
        )));
    }

    let metric = fields.depth_estimation_type.unwrap_or_default() == DepthEstimationType::Metric;
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
    encoder_rules("backbone", "backbone")
        .rename("neck.reassemble_stage.layers", "reassemble")
        .rename("neck.convs", "convolutions")
        .rename_each(
            "neck.fusion_stage.layers.*",
            "fusions.*",
            &[
                ("residual_layer1.convolution1", "residual.first"),
                ("residual_layer1.convolution2", "residual.second"),
                ("residual_layer2.convolution1", "refine.first"),
                ("residual_layer2.convolution2", "refine.second"),
                ("projection", "projection"),
            ],
        )
        .rename_each(
            "head",
            "head",
            &[("conv1", "0"), ("conv2", "1"), ("conv3", "2")],
        )
}
