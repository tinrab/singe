//! Reads a `transformers` `SuperGlue` config and imports its checkpoint, which holds `SuperPoint` under `keypoint_detector`.

use serde::Deserialize;

use crate::{
    error::{Error, Result},
    format::Rules,
    model::vision::{
        superglue::SuperGlueConfig,
        superpoint::transformers::{Fields as SuperPointFields, detector_rules, fields_config},
    },
    source::Source,
};

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Fields {
    keypoint_detector_config: SuperPointFields,
    hidden_size: i64,
    num_attention_heads: i64,
    keypoint_encoder_sizes: Vec<i64>,
    gnn_layers_types: Vec<String>,
    sinkhorn_iterations: usize,
    matching_threshold: f32,
}

pub(crate) fn config(source: &Source) -> Result<SuperGlueConfig> {
    let fields: Fields = source.json("config.json")?;
    let crosses = fields
        .gnn_layers_types
        .iter()
        .map(|kind| match kind.as_str() {
            "self" => Ok(false),
            "cross" => Ok(true),
            other => Err(Error::unsupported(format!("a SuperGlue `{other}` layer"))),
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(SuperGlueConfig {
        detector: fields_config(&fields.keypoint_detector_config)?,
        width: fields.hidden_size,
        heads: fields.num_attention_heads,
        encoder: fields.keypoint_encoder_sizes,
        crosses,
        sinkhorn_iterations: fields.sinkhorn_iterations,
        matching_threshold: fields.matching_threshold,
    })
}

pub(crate) fn rules(config: &SuperGlueConfig) -> Rules {
    let encoder = config.encoder.len();
    detector_rules("keypoint_detector", "detector")
        // A batch norm's count of the batches it saw in training, which evaluation never reads.
        .skip("keypoint_encoder.encoder.*.batch_norm.num_batches_tracked")
        .skip("gnn.layers.*.mlp.0.batch_norm.num_batches_tracked")
        .rename(
            &format!("keypoint_encoder.encoder.{encoder}"),
            "encoder_output",
        )
        .rename_each(
            "keypoint_encoder.encoder.*",
            "encoder.*",
            &[("linear", "linear"), ("batch_norm", "norm")],
        )
        .rename_each(
            "gnn.layers.*",
            "layers.*",
            &[
                ("attention.self.query", "attention.query"),
                ("attention.self.key", "attention.key"),
                ("attention.self.value", "attention.value"),
                ("attention.output.dense", "attention.output"),
                ("mlp.0.linear", "hidden.linear"),
                ("mlp.0.batch_norm", "hidden.norm"),
                ("mlp.1", "output"),
            ],
        )
        .rename("final_projection.final_proj", "projection")
        .rename("bin_score", "bin_score")
}
