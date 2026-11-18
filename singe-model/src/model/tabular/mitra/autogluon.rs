// Reads an AutoGluon Tab2D snapshot: its config and its tensors' names.

use serde::Deserialize;

use super::{MitraConfig, MitraTask};
use crate::{
    error::{Error, Result},
    format::Rules,
    source::Source,
};

#[derive(Debug, Deserialize)]
struct Fields {
    dim: i64,
    dim_output: i64,
    n_layers: i64,
    n_heads: i64,
    task: String,
}

pub(crate) fn config(source: &Source) -> Result<MitraConfig> {
    let fields: Fields = source.json("config.json")?;
    Ok(MitraConfig {
        width: fields.dim,
        outputs: fields.dim_output,
        layers: fields.n_layers,
        heads: fields.n_heads,
        task: match fields.task.as_str() {
            "CLASSIFICATION" => MitraTask::Classification,
            "REGRESSION" => MitraTask::Regression,
            other => return Err(Error::UnsupportedConfig(format!("Tab2D task {other}"))),
        },
    })
}

pub(crate) fn rules() -> Rules {
    let layer = |name: &str| format!("layers.*.{name}");
    let attention = |rules: Rules, from: &str, to: &str| {
        ["q", "k", "v", "o"]
            .iter()
            .zip(["query", "key", "value", "output"])
            .fold(rules, |rules, (stored, target)| {
                rules.rename(
                    &layer(&format!("{from}.{stored}")),
                    &format!("blocks.*.{to}.{target}"),
                )
            })
    };
    let rules = Rules::new()
        .rename("x_embedding.x_embedding", "cells")
        .rename("y_embedding.y_mask", "unknown")
        .rename(&layer("layer_norm1"), "blocks.*.rows_norm")
        .rename(&layer("layer_norm2"), "blocks.*.rows_feed_forward_norm")
        .rename(&layer("linear1"), "blocks.*.rows_feed_forward.up")
        .rename(&layer("linear2"), "blocks.*.rows_feed_forward.down")
        .rename(&layer("layer_norm3"), "blocks.*.cells_norm")
        .rename(&layer("layer_norm4"), "blocks.*.cells_feed_forward_norm")
        .rename(&layer("linear3"), "blocks.*.cells_feed_forward.up")
        .rename(&layer("linear4"), "blocks.*.cells_feed_forward.down")
        .rename("final_layer_norm", "norm")
        .rename("final_layer", "output");
    // A classifier's targets are a table of classes, and a regressor's a linear map of the value, both stored as `y_embedding`.
    attention(
        attention(rules, "attention1", "rows"),
        "attention2",
        "cells",
    )
    .rename("y_embedding.y_embedding", "targets")
}
