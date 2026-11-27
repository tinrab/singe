//! Reads an `AutoGluon` `Tab2D` snapshot: its config and its tensors' names.

use serde::Deserialize;

use crate::{
    error::{Error, Result},
    format::RuleSet,
    model::tabular::mitra::{MitraConfig, MitraTask},
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
            other => return Err(Error::unsupported(format!("Tab2D task {other}"))),
        },
    })
}

pub(crate) fn rules() -> RuleSet {
    let projections = [
        ("q", "query"),
        ("k", "key"),
        ("v", "value"),
        ("o", "output"),
    ];
    RuleSet::new()
        .rename("x_embedding.x_embedding", "cells")
        .rename("y_embedding.y_mask", "unknown")
        .rename_each(
            "layers.*",
            "blocks.*",
            &[
                ("layer_norm1", "rows_norm"),
                ("layer_norm2", "rows_feed_forward_norm"),
                ("linear1", "rows_feed_forward.up"),
                ("linear2", "rows_feed_forward.down"),
                ("layer_norm3", "cells_norm"),
                ("layer_norm4", "cells_feed_forward_norm"),
                ("linear3", "cells_feed_forward.up"),
                ("linear4", "cells_feed_forward.down"),
            ],
        )
        .rename("final_layer_norm", "norm")
        .rename("final_layer", "output")
        .rename_each("layers.*.attention1", "blocks.*.rows", &projections)
        .rename_each("layers.*.attention2", "blocks.*.cells", &projections)
        // A classifier's targets are a table of classes, and a regressor's a linear map of the value, both stored as `y_embedding`.
        .rename("y_embedding.y_embedding", "targets")
}
