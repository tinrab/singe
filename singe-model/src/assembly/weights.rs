// Loading a snapshot's weights into a model built on `Device::Meta`.

use std::path::Path;

use singe::{
    Device,
    nn::{CheckpointOptions, Model, StateNameMap, StatePiece},
};

use crate::error::Result;

// Replaces a schema's metadata with the snapshot's weights on `device`, cast to the schema's type.
pub fn load_weights<L>(
    schema: &Model<L>,
    snapshot: impl AsRef<Path>,
    device: Device,
    names: Option<&StateNameMap>,
) -> Result<Model<L>> {
    let state = schema.state().load_checkpoint(
        snapshot,
        &CheckpointOptions {
            device: Some(device),
            convert: true,
            names,
            pieces: None,
        },
    )?;
    Ok(schema.with_state(state)?)
}

fn expert_piece(name: &str, inner: i64) -> Option<StatePiece> {
    let (prefix, rest) = name.split_once(".mlp.experts.")?;
    let (expert, projection) = rest.split_once('.')?;
    let expert = expert.parse::<i64>().ok()?;
    let (packed, row) = match projection {
        "gate_proj.weight" => ("gate_up_proj", 0),
        "up_proj.weight" => ("gate_up_proj", inner),
        "down_proj.weight" => ("down_proj", 0),
        _ => return None,
    };
    Some(StatePiece {
        state: format!("{prefix}.mlp.experts.{packed}"),
        origin: vec![expert, row, 0],
    })
}

pub fn load_expert_weights<L>(
    schema: &Model<L>,
    snapshot: impl AsRef<Path>,
    device: Device,
    names: &StateNameMap,
    inner: i64,
) -> Result<Model<L>> {
    // Only the model's own experts, never those of a layer the names leave out, such as a multi-token prediction layer's.
    let pieces = move |name: &str| {
        name.starts_with("model.")
            .then(|| expert_piece(name, inner))
            .flatten()
    };
    let state = schema.state().load_checkpoint(
        snapshot,
        &CheckpointOptions {
            device: Some(device),
            convert: true,
            names: Some(names),
            pieces: Some(&pieces),
        },
    )?;
    Ok(schema.with_state(state)?)
}
