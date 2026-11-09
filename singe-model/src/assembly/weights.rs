// Loading a snapshot's weights into a model built on `Device::Meta`.

use std::path::Path;

use singe::{
    Device,
    nn::{CheckpointOptions, Model, StateNameMap},
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
