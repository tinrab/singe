mod alias;
mod families;
mod shape;

use self::{
    alias::redundant_metadata_or_transfer_input,
    families::{alias_redundant_input, shape_redundant_input},
};
use crate::{
    error::Result,
    graph::{Graph, ir::ValueId, node::NodeRef},
};

// Rewrites here never change a value: each replaces an operation with an input that has the same values and metadata.
pub fn redundant_operation_input(graph: &Graph, node: &NodeRef<'_>) -> Result<Option<ValueId>> {
    if let Some(input) = redundant_metadata_or_transfer_input(graph, node)? {
        return Ok(Some(input));
    }

    // Unsorted top-k does not promise the input order even when k spans the whole axis, so it is not an identity.
    if node.topk_values_identity().is_some() {
        return Ok(None);
    }

    if let Some(input) = shape_redundant_input(graph, node)? {
        return Ok(Some(input));
    }
    alias_redundant_input(graph, node)
}
