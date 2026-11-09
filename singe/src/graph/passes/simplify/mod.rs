mod operation_key;
mod redundant;
mod state;
mod topk;

use self::{operation_key::operation_key, state::SimplifyState};
use crate::{
    error::Result,
    graph::{Graph, ir::ValueId, passes::remap::remapped_node},
};

pub fn remove_redundant_ops_with_map(graph: &Graph) -> Result<(Graph, Vec<(ValueId, ValueId)>)> {
    let mut state = SimplifyState::create(graph);
    for old_index in 0..graph.node_count() {
        let old_id = ValueId::new(old_index);
        let node = &graph.required_node(old_id)?;
        if let Some(existing) = state.existing_input_constant(node) {
            state.replace(old_id, existing);
            continue;
        }

        if let Some(replacement) = state.redundant_replacement(graph, node)? {
            state.replace(old_id, replacement);
            continue;
        }

        let remapped = remapped_node(node, state.id_map())?;
        let operation_key = operation_key(&remapped.view());
        if let Some(existing) =
            state.existing_remapped_node(node, &remapped.view(), operation_key.as_ref())?
        {
            state.replace(old_id, existing);
            continue;
        }

        state.retain(old_id, node, remapped, operation_key)?;
    }
    state.finish(graph)
}
