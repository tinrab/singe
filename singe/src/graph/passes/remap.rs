use std::collections::HashMap;

use crate::{
    error::Result,
    graph::{
        Graph,
        ir::ValueId,
        node::{Node, NodeRef},
    },
};

pub fn prune_dead_nodes_with_map(graph: &Graph) -> Result<(Graph, Vec<(ValueId, ValueId)>)> {
    let mut graph = graph.clone();
    let source_count = graph.node_count();
    let remap = graph.prune_dead_operations()?;
    let remapped = (0..source_count)
        .map(ValueId::new)
        .filter_map(|old| Some((old, remap.get(old)?)))
        .collect();
    Ok((graph, remapped))
}

pub fn remapped_node(node: &NodeRef<'_>, id_map: &HashMap<ValueId, ValueId>) -> Result<Node> {
    node.map_inputs(|id| remap_node_id(id, id_map))
}

pub fn remap_node_id(id: ValueId, id_map: &HashMap<ValueId, ValueId>) -> Result<ValueId> {
    Graph::remapped_node(id, id_map)
}
