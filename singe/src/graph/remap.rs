use std::collections::{HashMap, HashSet};

use crate::{
    error::{Error, Result},
    graph::{Graph, ir::ValueId, node::Node, outputs::GraphOutput},
};

impl Graph {
    /// Creates a graph from retained nodes and metadata remapped from `source`.
    pub(crate) fn from_retained_nodes(
        source: &Self,
        id_map: &HashMap<ValueId, ValueId>,
        nodes: Vec<Node>,
    ) -> Result<Self> {
        let outputs = source
            .output_projections()
            .iter()
            .map(|output| Ok(output.with_node(Self::remapped_node(output.node(), id_map)?)))
            .collect::<Result<Vec<_>>>()?;
        let input_names = Self::input_names_for_nodes(&nodes);
        let initializers = source
            .initializers
            .iter()
            .filter(|(name, _)| input_names.contains(*name))
            .map(|(name, tensor)| (name.clone(), tensor.clone()))
            .collect();
        let static_initializers = source
            .static_initializers
            .iter()
            .filter(|name| input_names.contains(*name))
            .cloned()
            .collect();
        let input_dimension_guards = source
            .input_dimension_guards
            .iter()
            .filter(|((name, _), _)| input_names.contains(name))
            .map(|(key, guard)| (key.clone(), guard.clone()))
            .collect();

        let mut graph = Self::new();
        for node in nodes {
            graph.push_node(node);
        }

        graph.initializers = initializers;
        graph.static_initializers = static_initializers;
        graph.input_dimension_guards = input_dimension_guards;
        graph.state_kinds = source
            .state_kinds
            .iter()
            .filter(|(name, _)| input_names.contains(*name))
            .map(|(name, kind)| (name.clone(), *kind))
            .collect();
        graph.capture_info = source.capture_info;
        graph.input_writes.clone_from(&source.input_writes);
        graph.counted_outputs.clone_from(&source.counted_outputs);
        graph.generators.clone_from(&source.generators);
        graph.refresh_input_bindings();
        graph.set_output_values_with_names(
            outputs.iter().map(GraphOutput::node).collect(),
            outputs
                .iter()
                .map(|output| output.name().map(ToOwned::to_owned))
                .collect(),
        );
        Ok(graph)
    }

    pub(crate) fn remapped_node(
        id: ValueId,
        id_map: &HashMap<ValueId, ValueId>,
    ) -> Result<ValueId> {
        id_map
            .get(&id)
            .copied()
            .ok_or_else(|| Error::missing_node("remapped node"))
    }

    /// Returns `(old, new)` node id pairs for every source graph node.
    pub(crate) fn remapped_node_pairs(
        &self,
        id_map: &HashMap<ValueId, ValueId>,
    ) -> Result<Vec<(ValueId, ValueId)>> {
        self.node_ids()
            .map(|id| Ok((id, Self::remapped_node(id, id_map)?)))
            .collect()
    }

    fn input_names_for_nodes(nodes: &[Node]) -> HashSet<String> {
        nodes
            .iter()
            .filter_map(|node| {
                node.view()
                    .input_metadata()
                    .map(|(name, _, _, _)| name.to_string())
            })
            .collect()
    }
}
