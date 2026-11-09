use std::collections::HashMap;

use singe_cuda::graph::GraphNode;

use crate::{
    error::Result,
    graph::ir::{Graph, ValueId},
};

pub fn graph_node_positions(execution_order: &[ValueId]) -> HashMap<ValueId, usize> {
    execution_order
        .iter()
        .enumerate()
        .map(|(index, &id)| (id, index))
        .collect()
}

pub fn graph_users(
    ir: &Graph,
    execution_order: &[ValueId],
) -> Result<HashMap<ValueId, Vec<ValueId>>> {
    let mut users = HashMap::new();
    for &id in execution_order {
        for dependency in operation_dependencies(ir, id)? {
            users.entry(dependency).or_insert_with(Vec::new).push(id);
            // Lowering visits an operation by its first result, so a reader of a later one also reads that.
            if let Some(primary) = ir
                .operation_for_value(dependency)
                .and_then(|operation| operation.result_values().next())
                && primary != dependency
            {
                users.entry(primary).or_insert_with(Vec::new).push(id);
            }
        }
    }
    Ok(users)
}

pub fn operation_dependencies(ir: &Graph, id: ValueId) -> Result<Vec<ValueId>> {
    let operation = ir.required_operation_for_value(id)?;
    Ok(operation.inputs().to_vec())
}

/// Returns the native graph nodes that produce `dependencies`, each once.
///
/// Views of one value share its producer's node, and CUDA rejects a dependency list that names a node twice.
pub fn graph_native_dependencies(
    dependencies: &[ValueId],
    native_nodes: &HashMap<ValueId, GraphNode>,
) -> Vec<GraphNode> {
    let mut nodes = Vec::with_capacity(dependencies.len());
    for node in dependencies
        .iter()
        .filter_map(|dependency| native_nodes.get(dependency))
    {
        push_unique_cuda_graph_dependency(&mut nodes, node.clone());
    }
    nodes
}

pub fn push_unique_cuda_graph_dependency(dependencies: &mut Vec<GraphNode>, dependency: GraphNode) {
    if !dependencies.contains(&dependency) {
        dependencies.push(dependency);
    }
}
