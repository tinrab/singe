use std::collections::{HashMap, HashSet};

use crate::{
    backend::{
        cpu::interpreted,
        registry::{AliasBehavior, LayoutSupport, Operation},
    },
    error::Result,
    graph::{
        Graph, GraphOutput,
        ir::ValueId,
        node::{Node, NodeKind, NodeRef},
        passes::remap::{remap_node_id, remapped_node},
    },
    layout::Layout,
};

// Returns `graph` with a contiguous copy of each strided operand of an operation whose kernels read contiguous operands, and the map from old values to new ones, which is empty when no operand is strided.
// Eager kernels copy such operands themselves; a compiled graph runs the copy as a node of its own.
pub fn materialize_contiguous_operands(graph: &Graph) -> Result<(Graph, Vec<(ValueId, ValueId)>)> {
    let mut id_map = HashMap::with_capacity(graph.node_count());
    let mut nodes = Vec::with_capacity(graph.node_count());
    let mut changed = false;
    for id in graph.node_ids() {
        let node = graph.required_node(id)?;
        let mut copies = HashMap::new();
        if reads_contiguous(&node) {
            for &input in node.inputs() {
                let operand = graph.required_node(input)?;
                if copies.contains_key(&input) || !strided(operand.output_layout()) {
                    continue;
                }

                let copy = Node::create_with_shape(
                    NodeKind::contiguous(remap_node_id(input, &id_map)?),
                    None,
                    operand.output_data_type(),
                    operand.output_shape().clone(),
                    operand.output_device(),
                );
                copies.insert(input, push(&mut nodes, copy));
            }
        }

        changed |= !copies.is_empty();
        let node = node.map_inputs(|input| match copies.get(&input) {
            Some(&copy) => Ok(copy),
            None => remap_node_id(input, &id_map),
        })?;
        id_map.insert(id, push(&mut nodes, node));
    }

    if !changed {
        return Ok((graph.clone(), Vec::new()));
    }

    let remapped = graph.remapped_node_pairs(&id_map)?;
    Ok((Graph::from_retained_nodes(graph, &id_map, nodes)?, remapped))
}

fn reads_contiguous(node: &NodeRef<'_>) -> bool {
    node.schema()
        .is_some_and(|schema| !matches!(schema.support().layout(), LayoutSupport::AnyStrided))
}

fn strided(layout: &Layout) -> bool {
    !layout.is_contiguous() || layout.storage_offset() != 0
}

fn push(nodes: &mut Vec<Node>, node: Node) -> ValueId {
    nodes.push(node);
    ValueId::new(nodes.len() - 1)
}

// Returns `graph` with each explicit `contiguous` copy removed whose readers read strided operands with the same generated kernels, and the map from old values to new ones.
pub fn elide_contiguous_copies(
    graph: &Graph,
    for_cuda: bool,
) -> Result<(Graph, Vec<(ValueId, ValueId)>)> {
    let mut readers = HashMap::<ValueId, Vec<ValueId>>::new();
    for id in graph.node_ids() {
        for &input in graph.required_node(id)?.inputs() {
            readers.entry(input).or_default().push(id);
        }
    }

    let outputs = graph
        .output_projections()
        .iter()
        .map(GraphOutput::node)
        .collect::<HashSet<_>>();
    let mut id_map = HashMap::with_capacity(graph.node_count());
    let mut nodes = Vec::with_capacity(graph.node_count());
    let mut changed = false;
    for id in graph.node_ids() {
        let node = graph.required_node(id)?;
        let elided =
            if !outputs.contains(&id) && elidable(graph, &node, readers.get(&id), for_cuda)? {
                node.inputs().first().copied()
            } else {
                None
            };
        let new_id = if let Some(input) = elided {
            changed = true;
            remap_node_id(input, &id_map)?
        } else {
            push(&mut nodes, remapped_node(&node, &id_map)?)
        };
        id_map.insert(id, new_id);
    }

    if !changed {
        return Ok((graph.clone(), Vec::new()));
    }

    let remapped = graph.remapped_node_pairs(&id_map)?;
    Ok((Graph::from_retained_nodes(graph, &id_map, nodes)?, remapped))
}

fn elidable(
    graph: &Graph,
    node: &NodeRef<'_>,
    readers: Option<&Vec<ValueId>>,
    for_cuda: bool,
) -> Result<bool> {
    if node.operation() != Some(Operation::Contiguous)
        || !(for_cuda || node.output_device().is_cuda())
    {
        return Ok(false);
    }

    let Some(readers) = readers else {
        return Ok(false);
    };
    for &reader in readers {
        let reader = graph.required_node(reader)?;
        let Some(schema) = reader.schema() else {
            return Ok(false);
        };
        let Some(operation) = reader.operation() else {
            return Ok(false);
        };
        // Generated element-wise, reduction, and row kernels read any strides, while views need the copy's layout and matmuls would change provider.
        if !interpreted::interprets(operation)
            || schema.alias_behavior() != AliasBehavior::None
            || !matches!(schema.support().layout(), LayoutSupport::AnyStrided)
        {
            return Ok(false);
        }
    }
    Ok(true)
}
