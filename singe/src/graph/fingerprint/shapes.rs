use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
};

use crate::{
    backend::registry::BitHash,
    error::{Error, Result},
    graph::{Graph, fingerprint::stable::StableHasher, ir::ValueId, outputs::GraphOutput},
};

/// Hashes the logical shape recipe rather than a representative specialization.
/// Intermediate shapes follow from the named inputs, operations, and expressions.
pub fn hash_shape_program(graph: &Graph, hasher: &mut StableHasher) -> Result<()> {
    let mut live = HashSet::new();
    let mut pending = graph
        .output_projections()
        .iter()
        .map(GraphOutput::node)
        .collect::<Vec<_>>();
    pending.extend(graph.shape_program.constraints.keys().copied());
    while let Some(id) = pending.pop() {
        if !live.insert(id) {
            continue;
        }

        let node = &graph.required_node(id)?;
        // A shape value reads only the axes its expressions name.
        if !graph.shape_program.values.contains_key(&id) {
            pending.extend(node.inputs());
        }

        pending.extend(
            graph
                .shape_program
                .references(id)
                .into_iter()
                .map(ValueId::new),
        );
    }

    let order = graph
        .topological_sort()?
        .into_iter()
        .filter(|id| live.contains(id))
        .collect::<Vec<_>>();
    let values = order
        .iter()
        .enumerate()
        .map(|(index, id)| (*id, index))
        .collect::<HashMap<_, _>>();
    let index = |index| {
        values
            .get(&ValueId::new(index))
            .copied()
            .ok_or_else(|| Error::missing_node("hash shape program"))
    };
    order.len().hash(hasher);
    for id in order {
        let node = &graph.required_node(id)?;
        node.output_data_type().hash(hasher);
        let attributes = graph.shape_program.attributes.get(&id);
        if let Some(value) = graph.shape_program.values.get(&id) {
            "shape value".hash(hasher);
            value.shape()?.dimensions().hash(hasher);
            for dim in value.elements() {
                dim.hash_with(hasher, &index)?;
            }
        } else if let Some(attributes) = attributes
            && node.constant_tensor(0).is_some()
        {
            "symbolic literal".hash(hasher);
            attributes.hash_with(hasher, &index)?;
        } else if let Some((name, ..)) = node.input_metadata() {
            "input".hash(hasher);
            name.hash(hasher);
        } else if let Some(tensor) = node.constant_tensor(0) {
            "literal".hash(hasher);
            tensor.shape().dimensions().hash(hasher);
            tensor.to_bytes()?.hash(hasher);
        } else {
            node.operation()
                .ok_or_else(|| Error::missing_node("hash shape program"))?
                .hash(hasher);
            node.inputs().len().hash(hasher);
            for input in node.inputs() {
                values
                    .get(input)
                    .ok_or_else(|| Error::missing_input("hash shape program"))?
                    .hash(hasher);
            }

            attributes.is_some().hash(hasher);
            if let Some(attributes) = attributes {
                attributes.hash_with(hasher, &index)?;
            } else {
                node.attributes().bit_hash(hasher);
            }
        }
    }

    graph.output_projections().len().hash(hasher);
    for output in graph.output_projections() {
        output.name().hash(hasher);
        values
            .get(&output.node())
            .ok_or_else(|| Error::missing_output("hash shape program"))?
            .hash(hasher);
        output.output_index().hash(hasher);
    }

    graph.shape_program.constraints.len().hash(hasher);
    for (id, constraints) in &graph.shape_program.constraints {
        values
            .get(id)
            .ok_or_else(|| Error::missing_node("hash shape program"))?
            .hash(hasher);
        constraints.len().hash(hasher);
        for constraint in constraints {
            constraint.dimensions.hash(hasher);
        }
    }
    Ok(())
}
