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
#[expect(
    clippy::too_many_lines,
    reason = "the exhaustive shape-expression traversal keeps the fingerprint encoding visible together"
)]
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
        if let Some(value) = graph.shape_program.values.get(&id) {
            let mut sources = Vec::new();
            for dim in value.elements() {
                dim.referenced_values(&mut sources);
            }

            pending.extend(sources.into_iter().map(ValueId::new));
        } else {
            pending.extend(node.inputs());
        }
        if let Some(expression) = graph.shape_program.reshapes.get(&id) {
            let mut values = Vec::new();
            for dim in &expression.dimensions {
                dim.referenced_values(&mut values);
            }

            pending.extend(values.into_iter().map(ValueId::new));
        }
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
    order.len().hash(hasher);
    for id in order {
        let node = &graph.required_node(id)?;
        node.output_data_type().hash(hasher);
        if let Some(value) = graph.shape_program.values.get(&id) {
            "shape value".hash(hasher);
            value.shape()?.dimensions().hash(hasher);
            for dim in value.elements() {
                dim.hash_with(hasher, &|index| {
                    values
                        .get(&ValueId::new(index))
                        .copied()
                        .ok_or_else(|| Error::missing_node("hash shape program"))
                })?;
            }
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

            graph.shape_program.reshapes.contains_key(&id).hash(hasher);
            if let Some(expression) = graph.shape_program.reshapes.get(&id) {
                expression.allow_zero.hash(hasher);
                expression.dimensions.len().hash(hasher);
                for dim in &expression.dimensions {
                    dim.hash_with(hasher, &|index| {
                        values
                            .get(&ValueId::new(index))
                            .copied()
                            .ok_or_else(|| Error::missing_node("hash shape program"))
                    })?;
                }
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
