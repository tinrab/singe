use std::collections::{HashMap, HashSet};

use crate::{
    backend::registry::{CheckedOperationRequest, OperationRequest},
    dispatch,
    error::{Error, Result},
    graph::{
        GraphRunStats,
        guards::GraphGuards,
        ir::{Graph, OperationId, ValueId},
        symbolic::SymbolicShapeEnv,
    },
    tensor::Tensor,
};

pub fn execute_ir(
    ir: &Graph,
    order: &[OperationId],
    requests: &HashMap<OperationId, CheckedOperationRequest>,
    release_after: &[Vec<ValueId>],
    symbols: &SymbolicShapeEnv,
    mut values: HashMap<ValueId, Tensor>,
) -> Result<(Vec<Tensor>, GraphRunStats)> {
    let mut stats = GraphRunStats {
        peak_live_values: values.len(),
        ..GraphRunStats::default()
    };
    for (position, &id) in order.iter().enumerate() {
        let operation = ir.required_operation(id)?;
        if !operation
            .result_values()
            .all(|value| values.contains_key(&value))
        {
            let results = if operation.opcode().is_none() {
                operation
                    .result_values()
                    .map(|value| ir.required_constant_tensor(value).cloned())
                    .collect::<Result<Vec<_>>>()?
            } else {
                let inputs = operation
                    .inputs()
                    .iter()
                    .map(|value| {
                        values
                            .get(value)
                            .cloned()
                            .ok_or_else(|| Error::missing_input("execute ir"))
                    })
                    .collect::<Result<Vec<_>>>()?;
                stats.operations_executed += 1;
                let prepared = requests
                    .get(&id)
                    .ok_or_else(|| Error::missing_node("execute ir"))?;
                let rebound = if inputs.iter().zip(prepared.inputs()).all(|(input, spec)| {
                    input.data_type() == spec.data_type() && input.shape() == spec.shape()
                }) {
                    None
                } else {
                    Some(
                        OperationRequest::new(
                            prepared.operation(),
                            prepared.attributes().clone(),
                            inputs.iter().map(Tensor::value_spec).collect(),
                        )
                        .check()?,
                    )
                };
                dispatch::execute(rebound.as_ref().unwrap_or(prepared), &inputs)
                    .map_err(|error| error.with_graph_source(operation.source_location()))?
            };
            if results.len() != operation.results().len() {
                return Err(Error::missing_output("execute ir"));
            }

            for (value, tensor) in operation.result_values().zip(results) {
                GraphGuards::check_tensor_for_ir_value_with_symbols(
                    ir,
                    value,
                    &tensor,
                    symbols.value_symbols(value),
                )?;
                values.insert(value, tensor);
            }

            stats.peak_live_values = stats.peak_live_values.max(values.len());
        }
        for value in &release_after[position] {
            if values.remove(value).is_some() {
                stats.released_values += 1;
            }
        }
    }

    let outputs = ir
        .output_values()
        .iter()
        .map(|value| {
            values
                .get(value)
                .cloned()
                .ok_or_else(|| Error::missing_output("execute ir"))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((outputs, stats))
}

/// Last-use releases are computed once for prepared execution.
/// Dropping a descriptor releases only this interpreter's ownership; aliases and pending submissions retain their own owners.
/// Every public output is kept alive.
pub fn value_releases(ir: &Graph, order: &[OperationId]) -> Result<Vec<Vec<ValueId>>> {
    let outputs = ir.output_values().iter().copied().collect::<HashSet<_>>();
    let mut last_use = HashMap::new();
    for (position, &id) in order.iter().enumerate() {
        let operation = ir.required_operation(id)?;
        for result in operation.result_values() {
            last_use.entry(result).or_insert(position);
        }
        for &input in operation.inputs() {
            last_use.insert(input, position);
        }
    }

    let mut release_after = vec![Vec::new(); order.len()];
    for (value, position) in last_use {
        if !outputs.contains(&value) {
            release_after[position].push(value);
        }
    }
    Ok(release_after)
}
