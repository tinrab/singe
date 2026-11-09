//! A library matmul and the residual add after it as one cuBLASLt call, which reads the residual as its `C` with a `beta` of one.

use std::collections::HashMap;

use crate::{
    backend::{
        cuda::{
            compiled::{
                build::{
                    fusion::{GraphFusionCompletion, GraphFusionLowering},
                    lower::GraphOperationLoweringRequest,
                },
                plan::cublas::build_cublaslt_matmul_residual_graph,
            },
            graph_value::CudaGraphValue,
        },
        registry::Operation,
    },
    error::{Error, Result},
    graph::ir::ValueId,
    layout::Layout,
};

/// Returns the add that alone reads matmul `id`'s product and the residual it adds, when the residual has the product's data type and shape, is computed, and is laid out contiguously, as cuBLASLt's `C` must be.
///
/// # Errors
///
/// Returns an error if the graph's metadata is missing.
pub fn residual_add(
    request: &GraphOperationLoweringRequest<'_>,
    cache: &HashMap<ValueId, CudaGraphValue>,
) -> Result<Option<(ValueId, ValueId)>> {
    let ir = request.ir;
    let id = request.id;
    let (Some(&[add]), false) = (
        request.users.get(&id).map(Vec::as_slice),
        request.graph_outputs.contains(&id),
    ) else {
        return Ok(None);
    };
    let operation = ir.required_operation_for_value(add)?;
    let (Some(Operation::Add), &[a, b]) = (operation.opcode(), operation.inputs()) else {
        return Ok(None);
    };
    let residual = if a == id { b } else { a };
    let Some(value) = cache.get(&residual) else {
        return Ok(None);
    };
    let shape = ir.required_value_shape(id)?;
    let fits = residual != id
        && ir.required_value_data_type(residual)? == ir.required_value_data_type(id)?
        && ir.required_value_data_type(add)? == ir.required_value_data_type(id)?
        && ir.required_value_shape(add)? == shape
        && value.layout() == &Layout::contiguous(shape.dimensions())?;
    Ok(fits.then_some((add, residual)))
}

/// Lowers matmul `request.id` with inputs `inputs` and the residual add `add` of `residual` as one cuBLASLt call, and returns `false` without changing anything when cuBLASLt cannot run it.
///
/// # Errors
///
/// Returns an error if the output cannot be allocated or the call's node cannot be added.
pub fn try_cublaslt_matmul_residual_fusion(
    request: &GraphOperationLoweringRequest<'_>,
    inputs: [ValueId; 2],
    (add, residual): (ValueId, ValueId),
    lowering: &mut GraphFusionLowering<'_>,
) -> Result<bool> {
    let add_index = *request
        .positions
        .get(&add)
        .ok_or_else(|| Error::missing_node("try cublaslt matmul residual fusion"))?;
    let allocation =
        lowering
            .memory_planner
            .allocate(lowering.context.allocator(), request.ir, add)?;
    let allocation_dependencies = allocation.dependencies();
    let output = allocation.into_value();
    let Ok(graph) = build_cublaslt_matmul_residual_graph(
        request.runtime,
        request.ir,
        request.id,
        inputs,
        add,
        residual,
        &output,
        lowering.cache,
    ) else {
        return Ok(false);
    };
    let dependency_ids = [inputs[0], inputs[1], residual];
    lowering.finish_cublaslt_fusion(
        GraphFusionCompletion {
            ir: request.ir,
            dependency_ids: &dependency_ids,
            allocation_dependencies,
            output_id: add,
            output,
            realized_nodes: &[request.id, add],
            skipped_nodes: &[add],
            release_nodes: &[(request.id, request.index), (add, add_index)],
        },
        graph,
    )?;
    Ok(true)
}
