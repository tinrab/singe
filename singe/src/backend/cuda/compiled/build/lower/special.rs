use singe_cuda::graph::GraphNode;

use crate::{
    backend::{
        cuda::{
            compiled::{
                build::{
                    graph_node::add_cuda_graph_device_copy_node, operation::copy_to_input,
                    realized::record_realized_cuda_lowering, state::GraphLoweringState,
                    zero_work::graph_allows_zero_work_without_native_node,
                },
                plan::lowering::{CudaGraphProvider, LoweringPlan},
            },
            graph_value::CudaGraphValue,
        },
        registry::OperationSupport,
        spec,
    },
    error::{Error, Result},
    graph::ir::{Graph, ValueId},
    shape::Shape,
};

pub fn try_lower_empty_reduction(
    ir: &Graph,
    id: ValueId,
    state: &mut GraphLoweringState,
) -> Result<bool> {
    let value = id;
    let operation = ir.required_operation_for_value(id)?;
    let Some(attributes) = operation.attributes().reduction() else {
        return Ok(false);
    };
    let Some(opcode) = operation.opcode() else {
        return Ok(false);
    };
    let [input] = operation.inputs() else {
        return Err(Error::missing_input("try lower empty reduction"));
    };
    let problem = spec::reduction_problem(
        ir.required_value_shape(*input)?,
        &attributes.dimensions,
        attributes.keepdim,
    )?;
    let Some(fill) =
        spec::empty_reduction_fill(ir.required_value_shape(*input)?, &problem, opcode)?
    else {
        return Ok(false);
    };
    let output = fill.full(
        ir.required_value(value)?.device(),
        ir.required_value_data_type(value)?,
        &problem.output_shape,
    )?;
    let output = CudaGraphValue::from_tensor(output)?;
    state.cache.insert(id, output.clone());
    state.values.insert(id, output);
    state.stats.record_constant_node();
    Ok(true)
}

pub fn try_lower_device_copy(
    ir: &Graph,
    id: ValueId,
    output: &CudaGraphValue,
    dependencies: &[GraphNode],
    state: &mut GraphLoweringState,
) -> Result<bool> {
    let Some(input) = copy_to_input(ir, id)? else {
        return Ok(false);
    };
    let graph_node = add_cuda_graph_device_copy_node(
        ir,
        input,
        output,
        &state.cache,
        dependencies,
        &mut state.native_graph,
        &mut state.stats,
    )?;
    record_realized_cuda_lowering(
        &mut state.realized_lowering,
        id,
        ir,
        CudaGraphProvider::Alias,
    )?;
    state.cache.insert(id, output.clone());
    state.values.insert(id, output.clone());
    state.native_nodes.insert(id, graph_node);
    Ok(true)
}

pub fn try_lower_zero_work(
    ir: &Graph,
    id: ValueId,
    output_shape: &Shape,
    output: &CudaGraphValue,
    operation_support: Option<OperationSupport>,
    lowering_plan: &LoweringPlan,
    state: &mut GraphLoweringState,
) -> Result<bool> {
    if output_shape.element_count() != 0
        || !operation_support.is_some_and(graph_allows_zero_work_without_native_node)
    {
        return Ok(false);
    }

    if let Some(candidate) = lowering_plan.candidates(id).first() {
        record_realized_cuda_lowering(&mut state.realized_lowering, id, ir, candidate.provider())?;
    }

    state.cache.insert(id, output.clone());
    state.values.insert(id, output.clone());
    Ok(true)
}
