use std::collections::HashMap;

use singe_cuda::error::Error as CudaError;

use crate::{
    backend::cuda::{
        compiled::{
            build::state::GraphLoweringState, executable::CudaCompiledGraphStats,
            plan::memory::GraphMemoryPlanner,
        },
        context::CudaExecutionContext,
        graph_binding::validate_graph_node_value_tensor,
        graph_value::CudaGraphValue,
    },
    error::{Error, Result},
    graph::ir::{Graph, ValueBindingKind, ValueId},
    tensor::Tensor,
};

pub fn try_add_cuda_graph_terminal_node(
    context: &CudaExecutionContext,
    id: ValueId,
    ir: &Graph,
    state: &mut GraphLoweringState,
) -> Result<bool> {
    match terminal_binding(ir, id)? {
        Some(GraphTerminalBinding::Input { value, is_static }) => {
            // Runtime inputs bound in place are read where they are, like static inputs.
            let in_place = state.private.inputs_in_place.get(&value).cloned();
            if is_static || in_place.is_some() {
                let tensor = match &in_place {
                    Some(tensor) => tensor,
                    None => ir
                        .required_initializer_tensor(value)
                        .map_err(|_| Error::missing_input("try add cuda graph terminal node"))?,
                };
                add_cuda_graph_static_input(
                    context,
                    id,
                    ir,
                    tensor,
                    &mut state.cache,
                    &mut state.values,
                    &mut state.stats,
                )?;
            } else {
                add_cuda_graph_input_placeholder(
                    context,
                    id,
                    ir,
                    &mut state.memory_planner,
                    &mut state.cache,
                    &mut state.values,
                    &mut state.stats,
                )?;
            }
            Ok(true)
        }
        Some(GraphTerminalBinding::Constant { value }) => {
            let tensor = ir.required_constant_tensor(value)?;
            add_cuda_graph_constant(
                context,
                id,
                ir,
                tensor,
                &mut state.cache,
                &mut state.values,
                &mut state.stats,
            )?;
            Ok(true)
        }
        None => Ok(false),
    }
}

pub enum GraphTerminalBinding {
    Input { value: ValueId, is_static: bool },
    Constant { value: ValueId },
}

pub fn terminal_binding(ir: &Graph, id: ValueId) -> Result<Option<GraphTerminalBinding>> {
    let value = id;
    match ir.required_value_binding_kind(value)? {
        ValueBindingKind::StaticInput => Ok(Some(GraphTerminalBinding::Input {
            value,
            is_static: true,
        })),
        ValueBindingKind::RuntimeInput | ValueBindingKind::InitializerInput => {
            Ok(Some(GraphTerminalBinding::Input {
                value,
                is_static: false,
            }))
        }
        ValueBindingKind::Constant => Ok(Some(GraphTerminalBinding::Constant { value })),
        ValueBindingKind::OperationOutput => Ok(None),
    }
}

pub fn add_cuda_graph_input_placeholder(
    context: &CudaExecutionContext,
    id: ValueId,
    ir: &Graph,
    memory_planner: &mut GraphMemoryPlanner,
    cache: &mut HashMap<ValueId, CudaGraphValue>,
    values: &mut HashMap<ValueId, CudaGraphValue>,
    stats: &mut CudaCompiledGraphStats,
) -> Result<()> {
    let value = id;
    let layout = ir.required_value_layout(value)?;
    let data_type = ir.required_value_data_type(value)?;
    let elements = layout.storage_span_element_count()?;
    let byte_size = elements
        .checked_mul(data_type.size_bytes())
        .ok_or_else(|| Error::out_of_range("cuda graph input byte size"))?;
    let (placeholder, fresh) =
        memory_planner.buffer(context.allocator(), data_type, layout, elements)?;
    validate_graph_node_value_tensor(ir, id, &placeholder)?;
    cache.insert(id, placeholder.clone());
    values.insert(id, placeholder);
    stats.record_input_node(if fresh { byte_size } else { 0 })?;
    Ok(())
}

fn bind_tensor_value(
    context: &CudaExecutionContext,
    id: ValueId,
    ir: &Graph,
    tensor: &Tensor,
    cache: &mut HashMap<ValueId, CudaGraphValue>,
    values: &mut HashMap<ValueId, CudaGraphValue>,
) -> Result<()> {
    validate_graph_node_value_tensor(ir, id, tensor)?;
    let value = CudaGraphValue::from_tensor(tensor.clone())?;
    if value.region()?.region().context() != context.runtime().context() {
        return Err(CudaError::MemoryContextMismatch.into());
    }

    cache.insert(id, value.clone());
    values.insert(id, value);
    Ok(())
}

pub fn add_cuda_graph_constant(
    context: &CudaExecutionContext,
    id: ValueId,
    ir: &Graph,
    tensor: &Tensor,
    cache: &mut HashMap<ValueId, CudaGraphValue>,
    values: &mut HashMap<ValueId, CudaGraphValue>,
    stats: &mut CudaCompiledGraphStats,
) -> Result<()> {
    bind_tensor_value(context, id, ir, tensor, cache, values)?;
    stats.record_constant_node();
    Ok(())
}

pub fn add_cuda_graph_static_input(
    context: &CudaExecutionContext,
    id: ValueId,
    ir: &Graph,
    tensor: &Tensor,
    cache: &mut HashMap<ValueId, CudaGraphValue>,
    values: &mut HashMap<ValueId, CudaGraphValue>,
    stats: &mut CudaCompiledGraphStats,
) -> Result<()> {
    bind_tensor_value(context, id, ir, tensor, cache, values)?;
    stats.record_static_input_node();
    Ok(())
}
