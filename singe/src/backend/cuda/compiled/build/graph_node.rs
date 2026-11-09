use std::collections::HashMap;

use singe_cuda::{
    graph::{
        Graph as NativeCudaGraph, GraphNode, MemorySetNodeParams, raw::MemoryCopy1DNodeParams,
    },
    memory::MemoryCopyKind,
};

use crate::{
    backend::cuda::{
        compiled::{build::state::GraphLoweringState, executable::CudaCompiledGraphStats},
        graph_binding::{push_unique_cuda_graph_dependency, validate_graph_input_value_tensor},
        graph_value::CudaGraphValue,
        utility::device_ptr,
    },
    error::{Error, Result},
    graph::ir::{Graph, ValueId},
    utility::to_u32,
};

pub fn add_cuda_graph_device_copy_node(
    ir: &Graph,
    input: ValueId,
    output: &CudaGraphValue,
    cache: &HashMap<ValueId, CudaGraphValue>,
    dependencies: &[GraphNode],
    native_graph: &mut NativeCudaGraph,
    stats: &mut CudaCompiledGraphStats,
) -> Result<GraphNode> {
    let input_tensor = cache
        .get(&input)
        .ok_or_else(|| Error::missing_input("add cuda graph device copy node"))?;
    validate_graph_input_value_tensor(ir, input, input_tensor)?;
    if !input_tensor.is_contiguous() || !output.is_contiguous() {
        return Err(Error::ContiguousRequired);
    }

    let byte_size = output.byte_size()?;
    let input_byte_size = input_tensor.byte_size()?;
    if byte_size != input_byte_size {
        return Err(Error::length_mismatch(
            "cuda graph copy_to byte size",
            byte_size,
            input_byte_size,
        ));
    }

    let output_region = output.region()?.region();
    let input_region = input_tensor.region()?.region();
    // SAFETY: the regions belong to tensors that the compiled graph retains for every replay.
    let params = unsafe {
        MemoryCopy1DNodeParams::new(
            output_region.as_ptr().cast(),
            input_region.as_ptr().cast_const().cast(),
            byte_size,
            MemoryCopyKind::DeviceToDevice,
        )
    };
    // SAFETY: the regions belong to tensors that the compiled graph retains for every replay.
    let graph_node = unsafe { native_graph.add_memory_copy_node_1d(dependencies, &params)? };
    stats.record_memcpy_node();
    Ok(graph_node)
}

pub fn add_cuda_graph_memset_node(
    output: &CudaGraphValue,
    element_byte_width: usize,
    element_count: usize,
    dependencies: &[GraphNode],
    native_graph: &mut NativeCudaGraph,
    stats: &mut CudaCompiledGraphStats,
) -> Result<GraphNode> {
    let output_region = output.region()?.region();
    // Memset nodes write 1-, 2-, or 4-byte words, and zeroed elements of any width are zeroed words.
    let bytes = element_byte_width
        .checked_mul(element_count)
        .ok_or_else(|| Error::out_of_range("cuda graph memset byte count"))?;
    let word = [4, 2, 1]
        .into_iter()
        .find(|word| bytes % word == 0)
        .unwrap_or(1);
    // SAFETY: the regions belong to tensors that the compiled graph retains for every replay.
    let params = unsafe {
        MemorySetNodeParams::new(
            device_ptr(output_region.as_ptr()),
            to_u32(word, "cuda graph memset element byte width")?,
            bytes / word,
        )
    };
    // SAFETY: the regions belong to tensors that the compiled graph retains for every replay.
    let graph_node = unsafe { native_graph.add_memory_set_node(dependencies, &params)? };
    stats.record_memset_node();
    Ok(graph_node)
}

// Copies the values a graph writes into inputs it reads in place as its last nodes, after every other node, so replay leaves those inputs updated.
// Other written inputs are not known until replay, and the prepared executor copies them afterwards.
pub fn add_input_write_nodes(ir: &Graph, state: &mut GraphLoweringState) -> Result<()> {
    let names = ir.input_writes();
    let outputs = ir.output_values_vec();
    if names.is_empty() || state.native_nodes.is_empty() || !ir.counted_outputs().is_empty() {
        return Ok(());
    }

    let visible = outputs
        .len()
        .checked_sub(names.len())
        .ok_or_else(|| Error::missing_output("add input write nodes"))?;
    let bound = ir
        .input_values()
        .filter(|(id, _)| state.private.inputs_in_place.contains_key(id))
        .map(|(id, name)| (name.to_owned(), id))
        .collect::<HashMap<_, _>>();
    let mut dependencies = Vec::new();
    for node in state.native_nodes.values() {
        push_unique_cuda_graph_dependency(&mut dependencies, node.clone());
    }

    for (position, (name, &source)) in names.iter().zip(&outputs[visible..]).enumerate() {
        let Some(target) = bound.get(name).and_then(|id| state.values.get(id)).cloned() else {
            continue;
        };
        let Some(value) = state.values.get(&source) else {
            continue;
        };
        // A scatter already wrote this input where it is.
        if value.shares_storage(&target) && value.layout() == target.layout() {
            state.private.inputs_written_in_graph.insert(name.clone());
            state
                .private
                .outputs_written_in_graph
                .insert(visible + position);
            continue;
        }

        if value.shares_storage(&target)
            || !value.is_contiguous()
            || !target.is_contiguous()
            || value.data_type() != target.data_type()
            || value.shape() != target.shape()
        {
            continue;
        }

        let node = add_cuda_graph_device_copy_node(
            ir,
            source,
            &target,
            &state.cache,
            &dependencies,
            &mut state.native_graph,
            &mut state.stats,
        )?;
        // Several written inputs copy in output order, as the executor's copies do.
        dependencies = vec![node];
        state.private.inputs_written_in_graph.insert(name.clone());
        state
            .private
            .outputs_written_in_graph
            .insert(visible + position);
    }
    Ok(())
}
