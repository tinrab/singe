use std::{collections::HashMap, hash::RandomState, sync::Arc};

use singe_cuda::{
    graph::Graph as NativeCudaGraph,
    stream::{StreamCaptureMode, StreamFlags},
};

use crate::{
    backend::cuda::{
        CudaStream,
        compiled::plan::cublas::launch::launch_cublas_matmul,
        graph_binding::{validate_graph_matmul, validate_graph_node_value_tensor},
        graph_value::CudaGraphValue,
        runtime::CudaRuntime,
    },
    error::{Error, Result},
    graph::ir::{Graph, ValueId},
};

/// Captures cuBLAS matrix multiplication into a child CUDA graph.
/// The compiled graph retains input, output, and any workspace allocations through replay.
///
/// # Errors
///
/// Returns an error if operands or result metadata are missing or incompatible, tensor layouts or types are unsupported, required algorithm selection fails, or stream creation or captured native execution fails.
pub fn build_cublas_matmul_graph(
    runtime: &Arc<CudaRuntime>,
    ir: &Graph,
    id: ValueId,
    inputs: [ValueId; 2],
    output: &CudaGraphValue,
    cache: &HashMap<ValueId, CudaGraphValue, RandomState>,
) -> Result<NativeCudaGraph> {
    validate_graph_matmul(ir, id, inputs)?;

    let a = cache
        .get(&inputs[0])
        .ok_or_else(|| Error::missing_input("build cublas matmul graph"))?;
    let b = cache
        .get(&inputs[1])
        .ok_or_else(|| Error::missing_input("build cublas matmul graph"))?;
    validate_graph_node_value_tensor(ir, inputs[0], a)?;
    validate_graph_node_value_tensor(ir, inputs[1], b)?;
    validate_graph_node_value_tensor(ir, id, output)?;

    let output_value = id;
    // cuBLAS kernel selection can consult the legacy default stream.
    // A nonblocking capture stream avoids implicit dependencies on that stream.
    let stream = CudaStream::create_with_flags(
        ir.required_value(output_value)?.device(),
        StreamFlags::NON_BLOCKING,
    )?;
    let (graph, ()) = stream.capture(StreamCaptureMode::ThreadLocal, || {
        launch_cublas_matmul(runtime, a, b, output)
    })?;
    Ok(graph.into_native())
}
