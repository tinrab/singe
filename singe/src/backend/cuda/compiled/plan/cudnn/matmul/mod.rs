mod add;

use std::{collections::HashMap, hash::RandomState, sync::Arc};

use singe_cuda::graph::Graph as NativeCudaGraph;
use singe_cudnn::backend::execution::EngineChoice;

pub use self::add::{build_cudnn_matmul_add_graph, build_cudnn_matmul_add_relu_graph};
use crate::{
    backend::cuda::{
        CudaRuntime,
        compiled::plan::cudnn::{allocate_cudnn_workspace, cudnn_tensor_metadata_for_value},
        cudnn_backend::{
            CUDNN_GRAPH_INPUT0_TENSOR_ID, CUDNN_GRAPH_INPUT1_TENSOR_ID,
            CUDNN_GRAPH_OUTPUT_TENSOR_ID, CudnnBackendExecutable, CudnnBackendGraphKey,
            CudnnBackendGraphKind, CudnnGraphBinding, CudnnGraphBindingSource,
            build_cudnn_matmul_template, cached_cudnn_backend_graph_template,
            cudnn_backend_template_workspace_size, cudnn_backend_tensor_metadata_key,
            cudnn_engine_key_parameters,
        },
        graph_binding::validate_graph_node_value_tensor,
        graph_value::CudaGraphValue,
        validate_graph_matmul,
    },
    error::{Error, Result},
    graph::ir::{Graph, ValueId},
    utility::Locked,
};

/// Builds a child CUDA graph and cuDNN executable for matrix multiplication.
/// The returned executable retains the plan and workspace; graph values retain the bound tensor allocations through replay.
///
/// # Errors
///
/// Returns an error if required operation metadata or cached operands are missing or incompatible, tensor types or layouts are unsupported, cache or library mutexes are poisoned, or native plan creation, workspace allocation, tensor binding, or graph population fails.
pub fn build_cudnn_matmul_graph(
    runtime: &Arc<CudaRuntime>,
    ir: &Graph,
    id: ValueId,
    inputs: [ValueId; 2],
    output: &CudaGraphValue,
    cache: &HashMap<ValueId, CudaGraphValue, RandomState>,
    engine: Option<&EngineChoice>,
) -> Result<(NativeCudaGraph, CudnnBackendExecutable)> {
    validate_graph_matmul(ir, id, inputs)?;
    let value = id;
    let a_value = inputs[0];
    let b_value = inputs[1];
    let value_metadata = cudnn_tensor_metadata_for_value(ir, value)?;
    let a_metadata = cudnn_tensor_metadata_for_value(ir, a_value)?;
    let b_metadata = cudnn_tensor_metadata_for_value(ir, b_value)?;
    let value_shape = ir.required_value_shape(value)?;
    let value_layout = ir.required_value_layout(value)?;
    let a_shape = ir.required_value_shape(a_value)?;
    let a_layout = ir.required_value_layout(a_value)?;
    let b_shape = ir.required_value_shape(b_value)?;
    let b_layout = ir.required_value_layout(b_value)?;

    let a = cache
        .get(&inputs[0])
        .ok_or_else(|| Error::missing_input("build cudnn matmul graph"))?;
    let b = cache
        .get(&inputs[1])
        .ok_or_else(|| Error::missing_input("build cudnn matmul graph"))?;
    validate_graph_node_value_tensor(ir, inputs[0], a)?;
    validate_graph_node_value_tensor(ir, inputs[1], b)?;
    validate_graph_node_value_tensor(ir, id, output)?;

    let data_type = value_metadata.data_type();
    let key = CudnnBackendGraphKey::new(
        CudnnBackendGraphKind::Matmul,
        cudnn_engine_key_parameters(engine),
        vec![
            cudnn_backend_tensor_metadata_key(data_type, a_shape, a_layout),
            cudnn_backend_tensor_metadata_key(data_type, b_shape, b_layout),
        ],
        cudnn_backend_tensor_metadata_key(data_type, value_shape, value_layout),
    );
    let template = cached_cudnn_backend_graph_template(runtime, key, |cudnn| {
        build_cudnn_matmul_template(cudnn, &a_metadata, &b_metadata, &value_metadata, engine)
    })?;
    let workspace_size = cudnn_backend_template_workspace_size(&template)?;
    let a_id = CUDNN_GRAPH_INPUT0_TENSOR_ID;
    let b_id = CUDNN_GRAPH_INPUT1_TENSOR_ID;
    let output_id = CUDNN_GRAPH_OUTPUT_TENSOR_ID;
    let mut executable = CudnnBackendExecutable::new(
        template,
        allocate_cudnn_workspace(runtime.context(), workspace_size)?,
        vec![
            CudnnGraphBinding::new(a_id, CudnnGraphBindingSource::Node(inputs[0])),
            CudnnGraphBinding::new(b_id, CudnnGraphBindingSource::Node(inputs[1])),
            CudnnGraphBinding::new(output_id, CudnnGraphBindingSource::Output),
        ],
    );
    let mut child_graph = runtime.context().create_graph()?;
    let variant_pack = executable.variant_pack(output, cache)?;
    let mut cudnn = runtime.cudnn().locked("cuda runtime cache")?;
    // SAFETY: compiler graph values retain tensor allocations, and the returned executable retains its workspace and plan through graph replay resources.
    unsafe { executable.populate_graph(&mut cudnn, &variant_pack, &mut child_graph) }?;
    drop(cudnn);
    Ok((child_graph, executable))
}
