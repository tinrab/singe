use std::{collections::HashMap, hash::RandomState, sync::Arc};

use singe_cuda::graph::Graph as NativeCudaGraph;

use crate::{
    backend::cuda::{
        CudaRuntime,
        compiled::plan::cudnn::{
            CudnnMatmulAddPattern, CudnnMatmulAddReluPattern, allocate_cudnn_workspace,
            cudnn_tensor_metadata_for_value, validate_cudnn_same_shape_unary,
        },
        cudnn_backend::{
            CUDNN_GRAPH_BIAS_TENSOR_ID, CUDNN_GRAPH_INPUT0_TENSOR_ID, CUDNN_GRAPH_INPUT1_TENSOR_ID,
            CUDNN_GRAPH_OUTPUT_TENSOR_ID, CudnnBackendExecutable, CudnnBackendGraphKey,
            CudnnBackendGraphKind, CudnnGraphBinding, CudnnGraphBindingSource,
            build_cudnn_matmul_add_relu_template, build_cudnn_matmul_add_template,
            cached_cudnn_backend_graph_template, cudnn_backend_template_workspace_size,
            cudnn_backend_tensor_metadata_key,
        },
        graph_binding::validate_graph_node_value_tensor,
        graph_value::CudaGraphValue,
        validate_graph_matmul, validate_graph_matmul_add,
    },
    error::{Error, Result},
    graph::ir::{Graph, ValueId},
    utility::Locked,
};

/// Builds a child CUDA graph and cuDNN executable for matrix multiplication followed by bias addition.
/// The returned executable retains the plan and workspace; graph values retain the bound tensor allocations through replay.
///
/// # Errors
///
/// Returns an error if required operation metadata or cached operands are missing or incompatible, tensor types or layouts are unsupported, cache or library mutexes are poisoned, or native plan creation, workspace allocation, tensor binding, or graph population fails.
pub fn build_cudnn_matmul_add_graph(
    runtime: &Arc<CudaRuntime>,
    ir: &Graph,
    id: ValueId,
    inputs: [ValueId; 2],
    pattern: CudnnMatmulAddPattern,
    output: &CudaGraphValue,
    cache: &HashMap<ValueId, CudaGraphValue, RandomState>,
) -> Result<(NativeCudaGraph, CudnnBackendExecutable)> {
    validate_graph_matmul(ir, id, inputs)?;
    validate_graph_matmul_add(ir, id, pattern.add(), pattern.bias())?;
    let matmul_value = id;
    let a_value = inputs[0];
    let b_value = inputs[1];
    let bias_value = pattern.bias();
    let add_value = pattern.add();
    let matmul_metadata = cudnn_tensor_metadata_for_value(ir, matmul_value)?;
    let a_metadata = cudnn_tensor_metadata_for_value(ir, a_value)?;
    let b_metadata = cudnn_tensor_metadata_for_value(ir, b_value)?;
    let bias_metadata = cudnn_tensor_metadata_for_value(ir, bias_value)?;
    let add_metadata = cudnn_tensor_metadata_for_value(ir, add_value)?;
    let a_shape = ir.required_value_shape(a_value)?;
    let a_layout = ir.required_value_layout(a_value)?;
    let b_shape = ir.required_value_shape(b_value)?;
    let b_layout = ir.required_value_layout(b_value)?;
    let bias_shape = ir.required_value_shape(bias_value)?;
    let bias_layout = ir.required_value_layout(bias_value)?;
    let add_shape = ir.required_value_shape(add_value)?;
    let add_layout = ir.required_value_layout(add_value)?;

    let a = cache
        .get(&inputs[0])
        .ok_or_else(|| Error::missing_input("build cudnn matmul add graph"))?;
    let b = cache
        .get(&inputs[1])
        .ok_or_else(|| Error::missing_input("build cudnn matmul add graph"))?;
    let bias = cache
        .get(&pattern.bias())
        .ok_or_else(|| Error::missing_input("build cudnn matmul add graph"))?;
    validate_graph_node_value_tensor(ir, inputs[0], a)?;
    validate_graph_node_value_tensor(ir, inputs[1], b)?;
    validate_graph_node_value_tensor(ir, pattern.bias(), bias)?;
    validate_graph_node_value_tensor(ir, pattern.add(), output)?;

    let data_type = add_metadata.data_type();
    let key = CudnnBackendGraphKey::new(
        CudnnBackendGraphKind::MatmulAdd,
        vec![],
        vec![
            cudnn_backend_tensor_metadata_key(data_type, a_shape, a_layout),
            cudnn_backend_tensor_metadata_key(data_type, b_shape, b_layout),
            cudnn_backend_tensor_metadata_key(data_type, bias_shape, bias_layout),
        ],
        cudnn_backend_tensor_metadata_key(data_type, add_shape, add_layout),
    );
    let template = cached_cudnn_backend_graph_template(runtime, key, |cudnn| {
        build_cudnn_matmul_add_template(
            cudnn,
            &a_metadata,
            &b_metadata,
            &bias_metadata,
            &matmul_metadata,
            &add_metadata,
        )
    })?;
    let workspace_size = cudnn_backend_template_workspace_size(&template)?;
    let a_id = CUDNN_GRAPH_INPUT0_TENSOR_ID;
    let b_id = CUDNN_GRAPH_INPUT1_TENSOR_ID;
    let bias_id = CUDNN_GRAPH_BIAS_TENSOR_ID;
    let output_id = CUDNN_GRAPH_OUTPUT_TENSOR_ID;
    let mut executable = CudnnBackendExecutable::new(
        template,
        allocate_cudnn_workspace(runtime.context(), workspace_size)?,
        vec![
            CudnnGraphBinding::new(a_id, CudnnGraphBindingSource::Node(inputs[0])),
            CudnnGraphBinding::new(b_id, CudnnGraphBindingSource::Node(inputs[1])),
            CudnnGraphBinding::new(bias_id, CudnnGraphBindingSource::Node(pattern.bias())),
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

/// Builds a child CUDA graph and cuDNN executable for matrix multiplication, bias addition, and `Relu`.
/// The returned executable retains the plan and workspace; graph values retain the bound tensor allocations through replay.
///
/// # Errors
///
/// Returns an error if required operation metadata or cached operands are missing or incompatible, tensor types or layouts are unsupported, cache or library mutexes are poisoned, or native plan creation, workspace allocation, tensor binding, or graph population fails.
pub fn build_cudnn_matmul_add_relu_graph(
    runtime: &Arc<CudaRuntime>,
    ir: &Graph,
    id: ValueId,
    inputs: [ValueId; 2],
    pattern: CudnnMatmulAddReluPattern,
    output: &CudaGraphValue,
    cache: &HashMap<ValueId, CudaGraphValue, RandomState>,
) -> Result<(NativeCudaGraph, CudnnBackendExecutable)> {
    validate_graph_matmul(ir, id, inputs)?;
    validate_graph_matmul_add(ir, id, pattern.add(), pattern.bias())?;
    validate_cudnn_same_shape_unary(
        ir,
        pattern.relu(),
        pattern.add(),
        "cudnn matmul add relu output shape",
    )?;
    let matmul_value = id;
    let a_value = inputs[0];
    let b_value = inputs[1];
    let bias_value = pattern.bias();
    let add_value = pattern.add();
    let relu_value = pattern.relu();
    let matmul_metadata = cudnn_tensor_metadata_for_value(ir, matmul_value)?;
    let a_metadata = cudnn_tensor_metadata_for_value(ir, a_value)?;
    let b_metadata = cudnn_tensor_metadata_for_value(ir, b_value)?;
    let bias_metadata = cudnn_tensor_metadata_for_value(ir, bias_value)?;
    let add_metadata = cudnn_tensor_metadata_for_value(ir, add_value)?;
    let relu_metadata = cudnn_tensor_metadata_for_value(ir, relu_value)?;
    let a_shape = ir.required_value_shape(a_value)?;
    let a_layout = ir.required_value_layout(a_value)?;
    let b_shape = ir.required_value_shape(b_value)?;
    let b_layout = ir.required_value_layout(b_value)?;
    let bias_shape = ir.required_value_shape(bias_value)?;
    let bias_layout = ir.required_value_layout(bias_value)?;
    let relu_shape = ir.required_value_shape(relu_value)?;
    let relu_layout = ir.required_value_layout(relu_value)?;

    let a = cache
        .get(&inputs[0])
        .ok_or_else(|| Error::missing_input("build cudnn matmul add relu graph"))?;
    let b = cache
        .get(&inputs[1])
        .ok_or_else(|| Error::missing_input("build cudnn matmul add relu graph"))?;
    let bias = cache
        .get(&pattern.bias())
        .ok_or_else(|| Error::missing_input("build cudnn matmul add relu graph"))?;
    validate_graph_node_value_tensor(ir, inputs[0], a)?;
    validate_graph_node_value_tensor(ir, inputs[1], b)?;
    validate_graph_node_value_tensor(ir, pattern.bias(), bias)?;
    validate_graph_node_value_tensor(ir, pattern.relu(), output)?;

    let data_type = relu_metadata.data_type();
    let key = CudnnBackendGraphKey::new(
        CudnnBackendGraphKind::MatmulAddRelu,
        vec![],
        vec![
            cudnn_backend_tensor_metadata_key(data_type, a_shape, a_layout),
            cudnn_backend_tensor_metadata_key(data_type, b_shape, b_layout),
            cudnn_backend_tensor_metadata_key(data_type, bias_shape, bias_layout),
        ],
        cudnn_backend_tensor_metadata_key(data_type, relu_shape, relu_layout),
    );
    let template = cached_cudnn_backend_graph_template(runtime, key, |cudnn| {
        build_cudnn_matmul_add_relu_template(
            cudnn,
            &a_metadata,
            &b_metadata,
            &bias_metadata,
            &matmul_metadata,
            &add_metadata,
            &relu_metadata,
        )
    })?;
    let workspace_size = cudnn_backend_template_workspace_size(&template)?;
    let a_id = CUDNN_GRAPH_INPUT0_TENSOR_ID;
    let b_id = CUDNN_GRAPH_INPUT1_TENSOR_ID;
    let bias_id = CUDNN_GRAPH_BIAS_TENSOR_ID;
    let output_id = CUDNN_GRAPH_OUTPUT_TENSOR_ID;
    let mut executable = CudnnBackendExecutable::new(
        template,
        allocate_cudnn_workspace(runtime.context(), workspace_size)?,
        vec![
            CudnnGraphBinding::new(a_id, CudnnGraphBindingSource::Node(inputs[0])),
            CudnnGraphBinding::new(b_id, CudnnGraphBindingSource::Node(inputs[1])),
            CudnnGraphBinding::new(bias_id, CudnnGraphBindingSource::Node(pattern.bias())),
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
