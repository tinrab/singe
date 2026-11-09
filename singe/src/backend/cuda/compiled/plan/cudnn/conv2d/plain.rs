use std::{collections::HashMap, hash::RandomState, sync::Arc};

use singe_cuda::graph::Graph as NativeCudaGraph;

use crate::{
    backend::{
        cuda::{
            CudaRuntime,
            compiled::plan::cudnn::{
                allocate_cudnn_workspace,
                conv2d::{
                    executable::populate_cudnn_conv2d_graph,
                    validation::{validate_cudnn_conv2d_bias, validate_cudnn_convolution_nodes},
                },
                cudnn_tensor_metadata_for_value,
            },
            cudnn_backend::{
                CUDNN_GRAPH_BIAS_TENSOR_ID, CUDNN_GRAPH_INPUT0_TENSOR_ID,
                CUDNN_GRAPH_INPUT1_TENSOR_ID, CUDNN_GRAPH_OUTPUT_TENSOR_ID, CudnnBackendExecutable,
                CudnnBackendGraphKey, CudnnBackendGraphKind, CudnnGraphBinding,
                CudnnGraphBindingSource, build_cudnn_convolution_template,
                cached_cudnn_backend_graph_template, cudnn_backend_template_workspace_size,
                cudnn_backend_tensor_metadata_key, cudnn_convolution_geometry,
            },
            graph_binding::validate_graph_node_value_tensor,
            graph_value::CudaGraphValue,
        },
        spec::ConvolutionWindow,
    },
    error::{Error, Result},
    graph::ir::{Graph, ValueId},
};

/// Builds a child CUDA graph and cuDNN executable for convolution with an optional bias.
/// The returned executable retains the plan and workspace; graph values retain the bound tensor allocations through replay.
///
/// # Errors
///
/// Returns an error if required operation metadata or cached operands are missing or incompatible, tensor types or layouts are unsupported, cache or library mutexes are poisoned, or native plan creation, workspace allocation, tensor binding, or graph population fails.
pub fn build_cudnn_convolution_graph(
    runtime: &Arc<CudaRuntime>,
    ir: &Graph,
    id: ValueId,
    input: ValueId,
    weight: ValueId,
    bias: Option<ValueId>,
    window: &ConvolutionWindow,
    output: &CudaGraphValue,
    cache: &HashMap<ValueId, CudaGraphValue, RandomState>,
) -> Result<(NativeCudaGraph, CudnnBackendExecutable)> {
    validate_cudnn_convolution_nodes(ir, id, input, weight, window)?;
    let value = id;
    let input_value = input;
    let weight_value = weight;
    let value_metadata = cudnn_tensor_metadata_for_value(ir, value)?;
    let input_metadata = cudnn_tensor_metadata_for_value(ir, input_value)?;
    let weight_metadata = cudnn_tensor_metadata_for_value(ir, weight_value)?;
    let value_shape = ir.required_value_shape(value)?;
    let value_layout = ir.required_value_layout(value)?;
    let input_shape = ir.required_value_shape(input_value)?;
    let input_layout = ir.required_value_layout(input_value)?;
    let weight_shape = ir.required_value_shape(weight_value)?;
    let weight_layout = ir.required_value_layout(weight_value)?;

    let input_tensor = cache
        .get(&input)
        .ok_or_else(|| Error::missing_input("build cudnn convolution graph"))?;
    let weight_tensor = cache
        .get(&weight)
        .ok_or_else(|| Error::missing_input("build cudnn convolution graph"))?;
    validate_graph_node_value_tensor(ir, input, input_tensor)?;
    validate_graph_node_value_tensor(ir, weight, weight_tensor)?;
    validate_graph_node_value_tensor(ir, id, output)?;

    let data_type = value_metadata.data_type();
    let mut input_keys = vec![
        cudnn_backend_tensor_metadata_key(data_type, input_shape, input_layout),
        cudnn_backend_tensor_metadata_key(data_type, weight_shape, weight_layout),
    ];
    let bias_metadata = if let Some(bias) = bias {
        validate_cudnn_conv2d_bias(ir, id, weight, bias)?;
        let bias_tensor = cache
            .get(&bias)
            .ok_or_else(|| Error::missing_input("build cudnn convolution graph"))?;
        validate_graph_node_value_tensor(ir, bias, bias_tensor)?;
        let bias_value = bias;
        input_keys.push(cudnn_backend_tensor_metadata_key(
            ir.required_value_data_type(bias_value)?,
            ir.required_value_shape(bias_value)?,
            ir.required_value_layout(bias_value)?,
        ));
        Some(cudnn_tensor_metadata_for_value(ir, bias_value)?)
    } else {
        None
    };
    let key = CudnnBackendGraphKey::new(
        if window.spatial_rank() == 2 {
            CudnnBackendGraphKind::Conv2d
        } else {
            CudnnBackendGraphKind::Conv3d
        },
        window.parameters(),
        input_keys,
        cudnn_backend_tensor_metadata_key(data_type, value_shape, value_layout),
    );
    let template = cached_cudnn_backend_graph_template(runtime, key, |cudnn| {
        build_cudnn_convolution_template(
            cudnn,
            &input_metadata,
            &weight_metadata,
            bias_metadata.as_ref(),
            &value_metadata,
            cudnn_convolution_geometry(window)?,
        )
    })?;
    let workspace_size = cudnn_backend_template_workspace_size(&template)?;
    let input_id = CUDNN_GRAPH_INPUT0_TENSOR_ID;
    let weight_id = CUDNN_GRAPH_INPUT1_TENSOR_ID;
    let output_id = CUDNN_GRAPH_OUTPUT_TENSOR_ID;
    let mut bindings = vec![
        CudnnGraphBinding::new(input_id, CudnnGraphBindingSource::Node(input)),
        CudnnGraphBinding::new(weight_id, CudnnGraphBindingSource::Node(weight)),
        CudnnGraphBinding::new(output_id, CudnnGraphBindingSource::Output),
    ];
    if let Some(bias) = bias {
        bindings.push(CudnnGraphBinding::new(
            CUDNN_GRAPH_BIAS_TENSOR_ID,
            CudnnGraphBindingSource::Node(bias),
        ));
    }

    let executable = CudnnBackendExecutable::new(
        template,
        allocate_cudnn_workspace(runtime.context(), workspace_size)?,
        bindings,
    );
    populate_cudnn_conv2d_graph(runtime, executable, output, cache)
}
