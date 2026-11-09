use std::{collections::HashMap, hash::RandomState, sync::Arc};

use singe_cuda::graph::Graph as NativeCudaGraph;

use crate::{
    backend::cuda::{
        CudaRuntime,
        compiled::plan::cudnn::{
            CudnnConv2dBiasReluPattern, allocate_cudnn_workspace,
            conv2d::{
                executable::populate_cudnn_conv2d_graph,
                validation::{validate_cudnn_conv2d_bias, validate_cudnn_conv2d_nodes},
            },
            cudnn_tensor_metadata_for_value, validate_cudnn_same_shape_unary,
        },
        cudnn_backend::{
            CUDNN_GRAPH_BIAS_TENSOR_ID, CUDNN_GRAPH_INPUT0_TENSOR_ID, CUDNN_GRAPH_INPUT1_TENSOR_ID,
            CUDNN_GRAPH_OUTPUT_TENSOR_ID, CudnnBackendExecutable, CudnnBackendGraphKey,
            CudnnBackendGraphKind, CudnnBackendTensorMetadata, CudnnGraphBinding,
            CudnnGraphBindingSource, build_cudnn_conv2d_bias_relu_template,
            cached_cudnn_backend_graph_template, cudnn_backend_template_workspace_size,
            cudnn_backend_tensor_metadata_key,
        },
        graph_binding::validate_graph_node_value_tensor,
        graph_value::CudaGraphValue,
    },
    error::{Error, Result},
    graph::ir::{Graph, ValueId},
    layout::Layout,
    shape::Shape,
};

// The sliding window of a two-dimensional convolution.
#[derive(Debug, Clone, Copy)]
pub struct CudnnConv2dGeometry {
    pub stride: (i64, i64),
    pub padding: (i64, i64, i64, i64),
    pub dilation: (i64, i64),
    pub groups: i64,
}

/// Returns an error if required operation metadata or cached operands are missing or incompatible, tensor types or layouts are unsupported, cache or library mutexes are poisoned, or native plan creation, workspace allocation, tensor binding, or graph population fails.
pub fn build_cudnn_conv2d_bias_relu_graph(
    runtime: &Arc<CudaRuntime>,
    ir: &Graph,
    id: ValueId,
    operands: (ValueId, ValueId),
    pattern: CudnnConv2dBiasReluPattern,
    geometry: CudnnConv2dGeometry,
    output: &CudaGraphValue,
    cache: &HashMap<ValueId, CudaGraphValue, RandomState>,
) -> Result<(NativeCudaGraph, CudnnBackendExecutable)> {
    let (input, weight) = operands;
    let CudnnConv2dGeometry {
        stride,
        padding,
        dilation,
        groups,
    } = geometry;
    validate_cudnn_conv2d_nodes(ir, id, input, weight, stride, padding, dilation, groups)?;
    validate_cudnn_conv2d_bias(ir, id, weight, pattern.bias())?;
    validate_cudnn_same_shape_unary(
        ir,
        pattern.relu(),
        id,
        "cudnn conv2d bias relu output shape",
    )?;
    let conv_value = id;
    let input_value = input;
    let weight_value = weight;
    let bias_value = pattern.bias();
    let relu_value = pattern.relu();
    let conv_metadata = cudnn_tensor_metadata_for_value(ir, conv_value)?;
    let input_metadata = cudnn_tensor_metadata_for_value(ir, input_value)?;
    let weight_metadata = cudnn_tensor_metadata_for_value(ir, weight_value)?;
    let bias_metadata = cudnn_tensor_metadata_for_value(ir, bias_value)?;
    let relu_metadata = cudnn_tensor_metadata_for_value(ir, relu_value)?;
    let input_shape = ir.required_value_shape(input_value)?;
    let input_layout = ir.required_value_layout(input_value)?;
    let weight_shape = ir.required_value_shape(weight_value)?;
    let weight_layout = ir.required_value_layout(weight_value)?;
    let bias_shape = ir.required_value_shape(bias_value)?;
    let bias_layout = ir.required_value_layout(bias_value)?;
    let relu_shape = ir.required_value_shape(relu_value)?;
    let relu_layout = ir.required_value_layout(relu_value)?;

    let input_tensor = cache
        .get(&input)
        .ok_or_else(|| Error::missing_input("build cudnn conv2d bias relu graph"))?;
    let weight_tensor = cache
        .get(&weight)
        .ok_or_else(|| Error::missing_input("build cudnn conv2d bias relu graph"))?;
    let bias_tensor = cache
        .get(&pattern.bias())
        .ok_or_else(|| Error::missing_input("build cudnn conv2d bias relu graph"))?;
    validate_graph_node_value_tensor(ir, input, input_tensor)?;
    validate_graph_node_value_tensor(ir, weight, weight_tensor)?;
    validate_graph_node_value_tensor(ir, pattern.bias(), bias_tensor)?;
    validate_graph_node_value_tensor(ir, pattern.relu(), output)?;

    let executable = build_cudnn_conv2d_bias_relu_executable(
        runtime,
        CudnnConv2dBiasReluExecutableRequest {
            input,
            weight,
            pattern,
            stride,
            padding,
            dilation,
            groups,
            conv_metadata: &conv_metadata,
            input_metadata: &input_metadata,
            weight_metadata: &weight_metadata,
            bias_metadata: &bias_metadata,
            relu_metadata: &relu_metadata,
            input_shape,
            input_layout,
            weight_shape,
            weight_layout,
            bias_shape,
            bias_layout,
            relu_shape,
            relu_layout,
        },
    )?;
    populate_cudnn_conv2d_graph(runtime, executable, output, cache)
}

#[derive(Clone, Copy)]
pub struct CudnnConv2dBiasReluExecutableRequest<'a> {
    pub(crate) input: ValueId,
    pub(crate) weight: ValueId,
    pub(crate) pattern: CudnnConv2dBiasReluPattern,
    pub(crate) stride: (i64, i64),
    pub(crate) padding: (i64, i64, i64, i64),
    pub(crate) dilation: (i64, i64),
    pub(crate) groups: i64,
    pub(crate) conv_metadata: &'a CudnnBackendTensorMetadata,
    pub(crate) input_metadata: &'a CudnnBackendTensorMetadata,
    pub(crate) weight_metadata: &'a CudnnBackendTensorMetadata,
    pub(crate) bias_metadata: &'a CudnnBackendTensorMetadata,
    pub(crate) relu_metadata: &'a CudnnBackendTensorMetadata,
    pub(crate) input_shape: &'a Shape,
    pub(crate) input_layout: &'a Layout,
    pub(crate) weight_shape: &'a Shape,
    pub(crate) weight_layout: &'a Layout,
    pub(crate) bias_shape: &'a Shape,
    pub(crate) bias_layout: &'a Layout,
    pub(crate) relu_shape: &'a Shape,
    pub(crate) relu_layout: &'a Layout,
}

pub fn build_cudnn_conv2d_bias_relu_executable(
    runtime: &Arc<CudaRuntime>,
    request: CudnnConv2dBiasReluExecutableRequest<'_>,
) -> Result<CudnnBackendExecutable> {
    let data_type = request.relu_metadata.data_type();
    let key = CudnnBackendGraphKey::new(
        CudnnBackendGraphKind::Conv2dBiasRelu,
        vec![
            request.stride.0,
            request.stride.1,
            request.padding.0,
            request.padding.1,
            request.padding.2,
            request.padding.3,
            request.dilation.0,
            request.dilation.1,
            request.groups,
        ],
        vec![
            cudnn_backend_tensor_metadata_key(data_type, request.input_shape, request.input_layout),
            cudnn_backend_tensor_metadata_key(
                data_type,
                request.weight_shape,
                request.weight_layout,
            ),
            cudnn_backend_tensor_metadata_key(
                request.bias_metadata.data_type(),
                request.bias_shape,
                request.bias_layout,
            ),
        ],
        cudnn_backend_tensor_metadata_key(data_type, request.relu_shape, request.relu_layout),
    );
    let template = cached_cudnn_backend_graph_template(runtime, key, |cudnn| {
        build_cudnn_conv2d_bias_relu_template(
            cudnn,
            request.input_metadata,
            request.weight_metadata,
            request.bias_metadata,
            request.conv_metadata,
            request.relu_metadata,
            request.stride,
            request.padding,
            request.dilation,
        )
    })?;
    let workspace_size = cudnn_backend_template_workspace_size(&template)?;
    let input_id = CUDNN_GRAPH_INPUT0_TENSOR_ID;
    let weight_id = CUDNN_GRAPH_INPUT1_TENSOR_ID;
    let bias_id = CUDNN_GRAPH_BIAS_TENSOR_ID;
    let output_id = CUDNN_GRAPH_OUTPUT_TENSOR_ID;
    Ok(CudnnBackendExecutable::new(
        template,
        allocate_cudnn_workspace(runtime.context(), workspace_size)?,
        vec![
            CudnnGraphBinding::new(input_id, CudnnGraphBindingSource::Node(request.input)),
            CudnnGraphBinding::new(weight_id, CudnnGraphBindingSource::Node(request.weight)),
            CudnnGraphBinding::new(
                bias_id,
                CudnnGraphBindingSource::Node(request.pattern.bias()),
            ),
            CudnnGraphBinding::new(output_id, CudnnGraphBindingSource::Output),
        ],
    ))
}
