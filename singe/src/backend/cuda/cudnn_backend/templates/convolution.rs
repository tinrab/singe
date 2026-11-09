use singe_cudnn::{
    backend::{
        convolution::ConvolutionMode as CudnnConvolutionMode,
        execution::{
            OperationGraph as CudnnOperationGraph,
            OperationGraphConfig as CudnnOperationGraphConfig,
            convolution::{
                ConvolutionDescriptor as CudnnConvolutionDescriptor,
                ConvolutionDescriptorConfig as CudnnConvolutionDescriptorConfig,
                ConvolutionForwardOperation as CudnnConvolutionForwardOperation,
                ConvolutionGeometry as CudnnConvolutionGeometry,
            },
            pointwise::{
                PointwiseDescriptor as CudnnPointwiseDescriptor,
                PointwiseDescriptorConfig as CudnnPointwiseDescriptorConfig,
                PointwiseOperation as CudnnPointwiseOperation,
                PointwiseOperationConfig as CudnnPointwiseOperationConfig,
            },
        },
        pointwise::PointwiseMode as CudnnPointwiseMode,
    },
    context::Context as CudnnContext,
    data_type::DataType as CudnnDataType,
    math::NanPropagation as CudnnNanPropagation,
};

use crate::{
    backend::{
        cuda::cudnn_backend::{
            templates::tensors::{
                CUDNN_GRAPH_BIAS_TENSOR_ID, CUDNN_GRAPH_INPUT0_TENSOR_ID,
                CUDNN_GRAPH_INPUT1_TENSOR_ID, CUDNN_GRAPH_OUTPUT_TENSOR_ID,
                CUDNN_GRAPH_VIRTUAL0_TENSOR_ID, CUDNN_GRAPH_VIRTUAL1_TENSOR_ID,
                CudnnBackendTensorMetadata, cudnn_backend_tensor, cudnn_backend_virtual_tensor,
                cudnn_native_graph_execution_plan,
            },
            types::{
                CudnnBackendConv2dBiasReluTensors, CudnnBackendGraphTemplate, CudnnBackendOperation,
            },
        },
        spec::ConvolutionWindow,
    },
    error::Result,
};

/// Builds a cuDNN graph template for convolution, bias addition, and `Relu`.
/// Convolution uses cross-correlation with height and width strides and dilations.
/// Padding is ordered as height start, height end, width start, and width end.
///
/// # Errors
///
/// Returns an error if tensor types, layouts, or convolution geometry are unsupported, native descriptor or operation creation fails, no execution plan satisfies the numerical policy, or workspace querying fails.
pub fn build_cudnn_conv2d_bias_relu_template(
    cudnn: &CudnnContext,
    input: &CudnnBackendTensorMetadata,
    weight: &CudnnBackendTensorMetadata,
    bias: &CudnnBackendTensorMetadata,
    conv_output: &CudnnBackendTensorMetadata,
    output: &CudnnBackendTensorMetadata,
    stride: (i64, i64),
    padding: (i64, i64, i64, i64),
    dilation: (i64, i64),
) -> Result<CudnnBackendGraphTemplate> {
    let input_descriptor = cudnn_backend_tensor(CUDNN_GRAPH_INPUT0_TENSOR_ID, input)?;
    let weight_descriptor = cudnn_backend_tensor(CUDNN_GRAPH_INPUT1_TENSOR_ID, weight)?;
    let bias_descriptor = cudnn_backend_tensor(
        CUDNN_GRAPH_BIAS_TENSOR_ID,
        &bias.convolution_bias(output.rank())?,
    )?;
    let intermediate = conv_output.compute_intermediate()?;
    let conv_output_descriptor =
        cudnn_backend_virtual_tensor(CUDNN_GRAPH_VIRTUAL0_TENSOR_ID, &intermediate)?;
    let biased_descriptor =
        cudnn_backend_virtual_tensor(CUDNN_GRAPH_VIRTUAL1_TENSOR_ID, &intermediate)?;
    let output_descriptor = cudnn_backend_tensor(CUDNN_GRAPH_OUTPUT_TENSOR_ID, output)?;
    let convolution = cudnn_convolution_descriptor(
        input.compute_type()?,
        cudnn_convolution_geometry(&ConvolutionWindow::conv2d(stride, padding, dilation, 1))?,
    )?;
    let convolution_operation = CudnnConvolutionForwardOperation::create(
        &convolution,
        &input_descriptor,
        &weight_descriptor,
        &conv_output_descriptor,
        1.0,
        0.0,
    )?;
    let add = CudnnPointwiseDescriptor::create(
        CudnnPointwiseDescriptorConfig::new(CudnnPointwiseMode::Add, conv_output.compute_type()?)
            .with_nan_propagation(CudnnNanPropagation::Propagate),
    )?;
    let add_operation = CudnnPointwiseOperation::create(
        &add,
        CudnnPointwiseOperationConfig::Binary {
            x: &conv_output_descriptor,
            b: &bias_descriptor,
            y: &biased_descriptor,
            alpha1: 1.0,
            alpha2: 1.0,
        },
    )?;
    let relu = CudnnPointwiseDescriptor::create(
        CudnnPointwiseDescriptorConfig::new(CudnnPointwiseMode::ReluFwd, output.compute_type()?)
            .with_nan_propagation(CudnnNanPropagation::NotPropagate),
    )?;
    let relu_operation = CudnnPointwiseOperation::create(
        &relu,
        CudnnPointwiseOperationConfig::Unary {
            x: &biased_descriptor,
            y: &output_descriptor,
            alpha1: 1.0,
        },
    )?;
    let operation_graph = CudnnOperationGraph::create(
        cudnn,
        &[
            convolution_operation.descriptor(),
            add_operation.descriptor(),
            relu_operation.descriptor(),
        ],
        CudnnOperationGraphConfig::default(),
    )?;
    let plan = cudnn_native_graph_execution_plan(cudnn, &operation_graph, input.data_type())?;
    let workspace_size = plan.workspace_size()?;

    Ok(CudnnBackendGraphTemplate::new(
        plan,
        workspace_size,
        CudnnBackendOperation::conv2d_bias_relu(
            CudnnBackendConv2dBiasReluTensors::new(
                input_descriptor,
                weight_descriptor,
                bias_descriptor,
                conv_output_descriptor,
                biased_descriptor,
                output_descriptor,
            ),
            convolution,
            convolution_operation,
            add,
            add_operation,
            relu,
            relu_operation,
        ),
        operation_graph,
    ))
}

// Builds a cuDNN graph template for a cross-correlation over any number of spatial axes, with an optional bias.
pub fn build_cudnn_convolution_template(
    cudnn: &CudnnContext,
    input: &CudnnBackendTensorMetadata,
    weight: &CudnnBackendTensorMetadata,
    bias: Option<&CudnnBackendTensorMetadata>,
    output: &CudnnBackendTensorMetadata,
    geometry: CudnnConvolutionGeometry,
) -> Result<CudnnBackendGraphTemplate> {
    let input_descriptor = cudnn_backend_tensor(CUDNN_GRAPH_INPUT0_TENSOR_ID, input)?;
    let weight_descriptor = cudnn_backend_tensor(CUDNN_GRAPH_INPUT1_TENSOR_ID, weight)?;
    let conv_output_descriptor = if bias.is_some() {
        cudnn_backend_virtual_tensor(
            CUDNN_GRAPH_VIRTUAL0_TENSOR_ID,
            &output.compute_intermediate()?,
        )?
    } else {
        cudnn_backend_tensor(CUDNN_GRAPH_OUTPUT_TENSOR_ID, output)?
    };
    let convolution = cudnn_convolution_descriptor(input.compute_type()?, geometry)?;
    let operation = CudnnConvolutionForwardOperation::create(
        &convolution,
        &input_descriptor,
        &weight_descriptor,
        &conv_output_descriptor,
        1.0,
        0.0,
    )?;
    let (operation_graph, operation) = if let Some(bias) = bias {
        let bias_descriptor = cudnn_backend_tensor(
            CUDNN_GRAPH_BIAS_TENSOR_ID,
            &bias.convolution_bias(output.rank())?,
        )?;
        let output_descriptor = cudnn_backend_tensor(CUDNN_GRAPH_OUTPUT_TENSOR_ID, output)?;
        let add = CudnnPointwiseDescriptor::create(
            CudnnPointwiseDescriptorConfig::new(CudnnPointwiseMode::Add, output.compute_type()?)
                .with_nan_propagation(CudnnNanPropagation::Propagate),
        )?;
        let add_operation = CudnnPointwiseOperation::create(
            &add,
            CudnnPointwiseOperationConfig::Binary {
                x: &conv_output_descriptor,
                b: &bias_descriptor,
                y: &output_descriptor,
                alpha1: 1.0,
                alpha2: 1.0,
            },
        )?;
        let graph = CudnnOperationGraph::create(
            cudnn,
            &[operation.descriptor(), add_operation.descriptor()],
            CudnnOperationGraphConfig::default(),
        )?;
        let operation = CudnnBackendOperation::conv2d_bias(
            input_descriptor,
            weight_descriptor,
            bias_descriptor,
            conv_output_descriptor,
            output_descriptor,
            convolution,
            operation,
            add,
            add_operation,
        );
        (graph, operation)
    } else {
        let graph = CudnnOperationGraph::create(
            cudnn,
            &[operation.descriptor()],
            CudnnOperationGraphConfig::default(),
        )?;
        let operation = CudnnBackendOperation::conv2d(
            input_descriptor,
            weight_descriptor,
            conv_output_descriptor,
            convolution,
            operation,
        );
        (graph, operation)
    };
    let plan = cudnn_native_graph_execution_plan(cudnn, &operation_graph, input.data_type())?;
    let workspace_size = plan.workspace_size()?;

    Ok(CudnnBackendGraphTemplate::new(
        plan,
        workspace_size,
        operation,
        operation_graph,
    ))
}

pub fn cudnn_convolution_geometry(window: &ConvolutionWindow) -> Result<CudnnConvolutionGeometry> {
    let (before, after): (Vec<_>, Vec<_>) = window.padding.iter().copied().unzip();
    Ok(CudnnConvolutionGeometry::create(
        before,
        after,
        window.stride.clone(),
        window.dilation.clone(),
    )?)
}

fn cudnn_convolution_descriptor(
    compute_type: CudnnDataType,
    geometry: CudnnConvolutionGeometry,
) -> Result<CudnnConvolutionDescriptor> {
    Ok(CudnnConvolutionDescriptor::create(
        CudnnConvolutionDescriptorConfig::new(
            compute_type,
            CudnnConvolutionMode::CrossCorrelation,
            geometry,
        ),
    )?)
}
