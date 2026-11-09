use singe_cudnn::{
    backend::{
        execution::{
            OperationGraph as CudnnOperationGraph,
            OperationGraphConfig as CudnnOperationGraphConfig,
            matmul::{
                MatmulDescriptor as CudnnMatmulDescriptor,
                MatmulDescriptorConfig as CudnnMatmulDescriptorConfig,
                MatmulOperation as CudnnMatmulOperation,
                MatmulOperationConfig as CudnnMatmulOperationConfig,
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
    math::NanPropagation as CudnnNanPropagation,
};

use crate::{
    backend::cuda::cudnn_backend::{
        templates::tensors::{
            CUDNN_GRAPH_BIAS_TENSOR_ID, CUDNN_GRAPH_INPUT0_TENSOR_ID, CUDNN_GRAPH_INPUT1_TENSOR_ID,
            CUDNN_GRAPH_OUTPUT_TENSOR_ID, CUDNN_GRAPH_VIRTUAL0_TENSOR_ID,
            CUDNN_GRAPH_VIRTUAL1_TENSOR_ID, CudnnBackendTensorMetadata, cudnn_backend_tensor,
            cudnn_backend_virtual_tensor, cudnn_matmul_graph_execution_plan,
        },
        types::{
            CudnnBackendGraphTemplate, CudnnBackendMatmulAddReluTensors,
            CudnnBackendMatmulAddTensors, CudnnBackendOperation,
        },
    },
    error::Result,
};

/// Builds a cuDNN template for matrix multiplication followed by bias addition.
/// Leading unit axes promote native tensor descriptors to at least rank three.
///
/// # Errors
///
/// Returns an error if tensor ranks, storage types, or layouts are incompatible, native descriptor or operation creation fails, no matrix multiplication plan satisfies the numerical policy, or workspace querying fails.
pub fn build_cudnn_matmul_add_template(
    cudnn: &CudnnContext,
    a: &CudnnBackendTensorMetadata,
    b: &CudnnBackendTensorMetadata,
    bias: &CudnnBackendTensorMetadata,
    matmul_output: &CudnnBackendTensorMetadata,
    output: &CudnnBackendTensorMetadata,
) -> Result<CudnnBackendGraphTemplate> {
    let rank = output.rank().max(3);
    let output = output.promote_rank(rank)?;
    let a_descriptor =
        cudnn_backend_tensor(CUDNN_GRAPH_INPUT0_TENSOR_ID, a.promote_rank(rank)?.as_ref())?;
    let b_descriptor =
        cudnn_backend_tensor(CUDNN_GRAPH_INPUT1_TENSOR_ID, b.promote_rank(rank)?.as_ref())?;
    let bias_descriptor = cudnn_backend_tensor(
        CUDNN_GRAPH_BIAS_TENSOR_ID,
        bias.promote_rank(rank)?.as_ref(),
    )?;
    let matmul_output_descriptor = cudnn_backend_virtual_tensor(
        CUDNN_GRAPH_VIRTUAL0_TENSOR_ID,
        matmul_output.promote_rank(rank)?.as_ref(),
    )?;
    let output_descriptor = cudnn_backend_tensor(CUDNN_GRAPH_OUTPUT_TENSOR_ID, &output)?;
    let matmul =
        CudnnMatmulDescriptor::create(CudnnMatmulDescriptorConfig::new(a.compute_type()?))?;
    let matmul_operation = CudnnMatmulOperation::create(
        &matmul,
        &a_descriptor,
        &b_descriptor,
        &matmul_output_descriptor,
        CudnnMatmulOperationConfig::default(),
    )?;
    let add = CudnnPointwiseDescriptor::create(
        CudnnPointwiseDescriptorConfig::new(CudnnPointwiseMode::Add, output.compute_type()?)
            .with_nan_propagation(CudnnNanPropagation::Propagate),
    )?;
    let add_operation = CudnnPointwiseOperation::create(
        &add,
        CudnnPointwiseOperationConfig::Binary {
            x: &matmul_output_descriptor,
            b: &bias_descriptor,
            y: &output_descriptor,
            alpha1: 1.0,
            alpha2: 1.0,
        },
    )?;
    let operation_graph = CudnnOperationGraph::create(
        cudnn,
        &[matmul_operation.descriptor(), add_operation.descriptor()],
        CudnnOperationGraphConfig::default(),
    )?;
    let plan = cudnn_matmul_graph_execution_plan(cudnn, &operation_graph, a.data_type())?;
    let workspace_size = plan.workspace_size()?;

    Ok(CudnnBackendGraphTemplate::new(
        plan,
        workspace_size,
        CudnnBackendOperation::matmul_add(
            CudnnBackendMatmulAddTensors::new(
                a_descriptor,
                b_descriptor,
                bias_descriptor,
                matmul_output_descriptor,
                output_descriptor,
            ),
            matmul,
            matmul_operation,
            add,
            add_operation,
        ),
        operation_graph,
    ))
}

/// Builds a cuDNN template for matrix multiplication followed by bias addition and `Relu`.
/// Leading unit axes promote native tensor descriptors to at least rank three.
///
/// # Errors
///
/// Returns an error if tensor ranks, storage types, or layouts are incompatible, native descriptor or operation creation fails, no matrix multiplication plan satisfies the numerical policy, or workspace querying fails.
pub fn build_cudnn_matmul_add_relu_template(
    cudnn: &CudnnContext,
    a: &CudnnBackendTensorMetadata,
    b: &CudnnBackendTensorMetadata,
    bias: &CudnnBackendTensorMetadata,
    matmul_output: &CudnnBackendTensorMetadata,
    add_output: &CudnnBackendTensorMetadata,
    output: &CudnnBackendTensorMetadata,
) -> Result<CudnnBackendGraphTemplate> {
    let rank = output.rank().max(3);
    let output = output.promote_rank(rank)?;
    let a_descriptor =
        cudnn_backend_tensor(CUDNN_GRAPH_INPUT0_TENSOR_ID, a.promote_rank(rank)?.as_ref())?;
    let b_descriptor =
        cudnn_backend_tensor(CUDNN_GRAPH_INPUT1_TENSOR_ID, b.promote_rank(rank)?.as_ref())?;
    let bias_descriptor = cudnn_backend_tensor(
        CUDNN_GRAPH_BIAS_TENSOR_ID,
        bias.promote_rank(rank)?.as_ref(),
    )?;
    let matmul_output_descriptor = cudnn_backend_virtual_tensor(
        CUDNN_GRAPH_VIRTUAL0_TENSOR_ID,
        matmul_output.promote_rank(rank)?.as_ref(),
    )?;
    let biased_descriptor = cudnn_backend_virtual_tensor(
        CUDNN_GRAPH_VIRTUAL1_TENSOR_ID,
        add_output.promote_rank(rank)?.as_ref(),
    )?;
    let output_descriptor = cudnn_backend_tensor(CUDNN_GRAPH_OUTPUT_TENSOR_ID, &output)?;
    let matmul =
        CudnnMatmulDescriptor::create(CudnnMatmulDescriptorConfig::new(a.compute_type()?))?;
    let matmul_operation = CudnnMatmulOperation::create(
        &matmul,
        &a_descriptor,
        &b_descriptor,
        &matmul_output_descriptor,
        CudnnMatmulOperationConfig::default(),
    )?;
    let add = CudnnPointwiseDescriptor::create(
        CudnnPointwiseDescriptorConfig::new(CudnnPointwiseMode::Add, add_output.compute_type()?)
            .with_nan_propagation(CudnnNanPropagation::Propagate),
    )?;
    let add_operation = CudnnPointwiseOperation::create(
        &add,
        CudnnPointwiseOperationConfig::Binary {
            x: &matmul_output_descriptor,
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
            matmul_operation.descriptor(),
            add_operation.descriptor(),
            relu_operation.descriptor(),
        ],
        CudnnOperationGraphConfig::default(),
    )?;
    let plan = cudnn_matmul_graph_execution_plan(cudnn, &operation_graph, a.data_type())?;
    let workspace_size = plan.workspace_size()?;

    Ok(CudnnBackendGraphTemplate::new(
        plan,
        workspace_size,
        CudnnBackendOperation::matmul_add_relu(
            CudnnBackendMatmulAddReluTensors::new(
                a_descriptor,
                b_descriptor,
                bias_descriptor,
                matmul_output_descriptor,
                biased_descriptor,
                output_descriptor,
            ),
            matmul,
            matmul_operation,
            add,
            add_operation,
            relu,
            relu_operation,
        ),
        operation_graph,
    ))
}
