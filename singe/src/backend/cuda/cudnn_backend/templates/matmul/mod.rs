mod add;

use singe_cudnn::{
    backend::execution::{
        EngineChoice, OperationGraph as CudnnOperationGraph,
        OperationGraphConfig as CudnnOperationGraphConfig,
        matmul::{
            MatmulDescriptor as CudnnMatmulDescriptor,
            MatmulDescriptorConfig as CudnnMatmulDescriptorConfig,
            MatmulOperation as CudnnMatmulOperation,
            MatmulOperationConfig as CudnnMatmulOperationConfig,
        },
    },
    context::Context as CudnnContext,
};

pub use self::add::{build_cudnn_matmul_add_relu_template, build_cudnn_matmul_add_template};
use crate::{
    backend::cuda::cudnn_backend::{
        templates::tensors::{
            CUDNN_GRAPH_INPUT0_TENSOR_ID, CUDNN_GRAPH_INPUT1_TENSOR_ID,
            CUDNN_GRAPH_OUTPUT_TENSOR_ID, CudnnBackendTensorMetadata, cudnn_backend_tensor,
            cudnn_forced_execution_plan, cudnn_matmul_graph_execution_plan,
        },
        types::{CudnnBackendGraphTemplate, CudnnBackendOperation},
    },
    error::Result,
};

/// Builds a cuDNN matrix multiplication template with descriptor ranks promoted to at least three.
/// Compute precision derives from A's storage type.
///
/// # Errors
///
/// Returns an error if tensor types or layouts are unsupported, native descriptor or graph creation fails, no plan satisfies the numerical policy, or workspace querying fails.
// Builds a matmul template, with the engine and knobs a record states in place of what heuristics offer.
//
// # Errors
//
// Returns an error if cuDNN refuses the graph or that engine.
pub fn build_cudnn_matmul_template(
    cudnn: &CudnnContext,
    a: &CudnnBackendTensorMetadata,
    b: &CudnnBackendTensorMetadata,
    output: &CudnnBackendTensorMetadata,
    engine: Option<&EngineChoice>,
) -> Result<CudnnBackendGraphTemplate> {
    let rank = a.rank().max(b.rank()).max(output.rank()).max(3);
    let a = a.promote_rank(rank)?;
    let b = b.promote_rank(rank)?;
    let output = output.promote_rank(rank)?;
    let a_descriptor = cudnn_backend_tensor(CUDNN_GRAPH_INPUT0_TENSOR_ID, &a)?;
    let b_descriptor = cudnn_backend_tensor(CUDNN_GRAPH_INPUT1_TENSOR_ID, &b)?;
    let output_descriptor = cudnn_backend_tensor(CUDNN_GRAPH_OUTPUT_TENSOR_ID, &output)?;
    let matmul =
        CudnnMatmulDescriptor::create(CudnnMatmulDescriptorConfig::new(a.compute_type()?))?;
    let operation = CudnnMatmulOperation::create(
        &matmul,
        &a_descriptor,
        &b_descriptor,
        &output_descriptor,
        CudnnMatmulOperationConfig::default(),
    )?;
    let operation_graph = CudnnOperationGraph::create(
        cudnn,
        &[operation.descriptor()],
        CudnnOperationGraphConfig::default(),
    )?;
    let plan = match engine {
        Some(choice) => {
            cudnn_forced_execution_plan(cudnn, &operation_graph, a.data_type(), true, choice)?
        }
        None => cudnn_matmul_graph_execution_plan(cudnn, &operation_graph, a.data_type())?,
    };
    let workspace_size = plan.workspace_size()?;

    Ok(CudnnBackendGraphTemplate::new(
        plan,
        workspace_size,
        CudnnBackendOperation::matmul(
            a_descriptor,
            b_descriptor,
            output_descriptor,
            matmul,
            operation,
        ),
        operation_graph,
    ))
}
