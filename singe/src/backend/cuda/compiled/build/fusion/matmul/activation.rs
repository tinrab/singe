use crate::{
    backend::cuda::{
        compiled::{
            build::{
                fusion::{GraphFusionCompletion, GraphFusionLowering},
                lower::GraphOperationLoweringRequest,
            },
            plan::{
                cublas::build_cublaslt_matmul_activation_graph,
                cudnn::cudnn_matmul_activation_pattern, lowering::CudaGraphProvider,
            },
        },
        diagnostics::record_cuda_graph_provider_failure,
        graph_binding::validate_graph_node_value_tensor,
    },
    error::Result,
    graph::ir::ValueId,
};

pub fn try_cublaslt_matmul_activation_fusion(
    request: &GraphOperationLoweringRequest<'_>,
    inputs: [ValueId; 2],
    lowering: &mut GraphFusionLowering<'_>,
) -> Result<bool> {
    let Some(pattern) = cudnn_matmul_activation_pattern(
        request.ir,
        request.id,
        inputs,
        request.positions,
        request.users,
        request.graph_outputs,
        lowering.cache,
    )?
    else {
        return Ok(false);
    };
    let activation = pattern.activation();
    let allocation =
        lowering
            .memory_planner
            .allocate(lowering.context.allocator(), request.ir, activation)?;
    let allocation_dependencies = allocation.dependencies();
    let output = allocation.into_value();
    validate_graph_node_value_tensor(request.ir, activation, &output)?;
    let result = build_cublaslt_matmul_activation_graph(
        request.runtime,
        request.ir,
        request.id,
        inputs,
        activation,
        &output,
        lowering.cache,
        pattern.epilogue(),
    );
    match result {
        Ok(cublaslt_graph) => {
            let dependency_ids = [inputs[0], inputs[1]];
            lowering.finish_cublaslt_fusion(
                GraphFusionCompletion {
                    ir: request.ir,
                    dependency_ids: &dependency_ids,
                    allocation_dependencies,
                    output_id: activation,
                    output,
                    realized_nodes: &[request.id, activation],
                    skipped_nodes: &[activation],
                    release_nodes: &[
                        (request.id, request.index),
                        (activation, pattern.activation_index()),
                    ],
                },
                cublaslt_graph,
            )?;
            Ok(true)
        }
        Err(error) => {
            record_cuda_graph_provider_failure(
                lowering.provider_failures,
                CudaGraphProvider::CublasLt,
                error,
            )?;
            Ok(false)
        }
    }
}
