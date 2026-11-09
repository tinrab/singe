use crate::{
    backend::cuda::{
        compiled::{
            build::{
                fusion::{GraphFusionCompletion, GraphFusionLowering},
                lower::GraphOperationLoweringRequest,
            },
            plan::{
                cudnn::{build_cudnn_matmul_add_relu_graph, cudnn_matmul_add_relu_pattern},
                lowering::CudaGraphProvider,
            },
        },
        cudnn_backend::cudnn_takes_matmul,
        diagnostics::record_cuda_graph_provider_failure,
        graph_binding::validate_graph_node_value_tensor,
    },
    error::Result,
    graph::ir::ValueId,
};

pub fn try_cudnn_matmul_add_relu_fusion(
    request: &GraphOperationLoweringRequest<'_>,
    inputs: [ValueId; 2],
    lowering: &mut GraphFusionLowering<'_>,
) -> Result<bool> {
    let ir = request.ir;
    if !cudnn_takes_matmul(
        ir.required_value_data_type(inputs[0])?,
        ir.required_value_layout(inputs[0])?,
        ir.required_value_layout(inputs[1])?,
    ) {
        return Ok(false);
    }

    if let Some(pattern) = cudnn_matmul_add_relu_pattern(
        request.ir,
        request.id,
        inputs,
        request.positions,
        request.users,
        request.graph_outputs,
        lowering.cache,
    )? {
        let allocation = lowering.memory_planner.allocate(
            lowering.context.allocator(),
            request.ir,
            pattern.relu(),
        )?;
        let allocation_dependencies = allocation.dependencies();
        let output = allocation.into_value();
        validate_graph_node_value_tensor(request.ir, pattern.relu(), &output)?;
        let result = build_cudnn_matmul_add_relu_graph(
            request.runtime,
            request.ir,
            request.id,
            inputs,
            pattern,
            &output,
            lowering.cache,
        );
        if let Ok((child_graph, executable)) = result {
            let dependency_ids = [inputs[0], inputs[1], pattern.bias()];
            lowering.finish_cudnn_backend_fusion(
                GraphFusionCompletion {
                    ir: request.ir,
                    dependency_ids: &dependency_ids,
                    allocation_dependencies,
                    output_id: pattern.relu(),
                    output,
                    realized_nodes: &[request.id, pattern.add(), pattern.relu()],
                    skipped_nodes: &[pattern.add(), pattern.relu()],
                    release_nodes: &[
                        (request.id, request.index),
                        (pattern.add(), pattern.add_index()),
                        (pattern.relu(), pattern.relu_index()),
                    ],
                },
                child_graph,
                executable,
            )?;
            return Ok(true);
        } else if let Err(error) = result {
            record_cuda_graph_provider_failure(
                lowering.provider_failures,
                CudaGraphProvider::CudnnBackend,
                error,
            )?;
        }
    }
    Ok(false)
}
