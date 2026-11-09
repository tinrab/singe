use crate::{
    backend::cuda::{
        compiled::{
            build::{
                fusion::{GraphFusionCompletion, GraphFusionLowering},
                lower::GraphOperationLoweringRequest,
            },
            plan::{
                cublas::build_cublaslt_matmul_add_graph,
                cudnn::{build_cudnn_matmul_add_graph, cudnn_matmul_add_pattern},
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

pub fn try_matmul_add_fusion(
    request: &GraphOperationLoweringRequest<'_>,
    inputs: [ValueId; 2],
    lowering: &mut GraphFusionLowering<'_>,
) -> Result<bool> {
    let Some(pattern) = cudnn_matmul_add_pattern(
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

    let allocation = lowering.memory_planner.allocate(
        lowering.context.allocator(),
        request.ir,
        pattern.add(),
    )?;
    let allocation_dependencies = allocation.dependencies();
    let output = allocation.into_value();
    validate_graph_node_value_tensor(request.ir, pattern.add(), &output)?;

    let ir = request.ir;
    let takes = cudnn_takes_matmul(
        ir.required_value_data_type(inputs[0])?,
        ir.required_value_layout(inputs[0])?,
        ir.required_value_layout(inputs[1])?,
    );
    let result = takes.then(|| {
        build_cudnn_matmul_add_graph(
            request.runtime,
            request.ir,
            request.id,
            inputs,
            pattern,
            &output,
            lowering.cache,
        )
    });
    if let Some(Ok((child_graph, executable))) = result {
        let dependency_ids = [inputs[0], inputs[1], pattern.bias()];
        lowering.finish_cudnn_backend_fusion(
            GraphFusionCompletion {
                ir: request.ir,
                dependency_ids: &dependency_ids,
                allocation_dependencies,
                output_id: pattern.add(),
                output,
                realized_nodes: &[request.id, pattern.add()],
                skipped_nodes: &[pattern.add()],
                release_nodes: &[
                    (request.id, request.index),
                    (pattern.add(), pattern.add_index()),
                ],
            },
            child_graph,
            executable,
        )?;
        return Ok(true);
    } else if let Some(Err(error)) = result {
        record_cuda_graph_provider_failure(
            lowering.provider_failures,
            CudaGraphProvider::CudnnBackend,
            error,
        )?;
    }

    if !request
        .lowering_plan
        .should_try_provider(request.id, CudaGraphProvider::CublasLt)
    {
        return Ok(false);
    }

    let result = build_cublaslt_matmul_add_graph(
        request.runtime,
        request.ir,
        request.id,
        inputs,
        pattern.add(),
        pattern.bias(),
        &output,
        lowering.cache,
    );
    if let Ok(cublaslt_graph) = result {
        let dependency_ids = [inputs[0], inputs[1], pattern.bias()];
        lowering.finish_cublaslt_fusion(
            GraphFusionCompletion {
                ir: request.ir,
                dependency_ids: &dependency_ids,
                allocation_dependencies,
                output_id: pattern.add(),
                output,
                realized_nodes: &[request.id, pattern.add()],
                skipped_nodes: &[pattern.add()],
                release_nodes: &[
                    (request.id, request.index),
                    (pattern.add(), pattern.add_index()),
                ],
            },
            cublaslt_graph,
        )?;
        return Ok(true);
    } else if let Err(error) = result {
        record_cuda_graph_provider_failure(
            lowering.provider_failures,
            CudaGraphProvider::CublasLt,
            error,
        )?;
    }

    Ok(false)
}
