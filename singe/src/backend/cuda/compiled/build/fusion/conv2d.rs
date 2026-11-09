use crate::{
    backend::cuda::{
        compiled::{
            build::{
                fusion::{GraphFusionCompletion, GraphFusionLowering},
                lower::GraphOperationLoweringRequest,
            },
            plan::{
                cudnn::{
                    CudnnConv2dGeometry, build_cudnn_conv2d_bias_relu_graph,
                    cudnn_conv2d_bias_relu_pattern,
                },
                lowering::CudaGraphProvider,
            },
        },
        diagnostics::record_cuda_graph_provider_failure,
        graph_binding::validate_graph_node_value_tensor,
    },
    error::Result,
    graph::ir::ValueId,
};

pub fn try_add_cudnn_conv2d_bias_relu_fusion(
    request: &GraphOperationLoweringRequest<'_>,
    input: ValueId,
    weight: ValueId,
    bias: ValueId,
    stride: (i64, i64),
    padding: (i64, i64, i64, i64),
    dilation: (i64, i64),
    groups: i64,
    lowering: &mut GraphFusionLowering<'_>,
) -> Result<bool> {
    let Some(pattern) = cudnn_conv2d_bias_relu_pattern(
        request.ir,
        request.id,
        [input, weight, bias],
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
        pattern.relu(),
    )?;
    let allocation_dependencies = allocation.dependencies();
    let output = allocation.into_value();
    validate_graph_node_value_tensor(request.ir, pattern.relu(), &output)?;
    let result = build_cudnn_conv2d_bias_relu_graph(
        request.runtime,
        request.ir,
        request.id,
        (input, weight),
        pattern,
        CudnnConv2dGeometry {
            stride,
            padding,
            dilation,
            groups,
        },
        &output,
        lowering.cache,
    );
    match result {
        Ok((child_graph, executable)) => {
            let dependency_ids = [input, weight, bias];
            lowering.finish_cudnn_backend_fusion(
                GraphFusionCompletion {
                    ir: request.ir,
                    dependency_ids: &dependency_ids,
                    allocation_dependencies,
                    output_id: pattern.relu(),
                    output,
                    realized_nodes: &[request.id, pattern.relu()],
                    skipped_nodes: &[pattern.relu()],
                    release_nodes: &[
                        (request.id, request.index),
                        (pattern.relu(), pattern.relu_index()),
                    ],
                },
                child_graph,
                executable,
            )?;
            Ok(true)
        }
        Err(error) => {
            record_cuda_graph_provider_failure(
                lowering.provider_failures,
                CudaGraphProvider::CudnnBackend,
                error,
            )?;
            Ok(false)
        }
    }
}
