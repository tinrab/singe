use singe_cuda::graph::GraphNode;

use crate::{
    backend::cuda::{
        compiled::build::{
            lower::GraphOperationLoweringRequest,
            provider::{
                GraphProviderDiagnostics, GraphProviderNode, GraphProviderRequest,
                try_add_operation_provider_node,
            },
            state::GraphLoweringState,
        },
        diagnostics::GraphProviderFailures,
        graph_value::CudaGraphValue,
    },
    data_type::DataType,
    error::Result,
    shape::Shape,
};

pub enum GraphLoweredProviderOutput {
    Native(GraphNode),
    OutputOnly,
}

pub fn try_lower_provider_output(
    request: &GraphOperationLoweringRequest<'_>,
    output_data_type: DataType,
    output_shape: &Shape,
    output: &CudaGraphValue,
    dependencies: &[GraphNode],
    diagnostics: &GraphProviderDiagnostics<'_>,
    provider_failures: &mut GraphProviderFailures,
    state: &mut GraphLoweringState,
) -> Result<GraphLoweredProviderOutput> {
    let forced = state
        .planner
        .context()
        .forced_library(request.ir, request.id)?;
    let library_recorder = state.private.library_recorder.clone();
    let provider_node = state.with_provider_lowering(provider_failures, |cache, lowering| {
        try_add_operation_provider_node(
            GraphProviderRequest {
                runtime: request.runtime,
                ir: request.ir,
                id: request.id,
                output_data_type,
                output_shape,
                output,
                cache,
                dependencies,
                lowering_plan: request.lowering_plan,
                diagnostics,
                forced: forced.as_ref(),
                library_recorder: library_recorder.as_ref(),
            },
            lowering,
        )
    })?;

    match provider_node {
        Some(GraphProviderNode::Native(graph_node)) => {
            Ok(GraphLoweredProviderOutput::Native(graph_node))
        }
        Some(GraphProviderNode::OutputOnly) => Ok(GraphLoweredProviderOutput::OutputOnly),
        None => Err(diagnostics.lowering_failure(provider_failures)),
    }
}
