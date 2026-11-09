use singe_cuda::graph::GraphNode;

use crate::{
    backend::{
        cuda::{
            compiled::build::{
                graph_node::add_cuda_graph_memset_node,
                provider::{GraphProviderLowering, GraphProviderRequest},
                realized::record_realized_cuda_lowering,
            },
            graph_binding::graph_matmul_is_zero_work,
        },
        registry::{CudaGraphProvider, Operation, OperationSupport},
    },
    error::Result,
    graph::ir::ValueId,
};

pub enum GraphZeroWorkNode {
    Native(GraphNode),
    OutputOnly,
}

pub fn graph_allows_zero_work_without_native_node(support: OperationSupport) -> bool {
    if support.supports_cuda_graph_provider(CudaGraphProvider::Kernel) {
        return true;
    }
    matches!(
        support.operation(),
        Operation::Softmax
            | Operation::Cast
            | Operation::Contiguous
            | Operation::Expand
            | Operation::LayerNorm
            | Operation::RmsNorm
            | Operation::BatchNorm
            | Operation::LogSoftmax
            | Operation::Contraction
            | Operation::Conv2d
            | Operation::Concat
            | Operation::IndexSelect
            | Operation::Take
            | Operation::RepeatKv
            | Operation::Embedding
            | Operation::RotaryEmbedding
    )
}

pub fn try_add_zero_work_matmul_node(
    request: &GraphProviderRequest<'_>,
    inputs: [ValueId; 2],
    lowering: &mut GraphProviderLowering<'_>,
) -> Result<Option<GraphZeroWorkNode>> {
    if !graph_matmul_is_zero_work(request.ir, request.id, inputs)? {
        return Ok(None);
    }

    let output_count = request.output_shape.element_count();
    if let Some(candidate) = request.lowering_plan.candidates(request.id).first() {
        record_realized_cuda_lowering(
            lowering.realized_lowering,
            request.id,
            request.ir,
            candidate.provider(),
        )?;
    }

    if output_count == 0 {
        return Ok(Some(GraphZeroWorkNode::OutputOnly));
    }

    let graph_node = add_cuda_graph_memset_node(
        request.output,
        request.output_data_type.size_bytes(),
        output_count,
        request.dependencies,
        lowering.native_graph,
        lowering.stats,
    )?;
    Ok(Some(GraphZeroWorkNode::Native(graph_node)))
}
