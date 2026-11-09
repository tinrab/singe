use singe_cuda::graph::GraphNode;

use crate::{
    backend::cuda::compiled::{
        build::{
            attach::attach_kernel_graph,
            provider::{GraphProviderLowering, GraphProviderRequest, try_provider},
        },
        plan::{kernel::build_node_kernel_graph, lowering::CudaGraphProvider},
    },
    error::Result,
};

pub fn try_add_kernel_provider_node(
    request: &GraphProviderRequest<'_>,
    lowering: &mut GraphProviderLowering<'_>,
) -> Result<Option<GraphNode>> {
    try_provider(
        request,
        lowering,
        CudaGraphProvider::Kernel,
        |lowering| {
            build_node_kernel_graph(
                lowering.node_plans.remove(&request.id),
                lowering.planner,
                request.ir,
                request.id,
                request.output,
                request.cache,
                lowering.private,
            )
        },
        |child_graph, lowering| {
            attach_kernel_graph(
                lowering.native_graph,
                request.dependencies,
                lowering.stats,
                lowering.realized_lowering,
                lowering.library_resources,
                request.ir,
                request.id,
                child_graph,
            )
        },
    )
}
