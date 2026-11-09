use singe_cuda::graph::GraphNode;

use crate::{
    backend::cuda::compiled::{
        build::{
            attach::attach_cutensor_graph,
            operation::GraphIrContraction,
            provider::{GraphProviderLowering, GraphProviderRequest, try_provider},
        },
        plan::{cutensor::build_cutensor_contraction_graph, lowering::CudaGraphProvider},
    },
    error::Result,
};

pub fn try_add_cutensor_contraction_provider_node(
    request: &GraphProviderRequest<'_>,
    contraction: &GraphIrContraction,
    lowering: &mut GraphProviderLowering<'_>,
) -> Result<Option<GraphNode>> {
    try_provider(
        request,
        lowering,
        CudaGraphProvider::Cutensor,
        |_lowering| {
            build_cutensor_contraction_graph(
                request.runtime,
                request.ir,
                request.id,
                (contraction.a, &contraction.a_mode),
                (contraction.b, &contraction.b_mode),
                (contraction.c, &contraction.c_mode),
                &contraction.d_mode,
                contraction.alpha,
                contraction.beta,
                request.output,
                request.cache,
            )
        },
        |step, lowering| {
            attach_cutensor_graph(
                lowering.native_graph,
                request.dependencies,
                lowering.stats,
                lowering.realized_lowering,
                lowering.cutensor_steps,
                request.ir,
                request.id,
                request.output,
                step,
            )
        },
    )
}
