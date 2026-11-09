use singe_cuda::graph::GraphNode;

use crate::{
    backend::{
        cuda::{
            compiled::{
                build::{
                    attach::{CudnnGraphAttachment, attach_cudnn_graph},
                    operation::GraphIrConvolution,
                    provider::{GraphProviderLowering, GraphProviderRequest, try_provider},
                },
                plan::{
                    cudnn::{build_cudnn_attention_graph, build_cudnn_convolution_graph},
                    lowering::CudaGraphProvider,
                },
            },
            cudnn_backend::CudnnBackendExecutable,
            tuning::{ForcedLibrary, TunedChoice, build_forced, way_key},
        },
        registry::ScaledDotProductAttentionAttributes,
    },
    error::Result,
    graph::ir::ValueId,
};

pub fn try_add_convolution_provider_node(
    request: &GraphProviderRequest<'_>,
    convolution: &GraphIrConvolution,
    lowering: &mut GraphProviderLowering<'_>,
) -> Result<Option<GraphNode>> {
    try_provider(
        request,
        lowering,
        CudaGraphProvider::CudnnBackend,
        |_| {
            build_cudnn_convolution_graph(
                request.runtime,
                request.ir,
                request.id,
                convolution.input,
                convolution.weight,
                convolution.bias,
                &convolution.window,
                request.output,
                request.cache,
            )
        },
        |(child_graph, executable), lowering| {
            attach_cudnn_graph(
                lowering.native_graph,
                request.dependencies,
                lowering.stats,
                lowering.realized_lowering,
                lowering.cudnn_steps,
                request.ir,
                request.id,
                request.output,
                CudnnGraphAttachment {
                    child_graph,
                    executable,
                    update_on_replay: false,
                },
            )
        },
    )
}

pub fn try_add_attention_provider_node(
    request: &GraphProviderRequest<'_>,
    inputs: [ValueId; 3],
    key_lengths: Option<ValueId>,
    attention: &ScaledDotProductAttentionAttributes,
    lowering: &mut GraphProviderLowering<'_>,
) -> Result<Option<GraphNode>> {
    try_provider(
        request,
        lowering,
        CudaGraphProvider::CudnnBackend,
        |_| {
            let graph = build_forced(request.forced, ForcedLibrary::cudnn, |engine| {
                build_cudnn_attention_graph(
                    request.runtime,
                    request.ir,
                    request.id,
                    inputs,
                    key_lengths,
                    attention,
                    request.output,
                    request.cache,
                    engine,
                )
            })?;
            note_cudnn_engines(request, &graph.1, false)?;
            Ok(graph)
        },
        |(child_graph, executable), lowering| {
            attach_cudnn_graph(
                lowering.native_graph,
                request.dependencies,
                lowering.stats,
                lowering.realized_lowering,
                lowering.cudnn_steps,
                request.ir,
                request.id,
                request.output,
                CudnnGraphAttachment {
                    child_graph,
                    executable,
                    update_on_replay: false,
                },
            )
        },
    )
}

// Lists what cuDNN could run this node with, when tuning asks.
pub fn note_cudnn_engines(
    request: &GraphProviderRequest<'_>,
    executable: &CudnnBackendExecutable,
    matmul: bool,
) -> Result<()> {
    if let Some(recorder) = request.library_recorder {
        recorder.note(
            &way_key(request.ir, request.id)?,
            CudaGraphProvider::CudnnBackend,
            executable
                .engine_choices(request.runtime, request.output_data_type, matmul)?
                .into_iter()
                .map(TunedChoice::Cudnn)
                .collect(),
        )?;
    }
    Ok(())
}
