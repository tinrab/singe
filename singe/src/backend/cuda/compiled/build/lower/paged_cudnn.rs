use singe_cuda::graph::GraphNode;

use crate::{
    backend::{
        cuda::{
            compiled::{
                build::{
                    attach::{CudnnGraphAttachment, attach_cudnn_graph},
                    lower::GraphOperationLoweringRequest,
                    state::GraphLoweringState,
                },
                executable::LibraryResource,
                plan::{
                    cudnn::{CudnnPagedValues, build_cudnn_paged_attention_graph},
                    lowering::CudaGraphProvider,
                },
            },
            cudnn_backend::CudnnPaging,
            diagnostics::{GraphProviderFailures, record_cuda_graph_provider_failure},
            graph_value::CudaGraphValue,
            kernel_plan::{
                self, GraphStep, KernelInput, plan_cudnn_paging_indices, plan_cudnn_paging_results,
            },
            tensor_region::CudaTensorSource,
            tuning::{ForcedLibrary, TunedChoice, build_forced, way_key},
        },
        registry::ScaledDotProductAttentionAttributes,
    },
    data_type::DataType,
    error::{Error, Result},
    shape::Shape,
    tensor::Tensor,
    value::ValueSpec,
};

fn scratch(
    state: &mut GraphLoweringState,
    output: &CudaGraphValue,
    spec: &ValueSpec,
) -> Result<CudaGraphValue> {
    let buffer = Tensor::zeroed_on(output.device(), spec.data_type(), spec.shape())?;
    let value = CudaGraphValue::from_tensor(buffer.clone())?;
    state.private.scratch.push(buffer);
    Ok(value)
}

// Runs paged attention through cuDNN, between the generated kernels that turn paging tensors into its lengths and page table and its statistics into `lse`.
// Returns no node when cuDNN refuses, so a generated kernel runs instead.
pub fn try_lower_cudnn_paged_attention(
    request: &GraphOperationLoweringRequest<'_>,
    state: &mut GraphLoweringState,
    inputs: &[CudaGraphValue],
    metadata: &[KernelInput],
    (paging, attention): (&CudnnPaging, &ScaledDotProductAttentionAttributes),
    [output, lse]: [&CudaGraphValue; 2],
    kept: usize,
    dependencies: &[GraphNode],
    failures: &mut GraphProviderFailures,
) -> Result<Option<GraphNode>> {
    let ir = request.ir;
    let operation = ir.required_operation_for_value(request.id)?;
    let &[query, key, value, ..] = operation.inputs() else {
        return Err(Error::missing_input("cudnn paged attention"));
    };
    let index_metadata = metadata
        .get(3..)
        .ok_or_else(|| Error::missing_input("cudnn paged attention"))?;
    let indices = plan_cudnn_paging_indices(paging, index_metadata)?;
    let [offsets, query_lengths, key_lengths, pages] = indices.outputs.as_slice() else {
        return Err(Error::missing_output("cudnn paged attention"));
    };
    let offsets = scratch(state, output, offsets)?;
    let query_lengths = scratch(state, output, query_lengths)?;
    let key_lengths = scratch(state, output, key_lengths)?;
    let pages = scratch(state, output, pages)?;
    let statistics = scratch(
        state,
        output,
        &ValueSpec::new(
            DataType::F32,
            Shape::create([paging.sequences, paging.heads, paging.tokens])?,
        ),
    )?;
    let requested = if kept == 0 { output } else { lse };
    let forced = state.planner.context().forced_library(ir, request.id)?;
    let library = build_forced(forced.as_ref(), ForcedLibrary::cudnn, |engine| {
        build_cudnn_paged_attention_graph(
            request.runtime,
            ir,
            request.id,
            [query, key, value],
            paging,
            attention,
            CudnnPagedValues {
                output,
                offsets: &offsets,
                query_lengths: &query_lengths,
                key_lengths: &key_lengths,
                pages: &pages,
                stats: &statistics,
            },
            requested,
            &state.cache,
            engine,
        )
    });
    if let (Some(recorder), Ok((_, executable))) = (&state.private.library_recorder, &library) {
        recorder.note(
            &way_key(ir, request.id)?,
            CudaGraphProvider::CudnnBackend,
            executable
                .engine_choices(request.runtime, ir.required_value_data_type(query)?, false)?
                .into_iter()
                .map(TunedChoice::Cudnn)
                .collect(),
        )?;
    }

    let (library_graph, executable) = match library {
        Ok(built) => built,
        Err(error) => {
            record_cuda_graph_provider_failure(failures, CudaGraphProvider::CudnnBackend, error)?;
            return Ok(None);
        }
    };
    let index_sources = inputs
        .get(3..)
        .ok_or_else(|| Error::missing_input("cudnn paged attention"))?
        .iter()
        .map(|input| input as &dyn CudaTensorSource)
        .collect();
    let index_graph = kernel_plan::build_graph_steps(
        &[GraphStep {
            plan: &indices,
            inputs: index_sources,
        }],
        &[&offsets, &query_lengths, &key_lengths, &pages],
        &mut state.private,
    )?;
    let index_node = state
        .native_graph
        .add_child_graph_node(dependencies, &index_graph)?;
    state
        .library_resources
        .push(LibraryResource::Graph(index_graph));
    state.stats.record_kernel_child_graph_node();
    let library_node = attach_cudnn_graph(
        &mut state.native_graph,
        &[index_node],
        &mut state.stats,
        &mut state.realized_lowering,
        &mut state.cudnn_steps,
        ir,
        request.id,
        requested,
        CudnnGraphAttachment {
            child_graph: library_graph,
            executable,
            update_on_replay: false,
        },
    )?;
    let [query_offsets, page_offsets, _, last_page_lengths] = index_metadata else {
        return Err(Error::missing_input("cudnn paged attention"));
    };
    let stats_input = KernelInput::from_source(&statistics);
    let results = plan_cudnn_paging_results(
        paging,
        attention,
        output.data_type(),
        &[
            query_offsets.clone(),
            page_offsets.clone(),
            last_page_lengths.clone(),
            stats_input,
        ],
    )?;
    let result_graph = kernel_plan::build_graph_steps(
        &[GraphStep {
            plan: &results,
            inputs: vec![
                &inputs[3] as &dyn CudaTensorSource,
                &inputs[4],
                &inputs[6],
                &statistics,
            ],
        }],
        &[output, lse],
        &mut state.private,
    )?;
    let result_node = state
        .native_graph
        .add_child_graph_node(&[library_node], &result_graph)?;
    state
        .library_resources
        .push(LibraryResource::Graph(result_graph));
    state.stats.record_kernel_child_graph_node();
    Ok(Some(result_node))
}
