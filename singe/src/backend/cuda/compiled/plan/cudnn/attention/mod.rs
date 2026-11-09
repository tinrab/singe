use std::{collections::HashMap, hash::RandomState, sync::Arc};

use singe_cuda::graph::Graph as NativeCudaGraph;
use singe_cudnn::backend::{execution::EngineChoice, tensor::BackendTensorUid as CudnnTensorUid};

use crate::{
    backend::{
        cuda::{
            CudaRuntime,
            compiled::plan::cudnn::{allocate_cudnn_workspace, cudnn_tensor_metadata_for_value},
            cudnn_backend::{
                CUDNN_ATTENTION_KEY_LENGTHS_TENSOR_ID, CUDNN_ATTENTION_KEY_PAGES_TENSOR_ID,
                CUDNN_ATTENTION_KEY_TENSOR_ID, CUDNN_ATTENTION_OFFSETS_TENSOR_ID,
                CUDNN_ATTENTION_OUTPUT_TENSOR_ID, CUDNN_ATTENTION_QUERY_LENGTHS_TENSOR_ID,
                CUDNN_ATTENTION_QUERY_TENSOR_ID, CUDNN_ATTENTION_STATS_TENSOR_ID,
                CUDNN_ATTENTION_VALUE_PAGES_TENSOR_ID, CUDNN_ATTENTION_VALUE_TENSOR_ID,
                CudnnAttentionProblem, CudnnBackendExecutable, CudnnBackendGraphKey,
                CudnnBackendGraphKind, CudnnBackendTensorKey, CudnnBackendTensorMetadata,
                CudnnGraphBinding, CudnnGraphBindingSource, CudnnPaging,
                build_cudnn_attention_template, cached_cudnn_backend_graph_template,
                cudnn_attention_scalars, cudnn_attention_scale,
                cudnn_backend_template_workspace_size, cudnn_backend_tensor_metadata_key,
                cudnn_engine_key_parameters, cudnn_takes_attention,
            },
            graph_value::CudaGraphValue,
        },
        registry::ScaledDotProductAttentionAttributes,
    },
    error::{Error, Result},
    graph::ir::{Graph, ValueId},
    shape::Shape,
    tensor::Tensor,
    utility::Locked,
};

// The values paged attention reads and writes that are not the graph's own nodes.
#[derive(Debug, Clone, Copy)]
pub struct CudnnPagedValues<'a> {
    pub output: &'a CudaGraphValue,
    pub offsets: &'a CudaGraphValue,
    pub query_lengths: &'a CudaGraphValue,
    pub key_lengths: &'a CudaGraphValue,
    pub pages: &'a CudaGraphValue,
    pub stats: &'a CudaGraphValue,
}

// What one attention graph needs beyond its problem.
struct AttentionGraph<'a> {
    inputs: [ValueId; 3],
    // Dense attention's key lengths, which cuDNN reads where the graph holds them.
    key_lengths: Option<ValueId>,
    key: CudnnBackendGraphKey,
    problem: CudnnAttentionProblem,
    scalars: Vec<(CudnnTensorUid, u32)>,
    values: Vec<(CudnnTensorUid, CudaGraphValue)>,
    output_binding: bool,
    output: &'a CudaGraphValue,
}

fn finish(
    runtime: &Arc<CudaRuntime>,
    graph: AttentionGraph<'_>,
    cache: &HashMap<ValueId, CudaGraphValue, RandomState>,
) -> Result<(NativeCudaGraph, CudnnBackendExecutable)> {
    let AttentionGraph {
        inputs: [query, key, value],
        key_lengths,
        key: graph_key,
        problem,
        scalars,
        values,
        output_binding,
        output,
    } = graph;
    let template = cached_cudnn_backend_graph_template(runtime, graph_key, |cudnn| {
        build_cudnn_attention_template(cudnn, &problem)
    })?;
    let workspace_size = cudnn_backend_template_workspace_size(&template)?;
    let mut bindings = vec![
        CudnnGraphBinding::new(
            CUDNN_ATTENTION_QUERY_TENSOR_ID,
            CudnnGraphBindingSource::Node(query),
        ),
        CudnnGraphBinding::new(
            CUDNN_ATTENTION_KEY_TENSOR_ID,
            CudnnGraphBindingSource::Node(key),
        ),
        CudnnGraphBinding::new(
            CUDNN_ATTENTION_VALUE_TENSOR_ID,
            CudnnGraphBindingSource::Node(value),
        ),
    ];
    if let Some(lengths) = key_lengths {
        bindings.push(CudnnGraphBinding::new(
            CUDNN_ATTENTION_KEY_LENGTHS_TENSOR_ID,
            CudnnGraphBindingSource::Node(lengths),
        ));
    }

    if output_binding {
        bindings.push(CudnnGraphBinding::new(
            CUDNN_ATTENTION_OUTPUT_TENSOR_ID,
            CudnnGraphBindingSource::Output,
        ));
    }

    let mut executable = CudnnBackendExecutable::new(
        template,
        allocate_cudnn_workspace(runtime.context(), workspace_size)?,
        bindings,
    )
    .with_values(values)
    .with_scalars(scalars);
    let mut child_graph = runtime.context().create_graph()?;
    let variant_pack = executable.variant_pack(output, cache)?;
    let mut cudnn = runtime.cudnn().locked("cuda runtime cache")?;
    // SAFETY: compiler graph values retain tensor allocations, and the returned executable retains its workspace, its scalars, and the values only it binds.
    unsafe { executable.populate_graph(&mut cudnn, &variant_pack, &mut child_graph) }?;
    drop(cudnn);
    Ok((child_graph, executable))
}

fn key_material(
    ir: &Graph,
    values: [ValueId; 3],
    output: ValueId,
) -> Result<(Vec<CudnnBackendTensorKey>, CudnnBackendTensorKey)> {
    let tensor_key = |value: ValueId| -> Result<CudnnBackendTensorKey> {
        Ok(cudnn_backend_tensor_metadata_key(
            ir.required_value_data_type(value)?,
            ir.required_value_shape(value)?,
            ir.required_value_layout(value)?,
        ))
    };
    Ok((
        values
            .into_iter()
            .map(tensor_key)
            .collect::<Result<Vec<_>>>()?,
        tensor_key(output)?,
    ))
}

// Builds a child CUDA graph and cuDNN executable for attention whose tensors are the graph's own.
//
// # Errors
//
// Returns an error if cuDNN does not take these options, a value is missing, or no cuDNN engine plans it.
pub fn build_cudnn_attention_graph(
    runtime: &Arc<CudaRuntime>,
    ir: &Graph,
    id: ValueId,
    inputs: [ValueId; 3],
    key_lengths: Option<ValueId>,
    attention: &ScaledDotProductAttentionAttributes,
    output: &CudaGraphValue,
    cache: &HashMap<ValueId, CudaGraphValue, RandomState>,
    engine: Option<&EngineChoice>,
) -> Result<(NativeCudaGraph, CudnnBackendExecutable)> {
    let head_dimension = ir
        .required_value_shape(inputs[0])?
        .dimensions()
        .last()
        .copied()
        .unwrap_or(0);
    if !cudnn_takes_attention(ir.required_value_data_type(id)?, head_dimension, attention) {
        return Err(Error::unsupported_operation(
            "cudnn attention with these options",
        ));
    }

    let [query, key, value] = inputs;
    let dim = ir.required_value_shape(query)?.dimensions()[3];
    let (input_keys, output_key) = key_material(ir, inputs, id)?;
    let graph_key = CudnnBackendGraphKey::new(
        CudnnBackendGraphKind::Attention,
        [
            vec![
                i64::from(attention.causal),
                attention.window.unwrap_or(0),
                i64::from(key_lengths.is_some()),
            ],
            cudnn_engine_key_parameters(engine),
        ]
        .concat(),
        input_keys,
        output_key,
    );
    // cuDNN reads the query lengths with the key lengths: every batch's is the queries' count, which the executable holds.
    let values = key_lengths
        .map(|_| {
            let query_shape = ir.required_value_shape(query)?.dimensions();
            let count = i32::try_from(query_shape[2])
                .map_err(|_| Error::out_of_range("cudnn attention queries"))?;
            let batch = usize::try_from(query_shape[0])
                .map_err(|_| Error::out_of_range("cudnn attention batch"))?;
            let lengths = Tensor::from_vec(
                output.device(),
                Shape::create([query_shape[0]])?,
                vec![count; batch],
            )?;
            Ok::<_, Error>((
                CUDNN_ATTENTION_QUERY_LENGTHS_TENSOR_ID,
                CudaGraphValue::from_tensor(lengths)?,
            ))
        })
        .transpose()?
        .into_iter()
        .collect();
    finish(
        runtime,
        AttentionGraph {
            inputs,
            key_lengths,
            key: graph_key,
            problem: CudnnAttentionProblem {
                query: cudnn_tensor_metadata_for_value(ir, query)?,
                key: cudnn_tensor_metadata_for_value(ir, key)?,
                value: cudnn_tensor_metadata_for_value(ir, value)?,
                output: cudnn_tensor_metadata_for_value(ir, id)?,
                causal: attention.causal,
                window: attention.window,
                paging: None,
                key_lengths: key_lengths.is_some(),
                engine: engine.cloned(),
            },
            scalars: cudnn_attention_scalars(
                cudnn_attention_scale(attention, dim),
                attention.causal,
                attention.window,
                false,
            )?,
            values,
            output_binding: true,
            output,
        },
        cache,
    )
}

// Builds a child CUDA graph and cuDNN executable for attention over a paged cache.
// `id` is the node being lowered, which may be either of attention's results, so the output comes through `values`.
//
// # Errors
//
// Returns an error if a value is missing, or no cuDNN engine plans it.
pub fn build_cudnn_paged_attention_graph(
    runtime: &Arc<CudaRuntime>,
    ir: &Graph,
    id: ValueId,
    inputs: [ValueId; 3],
    paging: &CudnnPaging,
    attention: &ScaledDotProductAttentionAttributes,
    values: CudnnPagedValues<'_>,
    output: &CudaGraphValue,
    cache: &HashMap<ValueId, CudaGraphValue, RandomState>,
    engine: Option<&EngineChoice>,
) -> Result<(NativeCudaGraph, CudnnBackendExecutable)> {
    let [query, key, value] = inputs;
    let (input_keys, output_key) = key_material(ir, inputs, id)?;
    let graph_key = CudnnBackendGraphKey::new(
        CudnnBackendGraphKind::PagedAttention,
        [
            vec![
                i64::from(attention.causal),
                attention.window.unwrap_or(0),
                paging.sequences,
                paging.tokens,
                paging.table,
            ],
            cudnn_engine_key_parameters(engine),
        ]
        .concat(),
        input_keys,
        output_key,
    );
    let data_type = ir.required_value_data_type(query)?;
    let query_layout = ir.required_value_layout(query)?;
    let byte_alignment =
        CudnnBackendTensorMetadata::byte_alignment_for_layout(data_type, query_layout);
    let (packed_dimensions, packed_strides) = paging.packed_shape();
    let packed = CudnnBackendTensorMetadata::strided(
        data_type,
        packed_dimensions.to_vec(),
        packed_strides.to_vec(),
        byte_alignment,
    );
    let (cache_dimensions, cache_strides) = paging.cache_shape();
    let tensor_cache = |value: ValueId| -> Result<CudnnBackendTensorMetadata> {
        Ok(CudnnBackendTensorMetadata::strided(
            data_type,
            cache_dimensions.to_vec(),
            cache_strides.to_vec(),
            CudnnBackendTensorMetadata::byte_alignment_for_layout(
                data_type,
                ir.required_value_layout(value)?,
            ),
        ))
    };
    finish(
        runtime,
        AttentionGraph {
            inputs,
            key_lengths: None,
            key: graph_key,
            problem: CudnnAttentionProblem {
                query: packed.clone(),
                key: tensor_cache(key)?,
                value: tensor_cache(value)?,
                output: packed,
                causal: attention.causal,
                window: attention.window,
                paging: Some(*paging),
                key_lengths: false,
                engine: engine.cloned(),
            },
            scalars: cudnn_attention_scalars(
                cudnn_attention_scale(attention, paging.dim),
                attention.causal,
                attention.window,
                true,
            )?,
            values: vec![
                (CUDNN_ATTENTION_OUTPUT_TENSOR_ID, values.output.clone()),
                (CUDNN_ATTENTION_OFFSETS_TENSOR_ID, values.offsets.clone()),
                (
                    CUDNN_ATTENTION_QUERY_LENGTHS_TENSOR_ID,
                    values.query_lengths.clone(),
                ),
                (
                    CUDNN_ATTENTION_KEY_LENGTHS_TENSOR_ID,
                    values.key_lengths.clone(),
                ),
                (CUDNN_ATTENTION_KEY_PAGES_TENSOR_ID, values.pages.clone()),
                (CUDNN_ATTENTION_VALUE_PAGES_TENSOR_ID, values.pages.clone()),
                (CUDNN_ATTENTION_STATS_TENSOR_ID, values.stats.clone()),
            ],
            output_binding: false,
            output,
        },
        cache,
    )
}
