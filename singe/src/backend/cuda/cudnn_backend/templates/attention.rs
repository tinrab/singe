use singe_cudnn::{
    backend::{
        execution::{
            EngineChoice, OperationGraph as CudnnOperationGraph,
            OperationGraphConfig as CudnnOperationGraphConfig,
            attention::{
                AttentionSequenceInputs, DiagonalBandMaskBound, DiagonalBandMaskConfig,
                DiagonalBandMaskOperation, SdpaForwardConfig, SdpaForwardOperation,
                SdpaForwardSoftmax, SdpaForwardTensors, SdpaSubgraph, SequenceLength,
            },
        },
        pointwise::PointwiseMode,
        tensor::{
            BackendTensorUid as CudnnTensorUid, Shape as CudnnShape, Tensor as CudnnBackendTensor,
            TensorId, TensorSpec,
        },
    },
    context::Context as CudnnContext,
    data_type::DataType as CudnnDataType,
    error::Status as CudnnStatus,
};

use crate::{
    backend::{
        cuda::cudnn_backend::{
            data_type::storage_type,
            templates::tensors::{
                CudnnBackendTensorMetadata, cudnn_backend_tensor, cudnn_forced_execution_plan,
                cudnn_native_graph_execution_plan,
            },
            types::{CudnnBackendGraphTemplate, CudnnBackendOperation},
        },
        registry::ScaledDotProductAttentionAttributes,
    },
    data_type::DataType,
    error::{Error, Result},
    graph::{
        ComputePrecision, NumericalPolicy,
        ir::{Graph, ValueId},
    },
    layout::Layout,
    utility::to_i32,
};

pub const CUDNN_ATTENTION_QUERY_TENSOR_ID: CudnnTensorUid = CudnnTensorUid::new(1);

pub const CUDNN_ATTENTION_KEY_TENSOR_ID: CudnnTensorUid = CudnnTensorUid::new(2);

pub const CUDNN_ATTENTION_VALUE_TENSOR_ID: CudnnTensorUid = CudnnTensorUid::new(3);

pub const CUDNN_ATTENTION_OUTPUT_TENSOR_ID: CudnnTensorUid = CudnnTensorUid::new(4);

pub const CUDNN_ATTENTION_OFFSETS_TENSOR_ID: CudnnTensorUid = CudnnTensorUid::new(5);

pub const CUDNN_ATTENTION_QUERY_LENGTHS_TENSOR_ID: CudnnTensorUid = CudnnTensorUid::new(6);

pub const CUDNN_ATTENTION_KEY_LENGTHS_TENSOR_ID: CudnnTensorUid = CudnnTensorUid::new(7);

pub const CUDNN_ATTENTION_KEY_PAGES_TENSOR_ID: CudnnTensorUid = CudnnTensorUid::new(8);

pub const CUDNN_ATTENTION_VALUE_PAGES_TENSOR_ID: CudnnTensorUid = CudnnTensorUid::new(9);

pub const CUDNN_ATTENTION_STATS_TENSOR_ID: CudnnTensorUid = CudnnTensorUid::new(10);

const SCALE_TENSOR_ID: CudnnTensorUid = CudnnTensorUid::new(11);

const INFINITY_TENSOR_ID: CudnnTensorUid = CudnnTensorUid::new(12);

const WINDOW_TENSOR_ID: CudnnTensorUid = CudnnTensorUid::new(13);

const SCORES_TENSOR_ID: i64 = 14;

// Whether cuDNN can run attention of this type, head dimension, and options.
// Soft capping, masks, and sinks would need more of its score graph, so generated kernels keep them, and cuDNN planned nothing for a head dimension that is not a multiple of 8 up to 256, such as a tiny Llama's 12.
#[must_use]
pub fn cudnn_takes_attention(
    data_type: DataType,
    head_dimension: i64,
    attention: &ScaledDotProductAttentionAttributes,
) -> bool {
    matches!(data_type, DataType::F16 | DataType::Bf16)
        && head_dimension > 0
        && head_dimension % 8 == 0
        && head_dimension <= 256
        && attention.softcap.is_none()
        && !attention.masked
        && !attention.sinks
        && attention.window.is_none_or(|window| window > 0)
}

// Whether cuDNN reads `lengths`, an attention's key lengths, as they are: contiguous `I32` from their first element.
pub fn cudnn_reads_key_lengths(ir: &Graph, lengths: ValueId) -> Result<bool> {
    let layout = ir.required_value_layout(lengths)?;
    Ok(ir.required_value_data_type(lengths)? == DataType::I32
        && layout
            .addressing_strides()
            .iter()
            .all(|&stride| stride <= 1)
        && layout.storage_offset() == 0)
}

// Whether cuDNN can run a matmul of `lhs` and `rhs` of this type under the current numerical policy.
// It has no `F32` matmul that keeps full precision, refused `F64` matmuls of some shapes and every fused one, and planned none for an operand of equal rank below two or with a stride of zero, as a broadcast batch has, so those stay with the next library or generated kernels.
#[must_use]
pub fn cudnn_takes_matmul(data_type: DataType, lhs: &Layout, rhs: &Layout) -> bool {
    let typed = match data_type {
        DataType::F16 | DataType::Bf16 => true,
        DataType::F32 => NumericalPolicy::current().precision() != ComputePrecision::Full,
        _ => false,
    };
    let rank = lhs.dimensions().len();
    typed
        && rank >= 2
        && rank == rhs.dimensions().len()
        && [lhs, rhs]
            .iter()
            .all(|layout| layout.strides().iter().all(|&stride| stride > 0))
}

// The scale attention multiplies its scores by.
#[must_use]
#[expect(
    clippy::cast_precision_loss,
    reason = "head dimensions round to the score's precision"
)]
pub fn cudnn_attention_scale(attention: &ScaledDotProductAttentionAttributes, dim: i64) -> f32 {
    attention.scale.unwrap_or_else(|| 1.0 / (dim as f32).sqrt())
}

// The extents cuDNN's paged attention takes, and where each sits in `singe`'s `[T, H, D]` queries and `[P, S, G, D]` caches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudnnPaging {
    pub sequences: i64,
    pub tokens: i64,
    pub heads: i64,
    pub kv_heads: i64,
    pub dim: i64,
    pub page_size: i64,
    pub cache_pages: i64,
    // The longest list of pages a sequence can have.
    pub table: i64,
}

impl CudnnPaging {
    // Queries and outputs are `[B, H, S, D]` to cuDNN, each batch's tokens following the last's as they do in `[T, H, D]`.
    #[must_use]
    pub const fn packed_shape(&self) -> ([i64; 4], [i64; 4]) {
        let Self {
            sequences,
            tokens,
            heads,
            dim,
            ..
        } = *self;
        (
            [sequences, heads, tokens, dim],
            [tokens * heads * dim, dim, heads * dim, 1],
        )
    }

    // A cache is `[P, G, S, D]` to cuDNN, with its strides naming `singe`'s `[P, S, G, D]`.
    #[must_use]
    pub const fn cache_shape(&self) -> ([i64; 4], [i64; 4]) {
        let Self {
            kv_heads,
            dim,
            page_size,
            cache_pages,
            ..
        } = *self;
        (
            [cache_pages, kv_heads, page_size, dim],
            [page_size * kv_heads * dim, dim, kv_heads * dim, 1],
        )
    }

    #[must_use]
    pub const fn offsets_shape(&self) -> ([i64; 4], [i64; 4]) {
        ([self.sequences + 1, 1, 1, 1], [1, 1, 1, 1])
    }

    #[must_use]
    pub const fn lengths_shape(&self) -> ([i64; 4], [i64; 4]) {
        ([self.sequences, 1, 1, 1], [1, 1, 1, 1])
    }

    #[must_use]
    pub const fn table_shape(&self) -> ([i64; 4], [i64; 4]) {
        ([self.sequences, 1, self.table, 1], [self.table, 1, 1, 1])
    }

    // Statistics are `[B, H, S, 1]`, padded where a batch has fewer queries, and do not follow the packing.
    #[must_use]
    pub const fn stats_shape(&self) -> ([i64; 4], [i64; 4]) {
        let Self {
            sequences,
            tokens,
            heads,
            ..
        } = *self;
        (
            [sequences, heads, tokens, 1],
            [tokens * heads, tokens, 1, 1],
        )
    }

    #[must_use]
    pub const fn scores(&self) -> [i64; 4] {
        [
            self.sequences,
            self.heads,
            self.tokens,
            self.table * self.page_size,
        ]
    }

    // How far one query moves through the packed queries, in elements.
    #[must_use]
    pub const fn query_elements(&self) -> i64 {
        self.heads * self.dim
    }
}

// What cuDNN's attention reads and writes.
#[derive(Debug, Clone)]
pub struct CudnnAttentionProblem {
    pub query: CudnnBackendTensorMetadata,
    pub key: CudnnBackendTensorMetadata,
    pub value: CudnnBackendTensorMetadata,
    pub output: CudnnBackendTensorMetadata,
    pub causal: bool,
    pub window: Option<i64>,
    pub paging: Option<CudnnPaging>,
    // Whether each batch's count of keys, `[B]` of `I32`, hides the keys past it.
    pub key_lengths: bool,
    // The engine and knobs a record states, in place of what heuristics offer.
    pub engine: Option<EngineChoice>,
}

// The scalars attention's graph reads by value, as their bits, since cuDNN only reads its masks' as bound values.
pub fn cudnn_attention_scalars(
    scale: f32,
    causal: bool,
    window: Option<i64>,
    paged: bool,
) -> Result<Vec<(CudnnTensorUid, u32)>> {
    let mut scalars = vec![(SCALE_TENSOR_ID, scale.to_bits())];
    if causal || window.is_some() {
        scalars.push((INFINITY_TENSOR_ID, f32::NEG_INFINITY.to_bits()));
    }

    if let Some(window) = window {
        // A mask aligned to each batch's lengths counts in integers, and one aligned to the top left in floats.
        let bits = if paged {
            to_i32(window, "cudnn attention window")?.cast_unsigned()
        } else {
            #[expect(
                clippy::cast_precision_loss,
                reason = "windows are far below a float's exact integers"
            )]
            (window as f32).to_bits()
        };
        scalars.push((WINDOW_TENSOR_ID, bits));
    }
    Ok(scalars)
}

fn by_value(
    id: CudnnTensorUid,
    data_type: CudnnDataType,
    rank: usize,
) -> Result<CudnnBackendTensor> {
    let unit = vec![1; rank];
    Ok(CudnnBackendTensor::create_by_value(
        id, data_type, &unit, &unit, 4,
    )?)
}

fn indices(
    id: CudnnTensorUid,
    (dimensions, strides): ([i64; 4], [i64; 4]),
) -> Result<CudnnBackendTensor> {
    Ok(CudnnBackendTensor::create_with_id(
        id,
        CudnnDataType::I32,
        &dimensions,
        &strides,
        4,
    )?)
}

fn packed(
    id: CudnnTensorUid,
    metadata: &CudnnBackendTensorMetadata,
    offsets: &CudnnBackendTensor,
) -> Result<CudnnBackendTensor> {
    let shape = CudnnShape::with_strides(metadata.dimensions(), metadata.strides())?;
    let elements =
        usize::try_from(shape.dimensions()[1] * shape.dimensions()[2] * shape.dimensions()[3])
            .map_err(|_| Error::DimensionOutOfBounds)?;
    let mut spec = TensorSpec::new(storage_type(metadata.data_type())?, shape)
        .with_byte_alignment(metadata.byte_alignment())?
        .with_ragged_byte_capacity(elements * metadata.data_type().size_bytes());
    spec.ragged_offset = Some(TensorId::generate());
    Ok(CudnnBackendTensor::create_with_spec(
        id,
        &spec,
        Some(offsets),
    )?)
}

// The masks that hide scores from queries, in their own operation graph.
#[derive(Debug, Default)]
struct ScoreMasks {
    tensors: Vec<CudnnBackendTensor>,
    operations: Vec<DiagonalBandMaskOperation>,
    graph: Option<CudnnOperationGraph>,
    last: i64,
}

impl ScoreMasks {
    fn create(
        cudnn: &CudnnContext,
        scores: [i64; 4],
        causal: bool,
        window: Option<i64>,
        lengths: Option<(&CudnnBackendTensor, &CudnnBackendTensor)>,
    ) -> Result<Self> {
        let mut masks = Self {
            last: SCORES_TENSOR_ID,
            ..Self::default()
        };
        if !causal && window.is_none() {
            return Ok(masks);
        }

        let score = |id: i64| -> Result<CudnnBackendTensor> {
            Ok(CudnnBackendTensor::create_with_spec(
                CudnnTensorUid::new(id),
                &TensorSpec::new(CudnnDataType::F32, CudnnShape::contiguous(scores)?)
                    .virtual_tensor(),
                None,
            )?)
        };
        let sequence = lengths.map_or_else(AttentionSequenceInputs::default, |(query, key)| {
            AttentionSequenceInputs {
                query: Some(SequenceLength::PerBatch(query)),
                key_value: Some(SequenceLength::PerBatch(key)),
            }
        });
        let infinity = by_value(INFINITY_TENSOR_ID, CudnnDataType::F32, 4)?;
        let mut current = score(SCORES_TENSOR_ID)?;
        let mut next = SCORES_TENSOR_ID + 1;
        let mut tensors = Vec::new();
        if causal {
            let hidden = score(next)?;
            masks.operations.push(DiagonalBandMaskOperation::create(
                &current,
                &infinity,
                &hidden,
                DiagonalBandMaskConfig::new(PointwiseMode::CmpGe).with_sequence_inputs(sequence),
            )?);
            tensors.push(current);
            current = hidden;
            masks.last = next;
            next += 1;
        }

        if window.is_some() {
            let data_type = if lengths.is_some() {
                CudnnDataType::I32
            } else {
                CudnnDataType::F32
            };
            let bound = by_value(WINDOW_TENSOR_ID, data_type, 4)?;
            let hidden = score(next)?;
            masks.operations.push(DiagonalBandMaskOperation::create(
                &current,
                &infinity,
                &hidden,
                DiagonalBandMaskConfig::new(PointwiseMode::CmpGt)
                    .with_sequence_inputs(sequence)
                    .with_bound(DiagonalBandMaskBound::Left(&bound)),
            )?);
            tensors.push(current);
            tensors.push(bound);
            current = hidden;
            masks.last = next;
        }

        tensors.push(current);
        tensors.push(infinity);
        masks.tensors = tensors;
        let descriptors = masks
            .operations
            .iter()
            .map(DiagonalBandMaskOperation::descriptor)
            .collect::<Vec<_>>();
        masks.graph = Some(CudnnOperationGraph::create(
            cudnn,
            &descriptors,
            CudnnOperationGraphConfig::default(),
        )?);
        Ok(masks)
    }

    fn subgraph(&self) -> Option<SdpaSubgraph<'_>> {
        self.graph.as_ref().map(|graph| SdpaSubgraph {
            operation_graph: graph,
            input_id: CudnnTensorUid::new(SCORES_TENSOR_ID),
            output_id: CudnnTensorUid::new(self.last),
        })
    }
}

// The tensors only paged attention has.
struct PagedTensors {
    offsets: CudnnBackendTensor,
    query_lengths: CudnnBackendTensor,
    key_lengths: CudnnBackendTensor,
    key_pages: CudnnBackendTensor,
    value_pages: CudnnBackendTensor,
    stats: CudnnBackendTensor,
}

impl PagedTensors {
    fn create(paging: &CudnnPaging) -> Result<Self> {
        let (stats_dimensions, stats_strides) = paging.stats_shape();
        Ok(Self {
            offsets: indices(CUDNN_ATTENTION_OFFSETS_TENSOR_ID, paging.offsets_shape())?,
            query_lengths: indices(
                CUDNN_ATTENTION_QUERY_LENGTHS_TENSOR_ID,
                paging.lengths_shape(),
            )?,
            key_lengths: indices(
                CUDNN_ATTENTION_KEY_LENGTHS_TENSOR_ID,
                paging.lengths_shape(),
            )?,
            key_pages: indices(CUDNN_ATTENTION_KEY_PAGES_TENSOR_ID, paging.table_shape())?,
            value_pages: indices(CUDNN_ATTENTION_VALUE_PAGES_TENSOR_ID, paging.table_shape())?,
            stats: CudnnBackendTensor::create_with_id(
                CUDNN_ATTENTION_STATS_TENSOR_ID,
                CudnnDataType::F32,
                &stats_dimensions,
                &stats_strides,
                4,
            )?,
        })
    }

    fn into_tensors(self) -> [CudnnBackendTensor; 6] {
        [
            self.offsets,
            self.query_lengths,
            self.key_lengths,
            self.key_pages,
            self.value_pages,
            self.stats,
        ]
    }
}

// Builds a cuDNN attention template, whose scores are masked in a graph of their own.
//
// # Errors
//
// Returns an error if cuDNN refuses the tensors, the masks, or the attention, or no engine plans them under the numerical policy.
pub fn build_cudnn_attention_template(
    cudnn: &CudnnContext,
    problem: &CudnnAttentionProblem,
) -> Result<CudnnBackendGraphTemplate> {
    // cuDNN reports attention it does not take as bad parameters, which `cudnn_takes_attention` should have kept from it.
    build_template(cudnn, problem).map_err(|error| match &error {
        Error::Cudnn(cudnn_error) if cudnn_error.status() == Some(CudnnStatus::BadParam) => {
            Error::library_refused("cuDNN", error)
        }
        _ => error,
    })
}

fn build_template(
    cudnn: &CudnnContext,
    problem: &CudnnAttentionProblem,
) -> Result<CudnnBackendGraphTemplate> {
    let paged = problem
        .paging
        .as_ref()
        .map(PagedTensors::create)
        .transpose()?;
    let (query, output) = if let Some(paged) = &paged {
        (
            packed(
                CUDNN_ATTENTION_QUERY_TENSOR_ID,
                &problem.query,
                &paged.offsets,
            )?,
            packed(
                CUDNN_ATTENTION_OUTPUT_TENSOR_ID,
                &problem.output,
                &paged.offsets,
            )?,
        )
    } else {
        (
            cudnn_backend_tensor(CUDNN_ATTENTION_QUERY_TENSOR_ID, &problem.query)?,
            cudnn_backend_tensor(CUDNN_ATTENTION_OUTPUT_TENSOR_ID, &problem.output)?,
        )
    };
    let key = cudnn_backend_tensor(CUDNN_ATTENTION_KEY_TENSOR_ID, &problem.key)?;
    let value = cudnn_backend_tensor(CUDNN_ATTENTION_VALUE_TENSOR_ID, &problem.value)?;
    // Dense attention's key lengths are `[B]` to `singe`, and `[B, 1, 1, 1]` to cuDNN, in one buffer, and cuDNN reads the query lengths with them, each the queries' count.
    let lengths = (problem.key_lengths && paged.is_none())
        .then(|| {
            let shape = ([problem.query.dimensions()[0], 1, 1, 1], [1, 1, 1, 1]);
            Result::Ok((
                indices(CUDNN_ATTENTION_QUERY_LENGTHS_TENSOR_ID, shape)?,
                indices(CUDNN_ATTENTION_KEY_LENGTHS_TENSOR_ID, shape)?,
            ))
        })
        .transpose()?;
    let scale = by_value(SCALE_TENSOR_ID, CudnnDataType::F32, 1)?;
    let query_dimensions = problem.query.dimensions();
    let scores = problem.paging.as_ref().map_or_else(
        || {
            [
                query_dimensions[0],
                query_dimensions[1],
                query_dimensions[2],
                problem.key.dimensions()[2],
            ]
        },
        CudnnPaging::scores,
    );
    let masks = ScoreMasks::create(
        cudnn,
        scores,
        problem.causal,
        problem.window,
        paged
            .as_ref()
            .map(|paged| (&paged.query_lengths, &paged.key_lengths)),
    )?;
    let dense_sequence =
        lengths
            .as_ref()
            .map_or_else(AttentionSequenceInputs::default, |(query, key)| {
                AttentionSequenceInputs {
                    query: Some(SequenceLength::PerBatch(query)),
                    key_value: Some(SequenceLength::PerBatch(key)),
                }
            });
    let attention = SdpaForwardOperation::create(
        SdpaForwardTensors {
            query: &query,
            key: &key,
            value: &value,
            output: &output,
            scale: &scale,
        },
        SdpaForwardConfig {
            sequence: paged
                .as_ref()
                .map_or(dense_sequence, |paged| AttentionSequenceInputs {
                    query: Some(SequenceLength::PerBatch(&paged.query_lengths)),
                    key_value: Some(SequenceLength::PerBatch(&paged.key_lengths)),
                }),
            page_table_k: paged.as_ref().map(|paged| &paged.key_pages),
            page_table_v: paged.as_ref().map(|paged| &paged.value_pages),
            softmax: SdpaForwardSoftmax::Default {
                stats: paged.as_ref().map(|paged| &paged.stats),
            },
            subgraph: masks.subgraph(),
            ..SdpaForwardConfig::default()
        },
    )?;
    let operation_graph = CudnnOperationGraph::create(
        cudnn,
        &[attention.descriptor()],
        CudnnOperationGraphConfig::default(),
    )?;
    let plan = match &problem.engine {
        Some(choice) => cudnn_forced_execution_plan(
            cudnn,
            &operation_graph,
            problem.query.data_type(),
            false,
            choice,
        )?,
        None => {
            cudnn_native_graph_execution_plan(cudnn, &operation_graph, problem.query.data_type())?
        }
    };
    let workspace_size = plan.workspace_size()?;
    let ScoreMasks {
        tensors: mask_tensors,
        operations,
        graph,
        ..
    } = masks;
    let mut tensors = vec![query, key, value, output, scale];
    if let Some((query_lengths, key_lengths)) = lengths {
        tensors.push(query_lengths);
        tensors.push(key_lengths);
    }

    tensors.extend(mask_tensors);
    if let Some(paged) = paged {
        tensors.extend(paged.into_tensors());
    }
    Ok(CudnnBackendGraphTemplate::new(
        plan,
        workspace_size,
        CudnnBackendOperation::attention(tensors, operations, graph, attention),
        operation_graph,
    ))
}
