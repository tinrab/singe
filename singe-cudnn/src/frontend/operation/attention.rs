use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{
    data_type::DataType,
    error::{Error, Result},
    execution::advanced::{
        DiagonalBandMaskOperation, SdpaBackwardConfig, SdpaBackwardOperation, SdpaForwardConfig,
        SdpaForwardOperation, SoftmaxOperation,
    },
    frontend::{
        graph::Graph,
        lower::{
            LoweredOperation, LoweredSdpaSubgraph, LoweringContext, optional_tensor_at, tensor_at,
        },
        operation::FrontendOperationTensors,
    },
    math::NanPropagation,
    pointwise::PointwiseMode,
    tensor::{Tensor, TensorId},
};

/// Source for Philox RNG dropout seed data used by attention.
///
/// The cuDNN frontend SDPA API supports dropout configured with a probability,
/// seed tensor, and offset tensor. This Rust enum also allows a host seed for
/// helper paths that materialize the seed internally.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum AttentionDropoutSeedSource {
    Host(i64),
    Device(TensorId),
}

/// Dropout configuration for scaled dot product attention.
///
/// cuDNN SDPA can apply dropout after softmax using Philox RNG seed/offset
/// inputs, or a custom mask and scale through score modifiers. This config
/// represents the Philox-style probability and seed/offset path.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AttentionDropoutConfig {
    probability: f32,
    seed: AttentionDropoutSeedSource,
    offset: Option<TensorId>,
}

impl AttentionDropoutConfig {
    pub fn new(probability: f32, seed: i64) -> Self {
        Self {
            probability,
            seed: AttentionDropoutSeedSource::Host(seed),
            offset: None,
        }
    }

    pub fn with_seed_tensor(mut self, seed: TensorId) -> Self {
        self.seed = AttentionDropoutSeedSource::Device(seed);
        self
    }

    pub fn with_offset(mut self, offset: TensorId) -> Self {
        self.offset = Some(offset);
        self
    }

    pub fn probability(&self) -> f32 {
        self.probability
    }

    pub fn seed(&self) -> Option<i64> {
        match self.seed {
            AttentionDropoutSeedSource::Host(seed) => Some(seed),
            AttentionDropoutSeedSource::Device(_) => None,
        }
    }

    pub fn seed_tensor(&self) -> Option<TensorId> {
        match self.seed {
            AttentionDropoutSeedSource::Host(_) => None,
            AttentionDropoutSeedSource::Device(seed) => Some(seed),
        }
    }

    pub fn seed_source(&self) -> AttentionDropoutSeedSource {
        self.seed
    }

    pub fn offset(&self) -> Option<TensorId> {
        self.offset
    }
}

/// SDPA implementation selector.
///
/// The frontend SDPA operation supports `AUTO`, `COMPOSITE`, and `UNIFIED`
/// implementation choices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AttentionImplementation {
    Auto,
    Unified,
    Composite,
}

/// Mutually exclusive structural mask mode for high-level SDPA helpers.
///
/// This covers the masking modes that are encoded as independent booleans in
/// cuDNN frontend attributes while keeping Rust call sites from accidentally
/// combining incompatible causal alignments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum AttentionMaskMode {
    None,
    SequenceLengths {
        sequence_length_query: TensorId,
        sequence_length_key_value: TensorId,
    },
    Padding {
        sequence_length_query: TensorId,
        sequence_length_key_value: TensorId,
    },
    CausalTopLeft,
    CausalTopLeftWithPadding {
        sequence_length_query: TensorId,
        sequence_length_key_value: TensorId,
    },
    CausalBottomRight {
        sequence_length_query: TensorId,
        sequence_length_key_value: TensorId,
    },
    SlidingWindow {
        left_window: i64,
        right_window: i64,
    },
    SlidingWindowWithPadding {
        sequence_length_query: TensorId,
        sequence_length_key_value: TensorId,
        left_window: i64,
        right_window: i64,
    },
    Block(TensorId),
}

/// Dropout mode for high-level SDPA helpers.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum AttentionDropoutMode {
    None,
    Philox(AttentionDropoutConfig),
    CustomMask { mask: TensorId, scale: TensorId },
}

/// Variable-length sequence mode for direct cuDNN SDPA operations.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum DirectSdpaSequenceMode {
    #[default]
    None,
    SequenceLengths {
        sequence_length_query: TensorId,
        sequence_length_key_value: TensorId,
    },
    PaddingMask {
        sequence_length_query: TensorId,
        sequence_length_key_value: TensorId,
    },
}

impl DirectSdpaSequenceMode {
    pub fn sequence_lengths(
        sequence_length_query: TensorId,
        sequence_length_key_value: TensorId,
    ) -> Self {
        Self::SequenceLengths {
            sequence_length_query,
            sequence_length_key_value,
        }
    }

    pub fn padding_mask(
        sequence_length_query: TensorId,
        sequence_length_key_value: TensorId,
    ) -> Self {
        Self::PaddingMask {
            sequence_length_query,
            sequence_length_key_value,
        }
    }

    pub fn sequence_length_query(self) -> Option<TensorId> {
        match self {
            Self::SequenceLengths {
                sequence_length_query,
                ..
            }
            | Self::PaddingMask {
                sequence_length_query,
                ..
            } => Some(sequence_length_query),
            Self::None => None,
        }
    }

    pub fn sequence_length_key_value(self) -> Option<TensorId> {
        match self {
            Self::SequenceLengths {
                sequence_length_key_value,
                ..
            }
            | Self::PaddingMask {
                sequence_length_key_value,
                ..
            } => Some(sequence_length_key_value),
            Self::None => None,
        }
    }

    pub fn sequence_lengths_pair(self) -> Option<(TensorId, TensorId)> {
        match self {
            Self::SequenceLengths {
                sequence_length_query,
                sequence_length_key_value,
            }
            | Self::PaddingMask {
                sequence_length_query,
                sequence_length_key_value,
            } => Some((sequence_length_query, sequence_length_key_value)),
            Self::None => None,
        }
    }

    pub fn is_padding_mask(self) -> bool {
        matches!(self, Self::PaddingMask { .. })
    }
}

/// Options for the direct cuDNN SDPA forward operation.
///
/// Covers variable-length sequence tensors, padding mask, paged attention page
/// tables, block masks, dropout, random-number-generator dump, and the
/// unfused-FMA option described for frontend SDPA attributes.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DirectSdpaForwardConfig {
    sequence: DirectSdpaSequenceMode,
    page_table_k: Option<TensorId>,
    page_table_v: Option<TensorId>,
    block_mask: Option<TensorId>,
    dropout: Option<DirectSdpaDropout>,
    unfuse_fma: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DirectSdpaDropout {
    config: AttentionDropoutConfig,
    seed: Option<TensorId>,
    random_number_generator_dump: Option<TensorId>,
}

impl DirectSdpaDropout {
    pub fn new(config: AttentionDropoutConfig) -> Self {
        Self {
            config,
            seed: None,
            random_number_generator_dump: None,
        }
    }

    pub fn with_seed(mut self, seed: TensorId) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn with_random_number_generator_dump(
        mut self,
        random_number_generator_dump: TensorId,
    ) -> Self {
        self.random_number_generator_dump = Some(random_number_generator_dump);
        self
    }

    pub fn config(self) -> AttentionDropoutConfig {
        self.config
    }

    pub fn seed(self) -> Option<TensorId> {
        self.seed
    }

    pub fn random_number_generator_dump(self) -> Option<TensorId> {
        self.random_number_generator_dump
    }
}

impl DirectSdpaForwardConfig {
    pub fn new() -> Self {
        Self {
            sequence: DirectSdpaSequenceMode::None,
            page_table_k: None,
            page_table_v: None,
            block_mask: None,
            dropout: None,
            unfuse_fma: false,
        }
    }

    pub fn with_sequence_mode(mut self, sequence: DirectSdpaSequenceMode) -> Self {
        self.sequence = sequence;
        self
    }

    pub fn with_page_table_k(mut self, page_table_k: TensorId) -> Self {
        self.page_table_k = Some(page_table_k);
        self
    }

    pub fn with_page_table_v(mut self, page_table_v: TensorId) -> Self {
        self.page_table_v = Some(page_table_v);
        self
    }

    pub fn with_block_mask(mut self, block_mask: TensorId) -> Self {
        self.block_mask = Some(block_mask);
        self
    }

    pub fn with_dropout(mut self, dropout: DirectSdpaDropout) -> Self {
        self.dropout = Some(dropout);
        self
    }

    pub fn with_unfused_fma(mut self) -> Self {
        self.unfuse_fma = true;
        self
    }

    pub fn sequence_length_query(&self) -> Option<TensorId> {
        self.sequence.sequence_length_query()
    }

    pub fn sequence_length_key_value(&self) -> Option<TensorId> {
        self.sequence.sequence_length_key_value()
    }

    pub fn sequence_mode(&self) -> DirectSdpaSequenceMode {
        self.sequence
    }

    pub fn padding_mask(&self) -> bool {
        self.sequence.is_padding_mask()
    }

    pub fn page_table_k(&self) -> Option<TensorId> {
        self.page_table_k
    }

    pub fn page_table_v(&self) -> Option<TensorId> {
        self.page_table_v
    }

    pub fn block_mask(&self) -> Option<TensorId> {
        self.block_mask
    }

    pub fn dropout(&self) -> Option<DirectSdpaDropout> {
        self.dropout
    }

    pub fn unfuse_fma(&self) -> bool {
        self.unfuse_fma
    }
}

impl Default for DirectSdpaForwardConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Options for the direct cuDNN SDPA backward operation.
///
/// Covers variable-length sequences, padding/causal masks, score subgraphs,
/// sink gradients, and maximum total sequence lengths used by ragged THD
/// layouts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectSdpaBackwardConfig {
    sequence: DirectSdpaSequenceMode,
    causal_mask: bool,
    score_subgraph: Option<SdpaScoreSubgraph>,
    sink: Option<TensorId>,
    sink_gradient: Option<TensorId>,
    max_total_sequence_lengths: Option<DirectSdpaMaxTotalSequenceLengths>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DirectSdpaMaxTotalSequenceLengths {
    query: i64,
    key_value: i64,
}

impl DirectSdpaMaxTotalSequenceLengths {
    pub fn new(query: i64, key_value: i64) -> Self {
        Self { query, key_value }
    }

    pub fn query(self) -> i64 {
        self.query
    }

    pub fn key_value(self) -> i64 {
        self.key_value
    }
}

impl DirectSdpaBackwardConfig {
    pub fn new() -> Self {
        Self {
            sequence: DirectSdpaSequenceMode::None,
            causal_mask: false,
            score_subgraph: None,
            sink: None,
            sink_gradient: None,
            max_total_sequence_lengths: None,
        }
    }

    pub fn with_sequence_mode(mut self, sequence: DirectSdpaSequenceMode) -> Self {
        self.sequence = sequence;
        self
    }

    pub fn with_causal_mask(mut self) -> Self {
        self.causal_mask = true;
        self
    }

    pub fn with_score_subgraph(mut self, score_subgraph: SdpaScoreSubgraph) -> Self {
        self.score_subgraph = Some(score_subgraph);
        self
    }

    pub fn with_sink(mut self, sink: TensorId) -> Self {
        self.sink = Some(sink);
        self
    }

    pub fn with_sink_gradient(mut self, sink_gradient: TensorId) -> Self {
        self.sink_gradient = Some(sink_gradient);
        self
    }

    pub fn with_max_total_sequence_lengths(
        mut self,
        max_total_sequence_lengths: DirectSdpaMaxTotalSequenceLengths,
    ) -> Self {
        self.max_total_sequence_lengths = Some(max_total_sequence_lengths);
        self
    }

    pub fn sequence_length_query(&self) -> Option<TensorId> {
        self.sequence.sequence_length_query()
    }

    pub fn sequence_length_key_value(&self) -> Option<TensorId> {
        self.sequence.sequence_length_key_value()
    }

    pub fn sequence_mode(&self) -> DirectSdpaSequenceMode {
        self.sequence
    }

    pub fn padding_mask(&self) -> bool {
        self.sequence.is_padding_mask()
    }

    pub fn causal_mask(&self) -> bool {
        self.causal_mask
    }

    pub fn score_subgraph(&self) -> Option<&SdpaScoreSubgraph> {
        self.score_subgraph.as_ref()
    }

    pub fn sink(&self) -> Option<TensorId> {
        self.sink
    }

    pub fn sink_gradient(&self) -> Option<TensorId> {
        self.sink_gradient
    }

    pub fn max_total_sequence_lengths(&self) -> Option<DirectSdpaMaxTotalSequenceLengths> {
        self.max_total_sequence_lengths
    }
}

impl Default for DirectSdpaBackwardConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Frontend scaled dot product attention operation variants.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum SdpaOperation {
    Forward {
        q: TensorId,
        k: TensorId,
        v: TensorId,
        o: TensorId,
        scale: TensorId,
        stats: Option<TensorId>,
        logit_max: Option<TensorId>,
        score_sum_exp: Option<TensorId>,
        softmax_p: Option<TensorId>,
        softmax_s: Option<TensorId>,
        sink: Option<TensorId>,
        score_modifiers: SdpaScoreModifiers,
        score_subgraph: Box<Option<SdpaScoreSubgraph>>,
        config: Box<DirectSdpaForwardConfig>,
    },
    Backward {
        q: TensorId,
        k: TensorId,
        v: TensorId,
        o: TensorId,
        stats: TensorId,
        scale: TensorId,
        d_o: TensorId,
        d_q: TensorId,
        d_k: TensorId,
        d_v: TensorId,
        sink_gradient: Option<TensorId>,
        config: Box<DirectSdpaBackwardConfig>,
    },
}

impl FrontendOperationTensors for SdpaOperation {
    fn append_tensor_ids(&self, tensors: &mut Vec<TensorId>) {
        match self {
            Self::Forward {
                q,
                k,
                v,
                o,
                scale,
                stats,
                logit_max,
                score_sum_exp,
                softmax_p,
                softmax_s,
                sink,
                score_modifiers,
                score_subgraph: _,
                config,
            } => {
                tensors.extend([*q, *k, *v, *o, *scale]);
                tensors.extend(*stats);
                tensors.extend(*logit_max);
                tensors.extend(*score_sum_exp);
                tensors.extend(*softmax_p);
                tensors.extend(*softmax_s);
                tensors.extend(*sink);
                tensors.extend(score_modifiers.bias());
                tensors.extend(score_modifiers.alibi_slopes());
                tensors.extend(score_modifiers.additive_mask());
                tensors.extend(score_modifiers.dropout_mask());
                tensors.extend(score_modifiers.dropout_scale());
                tensors.extend(config.sequence_length_query());
                tensors.extend(config.sequence_length_key_value());
                tensors.extend(config.page_table_k());
                tensors.extend(config.page_table_v());
                tensors.extend(config.block_mask());
                tensors.extend(config.dropout().and_then(DirectSdpaDropout::seed));
                tensors.extend(
                    config
                        .dropout()
                        .and_then(|dropout| dropout.config().offset()),
                );
                tensors.extend(
                    config
                        .dropout()
                        .and_then(DirectSdpaDropout::random_number_generator_dump),
                );
            }
            Self::Backward {
                q,
                k,
                v,
                o,
                stats,
                scale,
                d_o,
                d_q,
                d_k,
                d_v,
                sink_gradient,
                config,
            } => {
                tensors.extend([*q, *k, *v, *o, *stats, *scale, *d_o, *d_q, *d_k, *d_v]);
                tensors.extend(config.sequence_length_query());
                tensors.extend(config.sequence_length_key_value());
                tensors.extend(config.sink());
                tensors.extend(*sink_gradient);
            }
        }
    }
}

impl SdpaOperation {
    pub(crate) fn lower(
        &self,
        context: &LoweringContext<'_>,
        subgraph: Option<LoweredSdpaSubgraph<'_>>,
    ) -> Result<LoweredOperation> {
        self.lower_with_tensors(context.backend_tensors(), subgraph)
    }

    pub(crate) fn lower_with_tensors(
        &self,
        tensors: &BTreeMap<TensorId, Tensor>,
        subgraph: Option<LoweredSdpaSubgraph<'_>>,
    ) -> Result<LoweredOperation> {
        match self {
            Self::Forward {
                q,
                k,
                v,
                o,
                scale,
                stats,
                logit_max,
                score_sum_exp,
                softmax_p,
                softmax_s,
                sink,
                score_modifiers: _,
                score_subgraph: _,
                config,
            } => {
                let softmax = match (*softmax_p, *softmax_s) {
                    (Some(softmax_p), Some(softmax_s)) => Some(SoftmaxOperation::create(
                        tensor_at(tensors, softmax_p)?,
                        tensor_at(tensors, softmax_s)?,
                        optional_tensor_at(tensors, *stats)?,
                        optional_tensor_at(tensors, *logit_max)?,
                        optional_tensor_at(tensors, *score_sum_exp)?,
                        optional_tensor_at(tensors, *sink)?,
                    )?),
                    (None, None) => None,
                    _ => {
                        return Err(Error::FrontendSdpaUnifiedSoftmaxDescriptorsIncomplete);
                    }
                };

                let (subgraph, subgraph_input_id, subgraph_output_id) =
                    if let Some(subgraph) = subgraph {
                        (
                            Some(subgraph.operation_graph),
                            Some(subgraph.input_id),
                            Some(subgraph.output_id),
                        )
                    } else {
                        (None, None, None)
                    };

                let dropout = config.dropout();

                Ok(LoweredOperation::SdpaForward(SdpaForwardOperation::create(
                    tensor_at(tensors, *q)?,
                    tensor_at(tensors, *k)?,
                    tensor_at(tensors, *v)?,
                    tensor_at(tensors, *o)?,
                    tensor_at(tensors, *scale)?,
                    optional_tensor_at(tensors, *stats)?,
                    SdpaForwardConfig {
                        sequence_length_query: config
                            .sequence_length_query()
                            .map(|tensor| tensor_at(tensors, tensor))
                            .transpose()?,
                        sequence_length_key_value: config
                            .sequence_length_key_value()
                            .map(|tensor| tensor_at(tensors, tensor))
                            .transpose()?,
                        page_table_k: config
                            .page_table_k()
                            .map(|tensor| tensor_at(tensors, tensor))
                            .transpose()?,
                        page_table_v: config
                            .page_table_v()
                            .map(|tensor| tensor_at(tensors, tensor))
                            .transpose()?,
                        block_mask: config
                            .block_mask()
                            .map(|tensor| tensor_at(tensors, tensor))
                            .transpose()?,
                        softmax: softmax.as_ref(),
                        subgraph,
                        subgraph_input_id,
                        subgraph_output_id,
                        dropout_seed: dropout
                            .and_then(DirectSdpaDropout::seed)
                            .map(|tensor| tensor_at(tensors, tensor))
                            .transpose()?,
                        dropout_offset: dropout
                            .and_then(|dropout| dropout.config().offset())
                            .map(|tensor| tensor_at(tensors, tensor))
                            .transpose()?,
                        dropout_random_number_generator_dump: dropout
                            .and_then(DirectSdpaDropout::random_number_generator_dump)
                            .map(|tensor| tensor_at(tensors, tensor))
                            .transpose()?,
                        dropout_probability: dropout.map(|dropout| dropout.config().probability()),
                        unfuse_fma: config.unfuse_fma(),
                    },
                )?))
            }
            Self::Backward {
                q,
                k,
                v,
                o,
                stats,
                scale,
                d_o,
                d_q,
                d_k,
                d_v,
                sink_gradient,
                config,
            } => {
                let (subgraph, subgraph_input_id, subgraph_output_id) =
                    if let Some(subgraph) = subgraph {
                        (
                            Some(subgraph.operation_graph),
                            Some(subgraph.input_id),
                            Some(subgraph.output_id),
                        )
                    } else {
                        (None, None, None)
                    };

                let max_total_sequence_lengths = config.max_total_sequence_lengths();

                Ok(LoweredOperation::SdpaBackward(
                    SdpaBackwardOperation::create(
                        tensor_at(tensors, *q)?,
                        tensor_at(tensors, *k)?,
                        tensor_at(tensors, *v)?,
                        tensor_at(tensors, *o)?,
                        tensor_at(tensors, *stats)?,
                        tensor_at(tensors, *scale)?,
                        tensor_at(tensors, *d_o)?,
                        tensor_at(tensors, *d_q)?,
                        tensor_at(tensors, *d_k)?,
                        tensor_at(tensors, *d_v)?,
                        SdpaBackwardConfig {
                            sequence_length_query: config
                                .sequence_length_query()
                                .map(|tensor| tensor_at(tensors, tensor))
                                .transpose()?,
                            sequence_length_key_value: config
                                .sequence_length_key_value()
                                .map(|tensor| tensor_at(tensors, tensor))
                                .transpose()?,
                            subgraph,
                            subgraph_input_id,
                            subgraph_output_id,
                            sink: config
                                .sink()
                                .map(|tensor| tensor_at(tensors, tensor))
                                .transpose()?,
                            sink_gradient: sink_gradient
                                .map(|tensor| tensor_at(tensors, tensor))
                                .transpose()?,
                            max_total_sequence_length_query: max_total_sequence_lengths
                                .map(DirectSdpaMaxTotalSequenceLengths::query),
                            max_total_sequence_length_key_value: max_total_sequence_lengths
                                .map(DirectSdpaMaxTotalSequenceLengths::key_value),
                        },
                    )?,
                ))
            }
        }
    }
}

/// Attributes for a standalone softmax operation in the frontend graph.
///
/// SDPA uses softmax over attention scores and may emit statistics needed by
/// backward training computation. This config stores the compute type and NaN
/// propagation for the Rust softmax helper.
#[derive(Debug, Clone, Copy)]
pub struct SoftmaxConfig {
    compute_type: DataType,
    nan_propagation: NanPropagation,
}

impl SoftmaxConfig {
    pub fn new(compute_type: DataType) -> Self {
        Self {
            compute_type,
            nan_propagation: NanPropagation::Propagate,
        }
    }

    pub fn with_nan_propagation(mut self, nan_propagation: NanPropagation) -> Self {
        self.nan_propagation = nan_propagation;
        self
    }

    pub fn compute_type(&self) -> DataType {
        self.compute_type
    }

    pub fn nan_propagation(&self) -> NanPropagation {
        self.nan_propagation
    }
}

/// Optional side-output tensors for a frontend softmax node.
///
/// These correspond to SDPA softmax statistics, logit max, and score sum-exp
/// outputs documented in the attention API.
#[derive(Debug, Clone, Copy)]
pub struct SoftmaxOperationConfig {
    stats: Option<TensorId>,
    max: Option<TensorId>,
    sum_exp: Option<TensorId>,
    sink: Option<TensorId>,
}

impl SoftmaxOperationConfig {
    pub fn new() -> Self {
        Self {
            stats: None,
            max: None,
            sum_exp: None,
            sink: None,
        }
    }

    pub fn with_stats(mut self, stats: TensorId) -> Self {
        self.stats = Some(stats);
        self
    }

    pub fn with_max(mut self, max: TensorId) -> Self {
        self.max = Some(max);
        self
    }

    pub fn with_sum_exp(mut self, sum_exp: TensorId) -> Self {
        self.sum_exp = Some(sum_exp);
        self
    }

    pub fn with_sink(mut self, sink: TensorId) -> Self {
        self.sink = Some(sink);
        self
    }

    pub fn stats(&self) -> Option<TensorId> {
        self.stats
    }

    pub fn max(&self) -> Option<TensorId> {
        self.max
    }

    pub fn sum_exp(&self) -> Option<TensorId> {
        self.sum_exp
    }

    pub fn sink(&self) -> Option<TensorId> {
        self.sink
    }
}

impl Default for SoftmaxOperationConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Frontend attention primitive operation variants.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum AttentionPrimitiveOperation {
    DiagonalBandMask {
        x: TensorId,
        b: TensorId,
        y: TensorId,
        comparison_mode: PointwiseMode,
        sequence_length_query: Option<TensorId>,
        sequence_length_key_value: Option<TensorId>,
        left_bound: Option<TensorId>,
        shift_right_bound: Option<TensorId>,
    },
    Softmax {
        x: TensorId,
        y: TensorId,
        stats: Option<TensorId>,
        max: Option<TensorId>,
        sum_exp: Option<TensorId>,
        sink: Option<TensorId>,
    },
}

impl FrontendOperationTensors for AttentionPrimitiveOperation {
    fn append_tensor_ids(&self, tensors: &mut Vec<TensorId>) {
        match self {
            Self::DiagonalBandMask {
                x,
                b,
                y,
                sequence_length_query,
                sequence_length_key_value,
                left_bound,
                shift_right_bound,
                ..
            } => {
                tensors.extend([*x, *b, *y]);
                tensors.extend(*sequence_length_query);
                tensors.extend(*sequence_length_key_value);
                tensors.extend(*left_bound);
                tensors.extend(*shift_right_bound);
            }
            Self::Softmax {
                x,
                y,
                stats,
                max,
                sum_exp,
                sink,
            } => {
                tensors.extend([*x, *y]);
                tensors.extend(*stats);
                tensors.extend(*max);
                tensors.extend(*sum_exp);
                tensors.extend(*sink);
            }
        }
    }
}

impl AttentionPrimitiveOperation {
    pub(crate) fn lower(&self, context: &LoweringContext<'_>) -> Result<LoweredOperation> {
        let tensors = context.backend_tensors();
        match self {
            Self::DiagonalBandMask {
                x,
                b,
                y,
                comparison_mode,
                sequence_length_query,
                sequence_length_key_value,
                left_bound,
                shift_right_bound,
            } => Ok(LoweredOperation::DiagonalBandMask(
                DiagonalBandMaskOperation::create(
                    tensor_at(tensors, *x)?,
                    tensor_at(tensors, *b)?,
                    tensor_at(tensors, *y)?,
                    *comparison_mode,
                    optional_tensor_at(tensors, *sequence_length_query)?,
                    optional_tensor_at(tensors, *sequence_length_key_value)?,
                    optional_tensor_at(tensors, *left_bound)?,
                    optional_tensor_at(tensors, *shift_right_bound)?,
                )?,
            )),
            Self::Softmax {
                x,
                y,
                stats,
                max,
                sum_exp,
                sink,
            } => Ok(LoweredOperation::Softmax(SoftmaxOperation::create(
                tensor_at(tensors, *x)?,
                tensor_at(tensors, *y)?,
                optional_tensor_at(tensors, *stats)?,
                optional_tensor_at(tensors, *max)?,
                optional_tensor_at(tensors, *sum_exp)?,
                optional_tensor_at(tensors, *sink)?,
            )?)),
        }
    }
}

/// Configuration for diagonal band masking of attention scores.
///
/// Diagonal masking is used for causal and sliding-window attention. The cuDNN
/// frontend exposes diagonal alignment and left/right bound attributes; this
/// Rust config stores tensor-backed bounds and optional sequence lengths.
#[derive(Debug, Clone, Copy)]
pub struct DiagonalBandMaskConfig {
    comparison_mode: PointwiseMode,
    sequence_length_query: Option<TensorId>,
    sequence_length_key_value: Option<TensorId>,
    left_bound: Option<TensorId>,
    shift_right_bound: Option<TensorId>,
}

impl DiagonalBandMaskConfig {
    pub fn new(comparison_mode: PointwiseMode) -> Self {
        Self {
            comparison_mode,
            sequence_length_query: None,
            sequence_length_key_value: None,
            left_bound: None,
            shift_right_bound: None,
        }
    }

    pub fn with_sequence_length_query(mut self, sequence_length_query: TensorId) -> Self {
        self.sequence_length_query = Some(sequence_length_query);
        self
    }

    pub fn with_sequence_length_key_value(mut self, sequence_length_key_value: TensorId) -> Self {
        self.sequence_length_key_value = Some(sequence_length_key_value);
        self
    }

    pub fn with_sequence_lengths(
        mut self,
        sequence_length_query: TensorId,
        sequence_length_key_value: TensorId,
    ) -> Self {
        self.sequence_length_query = Some(sequence_length_query);
        self.sequence_length_key_value = Some(sequence_length_key_value);
        self
    }

    pub fn with_left_bound(mut self, left_bound: TensorId) -> Self {
        self.left_bound = Some(left_bound);
        self
    }

    pub fn with_shift_right_bound(mut self, shift_right_bound: TensorId) -> Self {
        self.shift_right_bound = Some(shift_right_bound);
        self
    }

    pub fn comparison_mode(&self) -> PointwiseMode {
        self.comparison_mode
    }

    pub fn sequence_length_query(&self) -> Option<TensorId> {
        self.sequence_length_query
    }

    pub fn sequence_length_key_value(&self) -> Option<TensorId> {
        self.sequence_length_key_value
    }

    pub fn left_bound(&self) -> Option<TensorId> {
        self.left_bound
    }

    pub fn shift_right_bound(&self) -> Option<TensorId> {
        self.shift_right_bound
    }
}

/// Attributes for fused scaled dot product attention.
///
/// SDPA computes `softmax(QK^T / sqrt(d))V`, with optional attention scale,
/// causal masking, variable-length sequence tensors, and statistics outputs.
///
#[derive(Debug, Clone)]
pub struct SdpaConfig {
    name: Option<String>,
    aux_outputs: SdpaAuxOutputRequest,
    mask: SdpaFusedMaskMode,
    attention_scale: Option<f32>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum SdpaFusedMaskMode {
    #[default]
    None,
    CausalTopLeft,
    CausalBottomRight {
        sequence_length_query: TensorId,
        sequence_length_key_value: TensorId,
    },
}

impl SdpaFusedMaskMode {
    pub fn is_causal_top_left(self) -> bool {
        matches!(self, Self::CausalTopLeft)
    }

    pub fn is_causal_bottom_right(self) -> bool {
        matches!(self, Self::CausalBottomRight { .. })
    }

    pub fn sequence_length_query(self) -> Option<TensorId> {
        match self {
            Self::CausalBottomRight {
                sequence_length_query,
                ..
            } => Some(sequence_length_query),
            _ => None,
        }
    }

    pub fn sequence_length_key_value(self) -> Option<TensorId> {
        match self {
            Self::CausalBottomRight {
                sequence_length_key_value,
                ..
            } => Some(sequence_length_key_value),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SdpaAuxOutputRequest {
    softmax: SdpaSoftmaxAuxOutputRequest,
    quantized: SdpaQuantizedAuxOutputRequest,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum SdpaSoftmaxAuxOutputRequest {
    #[default]
    None,
    Stats,
    LogitMax,
    Descriptors,
    All,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum SdpaQuantizedAuxOutputRequest {
    #[default]
    None,
    Fp8Amax,
    Mxfp8Amax,
    AbsoluteMaxO,
}

impl SdpaAuxOutputRequest {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn stats_only() -> Self {
        Self {
            softmax: SdpaSoftmaxAuxOutputRequest::Stats,
            quantized: SdpaQuantizedAuxOutputRequest::new(),
        }
    }

    pub fn logit_max_only() -> Self {
        Self {
            softmax: SdpaSoftmaxAuxOutputRequest::LogitMax,
            quantized: SdpaQuantizedAuxOutputRequest::new(),
        }
    }

    pub fn softmax_descriptors() -> Self {
        Self {
            softmax: SdpaSoftmaxAuxOutputRequest::Descriptors,
            quantized: SdpaQuantizedAuxOutputRequest::new(),
        }
    }

    pub fn all_softmax() -> Self {
        Self {
            softmax: SdpaSoftmaxAuxOutputRequest::All,
            quantized: SdpaQuantizedAuxOutputRequest::new(),
        }
    }

    pub fn fp8_amax() -> Self {
        Self {
            softmax: SdpaSoftmaxAuxOutputRequest::Stats,
            quantized: SdpaQuantizedAuxOutputRequest::Fp8Amax,
        }
    }

    pub fn mxfp8_amax() -> Self {
        Self {
            softmax: SdpaSoftmaxAuxOutputRequest::Stats,
            quantized: SdpaQuantizedAuxOutputRequest::Mxfp8Amax,
        }
    }

    pub fn absolute_max_o_only() -> Self {
        Self {
            softmax: SdpaSoftmaxAuxOutputRequest::new(),
            quantized: SdpaQuantizedAuxOutputRequest::AbsoluteMaxO,
        }
    }

    pub fn stats(self) -> bool {
        self.softmax.stats()
    }

    pub fn logit_max(self) -> bool {
        self.softmax.logit_max()
    }

    pub fn score_sum_exp(self) -> bool {
        self.softmax.score_sum_exp()
    }

    pub fn absolute_max_s(self) -> bool {
        self.quantized.absolute_max_s()
    }

    pub fn absolute_max_o(self) -> bool {
        self.quantized.absolute_max_o()
    }

    pub(crate) fn requires_unified_cudnn_921(self) -> bool {
        self.softmax.requires_unified_cudnn_921()
    }

    pub(crate) fn has_direct_quantized_request(self) -> bool {
        self.quantized.has_any_request()
    }

    pub fn softmax(self) -> SdpaSoftmaxAuxOutputRequest {
        self.softmax
    }

    pub fn quantized(self) -> SdpaQuantizedAuxOutputRequest {
        self.quantized
    }
}

impl SdpaSoftmaxAuxOutputRequest {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn stats(self) -> bool {
        matches!(self, Self::Stats | Self::All)
    }

    pub fn logit_max(self) -> bool {
        matches!(self, Self::LogitMax | Self::Descriptors | Self::All)
    }

    pub fn score_sum_exp(self) -> bool {
        matches!(self, Self::Descriptors | Self::All)
    }

    pub(crate) fn requires_unified_cudnn_921(self) -> bool {
        self.logit_max() || self.score_sum_exp()
    }
}

impl SdpaQuantizedAuxOutputRequest {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn absolute_max_s(self) -> bool {
        matches!(self, Self::Fp8Amax)
    }

    pub fn absolute_max_o(self) -> bool {
        matches!(self, Self::Fp8Amax | Self::Mxfp8Amax | Self::AbsoluteMaxO)
    }

    pub(crate) fn has_any_request(self) -> bool {
        !matches!(self, Self::None)
    }
}

impl SdpaConfig {
    pub fn new() -> Self {
        Self {
            name: None,
            aux_outputs: SdpaAuxOutputRequest::new(),
            mask: SdpaFusedMaskMode::None,
            attention_scale: None,
        }
    }

    pub fn with_aux_outputs(mut self, aux_outputs: SdpaAuxOutputRequest) -> Self {
        self.aux_outputs = aux_outputs;
        self
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn with_mask(mut self, mask: SdpaFusedMaskMode) -> Self {
        self.mask = mask;
        self
    }

    pub fn with_attention_scale(mut self, attention_scale: f32) -> Self {
        self.attention_scale = Some(attention_scale);
        self
    }

    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    pub fn aux_outputs(&self) -> SdpaAuxOutputRequest {
        self.aux_outputs
    }

    pub fn mask(&self) -> SdpaFusedMaskMode {
        self.mask
    }

    pub fn attention_scale(&self) -> Option<f32> {
        self.attention_scale
    }
}

impl Default for SdpaConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// User-provided graph applied to SDPA score tensors.
///
/// The C++ frontend exposes score modification hooks through `set_score_mod`;
/// this type records a Rust frontend subgraph and its score input/output tensor
/// IDs for the same role.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SdpaScoreSubgraph {
    graph: Graph,
    input: TensorId,
    output: TensorId,
}

impl SdpaScoreSubgraph {
    pub fn new(graph: Graph, input: TensorId, output: TensorId) -> Self {
        Self {
            graph,
            input,
            output,
        }
    }

    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    pub fn input(&self) -> TensorId {
        self.input
    }

    pub fn output(&self) -> TensorId {
        self.output
    }
}

/// Optional SDPA score modifiers.
///
/// These cover additive bias, sink token, ALiBi, additive/dropout masks,
/// variable-length padding masks, causal masks, and sliding windows documented
/// for frontend SDPA score modification options.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SdpaScoreModifiers {
    bias: Option<TensorId>,
    sink_token: Option<TensorId>,
    alibi_slopes: Option<TensorId>,
    additive_mask: Option<TensorId>,
    dropout_mask: Option<TensorId>,
    dropout_scale: Option<TensorId>,
    mask_mode: AttentionMaskMode,
}

impl SdpaScoreModifiers {
    pub fn new() -> Self {
        Self {
            bias: None,
            sink_token: None,
            alibi_slopes: None,
            additive_mask: None,
            dropout_mask: None,
            dropout_scale: None,
            mask_mode: AttentionMaskMode::None,
        }
    }

    pub fn with_bias(mut self, bias: TensorId) -> Self {
        self.bias = Some(bias);
        self
    }

    pub fn with_sink_token(mut self, sink_token: TensorId) -> Self {
        self.sink_token = Some(sink_token);
        self
    }

    pub fn with_alibi(mut self, alibi_slopes: TensorId) -> Self {
        self.alibi_slopes = Some(alibi_slopes);
        self
    }

    pub fn with_additive_mask(mut self, additive_mask: TensorId) -> Self {
        self.additive_mask = Some(additive_mask);
        self
    }

    pub fn with_dropout(mut self, dropout_mask: TensorId, dropout_scale: TensorId) -> Self {
        self.dropout_mask = Some(dropout_mask);
        self.dropout_scale = Some(dropout_scale);
        self
    }

    fn set_dropout_mode(
        &mut self,
        dropout: &mut Option<AttentionDropoutConfig>,
        dropout_offset: &mut Option<TensorId>,
        dropout_mode: AttentionDropoutMode,
    ) {
        *dropout = None;
        *dropout_offset = None;
        self.dropout_mask = None;
        self.dropout_scale = None;
        match dropout_mode {
            AttentionDropoutMode::None => {}
            AttentionDropoutMode::Philox(config) => {
                *dropout = Some(config);
                *dropout_offset = config.offset();
            }
            AttentionDropoutMode::CustomMask { mask, scale } => {
                self.dropout_mask = Some(mask);
                self.dropout_scale = Some(scale);
            }
        }
    }

    fn dropout_mode(&self, dropout: Option<AttentionDropoutConfig>) -> AttentionDropoutMode {
        if let Some(dropout) = dropout {
            return AttentionDropoutMode::Philox(dropout);
        }
        match (self.dropout_mask, self.dropout_scale) {
            (Some(mask), Some(scale)) => AttentionDropoutMode::CustomMask { mask, scale },
            _ => AttentionDropoutMode::None,
        }
    }

    pub fn with_mask_mode(mut self, mask_mode: AttentionMaskMode) -> Self {
        self.mask_mode = mask_mode;
        self
    }

    pub fn mask_mode(&self) -> AttentionMaskMode {
        self.mask_mode
    }

    pub fn bias(&self) -> Option<TensorId> {
        self.bias
    }

    pub fn sink_token(&self) -> Option<TensorId> {
        self.sink_token
    }

    pub fn alibi_slopes(&self) -> Option<TensorId> {
        self.alibi_slopes
    }

    pub fn additive_mask(&self) -> Option<TensorId> {
        self.additive_mask
    }

    pub fn dropout_mask(&self) -> Option<TensorId> {
        self.dropout_mask
    }

    pub fn dropout_scale(&self) -> Option<TensorId> {
        self.dropout_scale
    }

    pub fn sequence_length_query(&self) -> Option<TensorId> {
        self.sequence_lengths()
            .map(|(sequence_length_query, _)| sequence_length_query)
    }

    pub fn sequence_length_key_value(&self) -> Option<TensorId> {
        self.sequence_lengths()
            .map(|(_, sequence_length_key_value)| sequence_length_key_value)
    }

    pub fn sequence_lengths(&self) -> Option<(TensorId, TensorId)> {
        match self.mask_mode {
            AttentionMaskMode::SequenceLengths {
                sequence_length_query,
                sequence_length_key_value,
            }
            | AttentionMaskMode::Padding {
                sequence_length_query,
                sequence_length_key_value,
            }
            | AttentionMaskMode::CausalTopLeftWithPadding {
                sequence_length_query,
                sequence_length_key_value,
            }
            | AttentionMaskMode::CausalBottomRight {
                sequence_length_query,
                sequence_length_key_value,
            }
            | AttentionMaskMode::SlidingWindowWithPadding {
                sequence_length_query,
                sequence_length_key_value,
                ..
            } => Some((sequence_length_query, sequence_length_key_value)),
            AttentionMaskMode::None
            | AttentionMaskMode::CausalTopLeft
            | AttentionMaskMode::SlidingWindow { .. }
            | AttentionMaskMode::Block(_) => None,
        }
    }

    pub fn padding_mask(&self) -> bool {
        matches!(
            self.mask_mode,
            AttentionMaskMode::Padding { .. }
                | AttentionMaskMode::CausalTopLeftWithPadding { .. }
                | AttentionMaskMode::SlidingWindowWithPadding { .. }
        )
    }

    pub fn causal_mask(&self) -> bool {
        matches!(
            self.mask_mode,
            AttentionMaskMode::CausalTopLeft | AttentionMaskMode::CausalTopLeftWithPadding { .. }
        )
    }

    pub fn causal_bottom_right(&self) -> bool {
        matches!(self.mask_mode, AttentionMaskMode::CausalBottomRight { .. })
    }

    pub fn sliding_window(&self) -> Option<(i64, i64)> {
        match self.mask_mode {
            AttentionMaskMode::SlidingWindow {
                left_window,
                right_window,
            }
            | AttentionMaskMode::SlidingWindowWithPadding {
                left_window,
                right_window,
                ..
            } => Some((left_window, right_window)),
            _ => None,
        }
    }

    pub fn block_mask(&self) -> Option<TensorId> {
        match self.mask_mode {
            AttentionMaskMode::Block(block_mask) => Some(block_mask),
            _ => None,
        }
    }

    pub fn has_custom_dropout(&self) -> bool {
        self.dropout_mask.is_some() || self.dropout_scale.is_some()
    }

    pub fn has_dropout_mask_without_scale(&self) -> bool {
        self.dropout_mask.is_some() != self.dropout_scale.is_some()
    }

    pub fn has_any_masking(&self) -> bool {
        !matches!(
            self.mask_mode,
            AttentionMaskMode::None | AttentionMaskMode::SequenceLengths { .. }
        )
    }

    pub fn has_pre_softmax_subgraph_modifier(&self) -> bool {
        self.bias.is_some()
            || self.alibi_slopes.is_some()
            || self.additive_mask.is_some()
            || self.causal_mask()
            || self.causal_bottom_right()
            || self.sliding_window().is_some()
    }

    pub fn has_causal_bottom_right_incompatible_modifier(&self) -> bool {
        self.bias.is_some() || self.alibi_slopes.is_some() || self.dropout_mask.is_some()
    }

    pub fn has_any_modifier(&self) -> bool {
        self.bias.is_some()
            || self.sink_token.is_some()
            || self.alibi_slopes.is_some()
            || self.additive_mask.is_some()
            || self.has_custom_dropout()
            || self.sequence_lengths().is_some()
            || self.has_any_masking()
    }
}

impl Default for SdpaScoreModifiers {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct AttentionScoreConfig {
    modifiers: SdpaScoreModifiers,
    score_subgraph: Option<SdpaScoreSubgraph>,
    dropout: Option<AttentionDropoutConfig>,
    dropout_offset: Option<TensorId>,
}

impl AttentionScoreConfig {
    pub fn new() -> Self {
        Self {
            modifiers: SdpaScoreModifiers::new(),
            score_subgraph: None,
            dropout: None,
            dropout_offset: None,
        }
    }

    pub fn with_score_subgraph(mut self, score_subgraph: SdpaScoreSubgraph) -> Self {
        self.score_subgraph = Some(score_subgraph);
        self
    }

    pub(crate) fn from_parts(
        modifiers: SdpaScoreModifiers,
        score_subgraph: Option<SdpaScoreSubgraph>,
    ) -> Self {
        Self {
            modifiers,
            score_subgraph,
            dropout: None,
            dropout_offset: None,
        }
    }

    pub fn with_bias(mut self, bias: TensorId) -> Self {
        self.modifiers = self.modifiers.with_bias(bias);
        self
    }

    pub fn with_sink_token(mut self, sink_token: TensorId) -> Self {
        self.modifiers = self.modifiers.with_sink_token(sink_token);
        self
    }

    pub fn with_alibi(mut self, alibi_slopes: TensorId) -> Self {
        self.modifiers = self.modifiers.with_alibi(alibi_slopes);
        self
    }

    pub fn with_additive_mask(mut self, additive_mask: TensorId) -> Self {
        self.modifiers = self.modifiers.with_additive_mask(additive_mask);
        self
    }

    pub fn with_mask_mode(mut self, mask_mode: AttentionMaskMode) -> Self {
        self.modifiers = self.modifiers.with_mask_mode(mask_mode);
        self
    }

    pub fn with_custom_dropout(mut self, mask: TensorId, scale: TensorId) -> Self {
        self.modifiers = self.modifiers.with_dropout(mask, scale);
        self
    }

    pub fn with_dropout_mode(mut self, dropout_mode: AttentionDropoutMode) -> Self {
        self.modifiers
            .set_dropout_mode(&mut self.dropout, &mut self.dropout_offset, dropout_mode);
        self
    }

    pub(crate) fn modifiers(&self) -> &SdpaScoreModifiers {
        &self.modifiers
    }

    pub fn mask_mode(&self) -> AttentionMaskMode {
        self.modifiers.mask_mode()
    }

    pub fn score_subgraph(&self) -> Option<&SdpaScoreSubgraph> {
        self.score_subgraph.as_ref()
    }

    pub fn dropout(&self) -> Option<AttentionDropoutConfig> {
        self.dropout
    }

    pub fn dropout_mode(&self) -> AttentionDropoutMode {
        self.modifiers.dropout_mode(self.dropout)
    }

    pub fn dropout_offset(&self) -> Option<TensorId> {
        self.dropout_offset
    }
}

impl Default for AttentionScoreConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// High-level attention configuration used by composite and fused SDPA helpers.
///
/// Combines fused SDPA attributes, softmax settings, score modifiers, dropout,
/// paged attention, block masks, and implementation selection.
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    fused: SdpaConfig,
    softmax: SoftmaxConfig,
    score: AttentionScoreConfig,
    implementation: AttentionImplementation,
    unfuse_fma: bool,
    paged: AttentionPagedKvCache,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionPagedCache {
    sequence: TensorId,
    page_table: TensorId,
}

impl AttentionPagedCache {
    pub fn new(sequence: TensorId, page_table: TensorId) -> Self {
        Self {
            sequence,
            page_table,
        }
    }

    pub fn sequence(&self) -> TensorId {
        self.sequence
    }

    pub fn page_table(&self) -> TensorId {
        self.page_table
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AttentionPagedKvCache {
    k: Option<AttentionPagedCache>,
    v: Option<AttentionPagedCache>,
    max_sequence_length_key_value: Option<i64>,
}

impl AttentionPagedKvCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_k(mut self, cache: AttentionPagedCache) -> Self {
        self.k = Some(cache);
        self
    }

    pub fn with_v(mut self, cache: AttentionPagedCache) -> Self {
        self.v = Some(cache);
        self
    }

    pub fn with_max_sequence_length_key_value(
        mut self,
        max_sequence_length_key_value: i64,
    ) -> Self {
        self.max_sequence_length_key_value = Some(max_sequence_length_key_value);
        self
    }

    pub fn k(&self) -> Option<AttentionPagedCache> {
        self.k
    }

    pub fn v(&self) -> Option<AttentionPagedCache> {
        self.v
    }

    pub fn max_sequence_length_key_value(&self) -> Option<i64> {
        self.max_sequence_length_key_value
    }

    pub fn is_enabled(&self) -> bool {
        self.k.is_some() || self.v.is_some()
    }

    pub fn has_any_request(&self) -> bool {
        self.is_enabled() || self.max_sequence_length_key_value.is_some()
    }
}

impl AttentionConfig {
    pub fn new(compute_type: DataType) -> Self {
        Self {
            fused: SdpaConfig::new(),
            softmax: SoftmaxConfig::new(compute_type),
            score: AttentionScoreConfig::new(),
            implementation: AttentionImplementation::Auto,
            unfuse_fma: false,
            paged: AttentionPagedKvCache::new(),
        }
    }

    pub fn with_aux_outputs(mut self, aux_outputs: SdpaAuxOutputRequest) -> Self {
        self.fused = self.fused.with_aux_outputs(aux_outputs);
        self
    }

    pub fn with_attention_scale(mut self, attention_scale: f32) -> Self {
        self.fused = self.fused.with_attention_scale(attention_scale);
        self
    }

    pub fn with_score_config(mut self, score: AttentionScoreConfig) -> Self {
        self.score = score;
        self
    }

    pub fn with_implementation(mut self, implementation: AttentionImplementation) -> Self {
        self.implementation = implementation;
        self
    }

    pub fn with_unfused_fma(mut self) -> Self {
        self.unfuse_fma = true;
        self
    }

    pub fn with_paged_cache(mut self, paged: AttentionPagedKvCache) -> Self {
        self.paged = paged;
        self
    }

    pub fn fused(&self) -> &SdpaConfig {
        &self.fused
    }

    pub fn softmax(&self) -> &SoftmaxConfig {
        &self.softmax
    }

    pub fn score_config(&self) -> &AttentionScoreConfig {
        &self.score
    }

    pub(crate) fn modifiers(&self) -> &SdpaScoreModifiers {
        self.score.modifiers()
    }

    pub fn mask_mode(&self) -> AttentionMaskMode {
        self.score.mask_mode()
    }

    pub fn score_subgraph(&self) -> Option<&SdpaScoreSubgraph> {
        self.score.score_subgraph()
    }

    pub fn implementation(&self) -> AttentionImplementation {
        self.implementation
    }

    pub fn dropout(&self) -> Option<AttentionDropoutConfig> {
        self.score.dropout()
    }

    pub fn dropout_mode(&self) -> AttentionDropoutMode {
        self.score.dropout_mode()
    }

    pub fn dropout_offset(&self) -> Option<TensorId> {
        self.score.dropout_offset()
    }

    pub fn unfuse_fma(&self) -> bool {
        self.unfuse_fma
    }

    pub fn paged(&self) -> AttentionPagedKvCache {
        self.paged
    }

    pub fn block_mask(&self) -> Option<TensorId> {
        self.score.modifiers().block_mask()
    }
}

/// High-level configuration for SDPA backward helpers.
///
/// Includes score modifiers, optional score subgraphs for forward and backward
/// score paths, dropout, attention scale, deterministic mode, and optional
/// gradients for bias, RNG dump, and sink token.
#[derive(Debug, Clone)]
pub struct AttentionBackwardConfig {
    softmax: SoftmaxConfig,
    score: AttentionScoreConfig,
    score_subgraph_bprop: Option<SdpaScoreSubgraph>,
    attention_scale: Option<f32>,
    deterministic_algorithm: bool,
    aux_gradients: AttentionBackwardAuxGradientRequest,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AttentionBackwardAuxGradientRequest {
    bias: bool,
    random_number_generator_dump: bool,
    sink_token: bool,
}

impl AttentionBackwardAuxGradientRequest {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn bias_gradient() -> Self {
        Self {
            bias: true,
            random_number_generator_dump: false,
            sink_token: false,
        }
    }

    pub fn rng_dump() -> Self {
        Self {
            bias: false,
            random_number_generator_dump: true,
            sink_token: false,
        }
    }

    pub fn sink_token_gradient() -> Self {
        Self {
            bias: false,
            random_number_generator_dump: false,
            sink_token: true,
        }
    }

    pub fn rng_dump_and_sink_token_gradient() -> Self {
        Self {
            bias: false,
            random_number_generator_dump: true,
            sink_token: true,
        }
    }

    pub fn all() -> Self {
        Self {
            bias: true,
            random_number_generator_dump: true,
            sink_token: true,
        }
    }

    pub fn bias(self) -> bool {
        self.bias
    }

    pub fn random_number_generator_dump(self) -> bool {
        self.random_number_generator_dump
    }

    pub fn sink_token(self) -> bool {
        self.sink_token
    }
}

impl AttentionBackwardConfig {
    pub fn new(compute_type: DataType) -> Self {
        Self {
            softmax: SoftmaxConfig::new(compute_type),
            score: AttentionScoreConfig::new(),
            score_subgraph_bprop: None,
            attention_scale: None,
            deterministic_algorithm: false,
            aux_gradients: AttentionBackwardAuxGradientRequest::new(),
        }
    }

    pub fn with_score_subgraph_bprop(mut self, score_subgraph_bprop: SdpaScoreSubgraph) -> Self {
        self.score_subgraph_bprop = Some(score_subgraph_bprop);
        self
    }

    pub fn with_score_config(mut self, score: AttentionScoreConfig) -> Self {
        self.score = score;
        self
    }

    pub fn with_attention_scale(mut self, attention_scale: f32) -> Self {
        self.attention_scale = Some(attention_scale);
        self
    }

    pub fn with_deterministic_algorithm(mut self) -> Self {
        self.deterministic_algorithm = true;
        self
    }

    pub fn with_aux_gradients(
        mut self,
        aux_gradients: AttentionBackwardAuxGradientRequest,
    ) -> Self {
        self.aux_gradients = aux_gradients;
        self
    }

    pub fn softmax(&self) -> SoftmaxConfig {
        self.softmax
    }

    pub fn score_config(&self) -> &AttentionScoreConfig {
        &self.score
    }

    pub(crate) fn modifiers(&self) -> &SdpaScoreModifiers {
        self.score.modifiers()
    }

    pub fn mask_mode(&self) -> AttentionMaskMode {
        self.score.mask_mode()
    }

    pub fn score_subgraph(&self) -> Option<&SdpaScoreSubgraph> {
        self.score.score_subgraph()
    }

    pub fn score_subgraph_bprop(&self) -> Option<&SdpaScoreSubgraph> {
        self.score_subgraph_bprop.as_ref()
    }

    pub fn dropout(&self) -> Option<AttentionDropoutConfig> {
        self.score.dropout()
    }

    pub fn dropout_mode(&self) -> AttentionDropoutMode {
        self.score.dropout_mode()
    }

    pub fn dropout_offset(&self) -> Option<TensorId> {
        self.score.dropout_offset()
    }

    pub fn attention_scale(&self) -> Option<f32> {
        self.attention_scale
    }

    pub fn deterministic_algorithm(&self) -> bool {
        self.deterministic_algorithm
    }

    pub fn aux_gradients(&self) -> AttentionBackwardAuxGradientRequest {
        self.aux_gradients
    }
}
