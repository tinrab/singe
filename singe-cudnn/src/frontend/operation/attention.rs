use serde::{Deserialize, Serialize};

use crate::{
    data_type::DataType, frontend::graph::Graph, math::NanPropagation, pointwise::PointwiseMode,
    tensor::TensorId,
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
}

/// Dropout mode for high-level SDPA helpers.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum AttentionDropoutMode {
    None,
    Philox(AttentionDropoutConfig),
    CustomMask { mask: TensorId, scale: TensorId },
}

/// Options for the direct cuDNN SDPA forward operation.
///
/// Covers variable-length sequence tensors, padding mask, paged attention page
/// tables, block masks, dropout, random-number-generator dump, and the
/// unfused-FMA option described for frontend SDPA attributes.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DirectSdpaForwardConfig {
    sequence_length_query: Option<TensorId>,
    sequence_length_key_value: Option<TensorId>,
    padding_mask: bool,
    page_table_k: Option<TensorId>,
    page_table_v: Option<TensorId>,
    block_mask: Option<TensorId>,
    dropout_seed: Option<TensorId>,
    dropout: Option<AttentionDropoutConfig>,
    random_number_generator_dump: Option<TensorId>,
    unfuse_fma: bool,
}

impl DirectSdpaForwardConfig {
    pub fn new() -> Self {
        Self {
            sequence_length_query: None,
            sequence_length_key_value: None,
            padding_mask: false,
            page_table_k: None,
            page_table_v: None,
            block_mask: None,
            dropout_seed: None,
            dropout: None,
            random_number_generator_dump: None,
            unfuse_fma: false,
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

    pub fn with_padding_mask(mut self) -> Self {
        self.padding_mask = true;
        self
    }

    pub fn without_padding_mask(mut self) -> Self {
        self.padding_mask = false;
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

    pub fn with_dropout(mut self, dropout: AttentionDropoutConfig) -> Self {
        self.dropout = Some(dropout);
        self
    }

    pub fn with_dropout_seed(mut self, dropout_seed: TensorId) -> Self {
        self.dropout_seed = Some(dropout_seed);
        self
    }

    pub fn with_random_number_generator_dump(
        mut self,
        random_number_generator_dump: TensorId,
    ) -> Self {
        self.random_number_generator_dump = Some(random_number_generator_dump);
        self
    }

    pub fn with_unfused_fma(mut self) -> Self {
        self.unfuse_fma = true;
        self
    }

    pub fn without_unfused_fma(mut self) -> Self {
        self.unfuse_fma = false;
        self
    }

    pub fn sequence_length_query(&self) -> Option<TensorId> {
        self.sequence_length_query
    }

    pub fn sequence_length_key_value(&self) -> Option<TensorId> {
        self.sequence_length_key_value
    }

    pub fn padding_mask(&self) -> bool {
        self.padding_mask
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

    pub fn dropout(&self) -> Option<AttentionDropoutConfig> {
        self.dropout
    }

    pub fn dropout_seed(&self) -> Option<TensorId> {
        self.dropout_seed
    }

    pub fn random_number_generator_dump(&self) -> Option<TensorId> {
        self.random_number_generator_dump
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
    sequence_length_query: Option<TensorId>,
    sequence_length_key_value: Option<TensorId>,
    padding_mask: bool,
    causal_mask: bool,
    score_subgraph: Option<SdpaScoreSubgraph>,
    sink: Option<TensorId>,
    sink_gradient: Option<TensorId>,
    max_total_sequence_length_query: Option<i64>,
    max_total_sequence_length_key_value: Option<i64>,
}

impl DirectSdpaBackwardConfig {
    pub fn new() -> Self {
        Self {
            sequence_length_query: None,
            sequence_length_key_value: None,
            padding_mask: false,
            causal_mask: false,
            score_subgraph: None,
            sink: None,
            sink_gradient: None,
            max_total_sequence_length_query: None,
            max_total_sequence_length_key_value: None,
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

    pub fn with_padding_mask(mut self) -> Self {
        self.padding_mask = true;
        self
    }

    pub fn without_padding_mask(mut self) -> Self {
        self.padding_mask = false;
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

    pub fn with_max_total_sequence_length_query(
        mut self,
        max_total_sequence_length_query: i64,
    ) -> Self {
        self.max_total_sequence_length_query = Some(max_total_sequence_length_query);
        self
    }

    pub fn with_max_total_sequence_length_key_value(
        mut self,
        max_total_sequence_length_key_value: i64,
    ) -> Self {
        self.max_total_sequence_length_key_value = Some(max_total_sequence_length_key_value);
        self
    }

    pub fn clear_max_total_sequence_lengths(mut self) -> Self {
        self.max_total_sequence_length_query = None;
        self.max_total_sequence_length_key_value = None;
        self
    }

    pub fn sequence_length_query(&self) -> Option<TensorId> {
        self.sequence_length_query
    }

    pub fn sequence_length_key_value(&self) -> Option<TensorId> {
        self.sequence_length_key_value
    }

    pub fn padding_mask(&self) -> bool {
        self.padding_mask
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

    pub fn max_total_sequence_length_query(&self) -> Option<i64> {
        self.max_total_sequence_length_query
    }

    pub fn max_total_sequence_length_key_value(&self) -> Option<i64> {
        self.max_total_sequence_length_key_value
    }
}

impl Default for DirectSdpaBackwardConfig {
    fn default() -> Self {
        Self::new()
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
    has_stats: bool,
    has_logit_max: bool,
    has_score_sum_exp: bool,
    has_absolute_max_s: bool,
    has_absolute_max_o: bool,
    causal_mask: bool,
    causal_bottom_right: bool,
    sequence_length_query: Option<TensorId>,
    sequence_length_key_value: Option<TensorId>,
    attention_scale: Option<f32>,
}

impl SdpaConfig {
    pub fn new() -> Self {
        Self {
            name: None,
            has_stats: false,
            has_logit_max: false,
            has_score_sum_exp: false,
            has_absolute_max_s: false,
            has_absolute_max_o: false,
            causal_mask: false,
            causal_bottom_right: false,
            sequence_length_query: None,
            sequence_length_key_value: None,
            attention_scale: None,
        }
    }

    pub fn with_stats(mut self) -> Self {
        self.has_stats = true;
        self
    }

    pub fn without_stats(mut self) -> Self {
        self.has_stats = false;
        self
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn with_logit_max(mut self) -> Self {
        self.has_logit_max = true;
        self
    }

    pub fn without_logit_max(mut self) -> Self {
        self.has_logit_max = false;
        self
    }

    pub fn with_score_sum_exp(mut self) -> Self {
        self.has_score_sum_exp = true;
        self
    }

    pub fn without_score_sum_exp(mut self) -> Self {
        self.has_score_sum_exp = false;
        self
    }

    pub fn with_absolute_max_s(mut self) -> Self {
        self.has_absolute_max_s = true;
        self
    }

    pub fn without_absolute_max_s(mut self) -> Self {
        self.has_absolute_max_s = false;
        self
    }

    pub fn with_absolute_max_o(mut self) -> Self {
        self.has_absolute_max_o = true;
        self
    }

    pub fn without_absolute_max_o(mut self) -> Self {
        self.has_absolute_max_o = false;
        self
    }

    pub fn with_causal_mask(mut self) -> Self {
        self.causal_mask = true;
        self
    }

    pub fn without_causal_mask(mut self) -> Self {
        self.causal_mask = false;
        self
    }

    pub fn with_causal_bottom_right(
        mut self,
        sequence_length_query: TensorId,
        sequence_length_key_value: TensorId,
    ) -> Self {
        self.causal_bottom_right = true;
        self.sequence_length_query = Some(sequence_length_query);
        self.sequence_length_key_value = Some(sequence_length_key_value);
        self
    }

    pub fn with_attention_scale(mut self, attention_scale: f32) -> Self {
        self.attention_scale = Some(attention_scale);
        self
    }

    pub fn has_stats(&self) -> bool {
        self.has_stats
    }

    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    pub fn has_logit_max(&self) -> bool {
        self.has_logit_max
    }

    pub fn has_score_sum_exp(&self) -> bool {
        self.has_score_sum_exp
    }

    pub fn has_absolute_max_s(&self) -> bool {
        self.has_absolute_max_s
    }

    pub fn has_absolute_max_o(&self) -> bool {
        self.has_absolute_max_o
    }

    pub fn has_causal_mask(&self) -> bool {
        self.causal_mask
    }

    pub fn has_causal_bottom_right(&self) -> bool {
        self.causal_bottom_right
    }

    pub fn sequence_length_query(&self) -> Option<TensorId> {
        self.sequence_length_query
    }

    pub fn sequence_length_key_value(&self) -> Option<TensorId> {
        self.sequence_length_key_value
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
    pub bias: Option<TensorId>,
    pub sink_token: Option<TensorId>,
    pub alibi_slopes: Option<TensorId>,
    pub additive_mask: Option<TensorId>,
    pub dropout_mask: Option<TensorId>,
    pub dropout_scale: Option<TensorId>,
    pub sequence_length_query: Option<TensorId>,
    pub sequence_length_key_value: Option<TensorId>,
    pub padding_mask: bool,
    pub causal_mask: bool,
    pub causal_bottom_right: bool,
    pub sliding_window: Option<(i64, i64)>,
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
            sequence_length_query: None,
            sequence_length_key_value: None,
            padding_mask: false,
            causal_mask: false,
            causal_bottom_right: false,
            sliding_window: None,
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

    pub fn with_sequence_lengths(
        mut self,
        sequence_length_query: TensorId,
        sequence_length_key_value: TensorId,
    ) -> Self {
        self.sequence_length_query = Some(sequence_length_query);
        self.sequence_length_key_value = Some(sequence_length_key_value);
        self
    }

    pub fn with_padding_mask(
        mut self,
        sequence_length_query: TensorId,
        sequence_length_key_value: TensorId,
    ) -> Self {
        self.sequence_length_query = Some(sequence_length_query);
        self.sequence_length_key_value = Some(sequence_length_key_value);
        self.padding_mask = true;
        self
    }

    pub fn with_causal_mask(mut self) -> Self {
        self.causal_mask = true;
        self
    }

    pub fn with_sliding_window(mut self, left_window: i64, right_window: i64) -> Self {
        self.sliding_window = Some((left_window, right_window));
        self
    }

    pub fn with_causal_bottom_right(
        mut self,
        sequence_length_query: TensorId,
        sequence_length_key_value: TensorId,
    ) -> Self {
        self.sequence_length_query = Some(sequence_length_query);
        self.sequence_length_key_value = Some(sequence_length_key_value);
        self.causal_bottom_right = true;
        self
    }

    pub fn with_mask_mode(mut self, mask_mode: AttentionMaskMode) -> Self {
        self.sequence_length_query = None;
        self.sequence_length_key_value = None;
        self.padding_mask = false;
        self.causal_mask = false;
        self.causal_bottom_right = false;
        self.sliding_window = None;

        match mask_mode {
            AttentionMaskMode::None => {}
            AttentionMaskMode::SequenceLengths {
                sequence_length_query,
                sequence_length_key_value,
            } => {
                self.sequence_length_query = Some(sequence_length_query);
                self.sequence_length_key_value = Some(sequence_length_key_value);
            }
            AttentionMaskMode::Padding {
                sequence_length_query,
                sequence_length_key_value,
            } => {
                self.sequence_length_query = Some(sequence_length_query);
                self.sequence_length_key_value = Some(sequence_length_key_value);
                self.padding_mask = true;
            }
            AttentionMaskMode::CausalTopLeft => {
                self.causal_mask = true;
            }
            AttentionMaskMode::CausalTopLeftWithPadding {
                sequence_length_query,
                sequence_length_key_value,
            } => {
                self.sequence_length_query = Some(sequence_length_query);
                self.sequence_length_key_value = Some(sequence_length_key_value);
                self.padding_mask = true;
                self.causal_mask = true;
            }
            AttentionMaskMode::CausalBottomRight {
                sequence_length_query,
                sequence_length_key_value,
            } => {
                self.sequence_length_query = Some(sequence_length_query);
                self.sequence_length_key_value = Some(sequence_length_key_value);
                self.causal_bottom_right = true;
            }
            AttentionMaskMode::SlidingWindow {
                left_window,
                right_window,
            } => {
                self.sliding_window = Some((left_window, right_window));
            }
            AttentionMaskMode::SlidingWindowWithPadding {
                sequence_length_query,
                sequence_length_key_value,
                left_window,
                right_window,
            } => {
                self.sequence_length_query = Some(sequence_length_query);
                self.sequence_length_key_value = Some(sequence_length_key_value);
                self.padding_mask = true;
                self.sliding_window = Some((left_window, right_window));
            }
        }
        self
    }

    pub fn mask_mode(&self) -> AttentionMaskMode {
        match (
            self.padding_mask,
            self.causal_mask,
            self.causal_bottom_right,
            self.sliding_window,
            self.sequence_length_query,
            self.sequence_length_key_value,
        ) {
            (
                false,
                false,
                false,
                None,
                Some(sequence_length_query),
                Some(sequence_length_key_value),
            ) => AttentionMaskMode::SequenceLengths {
                sequence_length_query,
                sequence_length_key_value,
            },
            (false, false, false, None, _, _) => AttentionMaskMode::None,
            (
                true,
                false,
                false,
                None,
                Some(sequence_length_query),
                Some(sequence_length_key_value),
            ) => AttentionMaskMode::Padding {
                sequence_length_query,
                sequence_length_key_value,
            },
            (false, true, false, None, _, _) => AttentionMaskMode::CausalTopLeft,
            (
                true,
                true,
                false,
                None,
                Some(sequence_length_query),
                Some(sequence_length_key_value),
            ) => AttentionMaskMode::CausalTopLeftWithPadding {
                sequence_length_query,
                sequence_length_key_value,
            },
            (
                false,
                false,
                true,
                None,
                Some(sequence_length_query),
                Some(sequence_length_key_value),
            ) => AttentionMaskMode::CausalBottomRight {
                sequence_length_query,
                sequence_length_key_value,
            },
            (false, false, false, Some((left_window, right_window)), _, _) => {
                AttentionMaskMode::SlidingWindow {
                    left_window,
                    right_window,
                }
            }
            (
                true,
                false,
                false,
                Some((left_window, right_window)),
                Some(sequence_length_query),
                Some(sequence_length_key_value),
            ) => AttentionMaskMode::SlidingWindowWithPadding {
                sequence_length_query,
                sequence_length_key_value,
                left_window,
                right_window,
            },
            _ => AttentionMaskMode::None,
        }
    }
}

impl Default for SdpaScoreModifiers {
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
    modifiers: SdpaScoreModifiers,
    score_subgraph: Option<SdpaScoreSubgraph>,
    implementation: AttentionImplementation,
    dropout: Option<AttentionDropoutConfig>,
    dropout_offset: Option<TensorId>,
    unfuse_fma: bool,
    max_sequence_length_key_value: Option<i64>,
    paged_k: Option<AttentionPagedCache>,
    paged_v: Option<AttentionPagedCache>,
    block_mask: Option<TensorId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionPagedCache {
    pub sequence: TensorId,
    pub page_table: TensorId,
}

impl AttentionPagedCache {
    pub fn new(sequence: TensorId, page_table: TensorId) -> Self {
        Self {
            sequence,
            page_table,
        }
    }
}

impl AttentionConfig {
    pub fn new(compute_type: DataType) -> Self {
        Self {
            fused: SdpaConfig::new(),
            softmax: SoftmaxConfig::new(compute_type),
            modifiers: SdpaScoreModifiers::new(),
            score_subgraph: None,
            implementation: AttentionImplementation::Auto,
            dropout: None,
            dropout_offset: None,
            unfuse_fma: false,
            max_sequence_length_key_value: None,
            paged_k: None,
            paged_v: None,
            block_mask: None,
        }
    }

    pub fn with_stats(mut self) -> Self {
        self.fused = self.fused.with_stats();
        self
    }

    pub fn without_stats(mut self) -> Self {
        self.fused = self.fused.without_stats();
        self
    }

    pub fn with_logit_max(mut self) -> Self {
        self.fused = self.fused.with_logit_max();
        self
    }

    pub fn without_logit_max(mut self) -> Self {
        self.fused = self.fused.without_logit_max();
        self
    }

    pub fn with_score_sum_exp(mut self) -> Self {
        self.fused = self.fused.with_score_sum_exp();
        self
    }

    pub fn without_score_sum_exp(mut self) -> Self {
        self.fused = self.fused.without_score_sum_exp();
        self
    }

    pub fn with_absolute_max_s(mut self) -> Self {
        self.fused = self.fused.with_absolute_max_s();
        self
    }

    pub fn without_absolute_max_s(mut self) -> Self {
        self.fused = self.fused.without_absolute_max_s();
        self
    }

    pub fn with_absolute_max_o(mut self) -> Self {
        self.fused = self.fused.with_absolute_max_o();
        self
    }

    pub fn without_absolute_max_o(mut self) -> Self {
        self.fused = self.fused.without_absolute_max_o();
        self
    }

    pub fn with_attention_scale(mut self, attention_scale: f32) -> Self {
        self.fused = self.fused.with_attention_scale(attention_scale);
        self
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

    pub fn with_score_subgraph(mut self, score_subgraph: SdpaScoreSubgraph) -> Self {
        self.score_subgraph = Some(score_subgraph);
        self
    }

    pub fn with_dropout(mut self, dropout_mask: TensorId, dropout_scale: TensorId) -> Self {
        self.modifiers = self.modifiers.with_dropout(dropout_mask, dropout_scale);
        self
    }

    pub fn with_mask_mode(mut self, mask_mode: AttentionMaskMode) -> Self {
        self.modifiers = self.modifiers.with_mask_mode(mask_mode);
        self
    }

    pub fn with_dropout_mode(mut self, dropout_mode: AttentionDropoutMode) -> Self {
        self.dropout = None;
        self.dropout_offset = None;
        self.modifiers.dropout_mask = None;
        self.modifiers.dropout_scale = None;
        match dropout_mode {
            AttentionDropoutMode::None => {}
            AttentionDropoutMode::Philox(dropout) => {
                self.dropout = Some(dropout);
                self.dropout_offset = dropout.offset();
            }
            AttentionDropoutMode::CustomMask { mask, scale } => {
                self.modifiers.dropout_mask = Some(mask);
                self.modifiers.dropout_scale = Some(scale);
            }
        }
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

    pub fn without_unfused_fma(mut self) -> Self {
        self.unfuse_fma = false;
        self
    }

    pub fn with_dropout_probability(mut self, probability: f32, seed: i64) -> Self {
        self.dropout = Some(AttentionDropoutConfig::new(probability, seed));
        self
    }

    pub fn with_dropout_probability_from_tensor(
        mut self,
        probability: f32,
        seed: TensorId,
        offset: TensorId,
    ) -> Self {
        self.dropout = Some(AttentionDropoutConfig::new(probability, 0).with_seed_tensor(seed));
        self.dropout_offset = Some(offset);
        self
    }

    pub fn clear_dropout_probability(mut self) -> Self {
        self.dropout = None;
        self.dropout_offset = None;
        self
    }

    pub fn with_dropout_offset(mut self, offset: TensorId) -> Self {
        self.dropout_offset = Some(offset);
        self
    }

    pub fn clear_dropout_offset(mut self) -> Self {
        self.dropout_offset = None;
        self
    }

    pub fn with_padding_mask(
        mut self,
        sequence_length_query: TensorId,
        sequence_length_key_value: TensorId,
    ) -> Self {
        self.modifiers = self
            .modifiers
            .with_padding_mask(sequence_length_query, sequence_length_key_value);
        self
    }

    pub fn with_sequence_lengths(
        mut self,
        sequence_length_query: TensorId,
        sequence_length_key_value: TensorId,
    ) -> Self {
        self.modifiers = self
            .modifiers
            .with_sequence_lengths(sequence_length_query, sequence_length_key_value);
        self
    }

    pub fn with_causal_mask(mut self) -> Self {
        self.modifiers = self.modifiers.with_causal_mask();
        self
    }

    pub fn with_sliding_window(mut self, left_window: i64, right_window: i64) -> Self {
        self.modifiers = self
            .modifiers
            .with_sliding_window(left_window, right_window);
        self
    }

    pub fn with_causal_bottom_right(
        mut self,
        sequence_length_query: TensorId,
        sequence_length_key_value: TensorId,
    ) -> Self {
        self.modifiers = self
            .modifiers
            .with_causal_bottom_right(sequence_length_query, sequence_length_key_value);
        self
    }

    pub fn with_paged_k(mut self, cache: AttentionPagedCache) -> Self {
        self.paged_k = Some(cache);
        self
    }

    pub fn with_paged_v(mut self, cache: AttentionPagedCache) -> Self {
        self.paged_v = Some(cache);
        self
    }

    pub fn with_block_mask(mut self, block_mask: TensorId) -> Self {
        self.block_mask = Some(block_mask);
        self
    }

    pub fn with_max_sequence_length_key_value(
        mut self,
        max_sequence_length_key_value: i64,
    ) -> Self {
        self.max_sequence_length_key_value = Some(max_sequence_length_key_value);
        self
    }

    pub fn fused(&self) -> &SdpaConfig {
        &self.fused
    }

    pub fn softmax(&self) -> &SoftmaxConfig {
        &self.softmax
    }

    pub fn modifiers(&self) -> &SdpaScoreModifiers {
        &self.modifiers
    }

    pub fn mask_mode(&self) -> AttentionMaskMode {
        self.modifiers.mask_mode()
    }

    pub fn score_subgraph(&self) -> Option<&SdpaScoreSubgraph> {
        self.score_subgraph.as_ref()
    }

    pub fn implementation(&self) -> AttentionImplementation {
        self.implementation
    }

    pub fn dropout(&self) -> Option<AttentionDropoutConfig> {
        self.dropout
    }

    pub fn dropout_mode(&self) -> AttentionDropoutMode {
        if let Some(dropout) = self.dropout {
            return AttentionDropoutMode::Philox(dropout);
        }
        match (self.modifiers.dropout_mask, self.modifiers.dropout_scale) {
            (Some(mask), Some(scale)) => AttentionDropoutMode::CustomMask { mask, scale },
            _ => AttentionDropoutMode::None,
        }
    }

    pub fn dropout_offset(&self) -> Option<TensorId> {
        self.dropout_offset
    }

    pub fn unfuse_fma(&self) -> bool {
        self.unfuse_fma
    }

    pub fn max_sequence_length_key_value(&self) -> Option<i64> {
        self.max_sequence_length_key_value
    }

    pub fn paged_k(&self) -> Option<AttentionPagedCache> {
        self.paged_k
    }

    pub fn paged_v(&self) -> Option<AttentionPagedCache> {
        self.paged_v
    }

    pub fn block_mask(&self) -> Option<TensorId> {
        self.block_mask
    }

    pub fn uses_modifiers(&self) -> bool {
        self.modifiers.bias.is_some()
            || self.modifiers.sink_token.is_some()
            || self.modifiers.alibi_slopes.is_some()
            || self.score_subgraph.is_some()
            || self.modifiers.additive_mask.is_some()
            || self.modifiers.dropout_mask.is_some()
            || self.modifiers.dropout_scale.is_some()
            || self.dropout.is_some()
            || self.modifiers.sequence_length_query.is_some()
            || self.modifiers.sequence_length_key_value.is_some()
            || self.modifiers.causal_mask
            || self.modifiers.causal_bottom_right
            || self.modifiers.sliding_window.is_some()
            || self.paged_k.is_some()
            || self.paged_v.is_some()
            || self.block_mask.is_some()
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
    modifiers: SdpaScoreModifiers,
    score_subgraph: Option<SdpaScoreSubgraph>,
    score_subgraph_bprop: Option<SdpaScoreSubgraph>,
    dropout: Option<AttentionDropoutConfig>,
    dropout_offset: Option<TensorId>,
    attention_scale: Option<f32>,
    deterministic_algorithm: bool,
    has_bias_gradient: bool,
    has_random_number_generator_dump: bool,
    has_sink_token_gradient: bool,
}

impl AttentionBackwardConfig {
    pub fn new(compute_type: DataType) -> Self {
        Self {
            softmax: SoftmaxConfig::new(compute_type),
            modifiers: SdpaScoreModifiers::new(),
            score_subgraph: None,
            score_subgraph_bprop: None,
            dropout: None,
            dropout_offset: None,
            attention_scale: None,
            deterministic_algorithm: false,
            has_bias_gradient: false,
            has_random_number_generator_dump: false,
            has_sink_token_gradient: false,
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

    pub fn with_score_subgraph(mut self, score_subgraph: SdpaScoreSubgraph) -> Self {
        self.score_subgraph = Some(score_subgraph);
        self
    }

    pub fn with_score_subgraph_bprop(mut self, score_subgraph_bprop: SdpaScoreSubgraph) -> Self {
        self.score_subgraph_bprop = Some(score_subgraph_bprop);
        self
    }

    pub fn with_dropout(mut self, dropout_mask: TensorId, dropout_scale: TensorId) -> Self {
        self.modifiers = self.modifiers.with_dropout(dropout_mask, dropout_scale);
        self
    }

    pub fn with_mask_mode(mut self, mask_mode: AttentionMaskMode) -> Self {
        self.modifiers = self.modifiers.with_mask_mode(mask_mode);
        self
    }

    pub fn with_dropout_mode(mut self, dropout_mode: AttentionDropoutMode) -> Self {
        self.dropout = None;
        self.dropout_offset = None;
        self.modifiers.dropout_mask = None;
        self.modifiers.dropout_scale = None;
        match dropout_mode {
            AttentionDropoutMode::None => {}
            AttentionDropoutMode::Philox(dropout) => {
                self.dropout = Some(dropout);
                self.dropout_offset = dropout.offset();
            }
            AttentionDropoutMode::CustomMask { mask, scale } => {
                self.modifiers.dropout_mask = Some(mask);
                self.modifiers.dropout_scale = Some(scale);
            }
        }
        self
    }

    pub fn with_dropout_probability(mut self, probability: f32, seed: i64) -> Self {
        self.dropout = Some(AttentionDropoutConfig::new(probability, seed));
        self
    }

    pub fn with_dropout_probability_from_tensor(
        mut self,
        probability: f32,
        seed: TensorId,
        offset: TensorId,
    ) -> Self {
        self.dropout = Some(AttentionDropoutConfig::new(probability, 0).with_seed_tensor(seed));
        self.dropout_offset = Some(offset);
        self
    }

    pub fn clear_dropout_probability(mut self) -> Self {
        self.dropout = None;
        self.dropout_offset = None;
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

    pub fn without_deterministic_algorithm(mut self) -> Self {
        self.deterministic_algorithm = false;
        self
    }

    pub fn with_dropout_offset(mut self, offset: TensorId) -> Self {
        self.dropout_offset = Some(offset);
        self
    }

    pub fn clear_dropout_offset(mut self) -> Self {
        self.dropout_offset = None;
        self
    }

    pub fn with_padding_mask(
        mut self,
        sequence_length_query: TensorId,
        sequence_length_key_value: TensorId,
    ) -> Self {
        self.modifiers = self
            .modifiers
            .with_padding_mask(sequence_length_query, sequence_length_key_value);
        self
    }

    pub fn with_sequence_lengths(
        mut self,
        sequence_length_query: TensorId,
        sequence_length_key_value: TensorId,
    ) -> Self {
        self.modifiers = self
            .modifiers
            .with_sequence_lengths(sequence_length_query, sequence_length_key_value);
        self
    }

    pub fn with_causal_mask(mut self) -> Self {
        self.modifiers = self.modifiers.with_causal_mask();
        self
    }

    pub fn with_sliding_window(mut self, left_window: i64, right_window: i64) -> Self {
        self.modifiers = self
            .modifiers
            .with_sliding_window(left_window, right_window);
        self
    }

    pub fn with_causal_bottom_right(
        mut self,
        sequence_length_query: TensorId,
        sequence_length_key_value: TensorId,
    ) -> Self {
        self.modifiers = self
            .modifiers
            .with_causal_bottom_right(sequence_length_query, sequence_length_key_value);
        self
    }

    pub fn with_bias_gradient(mut self) -> Self {
        self.has_bias_gradient = true;
        self
    }

    pub fn with_random_number_generator_dump(mut self) -> Self {
        self.has_random_number_generator_dump = true;
        self
    }

    pub fn with_sink_token_gradient(mut self) -> Self {
        self.has_sink_token_gradient = true;
        self
    }

    pub fn softmax(&self) -> SoftmaxConfig {
        self.softmax
    }

    pub fn modifiers(&self) -> &SdpaScoreModifiers {
        &self.modifiers
    }

    pub fn mask_mode(&self) -> AttentionMaskMode {
        self.modifiers.mask_mode()
    }

    pub fn score_subgraph(&self) -> Option<&SdpaScoreSubgraph> {
        self.score_subgraph.as_ref()
    }

    pub fn score_subgraph_bprop(&self) -> Option<&SdpaScoreSubgraph> {
        self.score_subgraph_bprop.as_ref()
    }

    pub fn dropout(&self) -> Option<AttentionDropoutConfig> {
        self.dropout
    }

    pub fn dropout_mode(&self) -> AttentionDropoutMode {
        if let Some(dropout) = self.dropout {
            return AttentionDropoutMode::Philox(dropout);
        }
        match (self.modifiers.dropout_mask, self.modifiers.dropout_scale) {
            (Some(mask), Some(scale)) => AttentionDropoutMode::CustomMask { mask, scale },
            _ => AttentionDropoutMode::None,
        }
    }

    pub fn dropout_offset(&self) -> Option<TensorId> {
        self.dropout_offset
    }

    pub fn attention_scale(&self) -> Option<f32> {
        self.attention_scale
    }

    pub fn deterministic_algorithm(&self) -> bool {
        self.deterministic_algorithm
    }

    pub fn has_bias_gradient(&self) -> bool {
        self.has_bias_gradient
    }

    pub fn has_random_number_generator_dump(&self) -> bool {
        self.has_random_number_generator_dump
    }

    pub fn has_sink_token_gradient(&self) -> bool {
        self.has_sink_token_gradient
    }
}
