use serde::{Deserialize, Serialize};

use crate::{
    backend::tensor::TensorId,
    data_type::DataType,
    frontend::operation::{
        AttentionDropoutConfig, AttentionDropoutMode, AttentionMaskMode,
        AttentionMaxTotalSequenceLengths, AttentionScoreConfig, SdpaAuxOutputRequest, SdpaConfig,
        SdpaScoreModifiers, SdpaScoreSubgraph, SoftmaxConfig, SoftmaxStatsBase,
    },
};

/// SDPA implementation selector.
///
/// The frontend SDPA operation supports `AUTO`, `COMPOSITE`, and `UNIFIED`
/// implementation choices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum AttentionImplementation {
    /// Selects unified attention when the request can use that path; otherwise selects supported composite lowering.
    Auto,
    /// Requires the unified native SDPA path; unsupported requests fail during preparation.
    Unified,
    /// Expands attention into component operations such as matmuls, score modifiers, and softmax.
    Composite,
}

/// High-level attention configuration used by composite and fused SDPA helpers.
///
/// Combines fused SDPA attributes, softmax settings, score modifiers, dropout,
/// paged attention, block masks, and implementation selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionConfig {
    pub(crate) fused: SdpaConfig,
    pub(crate) softmax: SoftmaxConfig,
    pub(crate) score: AttentionScoreConfig,
    pub(crate) implementation: AttentionImplementation,
    pub(crate) unfuse_fma: bool,
    pub(crate) paged: AttentionPagedKvCache,
    #[serde(default)]
    pub(crate) max_total_sequence_lengths: AttentionMaxTotalSequenceLengths,
}

/// Page-table tensor identifying blocks in one paged key or value cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttentionPagedCache {
    pub(crate) page_table: TensorId,
}

impl AttentionPagedCache {
    /// Uses the key/value sequence lengths from the attention score configuration.
    /// The page table may use ragged offsets for packed per-batch page lists.
    /// Offset values address page-table elements, independently of cache-token offsets.
    #[must_use]
    pub fn new(page_table: TensorId) -> Self {
        Self { page_table }
    }

    /// Returns the page-table tensor identity.
    #[must_use]
    pub fn page_table(&self) -> TensorId {
        self.page_table
    }
}

/// Independent key and value paging settings and an optional logical sequence-length hint.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttentionPagedKvCache {
    pub(crate) k: Option<AttentionPagedCache>,
    pub(crate) v: Option<AttentionPagedCache>,
    pub(crate) max_sequence_length_key_value: Option<i64>,
}

impl AttentionPagedKvCache {
    /// Creates cache settings with paging and the sequence-length hint disabled.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets or clears key paging while preserving value paging and the length hint.
    #[must_use]
    pub fn with_k(mut self, cache: impl Into<Option<AttentionPagedCache>>) -> Self {
        self.k = cache.into();
        self
    }

    /// Sets or clears value paging while preserving key paging and the length hint.
    #[must_use]
    pub fn with_v(mut self, cache: impl Into<Option<AttentionPagedCache>>) -> Self {
        self.v = cache.into();
        self
    }

    /// Sets the logical cache length for composite intermediates and score shapes, or clears the hint with `None`.
    /// Without a hint, a dense cache supplies the length, then bias or RNG-dump dimensions,
    /// then the smaller padded capacity when both caches are paged.
    /// Unified execution uses its native sequence descriptors for runtime limits.
    /// FP8 and MXFP8 requests may retain a hint without paging, matching C++.
    /// Dense operands retain their declared dimensions and strides.
    /// FP16/BF16 requests require at least one paged cache when a hint is present.
    #[must_use]
    pub fn with_max_sequence_length_key_value(
        mut self,
        max_sequence_length_key_value: impl Into<Option<i64>>,
    ) -> Self {
        self.max_sequence_length_key_value = max_sequence_length_key_value.into();
        self
    }

    /// Returns the key-cache paging configuration, if enabled.
    #[must_use]
    pub fn k(&self) -> Option<AttentionPagedCache> {
        self.k
    }

    /// Returns the value-cache paging configuration, if enabled.
    #[must_use]
    pub fn v(&self) -> Option<AttentionPagedCache> {
        self.v
    }

    /// Returns the optional logical key/value sequence-length hint in tokens.
    #[must_use]
    pub fn max_sequence_length_key_value(&self) -> Option<i64> {
        self.max_sequence_length_key_value
    }

    /// Returns whether key or value paging is enabled.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.k.is_some() || self.v.is_some()
    }

    /// Returns whether paging or a logical sequence-length hint is configured.
    #[must_use]
    pub fn has_any_request(&self) -> bool {
        self.is_enabled() || self.max_sequence_length_key_value.is_some()
    }
}

impl AttentionConfig {
    /// Selects softmax statistics encoding. SDPA backward requires natural log.
    #[must_use]
    pub fn with_stats_base(mut self, base: SoftmaxStatsBase) -> Self {
        self.softmax = self.softmax.with_stats_base(base);
        self.fused = self.fused.with_stats_base(base);
        self
    }

    /// Returns the logarithm base used to encode saved softmax statistics.
    #[must_use]
    pub fn stats_base(&self) -> SoftmaxStatsBase {
        self.fused.stats_base()
    }

    /// Records an attention request with inherited compute precision and automatic implementation selection.
    #[must_use]
    pub fn new() -> Self {
        Self {
            fused: SdpaConfig::new(),
            softmax: SoftmaxConfig::new(),
            score: AttentionScoreConfig::new(),
            implementation: AttentionImplementation::Auto,
            unfuse_fma: false,
            paged: AttentionPagedKvCache::new(),
            max_total_sequence_lengths: AttentionMaxTotalSequenceLengths::default(),
        }
    }

    /// Overrides attention compute precision, or restores graph inheritance with `None`.
    #[must_use]
    pub fn with_compute_type(mut self, compute_type: impl Into<Option<DataType>>) -> Self {
        self.softmax = self.softmax.with_compute_type(compute_type);
        self
    }

    pub(crate) fn with_aux_outputs(mut self, aux_outputs: SdpaAuxOutputRequest) -> Self {
        self.fused = self.fused.with_aux_outputs(aux_outputs);
        self
    }

    /// Sets or clears the numeric attention multiplier while retaining the scale tensor operand.
    #[must_use]
    pub fn with_attention_scale(mut self, attention_scale: impl Into<Option<f32>>) -> Self {
        self.fused = self.fused.with_attention_scale(attention_scale);
        self
    }

    /// Replaces score modifiers, score-subgraph, and dropout settings together.
    #[must_use]
    pub fn with_score_config(mut self, score: AttentionScoreConfig) -> Self {
        self.score = score;
        self
    }

    /// Selects automatic, unified, or composite attention lowering.
    #[must_use]
    pub fn with_implementation(mut self, implementation: AttentionImplementation) -> Self {
        self.implementation = implementation;
        self
    }

    /// Sets or clears the native unfused-FMA flag without changing other options.
    #[must_use]
    pub fn with_unfuse_fma(mut self, enabled: bool) -> Self {
        self.unfuse_fma = enabled;
        self
    }

    /// Replaces key/value paging settings and the logical sequence-length hint.
    #[must_use]
    pub fn with_paged_cache(mut self, paged: AttentionPagedKvCache) -> Self {
        self.paged = paged;
        self
    }

    /// Records optional packed-token bounds for query and key/value storage.
    /// At least one of Q, K, V or the declared output must have ragged offsets.
    /// Like C++, forward attention retains these as frontend hints without changing
    /// native workspace sizing or adding backend attributes.
    #[must_use]
    pub fn with_max_total_sequence_lengths(
        mut self,
        lengths: AttentionMaxTotalSequenceLengths,
    ) -> Self {
        self.max_total_sequence_lengths = lengths;
        self
    }

    /// Returns the optional packed-token storage bounds for query and key/value tensors.
    #[must_use]
    pub fn max_total_sequence_lengths(&self) -> AttentionMaxTotalSequenceLengths {
        self.max_total_sequence_lengths
    }

    /// Sets the attention operation name used by the fused configuration.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.fused = self.fused.with_name(name);
        self
    }

    /// Borrows the fused SDPA settings retained by this attention request.
    #[must_use]
    pub fn fused(&self) -> &SdpaConfig {
        &self.fused
    }

    /// Borrows softmax computation and statistics settings.
    #[must_use]
    pub fn softmax(&self) -> &SoftmaxConfig {
        &self.softmax
    }

    /// Borrows score modifiers, score-subgraph, and dropout settings.
    #[must_use]
    pub fn score_config(&self) -> &AttentionScoreConfig {
        &self.score
    }

    pub(crate) fn modifiers(&self) -> &SdpaScoreModifiers {
        self.score.modifiers()
    }

    /// Returns the structural mask mode.
    #[must_use]
    pub fn mask_mode(&self) -> AttentionMaskMode {
        self.score.mask_mode()
    }

    /// Borrows the optional graph applied to attention scores.
    #[must_use]
    pub fn score_subgraph(&self) -> Option<&SdpaScoreSubgraph> {
        self.score.score_subgraph()
    }

    /// Returns the requested attention implementation selector.
    #[must_use]
    pub fn implementation(&self) -> AttentionImplementation {
        self.implementation
    }

    /// Returns generated Philox dropout settings, or `None` for disabled or custom-mask dropout.
    #[must_use]
    pub fn dropout(&self) -> Option<AttentionDropoutConfig> {
        self.score.dropout()
    }

    pub(crate) fn remove_zero_probability_dropout(&mut self) {
        if self
            .dropout()
            .is_some_and(|dropout| dropout.probability() == 0.0)
        {
            self.score.dropout = None;
        }
    }

    /// Returns the complete dropout mode, including generated and caller-provided masks.
    #[must_use]
    pub fn dropout_mode(&self) -> AttentionDropoutMode {
        self.score.dropout_mode()
    }

    /// Returns the configured Philox offset tensor identity, if generated dropout uses one.
    #[must_use]
    pub fn dropout_offset(&self) -> Option<TensorId> {
        self.score.dropout_offset()
    }

    /// Returns whether native fused multiply-add operations are requested to be unfused.
    #[must_use]
    pub fn unfuse_fma(&self) -> bool {
        self.unfuse_fma
    }

    /// Returns the independent key/value paging settings and sequence-length hint.
    #[must_use]
    pub fn paged(&self) -> AttentionPagedKvCache {
        self.paged
    }

    /// Returns the block-mask tensor identity, or `None` when another mask mode is selected.
    #[must_use]
    pub fn block_mask(&self) -> Option<TensorId> {
        self.score.modifiers().block_mask()
    }
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self::new()
    }
}
