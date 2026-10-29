use std::{collections::BTreeMap, sync::Arc};

use serde::{Deserialize, Serialize};

use crate::{
    backend::{
        execution::attention::{
            SdpaBackwardConfig, SdpaBackwardOperation, SdpaForwardConfig, SdpaForwardDropout,
            SdpaForwardOperation, SdpaForwardSoftmax, SdpaSubgraph, SoftmaxOperation,
            SoftmaxOperationConfig,
        },
        tensor::{Tensor, TensorId},
    },
    error::Result,
    frontend::{
        error::{Error as FrontendError, SdpaRequirement},
        graph::{DataTypePolicy, Graph, SoftmaxOutputs},
        lower::{LoweredOperation, LoweringContext, optional_tensor_at, tensor_at},
        operation::{
            AttentionSequenceInputs, DirectSdpaDropout, SdpaBackwardSink, SdpaBackwardTensors,
            SdpaForwardQuantization, SdpaForwardTensors, SdpaFusedMaskMode, SdpaScoreModifiers,
            SdpaScoreSubgraph, SequenceLength, SoftmaxStatsBase,
        },
    },
};

/// Options for the direct cuDNN SDPA forward operation.
///
/// Covers variable-length sequence tensors, padding mask, paged attention page
/// tables, block masks, dropout, random-number-generator dump, and the
/// unfused-FMA option described for frontend SDPA attributes.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DirectSdpaForwardConfig {
    #[serde(default)]
    pub(crate) data_type_policy: DataTypePolicy,
    pub(crate) stats_base: SoftmaxStatsBase,
    pub(crate) sequence: AttentionSequenceInputs<TensorId>,
    pub(crate) padding_mask: bool,
    pub(crate) key_page_table: Option<TensorId>,
    pub(crate) value_page_table: Option<TensorId>,
    pub(crate) block_mask: Option<TensorId>,
    pub(crate) dropout: Option<DirectSdpaDropout>,
    pub(crate) unfuse_fma: bool,
    pub(crate) quantization: Option<SdpaForwardQuantization<TensorId>>,
    pub(crate) max_sequence_length_key_value: Option<i64>,
}

impl DirectSdpaForwardConfig {
    /// Creates direct forward settings with natural-log statistics and no masking, paging, dropout, quantization, or sequence-length hints.
    #[must_use]
    pub fn new() -> Self {
        Self {
            data_type_policy: DataTypePolicy::default(),
            stats_base: SoftmaxStatsBase::NaturalLog,
            sequence: AttentionSequenceInputs::default(),
            padding_mask: false,
            key_page_table: None,
            value_page_table: None,
            block_mask: None,
            dropout: None,
            unfuse_fma: false,
            quantization: None,
            max_sequence_length_key_value: None,
        }
    }

    /// Selects softmax statistics encoding. SDPA backward requires natural log.
    #[must_use]
    pub fn with_stats_base(mut self, base: SoftmaxStatsBase) -> Self {
        self.stats_base = base;
        self
    }

    /// Encoding of saved forward softmax statistics.
    #[must_use]
    pub fn stats_base(self) -> SoftmaxStatsBase {
        self.stats_base
    }

    /// Sets or clears the native scaling and Amax operand group.
    #[must_use]
    pub fn with_quantization(
        mut self,
        quantization: impl Into<Option<SdpaForwardQuantization<TensorId>>>,
    ) -> Self {
        self.quantization = quantization.into();
        self
    }

    /// Sets or clears the logical K/V length used for shape inference and score subgraphs.
    /// Unified cuDNN derives runtime limits from its cache and sequence descriptors.
    /// This hint is not a native SDPA attribute.
    /// Dense caches keep their declared geometry; the hint does not resize either operand.
    #[must_use]
    pub fn with_max_sequence_length_key_value(mut self, length: impl Into<Option<i64>>) -> Self {
        self.max_sequence_length_key_value = length.into();
        self
    }

    /// Returns the logical key/value sequence-length hint in tokens, if supplied.
    #[must_use]
    pub fn max_sequence_length_key_value(self) -> Option<i64> {
        self.max_sequence_length_key_value
    }

    /// Returns the optional native scaling and absolute-maximum operand group.
    #[must_use]
    pub fn quantization(self) -> Option<SdpaForwardQuantization<TensorId>> {
        self.quantization
    }

    /// Replaces query and key/value sequence-length inputs while preserving other settings.
    #[must_use]
    pub fn with_sequence_inputs(mut self, sequence: AttentionSequenceInputs<TensorId>) -> Self {
        self.sequence = sequence;
        self
    }

    /// Sets or clears the key page table independently of the value cache and sequence inputs.
    #[must_use]
    pub fn with_key_page_table(mut self, key_page_table: impl Into<Option<TensorId>>) -> Self {
        self.key_page_table = key_page_table.into();
        self
    }

    /// Sets or clears the value page table independently of the key cache and sequence inputs.
    #[must_use]
    pub fn with_value_page_table(mut self, value_page_table: impl Into<Option<TensorId>>) -> Self {
        self.value_page_table = value_page_table.into();
        self
    }

    /// Sets or clears the block mask without changing other masking options.
    #[must_use]
    pub fn with_block_mask(mut self, block_mask: impl Into<Option<TensorId>>) -> Self {
        self.block_mask = block_mask.into();
        self
    }

    /// Sets or clears the complete native Philox dropout request.
    #[must_use]
    pub fn with_dropout(mut self, dropout: impl Into<Option<DirectSdpaDropout>>) -> Self {
        self.dropout = dropout.into();
        self
    }

    /// Sets or clears the native unfused-FMA flag without changing other options.
    #[must_use]
    pub fn with_unfuse_fma(mut self, enabled: bool) -> Self {
        self.unfuse_fma = enabled;
        self
    }

    /// Returns the per-batch query-length tensor identity, or `None` for cumulative or absent lengths.
    pub fn sequence_length_query(self) -> Option<TensorId> {
        self.sequence.query.and_then(SequenceLength::per_batch)
    }

    /// Returns the per-batch key/value-length tensor identity, or `None` for cumulative or absent lengths.
    pub fn sequence_length_key_value(self) -> Option<TensorId> {
        self.sequence.key_value.and_then(SequenceLength::per_batch)
    }

    /// Returns the query and key/value sequence-length inputs.
    #[must_use]
    pub fn sequence_inputs(self) -> AttentionSequenceInputs<TensorId> {
        self.sequence
    }

    /// Returns whether the native padding-mask flag is enabled.
    #[must_use]
    pub fn padding_mask(self) -> bool {
        self.padding_mask
    }

    /// Sets or clears the native padding-mask flag while retaining sequence operands.
    #[must_use]
    pub fn with_padding_mask(mut self, enabled: bool) -> Self {
        self.padding_mask = enabled;
        self
    }

    /// Returns the key-cache page-table tensor identity, if supplied.
    #[must_use]
    pub fn key_page_table(self) -> Option<TensorId> {
        self.key_page_table
    }

    /// Returns the value-cache page-table tensor identity, if supplied.
    #[must_use]
    pub fn value_page_table(self) -> Option<TensorId> {
        self.value_page_table
    }

    /// Returns the optional block-mask tensor identity.
    #[must_use]
    pub fn block_mask(self) -> Option<TensorId> {
        self.block_mask
    }

    /// Returns the optional native Philox dropout request.
    #[must_use]
    pub fn dropout(self) -> Option<DirectSdpaDropout> {
        self.dropout
    }

    /// Returns whether native fused multiply-add operations are requested to be unfused.
    #[must_use]
    pub fn unfuse_fma(self) -> bool {
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
    #[serde(default)]
    pub(crate) data_type_policy: DataTypePolicy,
    pub(crate) sequence: AttentionSequenceInputs<TensorId>,
    pub(crate) padding_mask: bool,
    pub(crate) causal_mask: bool,
    pub(crate) score_subgraph: Option<SdpaScoreSubgraph>,
    pub(crate) sink: Option<SdpaBackwardSink<TensorId>>,
    pub(crate) max_total_sequence_lengths: Option<AttentionMaxTotalSequenceLengths>,
}

/// Optional bounds for the total token counts of packed query and key/value tensors.
/// Each side can be specified independently.
/// C++ forward attention preserves these frontend hints without lowering native attributes.
/// Quantized C++ backward composites validate packed layout but do not use these hints to size intermediates.
/// Direct native backward operations receive the corresponding backend attributes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttentionMaxTotalSequenceLengths {
    /// Optional upper bound on the total packed query-token count across batches.
    pub query: Option<i64>,
    /// Optional upper bound on the total packed key/value-token count across batches.
    pub key_value: Option<i64>,
}

impl AttentionMaxTotalSequenceLengths {
    /// Returns whether neither packed-token bound is supplied.
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.query.is_none() && self.key_value.is_none()
    }
}

impl DirectSdpaBackwardConfig {
    /// Creates direct backward settings without sequence inputs, masks, score subgraphs, sinks, or packed-token bounds.
    #[must_use]
    pub fn new() -> Self {
        Self {
            data_type_policy: DataTypePolicy::default(),
            sequence: AttentionSequenceInputs::default(),
            padding_mask: false,
            causal_mask: false,
            score_subgraph: None,
            sink: None,
            max_total_sequence_lengths: None,
        }
    }

    /// Replaces query and key/value sequence-length inputs while preserving other settings.
    #[must_use]
    pub fn with_sequence_inputs(mut self, sequence: AttentionSequenceInputs<TensorId>) -> Self {
        self.sequence = sequence;
        self
    }

    /// Sets or clears the built-in causal mask while retaining other score configuration.
    #[must_use]
    pub fn with_causal_mask(mut self, enabled: bool) -> Self {
        self.causal_mask = enabled;
        self
    }

    /// Sets or clears the score subgraph while retaining built-in mask and sequence settings.
    #[must_use]
    pub fn with_score_subgraph(
        mut self,
        score_subgraph: impl Into<Option<SdpaScoreSubgraph>>,
    ) -> Self {
        self.score_subgraph = score_subgraph.into();
        self
    }

    /// Selects sink logits and their optional gradient, or clears the complete sink request.
    #[must_use]
    pub fn with_sink(mut self, sink: impl Into<Option<SdpaBackwardSink<TensorId>>>) -> Self {
        self.sink = sink.into();
        self
    }

    /// Sets or clears the packed query and key/value token bounds lowered to native backward attributes.
    #[must_use]
    pub fn with_max_total_sequence_lengths(
        mut self,
        max_total_sequence_lengths: impl Into<Option<AttentionMaxTotalSequenceLengths>>,
    ) -> Self {
        self.max_total_sequence_lengths = max_total_sequence_lengths.into();
        self
    }

    /// Returns the per-batch query-length tensor identity, or `None` for cumulative or absent lengths.
    pub fn sequence_length_query(&self) -> Option<TensorId> {
        self.sequence.query.and_then(SequenceLength::per_batch)
    }

    /// Returns the per-batch key/value-length tensor identity, or `None` for cumulative or absent lengths.
    pub fn sequence_length_key_value(&self) -> Option<TensorId> {
        self.sequence.key_value.and_then(SequenceLength::per_batch)
    }

    /// Returns the query and key/value sequence-length inputs.
    #[must_use]
    pub fn sequence_inputs(&self) -> AttentionSequenceInputs<TensorId> {
        self.sequence
    }

    /// Returns whether the native padding-mask flag is enabled.
    #[must_use]
    pub fn padding_mask(&self) -> bool {
        self.padding_mask
    }

    /// Sets or clears the native padding-mask flag while retaining sequence operands.
    #[must_use]
    pub fn with_padding_mask(mut self, enabled: bool) -> Self {
        self.padding_mask = enabled;
        self
    }

    /// C++ causal-like support rule: a top-left-aligned right bound is present.
    #[must_use]
    pub fn causal_mask(&self) -> bool {
        self.causal_mask
    }

    /// Borrows the optional score-modifier graph used by backward attention.
    #[must_use]
    pub fn score_subgraph(&self) -> Option<&SdpaScoreSubgraph> {
        self.score_subgraph.as_ref()
    }

    /// Returns the sink logits and optional sink-gradient request, if configured.
    #[must_use]
    pub fn sink(&self) -> Option<SdpaBackwardSink<TensorId>> {
        self.sink
    }

    /// Returns the optional bounds for total packed query and key/value token counts.
    #[must_use]
    pub fn max_total_sequence_lengths(&self) -> Option<AttentionMaxTotalSequenceLengths> {
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
#[expect(
    clippy::large_enum_variant,
    reason = "graph nodes are built once per graph; boxing variants would only complicate matching"
)]
pub enum SdpaOperation {
    /// Direct native attention forward operation with optional explicit softmax and score-subgraph descriptors.
    Forward {
        /// Primary query, key, value, attention-scale, and output tensor identities.
        tensors: SdpaForwardTensors<TensorId>,
        /// Optional saved softmax log-sum-exp statistics output.
        stats: Option<TensorId>,
        /// Optional per-row logit maximum from an explicit softmax descriptor.
        logit_max: Option<TensorId>,
        /// Optional per-row exponential sum from an explicit softmax descriptor.
        score_sum_exp: Option<TensorId>,
        /// Input tensor for an explicit softmax descriptor; must be paired with `softmax_s`.
        softmax_p: Option<TensorId>,
        /// Output probability tensor for an explicit softmax descriptor; must be paired with `softmax_p`.
        softmax_s: Option<TensorId>,
        /// Optional sink-logit tensor for the explicit softmax descriptor.
        sink: Option<TensorId>,
        /// Frontend score modifiers retained for score-graph construction.
        score_modifiers: SdpaScoreModifiers,
        /// Optional frontend graph applied to attention scores.
        score_subgraph: Option<Box<SdpaScoreSubgraph>>,
        /// Direct native forward settings and optional quantization operands.
        config: Box<DirectSdpaForwardConfig>,
    },
    /// Direct native attention backward operation producing gradients from saved forward tensors.
    Backward {
        /// Forward activations, saved statistics, scale, and gradient tensor identities.
        tensors: SdpaBackwardTensors<TensorId>,
        /// Direct native backward settings and optional sink-gradient operands.
        config: Box<DirectSdpaBackwardConfig>,
    },
}

impl SdpaOperation {
    pub(crate) fn score_subgraph(&self) -> Option<&SdpaScoreSubgraph> {
        match self {
            Self::Forward { score_subgraph, .. } => score_subgraph.as_deref(),
            Self::Backward { config, .. } => config.score_subgraph(),
        }
    }

    /// Validates data ranks before score-graph construction interprets attention geometry.
    pub(crate) fn validate_graph(&self, graph: &Graph) -> Result<()> {
        match self {
            Self::Forward {
                tensors, config, ..
            } => {
                config.sequence_inputs().validate_forward()?;
                for (id, name) in [
                    (tensors.query, "sdpa query"),
                    (tensors.key, "sdpa key"),
                    (tensors.value, "sdpa value"),
                    (tensors.output, "sdpa output"),
                ] {
                    graph.validate_tensor_rank(id, 4, name)?;
                }
            }
            Self::Backward { tensors, .. } => {
                for id in tensors.activations_and_gradients() {
                    graph.validate_tensor_rank(id, 4, "sdpa backward data")?;
                }
                graph.validate_tensor_rank(tensors.stats, 4, "sdpa backward statistics")?;
            }
        }
        Ok(())
    }

    pub(crate) fn lower(
        &self,
        context: &LoweringContext<'_>,
        subgraph: Option<SdpaSubgraph<'_>>,
    ) -> Result<LoweredOperation> {
        self.lower_with_tensors(context.backend_tensors(), subgraph)
    }

    #[expect(
        clippy::too_many_lines,
        reason = "the composite attention operation builds and validates its complete tensor graph"
    )]
    pub(crate) fn lower_with_tensors(
        &self,
        tensors: &BTreeMap<TensorId, Tensor>,
        subgraph: Option<SdpaSubgraph<'_>>,
    ) -> Result<LoweredOperation> {
        match self {
            Self::Forward {
                tensors: forward_tensors,
                stats,
                logit_max,
                score_sum_exp,
                softmax_p,
                softmax_s,
                sink,
                config,
                ..
            } => {
                let softmax = match (*softmax_p, *softmax_s) {
                    (Some(softmax_p), Some(softmax_s)) => Some(SoftmaxOperation::create(
                        tensor_at(tensors, softmax_p)?,
                        SoftmaxOutputs {
                            output: softmax_s,
                            stats: *stats,
                            max: *logit_max,
                            sum_exp: *score_sum_exp,
                        }
                        .try_map(|id| tensor_at(tensors, id))?,
                        SoftmaxOperationConfig {
                            stats_base: config.stats_base,
                            sink: optional_tensor_at(tensors, *sink)?,
                        },
                    )?),
                    (None, None) => {
                        if stats.is_some() && config.stats_base == SoftmaxStatsBase::Log2 {
                            return Err(FrontendError::Compile(
                                "log2 statistics require explicit unified softmax descriptors"
                                    .into(),
                            )
                            .into());
                        }
                        None
                    }
                    _ => {
                        return Err(FrontendError::SdpaRequires {
                            requirement: SdpaRequirement::ProbabilityAndSumDescriptors,
                            operation: "unified sdpa softmax",
                        }
                        .into());
                    }
                };

                let dropout = config
                    .dropout()
                    .map(|dropout| -> Result<_> {
                        Ok(SdpaForwardDropout {
                            probability: dropout.probability(),
                            seed: tensor_at(tensors, dropout.seed())?,
                            offset: tensor_at(tensors, dropout.offset())?,
                            random_number_generator_dump: optional_tensor_at(
                                tensors,
                                dropout.random_number_generator_dump(),
                            )?,
                        })
                    })
                    .transpose()?;

                Ok(LoweredOperation::SdpaForward(SdpaForwardOperation::create(
                    forward_tensors.try_map(|id| tensor_at(tensors, id))?,
                    SdpaForwardConfig {
                        sequence: config
                            .sequence_inputs()
                            .try_map(|id| tensor_at(tensors, id))?,
                        page_table_k: config
                            .key_page_table()
                            .map(|tensor| tensor_at(tensors, tensor))
                            .transpose()?,
                        page_table_v: config
                            .value_page_table()
                            .map(|tensor| tensor_at(tensors, tensor))
                            .transpose()?,
                        block_mask: config
                            .block_mask()
                            .map(|tensor| tensor_at(tensors, tensor))
                            .transpose()?,
                        softmax: match softmax {
                            Some(softmax) => SdpaForwardSoftmax::Descriptor(Arc::new(softmax)),
                            None => SdpaForwardSoftmax::Default {
                                stats: optional_tensor_at(tensors, *stats)?,
                            },
                        },
                        subgraph,
                        dropout,
                        unfuse_fma: config.unfuse_fma(),
                        quantization: config
                            .quantization()
                            .map(|quantization| {
                                quantization.try_map(|tensor| tensor_at(tensors, tensor))
                            })
                            .transpose()?,
                    },
                )?))
            }
            Self::Backward {
                tensors: backward_tensors,
                config,
            } => {
                let max_total_sequence_lengths = config.max_total_sequence_lengths();

                Ok(LoweredOperation::SdpaBackward(
                    SdpaBackwardOperation::create(
                        (*backward_tensors).try_map(|id| tensor_at(tensors, id))?,
                        SdpaBackwardConfig {
                            sequence: config
                                .sequence_inputs()
                                .try_map(|id| tensor_at(tensors, id))?,
                            subgraph,
                            sink: config
                                .sink()
                                .map(|sink| sink.try_map(|id| tensor_at(tensors, id)))
                                .transpose()?,
                            max_total_sequence_length_query: max_total_sequence_lengths
                                .and_then(|lengths| lengths.query),
                            max_total_sequence_length_key_value: max_total_sequence_lengths
                                .and_then(|lengths| lengths.key_value),
                        },
                    )?,
                ))
            }
        }
    }
}

/// Attributes for fused scaled dot product attention.
///
/// SDPA computes `softmax(QK^T / sqrt(d))V`, with optional attention scale,
/// causal masking, variable-length sequence tensors, and statistics outputs.
///
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SdpaConfig {
    pub(crate) name: Option<String>,
    pub(crate) aux_outputs: SdpaAuxOutputRequest,
    pub(crate) stats_base: SoftmaxStatsBase,
    pub(crate) mask: SdpaFusedMaskMode,
    #[serde(default, with = "crate::scalar::float::optional_f32")]
    pub(crate) attention_scale: Option<f32>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SdpaAuxOutputRequest {
    pub(crate) softmax: SdpaSoftmaxAuxOutputRequest,
    pub(crate) quantized: SdpaQuantizedAuxOutputRequest,
    pub(crate) rng_dump: bool,
}

/// Independently requested softmax outputs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SdpaSoftmaxAuxOutputRequest {
    pub stats: bool,
    pub logit_max: bool,
    pub score_sum_exp: bool,
}

/// Independently requested pre-quantization absolute maxima.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SdpaQuantizedAuxOutputRequest {
    pub absolute_max_probabilities: bool,
    pub absolute_max_output: bool,
}

impl SdpaAuxOutputRequest {
    pub(crate) fn new(
        softmax: SdpaSoftmaxAuxOutputRequest,
        quantized: SdpaQuantizedAuxOutputRequest,
        rng_dump: bool,
    ) -> Self {
        Self {
            softmax,
            quantized,
            rng_dump,
        }
    }

    pub fn rng_dump(self) -> bool {
        self.rng_dump
    }

    pub fn stats(self) -> bool {
        self.softmax.stats
    }

    pub fn absolute_max_s(self) -> bool {
        self.quantized.absolute_max_probabilities
    }

    pub fn absolute_max_o(self) -> bool {
        self.quantized.absolute_max_output
    }

    pub fn softmax(self) -> SdpaSoftmaxAuxOutputRequest {
        self.softmax
    }

    pub fn quantized(self) -> SdpaQuantizedAuxOutputRequest {
        self.quantized
    }
}

impl SdpaSoftmaxAuxOutputRequest {
    pub fn stats(self) -> bool {
        self.stats
    }

    pub fn logit_max(self) -> bool {
        self.logit_max
    }

    pub fn score_sum_exp(self) -> bool {
        self.score_sum_exp
    }
}

impl SdpaQuantizedAuxOutputRequest {
    pub fn absolute_max_s(self) -> bool {
        self.absolute_max_probabilities
    }

    pub fn absolute_max_o(self) -> bool {
        self.absolute_max_output
    }
}

impl SdpaConfig {
    /// Creates fused attention settings with natural-log statistics, no masks or auxiliary outputs, and no numeric scale override.
    #[must_use]
    pub fn new() -> Self {
        Self {
            name: None,
            aux_outputs: SdpaAuxOutputRequest::default(),
            stats_base: SoftmaxStatsBase::NaturalLog,
            mask: SdpaFusedMaskMode::None,
            attention_scale: None,
        }
    }

    pub(crate) fn with_aux_outputs(mut self, aux_outputs: SdpaAuxOutputRequest) -> Self {
        self.aux_outputs = aux_outputs;
        self
    }

    /// Selects the encoding of requested statistics, including quantized attention.
    /// Current backward operations consume natural-log statistics.
    #[must_use]
    pub fn with_stats_base(mut self, base: SoftmaxStatsBase) -> Self {
        self.stats_base = base;
        self
    }

    /// Returns the encoding base for saved softmax statistics.
    #[must_use]
    pub fn stats_base(&self) -> SoftmaxStatsBase {
        self.stats_base
    }

    /// Names inferred attention results using the scope and native output-port names.
    /// Explicit tensor declarations retain their names; an empty scope disables inferred naming.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Replaces the fused attention mask mode.
    #[must_use]
    pub fn with_mask(mut self, mask: SdpaFusedMaskMode) -> Self {
        self.mask = mask;
        self
    }

    /// Sets the numeric attention factor, or removes it with `None` while retaining the tensor scale.
    #[must_use]
    pub fn with_attention_scale(mut self, attention_scale: impl Into<Option<f32>>) -> Self {
        self.attention_scale = attention_scale.into();
        self
    }

    /// Returns the nonempty attention name, or `None` when unnamed.
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref().filter(|name| !name.is_empty())
    }

    pub(crate) fn aux_outputs(&self) -> SdpaAuxOutputRequest {
        self.aux_outputs
    }

    /// Returns the fused attention mask mode.
    #[must_use]
    pub fn mask(&self) -> SdpaFusedMaskMode {
        self.mask
    }

    /// Returns the optional numeric attention multiplier applied in addition to the tensor scale.
    #[must_use]
    pub fn attention_scale(&self) -> Option<f32> {
        self.attention_scale
    }
}

impl Default for SdpaConfig {
    fn default() -> Self {
        Self::new()
    }
}
