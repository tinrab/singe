use serde::{Deserialize, Serialize};

use crate::{
    backend::tensor::TensorId,
    frontend::{
        graph::Graph,
        operation::{
            AttentionDiagonalAlignment, AttentionDiagonalBand, AttentionDropoutConfig,
            AttentionDropoutMask, AttentionDropoutMode, AttentionMaskMode, AttentionSequenceInputs,
            SequenceLength,
        },
    },
};

/// User-provided graph applied to SDPA score tensors.
///
/// The C++ frontend exposes score modification hooks through `set_score_mod`;
/// this type records a Rust frontend subgraph and its score input/output tensor
/// IDs for the same role.
/// Ordinary operation families retain their operands and attributes when imported, including
/// convolution, normalization, block scaling, RNG, `RoPE`, resampling, paged loads and `MoE`.
/// Native engine support still determines which score-modifier graph patterns can execute.
/// Direct native SDPA nodes preserve configuration-held tensor operands during import.
/// Nested score graphs follow remapped external captures while keeping their local tensor scopes.
/// Deferred attention requests expand against the enclosing target before import.
/// Backward heuristic overrides remain attached to the combined graph's automatic planning policy.
/// Automatic selection reports conflicting choices; preparation and explicit custom planning remain available.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SdpaScoreSubgraph {
    pub(crate) graph: Graph,
    pub(crate) input: TensorId,
    pub(crate) output: TensorId,
}

impl SdpaScoreSubgraph {
    /// Records the score graph and its local input and output tensor identities.
    #[must_use]
    pub fn new(graph: Graph, input: TensorId, output: TensorId) -> Self {
        Self {
            graph,
            input,
            output,
        }
    }

    /// Borrows the graph imported when this score modifier is expanded.
    #[must_use]
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// Returns the local tensor identity replaced by the incoming attention scores.
    #[must_use]
    pub fn input(&self) -> TensorId {
        self.input
    }

    /// Returns the local tensor identity used as the modified attention scores.
    #[must_use]
    pub fn output(&self) -> TensorId {
        self.output
    }
}

/// `ALiBi` slope source, independent of attention input/output precision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlibiBias {
    /// Uses the frontend-generated standard `ALiBi` slope schedule.
    Standard,
    /// Uses the supplied FP32 tensor as the `ALiBi` slopes.
    Slopes(TensorId),
}

/// Optional SDPA score modifiers.
///
/// These cover additive bias, sink token, `ALiBi`, additive/dropout masks,
/// variable-length padding masks, causal masks, and sliding windows documented
/// for frontend SDPA score modification options.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SdpaScoreModifiers {
    pub(crate) bias: Option<TensorId>,
    pub(crate) sink_token: Option<TensorId>,
    pub(crate) alibi: Option<AlibiBias>,
    pub(crate) additive_mask: Option<TensorId>,
    pub(crate) custom_dropout: Option<AttentionDropoutMask>,
    pub(crate) mask_mode: AttentionMaskMode,
    pub(crate) sequence: AttentionSequenceInputs<TensorId>,
}

impl SdpaScoreModifiers {
    /// Creates score modifiers with no bias, sink, masking, dropout, or sequence-length operands.
    #[must_use]
    pub fn new() -> Self {
        Self {
            bias: None,
            sink_token: None,
            alibi: None,
            additive_mask: None,
            custom_dropout: None,
            mask_mode: AttentionMaskMode::None,
            sequence: AttentionSequenceInputs::default(),
        }
    }

    /// Sets or clears the bias operand without changing other score modifiers.
    /// Bias tensors have four dimensions and a unit last stride, matching C++ forward and backward attention.
    #[must_use]
    pub fn with_bias(mut self, bias: impl Into<Option<TensorId>>) -> Self {
        self.bias = bias.into();
        self
    }

    /// Sets or clears the sink operand without changing other score modifiers.
    #[must_use]
    pub fn with_sink_token(mut self, sink_token: impl Into<Option<TensorId>>) -> Self {
        self.sink_token = sink_token.into();
        self
    }

    /// Selects the standard C++ `ALiBi` schedule, or clears the complete `ALiBi` request.
    /// Enabling replaces caller-supplied slopes with frontend-owned standard slopes.
    #[must_use]
    pub fn with_alibi(mut self, enabled: bool) -> Self {
        self.alibi = enabled.then_some(AlibiBias::Standard);
        self
    }

    /// Uses caller-supplied FP32 slopes instead of the standard schedule.
    /// `None` clears the complete `ALiBi` request.
    #[must_use]
    pub fn with_alibi_slopes(mut self, alibi_slopes: impl Into<Option<TensorId>>) -> Self {
        self.alibi = alibi_slopes.into().map(AlibiBias::Slopes);
        self
    }

    /// Sets or clears the additive mask independently of the mask mode.
    #[must_use]
    pub fn with_additive_mask(mut self, additive_mask: impl Into<Option<TensorId>>) -> Self {
        self.additive_mask = additive_mask.into();
        self
    }

    /// Sets the complete custom dropout request, or clears it with `None`.
    /// Preserves bias, masks, sequence inputs and other score modifiers.
    #[must_use]
    pub fn with_dropout(mut self, dropout: impl Into<Option<AttentionDropoutMask>>) -> Self {
        self.custom_dropout = dropout.into();
        self
    }

    fn set_dropout_mode(
        &mut self,
        dropout: &mut Option<AttentionDropoutConfig>,
        dropout_mode: AttentionDropoutMode,
    ) {
        *dropout = None;
        self.custom_dropout = None;
        match dropout_mode {
            AttentionDropoutMode::None => {}
            AttentionDropoutMode::Philox(config) => {
                *dropout = Some(config);
            }
            AttentionDropoutMode::CustomMask(mask) => {
                self.custom_dropout = Some(mask);
            }
        }
    }

    fn dropout_mode(&self, dropout: Option<AttentionDropoutConfig>) -> AttentionDropoutMode {
        if let Some(dropout) = dropout {
            return AttentionDropoutMode::Philox(dropout);
        }
        self.custom_dropout
            .map_or(AttentionDropoutMode::None, AttentionDropoutMode::CustomMask)
    }

    /// Replaces the mask mode while preserving independent additive masks and other score modifiers.
    #[must_use]
    pub fn with_mask_mode(mut self, mask_mode: AttentionMaskMode) -> Self {
        self.mask_mode = mask_mode;
        self
    }

    /// Returns the selected structural mask mode.
    #[must_use]
    pub fn mask_mode(&self) -> AttentionMaskMode {
        self.mask_mode
    }

    /// Returns the optional additive bias tensor identity.
    #[must_use]
    pub fn bias(&self) -> Option<TensorId> {
        self.bias
    }

    /// Returns the optional attention sink tensor identity.
    #[must_use]
    pub fn sink_token(&self) -> Option<TensorId> {
        self.sink_token
    }

    /// Returns caller-supplied slope tensor identity, or `None` for standard or disabled `ALiBi`.
    #[must_use]
    pub fn alibi_slopes(&self) -> Option<TensorId> {
        match self.alibi {
            Some(AlibiBias::Slopes(tensor)) => Some(tensor),
            _ => None,
        }
    }

    /// Returns the standard or caller-supplied `ALiBi` request, or `None` when disabled.
    #[must_use]
    pub fn alibi(&self) -> Option<AlibiBias> {
        self.alibi
    }

    /// Returns whether standard or caller-supplied `ALiBi` is enabled.
    #[must_use]
    pub fn has_alibi(&self) -> bool {
        self.alibi.is_some()
    }

    /// Returns the independent additive-mask tensor identity, if present.
    #[must_use]
    pub fn additive_mask(&self) -> Option<TensorId> {
        self.additive_mask
    }

    /// Returns the caller-provided dropout mask tensor identity, if configured.
    #[must_use]
    pub fn dropout_mask(&self) -> Option<TensorId> {
        self.custom_dropout.map(|dropout| dropout.mask)
    }

    /// Returns the caller-provided dropout scaling tensor identity, if configured.
    #[must_use]
    pub fn dropout_scale(&self) -> Option<TensorId> {
        self.custom_dropout.map(|dropout| dropout.scale)
    }

    /// The complete caller-provided dropout request, independently of generated Philox dropout.
    #[must_use]
    pub fn custom_dropout(&self) -> Option<AttentionDropoutMask> {
        self.custom_dropout
    }

    /// Replaces query and key/value sequence-length inputs while preserving other modifiers.
    #[must_use]
    pub fn with_sequence_inputs(mut self, sequence: AttentionSequenceInputs<TensorId>) -> Self {
        self.sequence = sequence;
        self
    }

    /// Returns the configured query and key/value sequence-length inputs.
    #[must_use]
    pub fn sequence_inputs(&self) -> AttentionSequenceInputs<TensorId> {
        self.sequence
    }

    /// Returns the per-batch query-length tensor identity, or `None` for cumulative or absent lengths.
    pub fn sequence_length_query(&self) -> Option<TensorId> {
        self.sequence.query.and_then(SequenceLength::per_batch)
    }

    /// Returns the per-batch key/value-length tensor identity, or `None` for cumulative or absent lengths.
    pub fn sequence_length_key_value(&self) -> Option<TensorId> {
        self.sequence.key_value.and_then(SequenceLength::per_batch)
    }

    /// Returns both per-batch length tensor identities when both are configured; otherwise returns `None`.
    #[must_use]
    pub fn sequence_lengths(&self) -> Option<(TensorId, TensorId)> {
        self.sequence.per_batch_pair()
    }

    /// Returns whether either query or key/value sequence-length operand is configured.
    #[must_use]
    pub fn has_sequence_lengths(&self) -> bool {
        !self.sequence.is_empty()
    }

    /// Returns whether padding masking is enabled directly or within a diagonal band.
    pub fn padding_mask(&self) -> bool {
        self.mask_mode == AttentionMaskMode::Padding
            || self
                .diagonal_band()
                .is_some_and(AttentionDiagonalBand::padding)
    }

    /// Returns whether a top-left diagonal band has a right bound, including nonzero right bounds.
    #[must_use]
    pub fn causal_mask(&self) -> bool {
        self.diagonal_band().is_some_and(|band| {
            band.alignment() == AttentionDiagonalAlignment::TopLeft && band.right_window().is_some()
        })
    }

    /// C++ bottom-right support rule, including nonzero right bounds and windows.
    #[must_use]
    pub fn causal_bottom_right(&self) -> bool {
        self.bottom_right_alignment() && self.right_window().is_some()
    }

    /// Returns whether a bottom-right diagonal band has at least one window bound.
    #[must_use]
    pub fn bottom_right_alignment(&self) -> bool {
        self.diagonal_band().is_some_and(|band| {
            band.alignment() == AttentionDiagonalAlignment::BottomRight && band.has_bounds()
        })
    }

    /// Returns the diagonal-band configuration, or `None` for another mask mode.
    #[must_use]
    pub fn diagonal_band(&self) -> Option<AttentionDiagonalBand> {
        match self.mask_mode {
            AttentionMaskMode::DiagonalBand(band) => Some(band),
            _ => None,
        }
    }

    /// Returns the diagonal band’s left window bound, or `None` when unbounded or another mask mode is selected.
    pub fn left_window(&self) -> Option<i64> {
        self.diagonal_band()
            .and_then(AttentionDiagonalBand::left_window)
    }

    /// Returns the diagonal band’s right window bound, or `None` when unbounded or another mask mode is selected.
    pub fn right_window(&self) -> Option<i64> {
        self.diagonal_band()
            .and_then(AttentionDiagonalBand::right_window)
    }

    /// Returns the block-mask tensor identity, or `None` for another mask mode.
    #[must_use]
    pub fn block_mask(&self) -> Option<TensorId> {
        match self.mask_mode {
            AttentionMaskMode::Block(block_mask) => Some(block_mask),
            _ => None,
        }
    }

    /// Returns whether a caller-provided dropout mask and scale are configured.
    #[must_use]
    pub fn has_custom_dropout(&self) -> bool {
        self.custom_dropout.is_some()
    }

    /// Returns whether the structural mask mode enables masking; a diagonal band requires a bound or padding.
    #[must_use]
    pub fn has_any_masking(&self) -> bool {
        match self.mask_mode {
            AttentionMaskMode::None => false,
            AttentionMaskMode::DiagonalBand(band) => band.has_bounds() || band.padding(),
            _ => true,
        }
    }

    /// Returns whether bias, `ALiBi`, an additive mask, or a bounded diagonal band modifies pre-softmax scores.
    pub fn has_pre_softmax_subgraph_modifier(&self) -> bool {
        self.bias.is_some()
            || self.has_alibi()
            || self.additive_mask.is_some()
            || self
                .diagonal_band()
                .is_some_and(AttentionDiagonalBand::has_bounds)
    }

    /// Returns whether bias, `ALiBi`, or custom dropout conflicts with the bottom-right causal path.
    #[must_use]
    pub fn has_causal_bottom_right_incompatible_modifier(&self) -> bool {
        self.bias.is_some() || self.has_alibi() || self.has_custom_dropout()
    }

    /// Returns whether bias, a sink, `ALiBi`, an additive mask, custom dropout, paired per-batch length tensors, or structural masking is present.
    #[must_use]
    pub fn has_any_modifier(&self) -> bool {
        self.bias.is_some()
            || self.sink_token.is_some()
            || self.has_alibi()
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

/// Attention score modifiers, an optional score subgraph, and generated or caller-provided dropout settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionScoreConfig {
    pub(crate) modifiers: SdpaScoreModifiers,
    pub(crate) score_subgraph: Option<SdpaScoreSubgraph>,
    pub(crate) dropout: Option<AttentionDropoutConfig>,
}

impl AttentionScoreConfig {
    /// Creates attention score settings with no modifiers, score subgraph, or dropout.
    #[must_use]
    pub fn new() -> Self {
        Self {
            modifiers: SdpaScoreModifiers::new(),
            score_subgraph: None,
            dropout: None,
        }
    }

    /// Sets or clears the score subgraph while retaining other score settings.
    #[must_use]
    pub fn with_score_subgraph(
        mut self,
        score_subgraph: impl Into<Option<SdpaScoreSubgraph>>,
    ) -> Self {
        self.score_subgraph = score_subgraph.into();
        self
    }

    /// Sets or clears the additive bias operand while preserving other score settings.
    #[must_use]
    pub fn with_bias(mut self, bias: impl Into<Option<TensorId>>) -> Self {
        self.modifiers = self.modifiers.with_bias(bias);
        self
    }

    /// Sets or clears the attention sink operand while preserving other score settings.
    #[must_use]
    pub fn with_sink_token(mut self, sink_token: impl Into<Option<TensorId>>) -> Self {
        self.modifiers = self.modifiers.with_sink_token(sink_token);
        self
    }

    /// Selects standard `ALiBi` or clears `ALiBi` while retaining other score options.
    #[must_use]
    pub fn with_alibi(mut self, enabled: bool) -> Self {
        self.modifiers = self.modifiers.with_alibi(enabled);
        self
    }

    /// Sets caller-supplied FP32 `ALiBi` slopes; `None` disables `ALiBi`.
    #[must_use]
    pub fn with_alibi_slopes(mut self, alibi_slopes: impl Into<Option<TensorId>>) -> Self {
        self.modifiers = self.modifiers.with_alibi_slopes(alibi_slopes);
        self
    }

    /// Sets or clears the additive mask independently of the structural mask mode.
    #[must_use]
    pub fn with_additive_mask(mut self, additive_mask: impl Into<Option<TensorId>>) -> Self {
        self.modifiers = self.modifiers.with_additive_mask(additive_mask);
        self
    }

    /// Replaces query and key/value sequence-length inputs.
    #[must_use]
    pub fn with_sequence_inputs(mut self, sequence: AttentionSequenceInputs<TensorId>) -> Self {
        self.modifiers = self.modifiers.with_sequence_inputs(sequence);
        self
    }

    /// Replaces the structural mask mode while preserving other score settings.
    #[must_use]
    pub fn with_mask_mode(mut self, mask_mode: AttentionMaskMode) -> Self {
        self.modifiers = self.modifiers.with_mask_mode(mask_mode);
        self
    }

    /// Replaces the dropout mode and clears seed or custom-mask state belonging to the previous mode.
    #[must_use]
    pub fn with_dropout_mode(mut self, dropout_mode: AttentionDropoutMode) -> Self {
        self.modifiers
            .set_dropout_mode(&mut self.dropout, dropout_mode);
        self
    }

    pub(crate) fn modifiers(&self) -> &SdpaScoreModifiers {
        &self.modifiers
    }

    /// Returns the structural mask mode.
    #[must_use]
    pub fn mask_mode(&self) -> AttentionMaskMode {
        self.modifiers.mask_mode()
    }

    /// Borrows the optional graph applied to attention scores.
    #[must_use]
    pub fn score_subgraph(&self) -> Option<&SdpaScoreSubgraph> {
        self.score_subgraph.as_ref()
    }

    /// Returns generated Philox dropout settings, or `None` for disabled or custom-mask dropout.
    #[must_use]
    pub fn dropout(&self) -> Option<AttentionDropoutConfig> {
        self.dropout
    }

    /// Returns the complete dropout mode, including generated and caller-provided masks.
    #[must_use]
    pub fn dropout_mode(&self) -> AttentionDropoutMode {
        self.modifiers.dropout_mode(self.dropout)
    }

    /// Returns the configured Philox offset tensor identity, if generated dropout uses one.
    #[must_use]
    pub fn dropout_offset(&self) -> Option<TensorId> {
        self.dropout.and_then(|dropout| dropout.offset())
    }
}

impl Default for AttentionScoreConfig {
    fn default() -> Self {
        Self::new()
    }
}
