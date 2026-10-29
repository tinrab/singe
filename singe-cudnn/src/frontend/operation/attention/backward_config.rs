use serde::{Deserialize, Serialize};

use crate::{
    backend::tensor::TensorId,
    data_type::DataType,
    frontend::operation::{
        AttentionBackwardWorkspaceLimit, AttentionDropoutConfig, AttentionDropoutMode,
        AttentionMaskMode, AttentionMaxTotalSequenceLengths, AttentionScoreConfig,
        SdpaScoreModifiers, SdpaScoreSubgraph, SoftmaxConfig,
    },
};

/// High-level configuration for SDPA backward helpers.
///
/// Includes score modifiers, optional score subgraphs for forward and backward
/// score paths, dropout, attention scale, deterministic mode, and optional
/// gradients for bias, RNG dump, and sink token.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionBackwardConfig {
    pub(crate) name: Option<String>,
    pub(crate) softmax: SoftmaxConfig,
    pub(crate) score: AttentionScoreConfig,
    pub(crate) score_subgraph_bprop: Option<SdpaScoreSubgraph>,
    #[serde(default, with = "crate::scalar::float::optional_f32")]
    pub(crate) attention_scale: Option<f32>,
    pub(crate) deterministic_algorithm: bool,
    pub(crate) aux_gradients: AttentionBackwardAuxGradients,
    pub(crate) max_total_sequence_lengths: AttentionMaxTotalSequenceLengths,
    pub(crate) dropout_scale_inverse: Option<TensorId>,
    pub(crate) probability_gradient_workspace_limit: Option<AttentionBackwardWorkspaceLimit>,
}

/// Auxiliary-output presence retained by recorded backward attention.
/// Recording and preparation derive these flags from the actual output identities.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttentionBackwardAuxGradients {
    pub(crate) bias: bool,
    pub(crate) random_number_generator_dump: bool,
    pub(crate) sink_token: bool,
}

impl AttentionBackwardAuxGradients {
    pub(crate) fn new(bias: bool, random_number_generator_dump: bool, sink_token: bool) -> Self {
        Self {
            bias,
            random_number_generator_dump,
            sink_token,
        }
    }

    /// Returns whether recorded outputs include a bias gradient.
    #[must_use]
    pub fn bias(self) -> bool {
        self.bias
    }

    /// Returns whether recorded outputs include an RNG dump.
    #[must_use]
    pub fn random_number_generator_dump(self) -> bool {
        self.random_number_generator_dump
    }

    /// Returns whether recorded outputs include a sink-token gradient.
    #[must_use]
    pub fn sink_token(self) -> bool {
        self.sink_token
    }
}

impl AttentionBackwardConfig {
    /// Records backward attention with inherited compute precision, defaulting to `F32`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            name: None,
            softmax: SoftmaxConfig::new(),
            score: AttentionScoreConfig::new(),
            score_subgraph_bprop: None,
            attention_scale: None,
            deterministic_algorithm: false,
            aux_gradients: AttentionBackwardAuxGradients::default(),
            max_total_sequence_lengths: AttentionMaxTotalSequenceLengths::default(),
            dropout_scale_inverse: None,
            probability_gradient_workspace_limit: None,
        }
    }

    /// Names inferred gradients and auxiliary results using the native output-port suffixes.
    /// Explicit tensor declarations retain their names; an empty scope disables inferred naming.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Returns the nonempty gradient-name scope, or `None` when unnamed.
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref().filter(|name| !name.is_empty())
    }

    /// Overrides backward compute precision, or restores graph inheritance with `None`.
    #[must_use]
    pub fn with_compute_type(mut self, compute_type: impl Into<Option<DataType>>) -> Self {
        self.softmax = self.softmax.with_compute_type(compute_type);
        self
    }

    /// Caps dP workspace for deterministic Hopper backward without a bias gradient.
    /// A cap below the padded dP requirement selects ordered FP32 dQ accumulation.
    /// Zero always requests that route.
    /// `Unlimited` retains the dP route.
    /// An explicit cap takes precedence over `CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT`.
    /// `None` removes the explicit cap and restores environment inheritance.
    #[must_use]
    pub fn with_probability_gradient_workspace_limit(
        mut self,
        limit: impl Into<Option<AttentionBackwardWorkspaceLimit>>,
    ) -> Self {
        self.probability_gradient_workspace_limit = limit.into();
        self
    }

    /// Returns the explicit cap, or `None` to inherit the environment setting.
    #[must_use]
    pub fn probability_gradient_workspace_limit(&self) -> Option<AttentionBackwardWorkspaceLimit> {
        self.probability_gradient_workspace_limit
    }

    /// Supplies C++'s inverse-scale operand for custom backward dropout.
    /// `None` removes this operand so Rust derives the inverse from the configured scale.
    /// The score configuration and inverse operand can be set in either order.
    /// Recording ignores the operand when the final dropout mode is Philox or disabled.
    #[must_use]
    pub fn with_dropout_scale_inverse(mut self, inverse: impl Into<Option<TensorId>>) -> Self {
        self.dropout_scale_inverse = inverse.into();
        self
    }

    /// Returns the inverse-scale operand used by the active custom-mask dropout mode.
    /// Philox derives its inverse from the probability, and disabled dropout ignores this operand.
    #[must_use]
    pub fn dropout_scale_inverse(&self) -> Option<TensorId> {
        match self.dropout_mode() {
            AttentionDropoutMode::CustomMask(_) => self.dropout_scale_inverse,
            AttentionDropoutMode::None | AttentionDropoutMode::Philox(_) => None,
        }
    }

    pub(crate) fn remove_inactive_dropout_scale_inverse(&mut self) {
        self.dropout_scale_inverse = self.dropout_scale_inverse();
    }

    /// Sets packed total-length hints without changing logical sequence dimensions.
    /// Quantized composites preserve the C++ validation-only semantics of these hints.
    /// Floating backward ignores both hints when either head width is not divisible by 16, or on SM8x/SM12x.
    /// Those native workarounds use full workspace sizing while retaining the original request in the recipe.
    #[must_use]
    pub fn with_max_total_sequence_lengths(
        mut self,
        lengths: AttentionMaxTotalSequenceLengths,
    ) -> Self {
        self.max_total_sequence_lengths = lengths;
        self
    }

    /// Returns requested packed query and key/value token bounds.
    #[must_use]
    pub fn max_total_sequence_lengths(&self) -> AttentionMaxTotalSequenceLengths {
        self.max_total_sequence_lengths
    }

    /// Sets or clears the backward score subgraph independently of the forward score configuration.
    #[must_use]
    pub fn with_score_subgraph_bprop(
        mut self,
        score_subgraph_bprop: impl Into<Option<SdpaScoreSubgraph>>,
    ) -> Self {
        self.score_subgraph_bprop = score_subgraph_bprop.into();
        self
    }

    /// Replaces score modifiers, the forward score subgraph, and dropout settings together.
    #[must_use]
    pub fn with_score_config(mut self, score: AttentionScoreConfig) -> Self {
        self.score = score;
        self
    }

    /// Sets or clears the numeric attention multiplier while retaining the attention-scale tensor operand.
    #[must_use]
    pub fn with_attention_scale(mut self, attention_scale: impl Into<Option<f32>>) -> Self {
        self.attention_scale = attention_scale.into();
        self
    }

    /// Sets or clears the explicit deterministic-algorithm request.
    /// Architecture-specific requirements still apply when this flag is false.
    #[must_use]
    pub fn with_deterministic_algorithm(mut self, enabled: bool) -> Self {
        self.deterministic_algorithm = enabled;
        self
    }

    pub(crate) fn set_aux_gradients(&mut self, aux_gradients: AttentionBackwardAuxGradients) {
        self.aux_gradients = aux_gradients;
    }

    /// Borrows backward softmax computation settings.
    #[must_use]
    pub fn softmax(&self) -> &SoftmaxConfig {
        &self.softmax
    }

    /// Borrows score modifiers, the forward score subgraph, and dropout settings.
    #[must_use]
    pub fn score_config(&self) -> &AttentionScoreConfig {
        &self.score
    }

    pub(crate) fn modifiers(&self) -> &SdpaScoreModifiers {
        self.score.modifiers()
    }

    /// Returns the configured structural attention mask mode.
    #[must_use]
    pub fn mask_mode(&self) -> AttentionMaskMode {
        self.score.mask_mode()
    }

    /// Borrows the optional forward score-modifier graph.
    #[must_use]
    pub fn score_subgraph(&self) -> Option<&SdpaScoreSubgraph> {
        self.score.score_subgraph()
    }

    /// Borrows the optional backward score-modifier graph.
    #[must_use]
    pub fn score_subgraph_bprop(&self) -> Option<&SdpaScoreSubgraph> {
        self.score_subgraph_bprop.as_ref()
    }

    /// Returns generated Philox dropout settings, or `None` for custom-mask or disabled dropout.
    #[must_use]
    pub fn dropout(&self) -> Option<AttentionDropoutConfig> {
        self.score.dropout()
    }

    /// Returns the complete generated, custom-mask, or disabled dropout request.
    #[must_use]
    pub fn dropout_mode(&self) -> AttentionDropoutMode {
        self.score.dropout_mode()
    }

    /// Returns the generated Philox dropout offset identity, if configured.
    #[must_use]
    pub fn dropout_offset(&self) -> Option<TensorId> {
        self.score.dropout_offset()
    }

    /// Returns the optional numeric attention multiplier applied in addition to the scale tensor.
    #[must_use]
    pub fn attention_scale(&self) -> Option<f32> {
        self.attention_scale
    }

    /// Returns whether deterministic algorithm selection is explicitly requested.
    #[must_use]
    pub fn deterministic_algorithm(&self) -> bool {
        self.deterministic_algorithm
    }

    /// Auxiliary-output presence derived from recorded output identities.
    #[must_use]
    pub fn aux_gradients(&self) -> AttentionBackwardAuxGradients {
        self.aux_gradients
    }
}

impl Default for AttentionBackwardConfig {
    fn default() -> Self {
        Self::new()
    }
}
