use std::borrow::Cow;

use serde::{Deserialize, Serialize};

use crate::{
    backend::{execution::EngineChoice, tensor::TensorId},
    error::Result,
    frontend::{
        composite::sdpa::{
            SdpaBackwardGradientTensors, SdpaBackwardInputs, SdpaFp8BackwardGradientTensors,
            SdpaFp8BackwardInputs, SdpaMxfp8BackwardGradientTensors, SdpaMxfp8BackwardInputs,
        },
        error::Error as FrontendError,
        lower::{LoweredOperation, LoweringContext},
        operation::{AttentionBackwardAuxGradients, AttentionBackwardConfig},
    },
};

/// An attention backward request resolved against the compilation target.
/// Declared output identities, layouts and encodings remain part of the recipe.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttentionBackwardRequest {
    /// Floating-point attention backward with independently optional auxiliary gradients.
    Floating {
        /// Saved forward operands, output gradient, statistics, and variant-specific scaling tensors.
        inputs: SdpaBackwardInputs,
        /// Declared primary-gradient and variant-specific auxiliary-output identities.
        outputs: SdpaBackwardGradientTensors,
        /// Target-resolved backward attention, score, dropout, and planning settings.
        config: AttentionBackwardConfig,
    },
    /// Per-tensor FP8 attention backward with explicit quantization operands and required gradient maxima.
    Fp8 {
        /// Saved forward operands, output gradient, statistics, and variant-specific scaling tensors.
        inputs: SdpaFp8BackwardInputs,
        /// Declared primary-gradient and variant-specific auxiliary-output identities.
        outputs: SdpaFp8BackwardGradientTensors,
        /// Target-resolved backward attention, score, dropout, and planning settings.
        config: AttentionBackwardConfig,
    },
    /// Block-scaled MXFP8 attention backward with independently optional gradient maxima.
    Mxfp8 {
        /// Saved forward operands, output gradient, statistics, and variant-specific scaling tensors.
        inputs: SdpaMxfp8BackwardInputs,
        /// Declared primary-gradient and variant-specific auxiliary-output identities.
        outputs: SdpaMxfp8BackwardGradientTensors,
        /// Target-resolved backward attention, score, dropout, and planning settings.
        config: AttentionBackwardConfig,
    },
}

/// Automatic engine choice supplied by an attention backward implementation.
/// Explicit custom planning remains independent of these heuristic overrides.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttentionBackwardHeuristicOverride {
    /// Engine index and knob constraints supplied to automatic backward-attention planning.
    #[serde(flatten)]
    pub choice: EngineChoice,
}

impl AttentionBackwardHeuristicOverride {
    pub(crate) fn matches(&self, candidate: &EngineChoice) -> bool {
        self.choice.engine_index == candidate.engine_index
            && self
                .choice
                .knob_choices
                .iter()
                .all(|choice| candidate.knob_choices.contains(choice))
    }

    /// Retains distinct overrides without choosing a planning policy during graph preparation.
    pub(crate) fn retain_into(current: &mut Vec<Self>, incoming: impl IntoIterator<Item = Self>) {
        for choice in incoming {
            if !current.contains(&choice) {
                current.push(choice);
            }
        }
    }

    /// Combines automatic choices only when heuristic selection or cache restoration needs them.
    pub(crate) fn resolve(overrides: &[Self]) -> Result<Option<Cow<'_, Self>>> {
        let Some((first, remaining)) = overrides.split_first() else {
            return Ok(None);
        };
        let mut resolved = Cow::Borrowed(first);
        for incoming in remaining {
            if resolved.choice.engine_index != incoming.choice.engine_index {
                return Err(FrontendError::SdpaHeuristicEngineConflict {
                    first: resolved.choice.engine_index,
                    second: incoming.choice.engine_index,
                }
                .into());
            }
            for &knob in &incoming.choice.knob_choices {
                match resolved
                    .choice
                    .knob_choices
                    .iter()
                    .find(|previous| previous.knob_type == knob.knob_type)
                {
                    Some(previous) if previous.value != knob.value => {
                        return Err(FrontendError::SdpaHeuristicKnobConflict {
                            engine_index: resolved.choice.engine_index,
                            knob_type: knob.knob_type,
                            first: previous.value,
                            second: knob.value,
                        }
                        .into());
                    }
                    Some(_) => {}
                    None => resolved.to_mut().choice.knob_choices.push(knob),
                }
            }
        }
        Ok(Some(resolved))
    }
}

impl AttentionBackwardRequest {
    /// Borrows backward settings independently of floating-point or quantization variant.
    #[must_use]
    pub fn config(&self) -> &AttentionBackwardConfig {
        match self {
            Self::Floating { config, .. }
            | Self::Fp8 { config, .. }
            | Self::Mxfp8 { config, .. } => config,
        }
    }

    pub(crate) fn config_mut(&mut self) -> &mut AttentionBackwardConfig {
        match self {
            Self::Floating { config, .. }
            | Self::Fp8 { config, .. }
            | Self::Mxfp8 { config, .. } => config,
        }
    }

    /// Makes retained configuration agree with the declared outputs and active dropout mode.
    pub(crate) fn normalize_configuration(&mut self) {
        let aux_gradients = match self {
            Self::Floating { outputs, .. } => AttentionBackwardAuxGradients::new(
                outputs.bias_gradient.is_some(),
                outputs.rng_dump.is_some(),
                outputs.sink_token_gradient.is_some(),
            ),
            Self::Fp8 { outputs, .. } => AttentionBackwardAuxGradients::new(
                false,
                false,
                outputs.sink_token_gradient.is_some(),
            ),
            Self::Mxfp8 { outputs, .. } => AttentionBackwardAuxGradients::new(
                false,
                false,
                outputs.sink_token_gradient.is_some(),
            ),
        };
        let config = self.config_mut();
        config.set_aux_gradients(aux_gradients);
        config.remove_inactive_dropout_scale_inverse();
    }

    /// Returns frontend tensor identities in query, key, value order.
    #[must_use]
    pub fn query_key_value(&self) -> [TensorId; 3] {
        match self {
            Self::Floating { inputs, .. } => [inputs.query, inputs.key, inputs.value],
            Self::Fp8 { inputs, .. } => [inputs.query, inputs.key, inputs.value],
            Self::Mxfp8 { inputs, .. } => [inputs.query, inputs.key, inputs.value],
        }
    }

    #[expect(
        clippy::unused_self,
        reason = "declarative requests share the operation lowering protocol but must be expanded before lowering"
    )]
    pub(crate) fn lower(&self, _context: &LoweringContext<'_>) -> Result<LoweredOperation> {
        Err(FrontendError::SdpaRequestNotExpanded.into())
    }
}

impl AttentionBackwardRequest {}
