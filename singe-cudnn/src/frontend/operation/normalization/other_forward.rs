use serde::{Deserialize, Serialize};

use crate::{backend::tensor::TensorId, frontend::operation::NormalizationStatistics};

/// Epsilon tensor and optional scope for instance normalization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstanceNormalizationConfig {
    pub(crate) name: Option<String>,
    pub(crate) epsilon: TensorId,
}

impl InstanceNormalizationConfig {
    /// Records the scalar epsilon tensor with no result-name scope.
    #[must_use]
    pub fn new(epsilon: TensorId) -> Self {
        Self {
            name: None,
            epsilon,
        }
    }

    /// Names inferred results `name::Y`, `name::MEAN` and `name::INV_VARIANCE`.
    /// Explicit tensor names are retained, and an empty scope disables naming.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Returns the nonempty result-name scope, or `None` when unnamed.
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref().filter(|name| !name.is_empty())
    }

    /// Returns the scalar numerical-stability tensor identity.
    #[must_use]
    pub fn epsilon(&self) -> TensorId {
        self.epsilon
    }
}

/// Epsilon tensor and optional scope for adaptive layer normalization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveLayerNormalizationConfig {
    pub(crate) name: Option<String>,
    pub(crate) epsilon: TensorId,
}

impl AdaptiveLayerNormalizationConfig {
    /// Records the scalar epsilon tensor with no result-name scope.
    #[must_use]
    pub fn new(epsilon: TensorId) -> Self {
        Self {
            name: None,
            epsilon,
        }
    }

    /// Names inferred results `name::Y`, `name::MEAN` and `name::INV_VARIANCE`.
    /// Explicit tensor names are retained, and an empty scope disables naming.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Returns the nonempty result-name scope, or `None` when unnamed.
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref().filter(|name| !name.is_empty())
    }

    /// Returns the scalar numerical-stability tensor identity.
    #[must_use]
    pub fn epsilon(&self) -> TensorId {
        self.epsilon
    }
}

/// Instance normalization input activation and affine operands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstanceNormalizationInputs {
    /// Input activation tensor identity.
    pub input: TensorId,
    /// Affine scale tensor identity.
    pub scale: TensorId,
    /// Affine bias tensor identity.
    pub bias: TensorId,
}

/// Instance normalization activation output and phase-specific saved-statistics request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstanceNormalizationOutputs<T = TensorId> {
    /// Normalized activation identity or its inference policy.
    pub output: T,
    /// Inference phase or saved mean and inverse-standard-deviation destinations.
    pub statistics: NormalizationStatistics<T>,
}

/// Adaptive layer normalization input activation and affine operands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdaptiveLayerNormalizationInputs {
    /// Input activation tensor identity.
    pub input: TensorId,
    /// Affine scale tensor identity.
    pub scale: TensorId,
    /// Optional independently broadcast affine bias tensor identity.
    pub bias: Option<TensorId>,
}

/// Adaptive layer normalization activation output and phase-specific saved-statistics request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdaptiveLayerNormalizationOutputs<T = TensorId> {
    /// Normalized activation identity or its inference policy.
    pub output: T,
    /// Inference phase or saved mean and inverse-standard-deviation destinations.
    pub statistics: NormalizationStatistics<T>,
}
