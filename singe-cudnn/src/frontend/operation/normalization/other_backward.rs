use serde::{Deserialize, Serialize};

use crate::{backend::tensor::TensorId, frontend::operation::NormalizationSavedStatistics};

/// Instance normalization backward operands and saved forward statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstanceNormalizationBackwardInputs {
    /// Input activation tensor identity.
    pub input: TensorId,
    /// Saved forward mean and inverse-standard-deviation tensor identities.
    pub statistics: NormalizationSavedStatistics,
    /// Gradient tensor identity for the forward output.
    pub output_gradient: TensorId,
    /// Affine scale tensor identity.
    pub scale: TensorId,
}

/// Instance normalization backward gradient destinations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstanceNormalizationBackwardOutputs<T = TensorId> {
    /// Input-gradient identity or its inference policy.
    pub input_gradient: T,
    /// Affine-scale gradient identity or its inference policy.
    pub scale_gradient: T,
    /// Affine-bias gradient identity or its inference policy.
    pub bias_gradient: T,
}

/// Adaptive layer normalization backward operands and saved forward statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdaptiveLayerNormalizationBackwardInputs {
    /// Input activation tensor identity.
    pub input: TensorId,
    /// Saved forward mean and inverse-standard-deviation tensor identities.
    pub statistics: NormalizationSavedStatistics,
    /// Gradient tensor identity for the forward output.
    pub output_gradient: TensorId,
    /// Affine scale tensor identity.
    pub scale: TensorId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
/// `AdaLayerNorm` backward results with an independently optional bias gradient.
/// Explicit affine gradients retain broadcast layouts; inference defaults to scale dimensions and row-major strides.
pub struct AdaptiveLayerNormalizationBackwardOutputs<T = TensorId> {
    /// Input-gradient identity or its inference policy.
    pub input_gradient: T,
    /// Affine-scale gradient identity or its inference policy.
    pub scale_gradient: T,
    /// Optional affine-bias gradient identity or its inference policy.
    pub bias_gradient: Option<T>,
}
