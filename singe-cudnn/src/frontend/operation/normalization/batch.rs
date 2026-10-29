use serde::{Deserialize, Serialize};

pub use crate::backend::execution::normalization::{
    BatchNormalizationBackwardWeightsInputs, BatchNormalizationBackwardWeightsOutputs,
    BatchNormalizationFinalizeInputs, BatchNormalizationFinalizeOutputs,
    BatchNormalizationRunningStatistics, BatchNormalizationRunningStats,
};
use crate::{
    backend::{execution::normalization::BatchNormalizationFinalizeStatsMode, tensor::TensorId},
    data_type::DataType,
    frontend::operation::NormalizationSavedStatistics,
};

/// Math precision and optional scope for `BatchNorm` weight gradients and equivalent coefficients.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BatchNormalizationBackwardWeightsConfig {
    pub(crate) name: Option<String>,
    pub(crate) compute_type: Option<DataType>,
}

impl BatchNormalizationBackwardWeightsConfig {
    /// Creates an unnamed weight-gradient request with inherited graph math precision.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Names inferred gradients and equivalent coefficients using C++ output-port suffixes.
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

    /// Overrides graph math precision independently of operand and result encodings.
    /// `None` restores inheritance, with an FP32 fallback when the graph leaves it unset.
    #[must_use]
    pub fn with_compute_type(mut self, compute_type: impl Into<Option<DataType>>) -> Self {
        self.compute_type = compute_type.into();
        self
    }

    /// Returns the math-precision override, or `None` to inherit graph policy.
    #[must_use]
    pub fn compute_type(&self) -> Option<DataType> {
        self.compute_type
    }
}

impl_inherited_compute_type!(BatchNormalizationBackwardWeightsConfig);

/// Batch-normalization training input and affine tensor identities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchNormalizationInputs {
    /// Input activation tensor identity.
    pub input: TensorId,
    /// Affine scale tensor identity.
    pub scale: TensorId,
    /// Affine bias tensor identity.
    pub bias: TensorId,
}

/// Batch-normalization training activation and saved-statistics outputs, with optional running-statistics updates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchNormalizationOutputs<T = TensorId> {
    /// Normalized activation tensor identity or its inference policy.
    pub output: T,
    /// Mean and inverse-standard-deviation tensors or their output inference policies.
    pub statistics: NormalizationSavedStatistics<T>,
    /// Next running mean and variance, required exactly when running-statistics inputs are supplied.
    pub running: Option<BatchNormalizationRunningStatistics<T>>,
}

/// Batch-normalization inference activation, affine operands, and cached mean and inverse standard deviation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchNormalizationInferenceInputs {
    /// Input activation tensor identity.
    pub input: TensorId,
    /// Cached or saved forward mean and inverse-standard-deviation tensor identities.
    pub statistics: NormalizationSavedStatistics,
    /// Affine scale tensor identity.
    pub scale: TensorId,
    /// Affine bias tensor identity.
    pub bias: TensorId,
}

/// Batch-normalization backward activation, output gradient, scale, and saved forward statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchNormalizationBackwardInputs {
    /// Input activation tensor identity.
    pub input: TensorId,
    /// Cached or saved forward mean and inverse-standard-deviation tensor identities.
    pub statistics: NormalizationSavedStatistics,
    /// Gradient tensor identity for the forward output.
    pub output_gradient: TensorId,
    /// Affine scale tensor identity.
    pub scale: TensorId,
}

/// Input, affine-scale, and affine-bias gradient outputs for batch normalization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchNormalizationBackwardOutputs<T = TensorId> {
    /// Input-gradient tensor identity or its inference policy.
    pub input_gradient: T,
    /// Scale-gradient tensor identity or its inference policy.
    pub scale_gradient: T,
    /// Bias-gradient tensor identity or its inference policy.
    pub bias_gradient: T,
}

/// Attributes for batch normalization finalize.
///
/// `bn_finalize` computes the equivalent scale/bias and statistics consumed by
/// later batchnorm steps, optionally including next running mean and variance.
///
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchNormalizationFinalizeConfig {
    pub(crate) name: Option<String>,
    pub(crate) compute_type: Option<DataType>,
    pub(crate) mode: BatchNormalizationFinalizeStatsMode,
    pub(crate) epsilon: TensorId,
    pub(crate) running: Option<BatchNormalizationRunningStats>,
}

impl BatchNormalizationFinalizeConfig {
    /// Creates a training finalization request with the supplied epsilon and inherited math precision.
    #[must_use]
    pub fn training(epsilon: TensorId) -> Self {
        Self {
            name: None,
            compute_type: None,
            mode: BatchNormalizationFinalizeStatsMode::Training,
            epsilon,
            running: None,
        }
    }

    /// Creates an inference finalization request with the supplied epsilon and inherited math precision.
    #[must_use]
    pub fn inference(epsilon: TensorId) -> Self {
        Self {
            name: None,
            compute_type: None,
            mode: BatchNormalizationFinalizeStatsMode::Inference,
            epsilon,
            running: None,
        }
    }

    /// Names inferred coefficients, saved statistics and running-statistics outputs using C++ port names.
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

    /// Overrides graph math precision independently of statistics and coefficient encodings.
    /// `None` restores inheritance, with an FP32 fallback when the graph leaves it unset.
    #[must_use]
    pub fn with_compute_type(mut self, compute_type: impl Into<Option<DataType>>) -> Self {
        self.compute_type = compute_type.into();
        self
    }

    /// Returns the math-precision override, or `None` to inherit graph policy.
    #[must_use]
    pub fn compute_type(&self) -> Option<DataType> {
        self.compute_type
    }

    /// Sets or clears running-statistics inputs without changing mode or epsilon.
    #[must_use]
    pub fn with_running_stats(
        mut self,
        running: impl Into<Option<BatchNormalizationRunningStats>>,
    ) -> Self {
        self.running = running.into();
        self
    }

    /// Returns the optional momentum and previous running-statistics inputs.
    #[must_use]
    pub fn running(&self) -> Option<BatchNormalizationRunningStats> {
        self.running
    }

    /// Returns the exponential averaging-factor tensor identity when running-statistics updates are requested.
    #[must_use]
    pub fn momentum(&self) -> Option<TensorId> {
        self.running.map(|running| running.momentum)
    }

    /// Returns the previous running-mean tensor identity, if configured.
    #[must_use]
    pub fn prev_running_mean(&self) -> Option<TensorId> {
        self.running.map(|running| running.previous.mean)
    }

    /// Returns the previous running-variance tensor identity, if configured.
    #[must_use]
    pub fn prev_running_var(&self) -> Option<TensorId> {
        self.running.map(|running| running.previous.variance)
    }
}

impl BatchNormalizationFinalizeConfig {
    /// Returns the native statistics-finalization mode.
    #[must_use]
    pub fn mode(&self) -> BatchNormalizationFinalizeStatsMode {
        self.mode
    }

    /// Returns the scalar numerical-stability tensor identity.
    #[must_use]
    pub fn epsilon(&self) -> TensorId {
        self.epsilon
    }
}

/// Attributes for batch normalization forward.
///
/// Batchnorm forward can optionally consume and produce running statistics, and
/// can include peer statistics for multi-GPU batch normalization patterns.
///
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchNormalizationConfig {
    pub(crate) name: Option<String>,
    pub(crate) epsilon: TensorId,
    pub(crate) running: Option<BatchNormalizationRunningStats>,
    pub(crate) peer_stats: Vec<TensorId>,
}

impl BatchNormalizationConfig {
    /// Records the epsilon tensor with no name, running-statistics update, or peers.
    #[must_use]
    pub fn new(epsilon: TensorId) -> Self {
        Self {
            name: None,
            epsilon,
            running: None,
            peer_stats: Vec::new(),
        }
    }

    /// Names inferred activations, saved statistics and next running statistics using C++ port names.
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

    /// Sets or clears running-statistics inputs without changing epsilon or peers.
    /// Supply the corresponding next statistics in [`BatchNormalizationOutputs::running`].
    #[must_use]
    pub fn with_running_stats(
        mut self,
        running: impl Into<Option<BatchNormalizationRunningStats>>,
    ) -> Self {
        self.running = running.into();
        self
    }

    /// Supplies peer-statistics buffers with caller-declared native layouts.
    /// cuDNN determines which layouts the selected engine supports.
    #[must_use]
    pub fn with_peer_stats(mut self, peer_stats: Vec<TensorId>) -> Self {
        self.peer_stats = peer_stats;
        self
    }

    /// Returns the scalar numerical-stability tensor identity.
    #[must_use]
    pub fn epsilon(&self) -> TensorId {
        self.epsilon
    }

    /// Returns the optional momentum and previous running-statistics inputs.
    #[must_use]
    pub fn running(&self) -> Option<BatchNormalizationRunningStats> {
        self.running
    }

    /// Borrows peer-statistics tensor identities in their native operand order.
    #[must_use]
    pub fn peer_stats(&self) -> &[TensorId] {
        &self.peer_stats
    }
}

/// Attributes for batch normalization backward.
///
/// DBN computes input, scale, and `bias_gradients` and can include peer
/// statistics for multi-GPU normalization patterns.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BatchNormalizationBackwardConfig {
    pub(crate) name: Option<String>,
    pub(crate) peer_stats: Vec<TensorId>,
}

impl BatchNormalizationBackwardConfig {
    /// Creates an unnamed backward request without peer-statistics operands.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Names inferred gradients `name::DX`, `name::DSCALE` and `name::DBIAS`.
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

    /// Supplies peer-statistics buffers with caller-declared native layouts.
    /// cuDNN determines which layouts the selected engine supports.
    #[must_use]
    pub fn with_peer_stats(mut self, peer_stats: Vec<TensorId>) -> Self {
        self.peer_stats = peer_stats;
        self
    }

    /// Borrows peer-statistics tensor identities in their native operand order.
    #[must_use]
    pub fn peer_stats(&self) -> &[TensorId] {
        &self.peer_stats
    }
}

impl_inherited_compute_type!(BatchNormalizationFinalizeConfig);
