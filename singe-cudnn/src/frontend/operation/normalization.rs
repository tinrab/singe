use serde::{Deserialize, Serialize};

use crate::{
    execution::normalization::BatchNormalizationFinalizeStatsMode,
    normalization::BackendNormalizationForwardPhase, tensor::TensorId,
};

/// Attributes for batch normalization inference.
///
/// Batch normalization computes `scale * (input - mean) / sqrt(variance + epsilon) + bias`.
/// This config stores the epsilon tensor used by the inference path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchNormalizationInferenceConfig {
    epsilon: TensorId,
}

impl BatchNormalizationInferenceConfig {
    pub fn new(epsilon: TensorId) -> Self {
        Self { epsilon }
    }

    pub fn epsilon(&self) -> TensorId {
        self.epsilon
    }
}

/// Attributes for DBN weight-gradient helper operations.
///
/// DBN is cuDNN frontend's batchnorm backward operation, which computes input,
/// scale, and bias_gradients during batchnorm backpropagation.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct DbnWeightConfig;

impl DbnWeightConfig {
    pub fn new() -> Self {
        Self
    }
}

/// Attributes for batch normalization finalize.
///
/// `bn_finalize` computes the equivalent scale/bias and statistics consumed by
/// later batchnorm steps, optionally including next running mean and variance.
///
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchNormalizationFinalizeConfig {
    mode: BatchNormalizationFinalizeStatsMode,
    epsilon: TensorId,
    running: Option<BatchNormalizationFinalizeRunningStats>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BatchNormalizationFinalizeRunningStats {
    pub momentum: TensorId,
    pub prev_running_mean: TensorId,
    pub prev_running_var: TensorId,
}

impl BatchNormalizationFinalizeRunningStats {
    pub fn new(
        momentum: TensorId,
        prev_running_mean: TensorId,
        prev_running_var: TensorId,
    ) -> Self {
        Self {
            momentum,
            prev_running_mean,
            prev_running_var,
        }
    }
}

impl BatchNormalizationFinalizeConfig {
    pub fn training(epsilon: TensorId) -> Self {
        Self {
            mode: BatchNormalizationFinalizeStatsMode::Training,
            epsilon,
            running: None,
        }
    }

    pub fn inference(epsilon: TensorId) -> Self {
        Self {
            mode: BatchNormalizationFinalizeStatsMode::Inference,
            epsilon,
            running: None,
        }
    }

    pub fn with_running_stats(mut self, running: BatchNormalizationFinalizeRunningStats) -> Self {
        self.running = Some(running);
        self
    }

    pub fn running(&self) -> Option<BatchNormalizationFinalizeRunningStats> {
        self.running
    }

    pub fn momentum(&self) -> Option<TensorId> {
        self.running.map(|running| running.momentum)
    }

    pub fn prev_running_mean(&self) -> Option<TensorId> {
        self.running.map(|running| running.prev_running_mean)
    }

    pub fn prev_running_var(&self) -> Option<TensorId> {
        self.running.map(|running| running.prev_running_var)
    }

    pub fn has_running_stats(&self) -> bool {
        self.running.is_some()
    }
}

impl BatchNormalizationFinalizeConfig {
    pub fn mode(&self) -> BatchNormalizationFinalizeStatsMode {
        self.mode
    }

    pub fn epsilon(&self) -> TensorId {
        self.epsilon
    }
}

/// Attributes for layer normalization forward.
///
/// Layer normalization normalizes across features independently for each sample
/// and may run in inference or training phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerNormalizationConfig {
    phase: BackendNormalizationForwardPhase,
    epsilon: TensorId,
}

impl LayerNormalizationConfig {
    pub fn inference(epsilon: TensorId) -> Self {
        Self {
            phase: BackendNormalizationForwardPhase::Inference,
            epsilon,
        }
    }

    pub fn training(epsilon: TensorId) -> Self {
        Self {
            phase: BackendNormalizationForwardPhase::Training,
            epsilon,
        }
    }

    pub fn phase(&self) -> BackendNormalizationForwardPhase {
        self.phase
    }

    pub fn epsilon(&self) -> TensorId {
        self.epsilon
    }
}

/// Attributes for RMS normalization forward.
///
/// RMS normalization is represented through the same frontend normalization
/// family as layer normalization, with an optional bias tensor in this Rust config.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RmsNormalizationConfig {
    phase: BackendNormalizationForwardPhase,
    epsilon: TensorId,
    bias: Option<TensorId>,
}

impl RmsNormalizationConfig {
    pub fn inference(epsilon: TensorId) -> Self {
        Self {
            phase: BackendNormalizationForwardPhase::Inference,
            epsilon,
            bias: None,
        }
    }

    pub fn training(epsilon: TensorId) -> Self {
        Self {
            phase: BackendNormalizationForwardPhase::Training,
            epsilon,
            bias: None,
        }
    }

    pub fn with_bias(mut self, bias: TensorId) -> Self {
        self.bias = Some(bias);
        self
    }

    pub fn phase(&self) -> BackendNormalizationForwardPhase {
        self.phase
    }

    pub fn epsilon(&self) -> TensorId {
        self.epsilon
    }

    pub fn bias(&self) -> Option<TensorId> {
        self.bias
    }
}

/// Attributes for layer normalization backward.
///
/// DLN computes input, scale, and bias_gradients during layernorm backpropagation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerNormalizationBackwardConfig {
    epsilon: TensorId,
}

impl LayerNormalizationBackwardConfig {
    pub fn new(epsilon: TensorId) -> Self {
        Self { epsilon }
    }

    pub fn epsilon(&self) -> TensorId {
        self.epsilon
    }
}

/// Attributes for RMS normalization backward.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RmsNormalizationBackwardConfig {
    has_bias_gradient: bool,
}

impl RmsNormalizationBackwardConfig {
    pub fn new() -> Self {
        Self {
            has_bias_gradient: false,
        }
    }

    pub fn with_bias_gradient(mut self) -> Self {
        self.has_bias_gradient = true;
        self
    }

    pub fn has_bias_gradient(&self) -> bool {
        self.has_bias_gradient
    }
}

impl Default for RmsNormalizationBackwardConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Attributes for instance normalization forward.
///
/// Instance normalization computes the standard normalization expression across each sample.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstanceNormalizationConfig {
    phase: BackendNormalizationForwardPhase,
    epsilon: TensorId,
}

impl InstanceNormalizationConfig {
    pub fn inference(epsilon: TensorId) -> Self {
        Self {
            phase: BackendNormalizationForwardPhase::Inference,
            epsilon,
        }
    }

    pub fn training(epsilon: TensorId) -> Self {
        Self {
            phase: BackendNormalizationForwardPhase::Training,
            epsilon,
        }
    }

    pub fn phase(&self) -> BackendNormalizationForwardPhase {
        self.phase
    }

    pub fn epsilon(&self) -> TensorId {
        self.epsilon
    }
}

/// Attributes for instance normalization backward.
///
/// Instance normalization backward computes gradients for the corresponding forward normalization operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstanceNormalizationBackwardConfig {
    epsilon: TensorId,
}

impl InstanceNormalizationBackwardConfig {
    pub fn new(epsilon: TensorId) -> Self {
        Self { epsilon }
    }

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
    epsilon: TensorId,
    running: Option<BatchNormalizationRunningStats>,
    peer_stats: Vec<TensorId>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BatchNormalizationRunningStats {
    pub momentum: TensorId,
    pub prev_mean: TensorId,
    pub prev_var: TensorId,
    pub next_mean: TensorId,
    pub next_var: TensorId,
}

impl BatchNormalizationRunningStats {
    pub fn new(
        momentum: TensorId,
        prev_mean: TensorId,
        prev_var: TensorId,
        next_mean: TensorId,
        next_var: TensorId,
    ) -> Self {
        Self {
            momentum,
            prev_mean,
            prev_var,
            next_mean,
            next_var,
        }
    }
}

impl BatchNormalizationConfig {
    pub fn new(epsilon: TensorId) -> Self {
        Self {
            epsilon,
            running: None,
            peer_stats: Vec::new(),
        }
    }

    pub fn with_running_stats(mut self, running: BatchNormalizationRunningStats) -> Self {
        self.running = Some(running);
        self
    }

    pub fn with_peer_stats(mut self, peer_stats: Vec<TensorId>) -> Self {
        self.peer_stats = peer_stats;
        self
    }

    pub fn epsilon(&self) -> TensorId {
        self.epsilon
    }

    pub fn running(&self) -> Option<BatchNormalizationRunningStats> {
        self.running
    }

    pub fn peer_stats(&self) -> &[TensorId] {
        &self.peer_stats
    }
}

/// Attributes for batch normalization backward.
///
/// DBN computes input, scale, and bias_gradients and can include peer
/// statistics for multi-GPU normalization patterns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchNormalizationBackwardConfig {
    epsilon: TensorId,
    peer_stats: Vec<TensorId>,
}

impl BatchNormalizationBackwardConfig {
    pub fn new(epsilon: TensorId) -> Self {
        Self {
            epsilon,
            peer_stats: Vec::new(),
        }
    }

    pub fn with_peer_stats(mut self, peer_stats: Vec<TensorId>) -> Self {
        self.peer_stats = peer_stats;
        self
    }

    pub fn epsilon(&self) -> TensorId {
        self.epsilon
    }

    pub fn peer_stats(&self) -> &[TensorId] {
        &self.peer_stats
    }
}

/// Attributes for adaptive layer normalization forward.
///
/// Adaptive layernorm normalizes across features independently for each sample,
/// with scale and bias varying across samples in a batch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveLayerNormalizationConfig {
    phase: BackendNormalizationForwardPhase,
    epsilon: TensorId,
    bias: Option<TensorId>,
}

impl AdaptiveLayerNormalizationConfig {
    pub fn inference(epsilon: TensorId) -> Self {
        Self {
            phase: BackendNormalizationForwardPhase::Inference,
            epsilon,
            bias: None,
        }
    }

    pub fn training(epsilon: TensorId) -> Self {
        Self {
            phase: BackendNormalizationForwardPhase::Training,
            epsilon,
            bias: None,
        }
    }

    pub fn with_bias(mut self, bias: TensorId) -> Self {
        self.bias = Some(bias);
        self
    }

    pub fn phase(&self) -> BackendNormalizationForwardPhase {
        self.phase
    }

    pub fn epsilon(&self) -> TensorId {
        self.epsilon
    }

    pub fn bias(&self) -> Option<TensorId> {
        self.bias
    }
}

/// Attributes for adaptive layer normalization backward.
///
/// DADALN computes input, scale, and bias_gradients during adaptive layernorm backpropagation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveLayerNormalizationBackwardConfig {
    has_bias_gradient: bool,
}

impl AdaptiveLayerNormalizationBackwardConfig {
    pub fn new() -> Self {
        Self {
            has_bias_gradient: false,
        }
    }

    pub fn with_bias_gradient(mut self) -> Self {
        self.has_bias_gradient = true;
        self
    }

    pub fn has_bias_gradient(&self) -> bool {
        self.has_bias_gradient
    }
}

impl Default for AdaptiveLayerNormalizationBackwardConfig {
    fn default() -> Self {
        Self::new()
    }
}
