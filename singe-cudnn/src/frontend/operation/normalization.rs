use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{
    data_type::DataType,
    error::Result,
    execution::normalization::{
        BatchNormalizationBackwardWeightsOperation, BatchNormalizationFinalizeStatsMode,
        BatchNormalizationFinalizeStatsOperation, NormalizationBackwardConfig,
        NormalizationBackwardOperation, NormalizationForwardConfig, NormalizationForwardOperation,
    },
    frontend::{
        lower::{
            LoweredOperation, LoweringContext, optional_tensor_at, tensor_at, tensor_slice_at,
        },
        operation::FrontendOperationTensors,
    },
    normalization::{BackendNormalizationForwardPhase, BackendNormalizationMode},
    tensor::{Tensor, TensorId},
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

/// Frontend normalization operation variants.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum NormalizationOperation {
    Layer {
        input: TensorId,
        scale: TensorId,
        bias: TensorId,
        output: TensorId,
        mean: TensorId,
        inv_variance: TensorId,
        config: LayerNormalizationConfig,
    },
    Rms {
        input: TensorId,
        scale: TensorId,
        bias: Option<TensorId>,
        output: TensorId,
        inv_variance: TensorId,
        config: RmsNormalizationConfig,
    },
    LayerBackward {
        input: TensorId,
        mean: TensorId,
        inv_variance: TensorId,
        dy: TensorId,
        scale: TensorId,
        dscale: TensorId,
        bias_gradient: TensorId,
        dx: TensorId,
        config: LayerNormalizationBackwardConfig,
    },
    RmsBackward {
        input: TensorId,
        inv_variance: TensorId,
        dy: TensorId,
        scale: TensorId,
        dscale: TensorId,
        bias_gradient: Option<TensorId>,
        dx: TensorId,
        config: RmsNormalizationBackwardConfig,
    },
    Instance {
        input: TensorId,
        scale: TensorId,
        bias: TensorId,
        output: TensorId,
        mean: TensorId,
        inv_variance: TensorId,
        config: InstanceNormalizationConfig,
    },
    InstanceBackward {
        input: TensorId,
        mean: TensorId,
        inv_variance: TensorId,
        dy: TensorId,
        scale: TensorId,
        dscale: TensorId,
        bias_gradient: TensorId,
        dx: TensorId,
        config: InstanceNormalizationBackwardConfig,
    },
    Batch {
        input: TensorId,
        scale: TensorId,
        bias: TensorId,
        output: TensorId,
        mean: TensorId,
        inv_variance: TensorId,
        config: BatchNormalizationConfig,
    },
    BatchInference {
        input: TensorId,
        mean: TensorId,
        inv_variance: TensorId,
        scale: TensorId,
        bias: TensorId,
        output: TensorId,
        config: BatchNormalizationInferenceConfig,
    },
    BatchFinalize {
        sum: TensorId,
        sq_sum: TensorId,
        scale: TensorId,
        bias: TensorId,
        next_running_mean: Option<TensorId>,
        next_running_var: Option<TensorId>,
        saved_mean: TensorId,
        saved_inv_variance: TensorId,
        eq_scale: TensorId,
        eq_bias: TensorId,
        accum_count: TensorId,
        config: BatchNormalizationFinalizeConfig,
    },
    BatchBackward {
        input: TensorId,
        mean: TensorId,
        inv_variance: TensorId,
        dy: TensorId,
        scale: TensorId,
        dscale: TensorId,
        bias_gradient: TensorId,
        dx: TensorId,
        config: BatchNormalizationBackwardConfig,
    },
    DbnWeight {
        dy: TensorId,
        input: TensorId,
        scale: TensorId,
        mean: TensorId,
        inv_variance: TensorId,
        dscale: TensorId,
        bias_gradient: TensorId,
        eq_bias: TensorId,
        eq_scale_dy: TensorId,
        eq_scale_x: TensorId,
        config: DbnWeightConfig,
    },
    AdaptiveLayer {
        input: TensorId,
        scale: TensorId,
        bias: Option<TensorId>,
        output: TensorId,
        mean: TensorId,
        inv_variance: TensorId,
        config: AdaptiveLayerNormalizationConfig,
    },
    AdaptiveLayerBackward {
        input: TensorId,
        mean: TensorId,
        inv_variance: TensorId,
        dy: TensorId,
        scale: TensorId,
        dscale: TensorId,
        bias_gradient: Option<TensorId>,
        dx: TensorId,
        config: AdaptiveLayerNormalizationBackwardConfig,
    },
}

impl FrontendOperationTensors for NormalizationOperation {
    fn append_tensor_ids(&self, tensors: &mut Vec<TensorId>) {
        match self {
            Self::Layer {
                input,
                scale,
                bias,
                output,
                mean,
                inv_variance,
                config,
            } => {
                tensors.extend([*input, *scale, *bias, *output, *mean, *inv_variance]);
                tensors.push(config.epsilon());
            }
            Self::Instance {
                input,
                scale,
                bias,
                output,
                mean,
                inv_variance,
                config,
            } => {
                tensors.extend([*input, *scale, *bias, *output, *mean, *inv_variance]);
                tensors.push(config.epsilon());
            }
            Self::Batch {
                input,
                scale,
                bias,
                output,
                mean,
                inv_variance,
                config,
            } => {
                tensors.extend([*input, *scale, *bias, *output, *mean, *inv_variance]);
                tensors.push(config.epsilon());
                if let Some(running) = config.running() {
                    tensors.extend([
                        running.momentum,
                        running.prev_mean,
                        running.prev_var,
                        running.next_mean,
                        running.next_var,
                    ]);
                }
                tensors.extend(config.peer_stats().iter().copied());
            }
            Self::Rms {
                input,
                scale,
                bias,
                output,
                inv_variance,
                config,
            } => {
                tensors.extend([*input, *scale, *output, *inv_variance]);
                tensors.extend(*bias);
                tensors.push(config.epsilon());
            }
            Self::LayerBackward {
                input,
                mean,
                inv_variance,
                dy,
                scale,
                dscale,
                bias_gradient,
                dx,
                config,
            } => {
                tensors.extend([
                    *input,
                    *mean,
                    *inv_variance,
                    *dy,
                    *scale,
                    *dscale,
                    *bias_gradient,
                    *dx,
                ]);
                tensors.push(config.epsilon());
            }
            Self::InstanceBackward {
                input,
                mean,
                inv_variance,
                dy,
                scale,
                dscale,
                bias_gradient,
                dx,
                config,
            } => {
                tensors.extend([
                    *input,
                    *mean,
                    *inv_variance,
                    *dy,
                    *scale,
                    *dscale,
                    *bias_gradient,
                    *dx,
                ]);
                tensors.push(config.epsilon());
            }
            Self::BatchBackward {
                input,
                mean,
                inv_variance,
                dy,
                scale,
                dscale,
                bias_gradient,
                dx,
                config,
            } => {
                tensors.extend([
                    *input,
                    *mean,
                    *inv_variance,
                    *dy,
                    *scale,
                    *dscale,
                    *bias_gradient,
                    *dx,
                ]);
                tensors.push(config.epsilon());
                tensors.extend(config.peer_stats().iter().copied());
            }
            Self::RmsBackward {
                input,
                inv_variance,
                dy,
                scale,
                dscale,
                bias_gradient,
                dx,
                config: _,
            } => {
                tensors.extend([*input, *inv_variance, *dy, *scale, *dscale, *dx]);
                tensors.extend(*bias_gradient);
            }
            Self::BatchInference {
                input,
                mean,
                inv_variance,
                scale,
                bias,
                output,
                config,
            } => {
                tensors.extend([*input, *mean, *inv_variance, *scale, *bias, *output]);
                tensors.push(config.epsilon());
            }
            Self::BatchFinalize {
                sum,
                sq_sum,
                scale,
                bias,
                next_running_mean,
                next_running_var,
                saved_mean,
                saved_inv_variance,
                eq_scale,
                eq_bias,
                accum_count,
                config,
            } => {
                tensors.extend([
                    *sum,
                    *sq_sum,
                    *scale,
                    *bias,
                    *saved_mean,
                    *saved_inv_variance,
                    *eq_scale,
                    *eq_bias,
                    *accum_count,
                    config.epsilon(),
                ]);
                tensors.extend(config.prev_running_mean());
                tensors.extend(config.prev_running_var());
                tensors.extend(*next_running_mean);
                tensors.extend(*next_running_var);
                tensors.extend(config.momentum());
            }
            Self::DbnWeight {
                dy,
                input,
                scale,
                mean,
                inv_variance,
                dscale,
                bias_gradient,
                eq_bias,
                eq_scale_dy,
                eq_scale_x,
                config: _,
            } => tensors.extend([
                *dy,
                *input,
                *scale,
                *mean,
                *inv_variance,
                *dscale,
                *bias_gradient,
                *eq_bias,
                *eq_scale_dy,
                *eq_scale_x,
            ]),
            Self::AdaptiveLayer {
                input,
                scale,
                bias,
                output,
                mean,
                inv_variance,
                config,
            } => {
                tensors.extend([*input, *scale, *output, *mean, *inv_variance]);
                tensors.extend(*bias);
                tensors.push(config.epsilon());
            }
            Self::AdaptiveLayerBackward {
                input,
                mean,
                inv_variance,
                dy,
                scale,
                dscale,
                bias_gradient,
                dx,
                config: _,
            } => {
                tensors.extend([*input, *mean, *inv_variance, *dy, *scale, *dscale, *dx]);
                tensors.extend(*bias_gradient);
            }
        }
    }
}

impl NormalizationOperation {
    pub(crate) fn lower(&self, context: &LoweringContext<'_>) -> Result<LoweredOperation> {
        let tensors = context.backend_tensors();
        Ok(match self {
            Self::Layer {
                input,
                scale,
                bias,
                output,
                mean,
                inv_variance,
                config,
            } => LoweredOperation::LayerNormalization(normalization_forward_op(
                tensors,
                BackendNormalizationMode::Layer,
                config.phase(),
                *input,
                Some(*mean),
                Some(*inv_variance),
                *scale,
                Some(*bias),
                config.epsilon(),
                None,
                &[],
                *output,
            )?),
            Self::Rms {
                input,
                scale,
                bias,
                output,
                inv_variance,
                config,
            } => LoweredOperation::RmsNormalization(normalization_forward_op(
                tensors,
                BackendNormalizationMode::Rms,
                config.phase(),
                *input,
                None,
                Some(*inv_variance),
                *scale,
                *bias,
                config.epsilon(),
                None,
                &[],
                *output,
            )?),
            Self::LayerBackward {
                input,
                mean,
                inv_variance,
                dy,
                scale,
                dscale,
                bias_gradient,
                dx,
                config,
            } => LoweredOperation::LayerNormalizationBackward(normalization_backward_op(
                tensors,
                BackendNormalizationMode::Layer,
                *input,
                Some(*mean),
                *inv_variance,
                *dy,
                *scale,
                Some(config.epsilon()),
                *dscale,
                Some(*bias_gradient),
                &[],
                *dx,
            )?),
            Self::RmsBackward {
                input,
                inv_variance,
                dy,
                scale,
                dscale,
                bias_gradient,
                dx,
                config: _,
            } => LoweredOperation::RmsNormalizationBackward(normalization_backward_op(
                tensors,
                BackendNormalizationMode::Rms,
                *input,
                None,
                *inv_variance,
                *dy,
                *scale,
                None,
                *dscale,
                *bias_gradient,
                &[],
                *dx,
            )?),
            Self::Instance {
                input,
                scale,
                bias,
                output,
                mean,
                inv_variance,
                config,
            } => LoweredOperation::InstanceNormalization(normalization_forward_op(
                tensors,
                BackendNormalizationMode::Instance,
                config.phase(),
                *input,
                Some(*mean),
                Some(*inv_variance),
                *scale,
                Some(*bias),
                config.epsilon(),
                None,
                &[],
                *output,
            )?),
            Self::InstanceBackward {
                input,
                mean,
                inv_variance,
                dy,
                scale,
                dscale,
                bias_gradient,
                dx,
                config,
            } => LoweredOperation::InstanceNormalizationBackward(normalization_backward_op(
                tensors,
                BackendNormalizationMode::Instance,
                *input,
                Some(*mean),
                *inv_variance,
                *dy,
                *scale,
                Some(config.epsilon()),
                *dscale,
                Some(*bias_gradient),
                &[],
                *dx,
            )?),
            Self::Batch {
                input,
                scale,
                bias,
                output,
                mean,
                inv_variance,
                config,
            } => LoweredOperation::BatchNormalization(normalization_forward_op(
                tensors,
                BackendNormalizationMode::Batch,
                BackendNormalizationForwardPhase::Training,
                *input,
                Some(*mean),
                Some(*inv_variance),
                *scale,
                Some(*bias),
                config.epsilon(),
                config.running(),
                config.peer_stats(),
                *output,
            )?),
            Self::BatchInference {
                input,
                mean,
                inv_variance,
                scale,
                bias,
                output,
                config,
            } => LoweredOperation::BatchNormalizationInference(normalization_forward_op(
                tensors,
                BackendNormalizationMode::Batch,
                BackendNormalizationForwardPhase::Inference,
                *input,
                Some(*mean),
                Some(*inv_variance),
                *scale,
                Some(*bias),
                config.epsilon(),
                None,
                &[],
                *output,
            )?),
            Self::BatchFinalize {
                sum,
                sq_sum,
                scale,
                bias,
                next_running_mean,
                next_running_var,
                saved_mean,
                saved_inv_variance,
                eq_scale,
                eq_bias,
                accum_count,
                config,
            } => LoweredOperation::BatchNormalizationFinalize(batch_normalization_finalize_op(
                tensors,
                *sum,
                *sq_sum,
                *scale,
                *bias,
                *next_running_mean,
                *next_running_var,
                *saved_mean,
                *saved_inv_variance,
                *eq_scale,
                *eq_bias,
                *accum_count,
                config,
            )?),
            Self::BatchBackward {
                input,
                mean,
                inv_variance,
                dy,
                scale,
                dscale,
                bias_gradient,
                dx,
                config,
            } => LoweredOperation::BatchNormalizationBackward(normalization_backward_op(
                tensors,
                BackendNormalizationMode::Batch,
                *input,
                Some(*mean),
                *inv_variance,
                *dy,
                *scale,
                None,
                *dscale,
                Some(*bias_gradient),
                config.peer_stats(),
                *dx,
            )?),
            Self::DbnWeight {
                dy,
                input,
                scale,
                mean,
                inv_variance,
                dscale,
                bias_gradient,
                eq_bias,
                eq_scale_dy,
                eq_scale_x,
                config: _,
            } => LoweredOperation::DbnWeight(batch_normalization_backward_weights_op(
                tensors,
                *dy,
                *input,
                *scale,
                *mean,
                *inv_variance,
                *dscale,
                *bias_gradient,
                *eq_bias,
                *eq_scale_dy,
                *eq_scale_x,
            )?),
            Self::AdaptiveLayer {
                input,
                scale,
                bias,
                output,
                mean,
                inv_variance,
                config,
            } => LoweredOperation::AdaptiveLayerNormalization(normalization_forward_op(
                tensors,
                BackendNormalizationMode::AdaLayerNorm,
                config.phase(),
                *input,
                Some(*mean),
                Some(*inv_variance),
                *scale,
                *bias,
                config.epsilon(),
                None,
                &[],
                *output,
            )?),
            Self::AdaptiveLayerBackward {
                input,
                mean,
                inv_variance,
                dy,
                scale,
                dscale,
                bias_gradient,
                dx,
                config: _,
            } => LoweredOperation::AdaptiveLayerNormalizationBackward(normalization_backward_op(
                tensors,
                BackendNormalizationMode::AdaLayerNorm,
                *input,
                Some(*mean),
                *inv_variance,
                *dy,
                *scale,
                None,
                *dscale,
                *bias_gradient,
                &[],
                *dx,
            )?),
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn normalization_forward_op(
    tensors: &BTreeMap<TensorId, Tensor>,
    mode: BackendNormalizationMode,
    phase: BackendNormalizationForwardPhase,
    input: TensorId,
    mean: Option<TensorId>,
    inv_variance: Option<TensorId>,
    scale: TensorId,
    bias: Option<TensorId>,
    epsilon: TensorId,
    running: Option<BatchNormalizationRunningStats>,
    peer_stats: &[TensorId],
    output: TensorId,
) -> Result<NormalizationForwardOperation> {
    let descriptor = NormalizationForwardConfig::new(mode, phase);
    let running_tensors = batch_running_tensors(tensors, running)?;
    NormalizationForwardOperation::create(
        &descriptor,
        tensor_at(tensors, input)?,
        optional_tensor_at(tensors, mean)?,
        optional_tensor_at(tensors, inv_variance)?,
        tensor_at(tensors, scale)?,
        optional_tensor_at(tensors, bias)?,
        Some(tensor_at(tensors, epsilon)?),
        running_tensors.0,
        running_tensors.1,
        running_tensors.2,
        running_tensors.3,
        running_tensors.4,
        &tensor_slice_at(tensors, peer_stats)?,
        tensor_at(tensors, output)?,
    )
}

#[allow(clippy::too_many_arguments)]
fn normalization_backward_op(
    tensors: &BTreeMap<TensorId, Tensor>,
    mode: BackendNormalizationMode,
    input: TensorId,
    mean: Option<TensorId>,
    inv_variance: TensorId,
    dy: TensorId,
    scale: TensorId,
    epsilon: Option<TensorId>,
    dscale: TensorId,
    bias_gradient: Option<TensorId>,
    peer_stats: &[TensorId],
    dx: TensorId,
) -> Result<NormalizationBackwardOperation> {
    let descriptor = NormalizationBackwardConfig::new(mode);
    NormalizationBackwardOperation::create(
        &descriptor,
        tensor_at(tensors, input)?,
        optional_tensor_at(tensors, mean)?,
        tensor_at(tensors, inv_variance)?,
        tensor_at(tensors, dy)?,
        tensor_at(tensors, scale)?,
        optional_tensor_at(tensors, epsilon)?,
        tensor_at(tensors, dscale)?,
        optional_tensor_at(tensors, bias_gradient)?,
        &tensor_slice_at(tensors, peer_stats)?,
        tensor_at(tensors, dx)?,
    )
}

#[allow(clippy::too_many_arguments)]
fn batch_normalization_finalize_op(
    tensors: &BTreeMap<TensorId, Tensor>,
    sum: TensorId,
    sq_sum: TensorId,
    scale: TensorId,
    bias: TensorId,
    next_running_mean: Option<TensorId>,
    next_running_var: Option<TensorId>,
    saved_mean: TensorId,
    saved_inv_variance: TensorId,
    eq_scale: TensorId,
    eq_bias: TensorId,
    accum_count: TensorId,
    config: &BatchNormalizationFinalizeConfig,
) -> Result<BatchNormalizationFinalizeStatsOperation> {
    BatchNormalizationFinalizeStatsOperation::create(
        config.mode(),
        DataType::F32,
        tensor_at(tensors, sum)?,
        tensor_at(tensors, sq_sum)?,
        tensor_at(tensors, scale)?,
        tensor_at(tensors, bias)?,
        optional_tensor_at(tensors, config.prev_running_mean())?,
        optional_tensor_at(tensors, config.prev_running_var())?,
        optional_tensor_at(tensors, next_running_mean)?,
        optional_tensor_at(tensors, next_running_var)?,
        tensor_at(tensors, saved_mean)?,
        tensor_at(tensors, saved_inv_variance)?,
        tensor_at(tensors, eq_scale)?,
        tensor_at(tensors, eq_bias)?,
        tensor_at(tensors, accum_count)?,
        tensor_at(tensors, config.epsilon())?,
        optional_tensor_at(tensors, config.momentum())?,
    )
}

#[allow(clippy::too_many_arguments)]
fn batch_normalization_backward_weights_op(
    tensors: &BTreeMap<TensorId, Tensor>,
    dy: TensorId,
    input: TensorId,
    scale: TensorId,
    mean: TensorId,
    inv_variance: TensorId,
    dscale: TensorId,
    bias_gradient: TensorId,
    eq_bias: TensorId,
    eq_scale_dy: TensorId,
    eq_scale_x: TensorId,
) -> Result<BatchNormalizationBackwardWeightsOperation> {
    BatchNormalizationBackwardWeightsOperation::create(
        DataType::F32,
        tensor_at(tensors, mean)?,
        tensor_at(tensors, inv_variance)?,
        tensor_at(tensors, scale)?,
        tensor_at(tensors, input)?,
        tensor_at(tensors, dy)?,
        tensor_at(tensors, dscale)?,
        tensor_at(tensors, bias_gradient)?,
        tensor_at(tensors, eq_scale_dy)?,
        tensor_at(tensors, eq_scale_x)?,
        tensor_at(tensors, eq_bias)?,
    )
}

type BatchRunningTensors<'a> = (
    Option<&'a Tensor>,
    Option<&'a Tensor>,
    Option<&'a Tensor>,
    Option<&'a Tensor>,
    Option<&'a Tensor>,
);

fn batch_running_tensors(
    tensors: &BTreeMap<TensorId, Tensor>,
    running: Option<BatchNormalizationRunningStats>,
) -> Result<BatchRunningTensors<'_>> {
    let Some(BatchNormalizationRunningStats {
        momentum,
        prev_mean,
        prev_var,
        next_mean,
        next_var,
    }) = running
    else {
        return Ok((None, None, None, None, None));
    };
    Ok((
        Some(tensor_at(tensors, momentum)?),
        Some(tensor_at(tensors, prev_mean)?),
        Some(tensor_at(tensors, prev_var)?),
        Some(tensor_at(tensors, next_mean)?),
        Some(tensor_at(tensors, next_var)?),
    ))
}
