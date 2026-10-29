use crate::{
    backend::{
        execution::normalization::{
            BatchNormalizationBackwardWeightsOperation, BatchNormalizationFinalizeStatsConfig,
            BatchNormalizationFinalizeStatsOperation, NormalizationBackwardConfig,
            NormalizationBackwardOperation, NormalizationBackwardTensors,
            NormalizationForwardConfig, NormalizationForwardOperation, NormalizationForwardTensors,
        },
        normalization::{BackendNormalizationForwardPhase, BackendNormalizationMode},
    },
    error::Result,
    frontend::{
        graph::Graph,
        lower::{LoweredOperation, LoweringContext, tensor_at},
        operation::NormalizationOperation,
    },
};

impl NormalizationOperation {
    pub(crate) fn validate_graph(&self, graph: &Graph) -> Result<()> {
        match self {
            Self::Layer {
                inputs,
                outputs,
                config,
            } => graph.validate_layer_normalization(*inputs, *outputs, config),
            Self::Rms {
                inputs,
                outputs,
                config,
            } => graph.validate_rms_normalization(*inputs, *outputs, config),
            Self::LayerBackward {
                inputs,
                outputs,
                config,
            } => graph.validate_layer_normalization_backward(*inputs, *outputs, config),
            Self::RmsBackward {
                inputs,
                outputs,
                config,
            } => graph.validate_rms_normalization_backward(*inputs, *outputs, config),
            Self::Instance {
                inputs,
                outputs,
                config,
            } => graph.validate_instance_normalization(*inputs, *outputs, config),
            Self::AdaptiveLayer {
                inputs,
                outputs,
                config,
            } => graph.validate_adaptive_layer_normalization(*inputs, *outputs, config),
            Self::InstanceBackward {
                inputs, outputs, ..
            } => graph.validate_instance_normalization_backward(*inputs, *outputs),
            Self::AdaptiveLayerBackward {
                inputs, outputs, ..
            } => graph.validate_adaptive_layer_normalization_backward(*inputs, *outputs),
            Self::Batch {
                inputs,
                outputs,
                config,
            } => graph.validate_batch_normalization(*inputs, *outputs, config),
            Self::BatchInference { inputs, output, .. } => {
                graph.validate_batch_normalization_inference(*inputs, *output)
            }
            Self::BatchBackward {
                inputs,
                outputs,
                config,
            } => graph.validate_batch_normalization_backward(*inputs, *outputs, config),
            Self::BatchFinalize {
                inputs,
                outputs,
                config,
            } => graph.validate_batch_normalization_finalize(*inputs, *outputs, config),
            Self::BatchNormalizationBackwardWeights {
                inputs, outputs, ..
            } => graph.validate_batch_normalization_backward_weights(*inputs, *outputs),
        }
    }

    #[expect(
        clippy::too_many_lines,
        reason = "the normalization graph keeps its coupled shape and execution contracts together"
    )]
    pub(crate) fn lower(&self, context: &LoweringContext<'_>) -> Result<LoweredOperation> {
        let tensors = context.backend_tensors();
        Ok(match self {
            Self::Layer {
                inputs,
                outputs,
                config,
            } => LoweredOperation::LayerNormalization(NormalizationForwardOperation::create(
                &NormalizationForwardConfig::new(
                    BackendNormalizationMode::Layer,
                    outputs.statistics.phase(),
                ),
                NormalizationForwardTensors {
                    mean: outputs.statistics.saved().map(|stats| stats.mean),
                    inv_variance: outputs.statistics.saved().map(|stats| stats.inv_variance),
                    scale: inputs.affine.scale(),
                    bias: inputs.affine.bias(),
                    epsilon: Some(config.epsilon()),
                    ..NormalizationForwardTensors::new(inputs.input, outputs.output)
                }
                .try_map(|id| tensor_at(tensors, id))?,
            )?),
            Self::Rms {
                inputs,
                outputs,
                config,
            } => LoweredOperation::RmsNormalization(NormalizationForwardOperation::create(
                &NormalizationForwardConfig::new(
                    BackendNormalizationMode::Rms,
                    outputs.statistics.phase(),
                ),
                NormalizationForwardTensors {
                    inv_variance: outputs.statistics.inv_variance(),
                    scale: inputs.scale,
                    bias: inputs.bias,
                    epsilon: Some(config.epsilon()),
                    ..NormalizationForwardTensors::new(inputs.input, outputs.output)
                }
                .try_map(|id| tensor_at(tensors, id))?,
            )?),
            Self::LayerBackward {
                inputs, outputs, ..
            } => LoweredOperation::LayerNormalizationBackward(
                NormalizationBackwardOperation::create(
                    &NormalizationBackwardConfig::new(BackendNormalizationMode::Layer),
                    NormalizationBackwardTensors {
                        mean: Some(inputs.statistics.mean),
                        scale: inputs.scale,
                        scale_gradient: outputs.affine.map(|affine| affine.scale_gradient),
                        bias_gradient: outputs.affine.map(|affine| affine.bias_gradient),
                        ..NormalizationBackwardTensors::new(
                            inputs.input,
                            inputs.output_gradient,
                            outputs.input_gradient,
                            inputs.statistics.inv_variance,
                        )
                    }
                    .try_map(|id| tensor_at(tensors, id))?,
                )?,
            ),
            Self::RmsBackward {
                inputs, outputs, ..
            } => {
                LoweredOperation::RmsNormalizationBackward(NormalizationBackwardOperation::create(
                    &NormalizationBackwardConfig::new(BackendNormalizationMode::Rms),
                    NormalizationBackwardTensors {
                        scale: inputs.scale,
                        scale_gradient: outputs.affine.map(|affine| affine.scale_gradient),
                        bias_gradient: outputs.affine.and_then(|affine| affine.bias_gradient),
                        ..NormalizationBackwardTensors::new(
                            inputs.input,
                            inputs.output_gradient,
                            outputs.input_gradient,
                            inputs.inv_variance,
                        )
                    }
                    .try_map(|id| tensor_at(tensors, id))?,
                )?)
            }
            Self::Instance {
                inputs,
                outputs,
                config,
            } => LoweredOperation::InstanceNormalization(NormalizationForwardOperation::create(
                &NormalizationForwardConfig::new(
                    BackendNormalizationMode::Instance,
                    outputs.statistics.phase(),
                ),
                NormalizationForwardTensors {
                    mean: outputs.statistics.saved().map(|stats| stats.mean),
                    inv_variance: outputs.statistics.saved().map(|stats| stats.inv_variance),
                    scale: Some(inputs.scale),
                    bias: Some(inputs.bias),
                    epsilon: Some(config.epsilon()),
                    ..NormalizationForwardTensors::new(inputs.input, outputs.output)
                }
                .try_map(|id| tensor_at(tensors, id))?,
            )?),
            Self::InstanceBackward {
                inputs, outputs, ..
            } => LoweredOperation::InstanceNormalizationBackward(
                NormalizationBackwardOperation::create(
                    &NormalizationBackwardConfig::new(BackendNormalizationMode::Instance),
                    NormalizationBackwardTensors {
                        mean: Some(inputs.statistics.mean),
                        scale: Some(inputs.scale),
                        scale_gradient: Some(outputs.scale_gradient),
                        bias_gradient: Some(outputs.bias_gradient),
                        ..NormalizationBackwardTensors::new(
                            inputs.input,
                            inputs.output_gradient,
                            outputs.input_gradient,
                            inputs.statistics.inv_variance,
                        )
                    }
                    .try_map(|id| tensor_at(tensors, id))?,
                )?,
            ),
            Self::Batch {
                inputs,
                outputs,
                config,
            } => LoweredOperation::BatchNormalization(NormalizationForwardOperation::create(
                &NormalizationForwardConfig::new(
                    BackendNormalizationMode::Batch,
                    BackendNormalizationForwardPhase::Training,
                ),
                NormalizationForwardTensors {
                    mean: Some(outputs.statistics.mean),
                    inv_variance: Some(outputs.statistics.inv_variance),
                    scale: Some(inputs.scale),
                    bias: Some(inputs.bias),
                    epsilon: Some(config.epsilon()),
                    averaging_factor: config.running().map(|running| running.momentum),
                    previous_running_mean: config.running().map(|running| running.previous.mean),
                    previous_running_variance: config
                        .running()
                        .map(|running| running.previous.variance),
                    next_running_mean: outputs.running.map(|running| running.mean),
                    next_running_variance: outputs.running.map(|running| running.variance),
                    peer_statistics: config.peer_stats().to_vec(),
                    ..NormalizationForwardTensors::new(inputs.input, outputs.output)
                }
                .try_map(|id| tensor_at(tensors, id))?,
            )?),
            Self::BatchInference { inputs, output, .. } => {
                LoweredOperation::BatchNormalizationInference(
                    NormalizationForwardOperation::create(
                        &NormalizationForwardConfig::new(
                            BackendNormalizationMode::Batch,
                            BackendNormalizationForwardPhase::Inference,
                        ),
                        NormalizationForwardTensors {
                            mean: Some(inputs.statistics.mean),
                            inv_variance: Some(inputs.statistics.inv_variance),
                            scale: Some(inputs.scale),
                            bias: Some(inputs.bias),
                            ..NormalizationForwardTensors::new(inputs.input, *output)
                        }
                        .try_map(|id| tensor_at(tensors, id))?,
                    )?,
                )
            }
            Self::BatchFinalize {
                inputs,
                outputs,
                config,
            } => LoweredOperation::BatchNormalizationFinalize(
                BatchNormalizationFinalizeStatsOperation::create(
                    (*inputs).try_map(|id| tensor_at(tensors, id))?,
                    (*outputs).try_map(|id| tensor_at(tensors, id))?,
                    BatchNormalizationFinalizeStatsConfig::new(
                        config.mode(),
                        context.compute_data_type(config.compute_type()),
                        config.epsilon(),
                    )
                    .with_running_stats(config.running())
                    .try_map(|id| tensor_at(tensors, id))?,
                )?,
            ),
            Self::BatchBackward {
                inputs,
                outputs,
                config,
            } => LoweredOperation::BatchNormalizationBackward(
                NormalizationBackwardOperation::create(
                    &NormalizationBackwardConfig::new(BackendNormalizationMode::Batch),
                    NormalizationBackwardTensors {
                        mean: Some(inputs.statistics.mean),
                        scale: Some(inputs.scale),
                        scale_gradient: Some(outputs.scale_gradient),
                        bias_gradient: Some(outputs.bias_gradient),
                        peer_statistics: config.peer_stats().to_vec(),
                        ..NormalizationBackwardTensors::new(
                            inputs.input,
                            inputs.output_gradient,
                            outputs.input_gradient,
                            inputs.statistics.inv_variance,
                        )
                    }
                    .try_map(|id| tensor_at(tensors, id))?,
                )?,
            ),
            Self::BatchNormalizationBackwardWeights {
                inputs,
                outputs,
                config,
            } => LoweredOperation::BatchNormalizationBackwardWeights(
                BatchNormalizationBackwardWeightsOperation::create(
                    context.compute_data_type(config.compute_type()),
                    (*inputs).try_map(|id| tensor_at(tensors, id))?,
                    (*outputs).try_map(|id| tensor_at(tensors, id))?,
                )?,
            ),
            Self::AdaptiveLayer {
                inputs,
                outputs,
                config,
            } => {
                LoweredOperation::AdaptiveLayerNormalization(NormalizationForwardOperation::create(
                    &NormalizationForwardConfig::new(
                        BackendNormalizationMode::AdaLayerNorm,
                        outputs.statistics.phase(),
                    ),
                    NormalizationForwardTensors {
                        mean: outputs.statistics.saved().map(|stats| stats.mean),
                        inv_variance: outputs.statistics.saved().map(|stats| stats.inv_variance),
                        scale: Some(inputs.scale),
                        bias: inputs.bias,
                        epsilon: Some(config.epsilon()),
                        ..NormalizationForwardTensors::new(inputs.input, outputs.output)
                    }
                    .try_map(|id| tensor_at(tensors, id))?,
                )?)
            }
            Self::AdaptiveLayerBackward {
                inputs, outputs, ..
            } => LoweredOperation::AdaptiveLayerNormalizationBackward(
                NormalizationBackwardOperation::create(
                    &NormalizationBackwardConfig::new(BackendNormalizationMode::AdaLayerNorm),
                    NormalizationBackwardTensors {
                        mean: Some(inputs.statistics.mean),
                        scale: Some(inputs.scale),
                        scale_gradient: Some(outputs.scale_gradient),
                        bias_gradient: outputs.bias_gradient,
                        ..NormalizationBackwardTensors::new(
                            inputs.input,
                            inputs.output_gradient,
                            outputs.input_gradient,
                            inputs.statistics.inv_variance,
                        )
                    }
                    .try_map(|id| tensor_at(tensors, id))?,
                )?,
            ),
        })
    }
}
