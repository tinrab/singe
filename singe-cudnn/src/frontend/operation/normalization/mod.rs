mod batch;
mod layer_rms;
mod lower;
mod other_backward;
mod other_forward;
mod statistics;

use serde::{Deserialize, Serialize};

use crate::backend::tensor::TensorId;
pub use crate::frontend::operation::normalization::{
    batch::{
        BatchNormalizationBackwardConfig, BatchNormalizationBackwardInputs,
        BatchNormalizationBackwardOutputs, BatchNormalizationBackwardWeightsConfig,
        BatchNormalizationBackwardWeightsInputs, BatchNormalizationBackwardWeightsOutputs,
        BatchNormalizationConfig, BatchNormalizationFinalizeConfig,
        BatchNormalizationFinalizeInputs, BatchNormalizationFinalizeOutputs,
        BatchNormalizationInferenceInputs, BatchNormalizationInputs, BatchNormalizationOutputs,
        BatchNormalizationRunningStatistics, BatchNormalizationRunningStats,
    },
    layer_rms::{
        LayerNormalizationAffine, LayerNormalizationAffineGradients,
        LayerNormalizationBackwardConfig, LayerNormalizationBackwardInputs,
        LayerNormalizationBackwardOutputs, LayerNormalizationConfig, LayerNormalizationInputs,
        LayerNormalizationOutputs, RmsNormalizationAffineGradients, RmsNormalizationBackwardConfig,
        RmsNormalizationBackwardInputs, RmsNormalizationBackwardOutputs, RmsNormalizationConfig,
        RmsNormalizationInputs, RmsNormalizationOutputs, RmsNormalizationStatistics,
    },
    other_backward::{
        AdaptiveLayerNormalizationBackwardInputs, AdaptiveLayerNormalizationBackwardOutputs,
        InstanceNormalizationBackwardInputs, InstanceNormalizationBackwardOutputs,
    },
    other_forward::{
        AdaptiveLayerNormalizationConfig, AdaptiveLayerNormalizationInputs,
        AdaptiveLayerNormalizationOutputs, InstanceNormalizationConfig,
        InstanceNormalizationInputs, InstanceNormalizationOutputs,
    },
    statistics::{NormalizationSavedStatistics, NormalizationStatistics},
};

/// Common attributes for normalization operations without additional operation options.
/// Used by `BatchNorm` inference, `InstanceNorm` backward and `AdaLayerNorm` backward.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NormalizationConfig {
    pub(crate) name: Option<String>,
}

impl NormalizationConfig {
    /// Creates an unnamed normalization request.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Names inferred results using C++ output-port suffixes.
    /// Explicit tensor names are retained, and an empty scope disables naming.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Returns the nonempty output-name scope, or `None` when unnamed.
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref().filter(|name| !name.is_empty())
    }
}

/// Frontend normalization operation variants.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum NormalizationOperation {
    /// Layer normalization forward.
    Layer {
        /// Input and saved-statistic operands for this normalization stage.
        inputs: LayerNormalizationInputs,
        /// Activation, statistic, or gradient destinations for this normalization stage.
        outputs: LayerNormalizationOutputs,
        /// Operation name and stage-specific normalization settings.
        config: LayerNormalizationConfig,
    },
    /// Root-mean-square normalization forward.
    Rms {
        /// Input and saved-statistic operands for this normalization stage.
        inputs: RmsNormalizationInputs,
        /// Activation, statistic, or gradient destinations for this normalization stage.
        outputs: RmsNormalizationOutputs,
        /// Operation name and stage-specific normalization settings.
        config: RmsNormalizationConfig,
    },
    /// Layer normalization backward with optional affine gradients.
    LayerBackward {
        /// Input and saved-statistic operands for this normalization stage.
        inputs: LayerNormalizationBackwardInputs,
        /// Activation, statistic, or gradient destinations for this normalization stage.
        outputs: LayerNormalizationBackwardOutputs,
        /// Operation name and stage-specific normalization settings.
        config: LayerNormalizationBackwardConfig,
    },
    /// Root-mean-square normalization backward with optional affine gradients.
    RmsBackward {
        /// Input and saved-statistic operands for this normalization stage.
        inputs: RmsNormalizationBackwardInputs,
        /// Activation, statistic, or gradient destinations for this normalization stage.
        outputs: RmsNormalizationBackwardOutputs,
        /// Operation name and stage-specific normalization settings.
        config: RmsNormalizationBackwardConfig,
    },
    /// Instance normalization forward.
    Instance {
        /// Input and saved-statistic operands for this normalization stage.
        inputs: InstanceNormalizationInputs,
        /// Activation, statistic, or gradient destinations for this normalization stage.
        outputs: InstanceNormalizationOutputs,
        /// Operation name and stage-specific normalization settings.
        config: InstanceNormalizationConfig,
    },
    /// Instance normalization backward.
    InstanceBackward {
        /// Input and saved-statistic operands for this normalization stage.
        inputs: InstanceNormalizationBackwardInputs,
        /// Activation, statistic, or gradient destinations for this normalization stage.
        outputs: InstanceNormalizationBackwardOutputs,
        /// Operation name and stage-specific normalization settings.
        config: NormalizationConfig,
    },
    /// Batch normalization training with saved statistics and optional running-statistics updates.
    Batch {
        /// Input and saved-statistic operands for this normalization stage.
        inputs: BatchNormalizationInputs,
        /// Activation, statistic, or gradient destinations for this normalization stage.
        outputs: BatchNormalizationOutputs,
        /// Operation name and stage-specific normalization settings.
        config: BatchNormalizationConfig,
    },
    /// Batch normalization inference using supplied statistics.
    BatchInference {
        /// Input and saved-statistic operands for this normalization stage.
        inputs: BatchNormalizationInferenceInputs,
        /// Normalized inference output tensor identity.
        output: TensorId,
        /// Operation name and stage-specific normalization settings.
        config: NormalizationConfig,
    },
    /// Finalizes batch statistics and equivalent affine parameters from accumulated reductions.
    BatchFinalize {
        /// Input and saved-statistic operands for this normalization stage.
        inputs: BatchNormalizationFinalizeInputs,
        /// Activation, statistic, or gradient destinations for this normalization stage.
        outputs: BatchNormalizationFinalizeOutputs,
        /// Operation name and stage-specific normalization settings.
        config: BatchNormalizationFinalizeConfig,
    },
    /// Batch normalization backward.
    BatchBackward {
        /// Input and saved-statistic operands for this normalization stage.
        inputs: BatchNormalizationBackwardInputs,
        /// Activation, statistic, or gradient destinations for this normalization stage.
        outputs: BatchNormalizationBackwardOutputs,
        /// Operation name and stage-specific normalization settings.
        config: BatchNormalizationBackwardConfig,
    },
    /// Computes batch-normalization parameter gradients from saved statistics and reductions.
    BatchNormalizationBackwardWeights {
        /// Input and saved-statistic operands for this normalization stage.
        inputs: BatchNormalizationBackwardWeightsInputs,
        /// Activation, statistic, or gradient destinations for this normalization stage.
        outputs: BatchNormalizationBackwardWeightsOutputs,
        /// Operation name and stage-specific normalization settings.
        config: BatchNormalizationBackwardWeightsConfig,
    },
    /// Adaptive layer normalization forward with modulation operands.
    AdaptiveLayer {
        /// Input and saved-statistic operands for this normalization stage.
        inputs: AdaptiveLayerNormalizationInputs,
        /// Activation, statistic, or gradient destinations for this normalization stage.
        outputs: AdaptiveLayerNormalizationOutputs,
        /// Operation name and stage-specific normalization settings.
        config: AdaptiveLayerNormalizationConfig,
    },
    /// Adaptive layer normalization backward with modulation gradients.
    AdaptiveLayerBackward {
        /// Input and saved-statistic operands for this normalization stage.
        inputs: AdaptiveLayerNormalizationBackwardInputs,
        /// Activation, statistic, or gradient destinations for this normalization stage.
        outputs: AdaptiveLayerNormalizationBackwardOutputs,
        /// Operation name and stage-specific normalization settings.
        config: NormalizationConfig,
    },
}
