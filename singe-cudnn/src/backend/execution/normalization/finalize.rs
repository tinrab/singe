use std::result::Result as StdResult;

use num_enum::{IntoPrimitive, TryFromPrimitive};

use serde::{Deserialize, Serialize};

use singe_core::{impl_enum_conversion, impl_enum_display};

use singe_cudnn_sys as sys;

use crate::{
    backend::{
        attribute::BackendAttributeName,
        descriptor::{BackendDescriptor, BackendDescriptorType},
        execution::normalization::{
            BatchNormalizationRunningStatistics, BatchNormalizationRunningStats,
            NormalizationSavedStatistics,
        },
        tensor::{BackendTensorUid, Tensor, TensorId},
    },
    data_type::DataType,
    error::Result,
};

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive, Serialize, Deserialize,
)]
/// Mathematical mode that converts batch normalization statistics and trained scale/bias
/// into equivalent scale/bias for the next normalization stage.
#[repr(u32)]
#[non_exhaustive]
pub enum BatchNormalizationFinalizeStatsMode {
    ///< Training mode finalization. @since cuDNN 9.0.0
    Training = sys::cudnnBnFinalizeStatsMode_t::CUDNN_BN_FINALIZE_STATISTICS_TRAINING as _,
    ///< Inference mode finalization. @since cuDNN 9.0.0
    Inference = sys::cudnnBnFinalizeStatsMode_t::CUDNN_BN_FINALIZE_STATISTICS_INFERENCE as _,
}

impl_enum_conversion!(
    sys::cudnnBnFinalizeStatsMode_t,
    BatchNormalizationFinalizeStatsMode
);

impl_enum_display!(BatchNormalizationFinalizeStatsMode, {
    Self::Training => "CUDNN_BN_FINALIZE_STATISTICS_TRAINING",
    Self::Inference => "CUDNN_BN_FINALIZE_STATISTICS_INFERENCE",
});

/// Accumulated statistics and affine values consumed by `BatchNorm` finalization.
/// Shared by frontend identities, native tensor references and retained UIDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchNormalizationFinalizeInputs<T = TensorId> {
    /// Accumulated per-channel sum tensor.
    pub sum: T,
    /// Accumulated per-channel sum-of-squares tensor.
    pub squared_sum: T,
    /// Affine scale tensor.
    pub scale: T,
    /// Affine bias tensor.
    pub bias: T,
    /// Tensor containing the number of elements contributing to each accumulated statistic.
    pub accumulation_count: T,
}

impl<T> BatchNormalizationFinalizeInputs<T> {
    /// Borrows accumulated statistics, affine operands and the accumulation count.
    pub fn as_ref(&self) -> BatchNormalizationFinalizeInputs<&T> {
        BatchNormalizationFinalizeInputs {
            sum: &self.sum,
            squared_sum: &self.squared_sum,
            scale: &self.scale,
            bias: &self.bias,
            accumulation_count: &self.accumulation_count,
        }
    }

    /// Maps tensor operands while preserving their statistical roles and optional omissions.
    pub fn map<U>(self, mut map: impl FnMut(T) -> U) -> BatchNormalizationFinalizeInputs<U> {
        BatchNormalizationFinalizeInputs {
            sum: map(self.sum),
            squared_sum: map(self.squared_sum),
            scale: map(self.scale),
            bias: map(self.bias),
            accumulation_count: map(self.accumulation_count),
        }
    }

    /// Maps tensor operands in declaration order while preserving their statistical roles and optional omissions.
    ///
    /// # Errors
    ///
    /// Returns the first error from the supplied tensor mapping; later operands are not mapped.
    pub fn try_map<U, E>(
        self,
        mut map: impl FnMut(T) -> StdResult<U, E>,
    ) -> StdResult<BatchNormalizationFinalizeInputs<U>, E> {
        Ok(BatchNormalizationFinalizeInputs {
            sum: map(self.sum)?,
            squared_sum: map(self.squared_sum)?,
            scale: map(self.scale)?,
            bias: map(self.bias)?,
            accumulation_count: map(self.accumulation_count)?,
        })
    }
}

/// Equivalent affine values, saved statistics and optional updated running statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchNormalizationFinalizeOutputs<T = TensorId> {
    /// Equivalent affine scale tensor for the finalized normalization step.
    pub equivalent_scale: T,
    /// Equivalent bias tensor for the subsequent normalization step.
    pub equivalent_bias: T,
    /// Saved mean and inverse-standard-deviation tensors.
    pub statistics: NormalizationSavedStatistics<T>,
    /// Optional updated running mean and variance destinations.
    pub running: Option<BatchNormalizationRunningStatistics<T>>,
}

impl<T> BatchNormalizationFinalizeOutputs<T> {
    /// Borrows equivalent coefficients and statistics while preserving omitted running outputs.
    pub fn as_ref(&self) -> BatchNormalizationFinalizeOutputs<&T> {
        BatchNormalizationFinalizeOutputs {
            equivalent_scale: &self.equivalent_scale,
            equivalent_bias: &self.equivalent_bias,
            statistics: self.statistics.as_ref(),
            running: self
                .running
                .as_ref()
                .map(BatchNormalizationRunningStatistics::as_ref),
        }
    }

    /// Maps tensor operands while preserving their statistical roles and optional omissions.
    pub fn map<U>(self, mut map: impl FnMut(T) -> U) -> BatchNormalizationFinalizeOutputs<U> {
        BatchNormalizationFinalizeOutputs {
            equivalent_scale: map(self.equivalent_scale),
            equivalent_bias: map(self.equivalent_bias),
            statistics: self.statistics.map(&mut map),
            running: self.running.map(|running| running.map(map)),
        }
    }

    /// Maps tensor operands in declaration order while preserving their statistical roles and optional omissions.
    ///
    /// # Errors
    ///
    /// Returns the first error from the supplied tensor mapping; later operands are not mapped.
    pub fn try_map<U, E>(
        self,
        mut map: impl FnMut(T) -> StdResult<U, E>,
    ) -> StdResult<BatchNormalizationFinalizeOutputs<U>, E> {
        Ok(BatchNormalizationFinalizeOutputs {
            equivalent_scale: map(self.equivalent_scale)?,
            equivalent_bias: map(self.equivalent_bias)?,
            statistics: self.statistics.try_map(&mut map)?,
            running: self
                .running
                .map(|running| running.try_map(map))
                .transpose()?,
        })
    }
}

/// Resolved native finalization mode, math precision and scalar operands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchNormalizationFinalizeStatsConfig<T = TensorId> {
    /// Native statistics mode.
    pub mode: BatchNormalizationFinalizeStatsMode,
    /// Math precision selected independently of tensor storage encodings.
    pub compute_type: DataType,
    /// Scalar numerical-stability tensor.
    pub epsilon: T,
    /// Optional exponential averaging factor and previous running mean and variance.
    pub running: Option<BatchNormalizationRunningStats<T>>,
}

impl<T> BatchNormalizationFinalizeStatsConfig<T> {
    /// Borrows scalar and running-statistic operands while retaining mode and math precision.
    pub fn as_ref(&self) -> BatchNormalizationFinalizeStatsConfig<&T> {
        BatchNormalizationFinalizeStatsConfig {
            mode: self.mode,
            compute_type: self.compute_type,
            epsilon: &self.epsilon,
            running: self
                .running
                .as_ref()
                .map(BatchNormalizationRunningStats::as_ref),
        }
    }

    /// Records native mode, math precision, and epsilon without a running-statistics update.
    pub fn new(
        mode: BatchNormalizationFinalizeStatsMode,
        compute_type: DataType,
        epsilon: T,
    ) -> Self {
        Self {
            mode,
            compute_type,
            epsilon,
            running: None,
        }
    }

    /// Sets or clears the averaging factor and previous running-statistics inputs while retaining mode, math precision, and epsilon.
    #[must_use]
    pub fn with_running_stats(
        mut self,
        running: impl Into<Option<BatchNormalizationRunningStats<T>>>,
    ) -> Self {
        self.running = running.into();
        self
    }

    /// Maps tensor operands while preserving their statistical roles and optional omissions.
    pub fn map<U>(self, mut map: impl FnMut(T) -> U) -> BatchNormalizationFinalizeStatsConfig<U> {
        BatchNormalizationFinalizeStatsConfig {
            mode: self.mode,
            compute_type: self.compute_type,
            epsilon: map(self.epsilon),
            running: self.running.map(|running| running.map(map)),
        }
    }

    /// Maps tensor operands in declaration order while preserving their statistical roles and optional omissions.
    ///
    /// # Errors
    ///
    /// Returns the first error from the supplied tensor mapping; later operands are not mapped.
    pub fn try_map<U, E>(
        self,
        mut map: impl FnMut(T) -> StdResult<U, E>,
    ) -> StdResult<BatchNormalizationFinalizeStatsConfig<U>, E> {
        Ok(BatchNormalizationFinalizeStatsConfig {
            mode: self.mode,
            compute_type: self.compute_type,
            epsilon: map(self.epsilon)?,
            running: self
                .running
                .map(|running| running.try_map(map))
                .transpose()?,
        })
    }
}

/// Native finalization attributes and operand identities, independent of original tensor owners.
/// Absent tensor attributes remain `None`, independently of the selected native mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchNormalizationFinalizeStatsOperationMetadata {
    /// Retained native input identities, preserving independently omitted attributes.
    pub inputs: BatchNormalizationFinalizeInputs<Option<BackendTensorUid>>,
    /// Retained native output identities, preserving independently omitted attributes.
    pub outputs: BatchNormalizationFinalizeOutputs<Option<BackendTensorUid>>,
    /// Retained native mode, math precision, scalar identities, and optional running-statistics inputs.
    pub config: BatchNormalizationFinalizeStatsConfig<Option<BackendTensorUid>>,
}

/// Finalized native batch-statistics operation retaining equivalent coefficients and statistic operand identities.
#[derive(Debug)]
pub struct BatchNormalizationFinalizeStatsOperation {
    descriptor: BackendDescriptor,
    metadata: BatchNormalizationFinalizeStatsOperationMetadata,
}

impl BatchNormalizationFinalizeStatsOperation {
    /// Creates finalization with shared accumulated-statistic and result groups.
    /// Previous and updated running statistics retain their distinct input and output roles.
    /// Groups accept tensor references or optional references to omit individual native attributes.
    /// Each group chooses its reference representation independently.
    /// cuDNN decides which combinations are supported for the selected mode.
    ///
    /// # Errors
    /// Returns an error if cuDNN rejects the mode, precision, tensors or descriptor finalization.
    pub fn create<'a>(
        inputs: BatchNormalizationFinalizeInputs<impl Into<Option<&'a Tensor>>>,
        outputs: BatchNormalizationFinalizeOutputs<impl Into<Option<&'a Tensor>>>,
        config: BatchNormalizationFinalizeStatsConfig<impl Into<Option<&'a Tensor>>>,
    ) -> Result<Self> {
        let inputs: BatchNormalizationFinalizeInputs<Option<&Tensor>> = inputs.map(Into::into);
        let outputs: BatchNormalizationFinalizeOutputs<Option<&Tensor>> = outputs.map(Into::into);
        let config: BatchNormalizationFinalizeStatsConfig<Option<&Tensor>> = config.map(Into::into);
        let mut descriptor =
            BackendDescriptor::create(BackendDescriptorType::OperationBnFinalizeStatistics)?;
        descriptor.set_attribute_enum(
            BackendAttributeName::OperationBnFinalizeStatsMode,
            config.mode,
        )?;
        descriptor.set_attribute_enum(
            BackendAttributeName::OperationBnFinalizeMathPrec,
            config.compute_type,
        )?;
        for (attribute, tensor) in [
            (
                BackendAttributeName::OperationBnFinalizeYSumDesc,
                inputs.sum,
            ),
            (
                BackendAttributeName::OperationBnFinalizeYSqSumDesc,
                inputs.squared_sum,
            ),
            (
                BackendAttributeName::OperationBnFinalizeScaleDesc,
                inputs.scale,
            ),
            (
                BackendAttributeName::OperationBnFinalizeBiasDesc,
                inputs.bias,
            ),
            (
                BackendAttributeName::OperationBnFinalizePrevRunningMeanDesc,
                config.running.and_then(|running| running.previous.mean),
            ),
            (
                BackendAttributeName::OperationBnFinalizePrevRunningVarDesc,
                config.running.and_then(|running| running.previous.variance),
            ),
            (
                BackendAttributeName::OperationBnFinalizeUpdatedRunningMeanDesc,
                outputs.running.and_then(|running| running.mean),
            ),
            (
                BackendAttributeName::OperationBnFinalizeUpdatedRunningVarDesc,
                outputs.running.and_then(|running| running.variance),
            ),
            (
                BackendAttributeName::OperationBnFinalizeSavedMeanDesc,
                outputs.statistics.mean,
            ),
            (
                BackendAttributeName::OperationBnFinalizeSavedInvStdDesc,
                outputs.statistics.inv_variance,
            ),
            (
                BackendAttributeName::OperationBnFinalizeEqScaleDesc,
                outputs.equivalent_scale,
            ),
            (
                BackendAttributeName::OperationBnFinalizeEqBiasDesc,
                outputs.equivalent_bias,
            ),
            (
                BackendAttributeName::OperationBnFinalizeAccumCountDesc,
                inputs.accumulation_count,
            ),
            (
                BackendAttributeName::OperationBnFinalizeEpsilonDesc,
                config.epsilon,
            ),
            (
                BackendAttributeName::OperationBnFinalizeExpAverateFactorDesc,
                config.running.and_then(|running| running.momentum),
            ),
        ] {
            if let Some(tensor) = tensor {
                descriptor.set_attribute_descriptor(attribute, tensor.descriptor())?;
            }
        }
        descriptor.finalize()?;

        Ok(Self {
            descriptor,
            metadata: BatchNormalizationFinalizeStatsOperationMetadata {
                inputs: inputs.map(|tensor| tensor.map(Tensor::id)),
                outputs: outputs.map(|tensor| tensor.map(Tensor::id)),
                config: config.map(|tensor| tensor.map(Tensor::id)),
            },
        })
    }

    /// Retained native attributes and operand roles without descriptor queries or allocation.
    #[must_use]
    pub fn metadata(&self) -> &BatchNormalizationFinalizeStatsOperationMetadata {
        &self.metadata
    }

    /// Borrows the finalized native statistics-helper descriptor.
    #[must_use]
    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}
