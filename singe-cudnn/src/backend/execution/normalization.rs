//! Native normalization descriptors and per-channel statistics helpers.

mod backward_weights;
mod finalize;
mod genstats;

use std::{convert::Infallible, result::Result as StdResult};

use serde::{Deserialize, Serialize};

pub use crate::backend::execution::normalization::{
    backward_weights::{
        BatchNormalizationBackwardWeightsInputs, BatchNormalizationBackwardWeightsOperation,
        BatchNormalizationBackwardWeightsOperationMetadata,
        BatchNormalizationBackwardWeightsOutputs,
    },
    finalize::{
        BatchNormalizationFinalizeInputs, BatchNormalizationFinalizeOutputs,
        BatchNormalizationFinalizeStatsConfig, BatchNormalizationFinalizeStatsMode,
        BatchNormalizationFinalizeStatsOperation, BatchNormalizationFinalizeStatsOperationMetadata,
    },
    genstats::{GenStatsMode, GenStatsOperation, GenStatsOperationMetadata, GenStatsOutputs},
};
use crate::{
    backend::{
        attribute::BackendAttributeName,
        descriptor::{BackendDescriptor, BackendDescriptorType},
        normalization::{BackendNormalizationForwardPhase, BackendNormalizationMode},
        tensor::{BackendTensorUid, Tensor, TensorId},
    },
    error::Result,
};

/// Saved mean and inverse standard deviation, shared by graph and native normalization APIs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct NormalizationSavedStatistics<T = TensorId> {
    /// Saved mean tensor.
    pub mean: T,
    /// Saved inverse-standard-deviation tensor, including the forward epsilon.
    pub inv_variance: T,
}

impl<T> NormalizationSavedStatistics<T> {
    /// Borrows saved-statistic owners without changing their roles.
    pub fn as_ref(&self) -> NormalizationSavedStatistics<&T> {
        NormalizationSavedStatistics {
            mean: &self.mean,
            inv_variance: &self.inv_variance,
        }
    }

    /// Maps both statistics without changing their roles.
    pub fn map<U>(self, mut map: impl FnMut(T) -> U) -> NormalizationSavedStatistics<U> {
        NormalizationSavedStatistics {
            mean: map(self.mean),
            inv_variance: map(self.inv_variance),
        }
    }

    /// Resolves both statistics through a fallible mapping.
    ///
    /// # Errors
    ///
    /// Returns the first error from the supplied tensor mapping; later operands are not mapped.
    pub fn try_map<U, E>(
        self,
        mut map: impl FnMut(T) -> StdResult<U, E>,
    ) -> StdResult<NormalizationSavedStatistics<U>, E> {
        Ok(NormalizationSavedStatistics {
            mean: map(self.mean)?,
            inv_variance: map(self.inv_variance)?,
        })
    }
}

/// Running mean and variance, distinct from saved inverse standard deviation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchNormalizationRunningStatistics<T = TensorId> {
    /// Running mean tensor.
    pub mean: T,
    /// Running variance tensor; this is not the saved inverse standard deviation.
    pub variance: T,
}

impl<T> BatchNormalizationRunningStatistics<T> {
    /// Borrows running mean and variance without copying tensor owners.
    pub fn as_ref(&self) -> BatchNormalizationRunningStatistics<&T> {
        BatchNormalizationRunningStatistics {
            mean: &self.mean,
            variance: &self.variance,
        }
    }

    /// Maps running mean and variance tensors while preserving their statistical roles.
    pub fn map<U>(self, mut map: impl FnMut(T) -> U) -> BatchNormalizationRunningStatistics<U> {
        BatchNormalizationRunningStatistics {
            mean: map(self.mean),
            variance: map(self.variance),
        }
    }

    /// Maps running mean followed by running variance through a fallible tensor mapping.
    ///
    /// # Errors
    ///
    /// Returns the first error from the supplied tensor mapping; later operands are not mapped.
    pub fn try_map<U, E>(
        self,
        mut map: impl FnMut(T) -> StdResult<U, E>,
    ) -> StdResult<BatchNormalizationRunningStatistics<U>, E> {
        Ok(BatchNormalizationRunningStatistics {
            mean: map(self.mean)?,
            variance: map(self.variance)?,
        })
    }
}

/// Momentum and previous running statistics consumed by `BatchNorm` training or finalization.
/// Updated statistics belong to the operation's output group.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchNormalizationRunningStats<T = TensorId> {
    /// Exponential averaging-factor tensor for updating running statistics.
    pub momentum: T,
    /// Previous running mean and variance consumed by the update.
    pub previous: BatchNormalizationRunningStatistics<T>,
}

impl<T> BatchNormalizationRunningStats<T> {
    /// Borrows momentum and previous statistics as one request.
    pub fn as_ref(&self) -> BatchNormalizationRunningStats<&T> {
        BatchNormalizationRunningStats {
            momentum: &self.momentum,
            previous: self.previous.as_ref(),
        }
    }

    /// Maps momentum and previous running statistics without changing the update request.
    pub fn map<U>(self, mut map: impl FnMut(T) -> U) -> BatchNormalizationRunningStats<U> {
        BatchNormalizationRunningStats {
            momentum: map(self.momentum),
            previous: self.previous.map(map),
        }
    }

    /// Maps momentum, previous mean, and previous variance through a fallible tensor mapping.
    ///
    /// # Errors
    ///
    /// Returns the first error from the supplied tensor mapping; later operands are not mapped.
    pub fn try_map<U, E>(
        self,
        mut map: impl FnMut(T) -> StdResult<U, E>,
    ) -> StdResult<BatchNormalizationRunningStats<U>, E> {
        Ok(BatchNormalizationRunningStats {
            momentum: map(self.momentum)?,
            previous: self.previous.try_map(map)?,
        })
    }
}

/// Native normalization mode and training or inference phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NormalizationForwardConfig {
    mode: BackendNormalizationMode,
    phase: BackendNormalizationForwardPhase,
}

impl NormalizationForwardConfig {
    /// Records the native normalization mode and forward phase.
    #[must_use]
    pub fn new(mode: BackendNormalizationMode, phase: BackendNormalizationForwardPhase) -> Self {
        Self { mode, phase }
    }

    /// Returns the native normalization mode.
    #[must_use]
    pub fn mode(&self) -> BackendNormalizationMode {
        self.mode
    }

    /// Returns the native training or inference phase.
    #[must_use]
    pub fn phase(&self) -> BackendNormalizationForwardPhase {
        self.phase
    }
}

/// Native normalization mode selected for backward execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NormalizationBackwardConfig {
    mode: BackendNormalizationMode,
}

impl NormalizationBackwardConfig {
    /// Records the native normalization mode for backward execution.
    #[must_use]
    pub fn new(mode: BackendNormalizationMode) -> Self {
        Self { mode }
    }

    /// Returns the native normalization mode.
    #[must_use]
    pub fn mode(&self) -> BackendNormalizationMode {
        self.mode
    }
}

/// Tensor operands for a native normalization-forward descriptor.
/// Mean and inverse variance are inputs for batch inference and outputs for training.
/// The same roles can carry frontend identities, borrowed tensors or native UIDs.
/// Input is required; other descriptors may be omitted as in the C++ native builder.
/// cuDNN determines which combinations are supported for the configured mode and phase.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormalizationForwardTensors<T> {
    /// Input activation tensor.
    pub input: T,
    /// Optional normalized activation output.
    pub output: Option<T>,
    /// Optional computed or cached mean; the forward mode and phase determine its input or output role.
    pub mean: Option<T>,
    /// Optional computed or cached inverse standard deviation; its role follows the mode and phase.
    pub inv_variance: Option<T>,
    /// Optional affine scale tensor.
    pub scale: Option<T>,
    /// Optional affine bias tensor.
    pub bias: Option<T>,
    /// Optional scalar numerical-stability tensor.
    pub epsilon: Option<T>,
    /// Optional exponential averaging-factor tensor for running-statistics updates.
    pub averaging_factor: Option<T>,
    /// Optional previous running-mean tensor.
    pub previous_running_mean: Option<T>,
    /// Optional previous running-variance tensor.
    pub previous_running_variance: Option<T>,
    /// Optional destination for the updated running mean.
    pub next_running_mean: Option<T>,
    /// Optional destination for the updated running variance.
    pub next_running_variance: Option<T>,
    /// Peer-statistics tensors in the order supplied to cuDNN.
    pub peer_statistics: Vec<T>,
}

/// Tensor operands for a native normalization-backward descriptor.
/// Input is required; gradient and saved-state descriptors may be omitted independently.
/// cuDNN determines which combinations are supported for the configured mode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormalizationBackwardTensors<T> {
    /// Input activation tensor.
    pub input: T,
    /// Optional gradient with respect to the forward output.
    pub output_gradient: Option<T>,
    /// Optional destination for the input gradient.
    pub input_gradient: Option<T>,
    /// Optional saved forward mean tensor.
    pub mean: Option<T>,
    /// Optional saved forward inverse-standard-deviation tensor.
    pub inv_variance: Option<T>,
    /// Optional affine scale tensor.
    pub scale: Option<T>,
    /// Optional scalar numerical-stability tensor.
    pub epsilon: Option<T>,
    /// Optional destination for the affine scale gradient.
    pub scale_gradient: Option<T>,
    /// Optional destination for the affine bias gradient.
    pub bias_gradient: Option<T>,
    /// Peer-statistics tensors in the order supplied to cuDNN.
    pub peer_statistics: Vec<T>,
}

impl<T> NormalizationForwardTensors<T> {
    /// Maps supplied operands and peer statistics while preserving omitted attributes.
    pub fn map<U>(self, mut map: impl FnMut(T) -> U) -> NormalizationForwardTensors<U> {
        match self.try_map::<_, Infallible>(|tensor| Ok(map(tensor))) {
            Ok(tensors) => tensors,
            Err(never) => match never {},
        }
    }

    /// Maps all required, optional and peer operands while preserving their roles.
    ///
    /// # Errors
    ///
    /// Returns the first error from the supplied tensor mapping; later operands are not mapped.
    pub fn try_map<U, E>(
        self,
        mut map: impl FnMut(T) -> StdResult<U, E>,
    ) -> StdResult<NormalizationForwardTensors<U>, E> {
        Ok(NormalizationForwardTensors {
            input: map(self.input)?,
            output: self.output.map(&mut map).transpose()?,
            mean: self.mean.map(&mut map).transpose()?,
            inv_variance: self.inv_variance.map(&mut map).transpose()?,
            scale: self.scale.map(&mut map).transpose()?,
            bias: self.bias.map(&mut map).transpose()?,
            epsilon: self.epsilon.map(&mut map).transpose()?,
            averaging_factor: self.averaging_factor.map(&mut map).transpose()?,
            previous_running_mean: self.previous_running_mean.map(&mut map).transpose()?,
            previous_running_variance: self.previous_running_variance.map(&mut map).transpose()?,
            next_running_mean: self.next_running_mean.map(&mut map).transpose()?,
            next_running_variance: self.next_running_variance.map(&mut map).transpose()?,
            peer_statistics: self
                .peer_statistics
                .into_iter()
                .map(&mut map)
                .collect::<StdResult<_, _>>()?,
        })
    }

    /// Starts with the required input and a supplied or omitted output.
    pub fn new(input: T, output: impl Into<Option<T>>) -> Self {
        Self {
            input,
            output: output.into(),
            mean: None,
            inv_variance: None,
            scale: None,
            bias: None,
            epsilon: None,
            averaging_factor: None,
            previous_running_mean: None,
            previous_running_variance: None,
            next_running_mean: None,
            next_running_variance: None,
            peer_statistics: Vec::new(),
        }
    }
}

impl<T> NormalizationBackwardTensors<T> {
    /// Maps supplied gradients, saved state and peer statistics without changing their roles.
    pub fn map<U>(self, mut map: impl FnMut(T) -> U) -> NormalizationBackwardTensors<U> {
        match self.try_map::<_, Infallible>(|tensor| Ok(map(tensor))) {
            Ok(tensors) => tensors,
            Err(never) => match never {},
        }
    }

    /// Maps all required, optional and peer operands while preserving their roles.
    ///
    /// # Errors
    ///
    /// Returns the first error from the supplied tensor mapping; later operands are not mapped.
    pub fn try_map<U, E>(
        self,
        mut map: impl FnMut(T) -> StdResult<U, E>,
    ) -> StdResult<NormalizationBackwardTensors<U>, E> {
        Ok(NormalizationBackwardTensors {
            input: map(self.input)?,
            output_gradient: self.output_gradient.map(&mut map).transpose()?,
            input_gradient: self.input_gradient.map(&mut map).transpose()?,
            mean: self.mean.map(&mut map).transpose()?,
            inv_variance: self.inv_variance.map(&mut map).transpose()?,
            scale: self.scale.map(&mut map).transpose()?,
            epsilon: self.epsilon.map(&mut map).transpose()?,
            scale_gradient: self.scale_gradient.map(&mut map).transpose()?,
            bias_gradient: self.bias_gradient.map(&mut map).transpose()?,
            peer_statistics: self
                .peer_statistics
                .into_iter()
                .map(&mut map)
                .collect::<StdResult<_, _>>()?,
        })
    }

    /// Starts with the required input and independently supplied or omitted gradient and saved state.
    pub fn new(
        input: T,
        output_gradient: impl Into<Option<T>>,
        input_gradient: impl Into<Option<T>>,
        inv_variance: impl Into<Option<T>>,
    ) -> Self {
        Self {
            input,
            output_gradient: output_gradient.into(),
            input_gradient: input_gradient.into(),
            inv_variance: inv_variance.into(),
            mean: None,
            scale: None,
            epsilon: None,
            scale_gradient: None,
            bias_gradient: None,
            peer_statistics: Vec::new(),
        }
    }
}

/// Retained forward mode, phase and native tensor identities.
/// Statistics retain their phase-specific input or output roles, including omitted attributes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormalizationForwardOperationMetadata {
    /// Retained native normalization mode and forward phase.
    pub config: NormalizationForwardConfig,
    /// Native identities of supplied operands and ordered peer statistics, preserving omitted attributes.
    pub tensors: NormalizationForwardTensors<BackendTensorUid>,
}

/// Finalized native normalization-forward operation with retained mode, phase, and tensor identities.
#[derive(Debug)]
pub struct NormalizationForwardOperation {
    descriptor: BackendDescriptor,
    metadata: NormalizationForwardOperationMetadata,
}

impl NormalizationForwardOperation {
    /// Creates a native forward operation with the supplied operand attributes.
    /// Omitted descriptors remain unset, matching the C++ native builder.
    ///
    /// # Errors
    /// Returns an error if cuDNN rejects descriptor creation, attributes or finalization.
    pub fn create(
        norm: &NormalizationForwardConfig,
        tensors: NormalizationForwardTensors<&Tensor>,
    ) -> Result<Self> {
        let mut descriptor =
            BackendDescriptor::create(BackendDescriptorType::OperationNormForward)?;
        descriptor.set_attribute_enum(BackendAttributeName::OperationNormFwdMode, norm.mode())?;
        descriptor.set_attribute_enum(BackendAttributeName::OperationNormFwdPhase, norm.phase())?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationNormFwdXDesc,
            tensors.input.descriptor(),
        )?;
        for (attribute, tensor) in [
            (BackendAttributeName::OperationNormFwdMeanDesc, tensors.mean),
            (
                BackendAttributeName::OperationNormFwdInvVarianceDesc,
                tensors.inv_variance,
            ),
            (
                BackendAttributeName::OperationNormFwdScaleDesc,
                tensors.scale,
            ),
            (BackendAttributeName::OperationNormFwdBiasDesc, tensors.bias),
            (
                BackendAttributeName::OperationNormFwdEpsilonDesc,
                tensors.epsilon,
            ),
            (
                BackendAttributeName::OperationNormFwdExpAvgFactorDesc,
                tensors.averaging_factor,
            ),
            (
                BackendAttributeName::OperationNormFwdInputRunningMeanDesc,
                tensors.previous_running_mean,
            ),
            (
                BackendAttributeName::OperationNormFwdInputRunningVarDesc,
                tensors.previous_running_variance,
            ),
            (
                BackendAttributeName::OperationNormFwdOutputRunningMeanDesc,
                tensors.next_running_mean,
            ),
            (
                BackendAttributeName::OperationNormFwdOutputRunningVarDesc,
                tensors.next_running_variance,
            ),
        ] {
            if let Some(tensor) = tensor {
                descriptor.set_attribute_descriptor(attribute, tensor.descriptor())?;
            }
        }
        if !tensors.peer_statistics.is_empty() {
            let peer_stats = tensors
                .peer_statistics
                .iter()
                .map(|tensor| tensor.descriptor())
                .collect::<Vec<_>>();
            descriptor.set_attribute_descriptor_slice(
                BackendAttributeName::OperationNormFwdPeerStatDescs,
                &peer_stats,
            )?;
        }
        if let Some(output) = tensors.output {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationNormFwdYDesc,
                output.descriptor(),
            )?;
        }
        descriptor.finalize()?;

        Ok(Self {
            descriptor,
            metadata: NormalizationForwardOperationMetadata {
                config: *norm,
                tensors: tensors.map(Tensor::id),
            },
        })
    }

    /// Retained mode, phase and native operand UIDs, including ordered peer statistics.
    /// Does not query cuDNN or allocate, and does not borrow the original tensor wrappers.
    #[must_use]
    pub fn metadata(&self) -> &NormalizationForwardOperationMetadata {
        &self.metadata
    }

    /// Borrows the finalized native normalization descriptor.
    #[must_use]
    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}

/// Retained backward mode, native gradient identities and optional saved state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormalizationBackwardOperationMetadata {
    /// Retained native normalization mode.
    pub config: NormalizationBackwardConfig,
    /// Native identities of supplied operands and ordered peer statistics, preserving omitted attributes.
    pub tensors: NormalizationBackwardTensors<BackendTensorUid>,
}

/// Finalized native normalization-backward operation with retained mode and tensor identities.
#[derive(Debug)]
pub struct NormalizationBackwardOperation {
    descriptor: BackendDescriptor,
    metadata: NormalizationBackwardOperationMetadata,
}

impl NormalizationBackwardOperation {
    /// Creates a native backward operation with independently optional gradients and saved state.
    /// Omitted descriptors remain unset; cuDNN decides which combinations are supported.
    ///
    /// # Errors
    /// Returns an error if cuDNN rejects descriptor creation, attributes or finalization.
    pub fn create(
        norm: &NormalizationBackwardConfig,
        tensors: NormalizationBackwardTensors<&Tensor>,
    ) -> Result<Self> {
        let mut descriptor =
            BackendDescriptor::create(BackendDescriptorType::OperationNormBackward)?;
        descriptor.set_attribute_enum(BackendAttributeName::OperationNormBwdMode, norm.mode())?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationNormBwdXDesc,
            tensors.input.descriptor(),
        )?;
        for (attribute, tensor) in [
            (BackendAttributeName::OperationNormBwdMeanDesc, tensors.mean),
            (
                BackendAttributeName::OperationNormBwdInvVarianceDesc,
                tensors.inv_variance,
            ),
            (
                BackendAttributeName::OperationNormBwdDyDesc,
                tensors.output_gradient,
            ),
            (
                BackendAttributeName::OperationNormBwdScaleDesc,
                tensors.scale,
            ),
            (
                BackendAttributeName::OperationNormBwdEpsilonDesc,
                tensors.epsilon,
            ),
            (
                BackendAttributeName::OperationNormBwdDScaleDesc,
                tensors.scale_gradient,
            ),
            (
                BackendAttributeName::OperationNormBwdDBiasDesc,
                tensors.bias_gradient,
            ),
        ] {
            if let Some(tensor) = tensor {
                descriptor.set_attribute_descriptor(attribute, tensor.descriptor())?;
            }
        }
        if !tensors.peer_statistics.is_empty() {
            let peer_stats = tensors
                .peer_statistics
                .iter()
                .map(|tensor| tensor.descriptor())
                .collect::<Vec<_>>();
            descriptor.set_attribute_descriptor_slice(
                BackendAttributeName::OperationNormBwdPeerStatDescs,
                &peer_stats,
            )?;
        }
        if let Some(input_gradient) = tensors.input_gradient {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationNormBwdDxDesc,
                input_gradient.descriptor(),
            )?;
        }
        descriptor.finalize()?;

        Ok(Self {
            descriptor,
            metadata: NormalizationBackwardOperationMetadata {
                config: *norm,
                tensors: tensors.map(Tensor::id),
            },
        })
    }

    /// Retained mode, native operand UIDs and omitted attributes, including peer order.
    /// Does not query cuDNN or allocate, and does not borrow the original tensor wrappers.
    #[must_use]
    pub fn metadata(&self) -> &NormalizationBackwardOperationMetadata {
        &self.metadata
    }

    /// Borrows the finalized native normalization descriptor.
    #[must_use]
    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}
