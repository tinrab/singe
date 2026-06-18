use num_enum::{IntoPrimitive, TryFromPrimitive};
use serde::{Deserialize, Serialize};
use singe_core::{impl_enum_conversion, impl_enum_display};
use singe_cudnn_sys as sys;

use crate::{
    attribute::{BackendAttributeName, BackendAttributeType},
    data_type::DataType,
    descriptor::{BackendDescriptor, BackendDescriptorType},
    error::Result,
    normalization::{BackendNormalizationForwardPhase, BackendNormalizationMode},
    tensor::Tensor,
};

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive, Serialize, Deserialize,
)]
/// Statistics mode for backend statistics generation operations.
#[repr(u32)]
#[non_exhaustive]
pub enum GenStatsMode {
    /// Computes and writes the sum and sum of squares of the input tensor along the specified dimensions.
    /// Currently supported reduction dimensions are limited to per-channel reductions.
    SumSqsum = sys::cudnnGenStatsMode_t::CUDNN_GENSTATS_SUM_SQSUM as _,
}

impl_enum_conversion!(sys::cudnnGenStatsMode_t, GenStatsMode);

impl_enum_display!(GenStatsMode, {
    Self::SumSqsum => "CUDNN_GENSTATS_SUM_SQSUM",
});

#[derive(Debug)]
pub struct GenStatsOperation {
    descriptor: BackendDescriptor,
}

impl GenStatsOperation {
    pub fn create(mode: GenStatsMode, x: &Tensor, sum: &Tensor, sqsum: &Tensor) -> Result<Self> {
        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::OperationGenStats)?;
        descriptor.set_attribute_enum(
            BackendAttributeName::OperationGenStatsMode,
            BackendAttributeType::GenStatsMode,
            mode,
        )?;
        descriptor.set_attribute_enum(
            BackendAttributeName::OperationGenStatsMathPrec,
            BackendAttributeType::DataType,
            x.data_type(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationGenStatsXDesc,
            x.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationGenStatsSumDesc,
            sum.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationGenStatsSqSumDesc,
            sqsum.descriptor(),
        )?;
        descriptor.finalize()?;

        Ok(Self { descriptor })
    }

    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive, Serialize, Deserialize,
)]
/// Mathematical mode that converts batch normalization statistics and trained scale/bias
/// into equivalent scale/bias for the next normalization stage.
#[repr(u32)]
#[non_exhaustive]
pub enum BatchNormalizationFinalizeStatsMode {
    Training = sys::cudnnBnFinalizeStatsMode_t::CUDNN_BN_FINALIZE_STATISTICS_TRAINING as _,
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

#[derive(Debug)]
pub struct BatchNormalizationFinalizeStatsOperation {
    descriptor: BackendDescriptor,
}

impl BatchNormalizationFinalizeStatsOperation {
    pub fn create(
        mode: BatchNormalizationFinalizeStatsMode,
        compute_type: DataType,
        y_sum: &Tensor,
        y_sq_sum: &Tensor,
        scale: &Tensor,
        bias: &Tensor,
        prev_running_mean: Option<&Tensor>,
        prev_running_var: Option<&Tensor>,
        updated_running_mean: Option<&Tensor>,
        updated_running_var: Option<&Tensor>,
        saved_mean: &Tensor,
        saved_inv_std: &Tensor,
        eq_scale: &Tensor,
        eq_bias: &Tensor,
        accum_count: &Tensor,
        epsilon: &Tensor,
        exp_average_factor: Option<&Tensor>,
    ) -> Result<Self> {
        let mut descriptor =
            BackendDescriptor::create(BackendDescriptorType::OperationBnFinalizeStatistics)?;
        descriptor.set_attribute_enum(
            BackendAttributeName::OperationBnFinalizeStatsMode,
            BackendAttributeType::BnFinalizeStatsMode,
            mode,
        )?;
        descriptor.set_attribute_enum(
            BackendAttributeName::OperationBnFinalizeMathPrec,
            BackendAttributeType::DataType,
            compute_type,
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationBnFinalizeYSumDesc,
            y_sum.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationBnFinalizeYSqSumDesc,
            y_sq_sum.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationBnFinalizeScaleDesc,
            scale.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationBnFinalizeBiasDesc,
            bias.descriptor(),
        )?;
        if let Some(prev_running_mean) = prev_running_mean {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationBnFinalizePrevRunningMeanDesc,
                prev_running_mean.descriptor(),
            )?;
        }
        if let Some(prev_running_var) = prev_running_var {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationBnFinalizePrevRunningVarDesc,
                prev_running_var.descriptor(),
            )?;
        }
        if let Some(updated_running_mean) = updated_running_mean {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationBnFinalizeUpdatedRunningMeanDesc,
                updated_running_mean.descriptor(),
            )?;
        }
        if let Some(updated_running_var) = updated_running_var {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationBnFinalizeUpdatedRunningVarDesc,
                updated_running_var.descriptor(),
            )?;
        }
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationBnFinalizeSavedMeanDesc,
            saved_mean.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationBnFinalizeSavedInvStdDesc,
            saved_inv_std.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationBnFinalizeEqScaleDesc,
            eq_scale.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationBnFinalizeEqBiasDesc,
            eq_bias.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationBnFinalizeAccumCountDesc,
            accum_count.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationBnFinalizeEpsilonDesc,
            epsilon.descriptor(),
        )?;
        if let Some(exp_average_factor) = exp_average_factor {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationBnFinalizeExpAverateFactorDesc,
                exp_average_factor.descriptor(),
            )?;
        }
        descriptor.finalize()?;

        Ok(Self { descriptor })
    }

    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}

#[derive(Debug)]
pub struct BatchNormalizationBackwardWeightsOperation {
    descriptor: BackendDescriptor,
}

impl BatchNormalizationBackwardWeightsOperation {
    pub fn create(
        compute_type: DataType,
        mean: &Tensor,
        inv_std: &Tensor,
        bn_scale: &Tensor,
        x: &Tensor,
        dy: &Tensor,
        dbn_scale: &Tensor,
        dbn_bias: &Tensor,
        eq_dy_scale: &Tensor,
        eq_x_scale: &Tensor,
        eq_bias: &Tensor,
    ) -> Result<Self> {
        let mut descriptor =
            BackendDescriptor::create(BackendDescriptorType::OperationBnBwdWeights)?;
        descriptor.set_attribute_enum(
            BackendAttributeName::OperationBnBwdWeightsMathPrec,
            BackendAttributeType::DataType,
            compute_type,
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationBnBwdWeightsMeanDesc,
            mean.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationBnBwdWeightsInvStdDesc,
            inv_std.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationBnBwdWeightsBnScaleDesc,
            bn_scale.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationBnBwdWeightsXDesc,
            x.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationBnBwdWeightsDyDesc,
            dy.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationBnBwdWeightsDbnScaleDesc,
            dbn_scale.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationBnBwdWeightsDbnBiasDesc,
            dbn_bias.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationBnBwdWeightsEqDyScaleDesc,
            eq_dy_scale.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationBnBwdWeightsEqXScaleDesc,
            eq_x_scale.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationBnBwdWeightsEqBias,
            eq_bias.descriptor(),
        )?;
        descriptor.finalize()?;

        Ok(Self { descriptor })
    }

    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}

#[derive(Debug)]
pub struct NormalizationForwardConfig {
    mode: BackendNormalizationMode,
    phase: BackendNormalizationForwardPhase,
}

impl NormalizationForwardConfig {
    pub fn new(mode: BackendNormalizationMode, phase: BackendNormalizationForwardPhase) -> Self {
        Self { mode, phase }
    }

    pub fn mode(&self) -> BackendNormalizationMode {
        self.mode
    }

    pub fn phase(&self) -> BackendNormalizationForwardPhase {
        self.phase
    }
}

#[derive(Debug, Clone, Copy)]
pub struct NormalizationBackwardConfig {
    mode: BackendNormalizationMode,
}

impl NormalizationBackwardConfig {
    pub fn new(mode: BackendNormalizationMode) -> Self {
        Self { mode }
    }

    pub fn mode(&self) -> BackendNormalizationMode {
        self.mode
    }
}

#[derive(Debug)]
pub struct NormalizationForwardOperation {
    descriptor: BackendDescriptor,
}

impl NormalizationForwardOperation {
    pub fn create(
        norm: &NormalizationForwardConfig,
        x: &Tensor,
        mean: Option<&Tensor>,
        inv_variance: Option<&Tensor>,
        scale: &Tensor,
        bias: Option<&Tensor>,
        epsilon: Option<&Tensor>,
        exp_avg_factor: Option<&Tensor>,
        input_running_mean: Option<&Tensor>,
        input_running_var: Option<&Tensor>,
        output_running_mean: Option<&Tensor>,
        output_running_var: Option<&Tensor>,
        peer_stats: &[&Tensor],
        y: &Tensor,
    ) -> Result<Self> {
        let mut descriptor =
            BackendDescriptor::create(BackendDescriptorType::OperationNormForward)?;
        descriptor.set_attribute_enum(
            BackendAttributeName::OperationNormFwdMode,
            BackendAttributeType::NormMode,
            norm.mode(),
        )?;
        descriptor.set_attribute_enum(
            BackendAttributeName::OperationNormFwdPhase,
            BackendAttributeType::NormFwdPhase,
            norm.phase(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationNormFwdXDesc,
            x.descriptor(),
        )?;
        if let Some(mean) = mean {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationNormFwdMeanDesc,
                mean.descriptor(),
            )?;
        }
        if let Some(inv_variance) = inv_variance {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationNormFwdInvVarianceDesc,
                inv_variance.descriptor(),
            )?;
        }
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationNormFwdScaleDesc,
            scale.descriptor(),
        )?;
        if let Some(bias) = bias {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationNormFwdBiasDesc,
                bias.descriptor(),
            )?;
        }
        if let Some(epsilon) = epsilon {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationNormFwdEpsilonDesc,
                epsilon.descriptor(),
            )?;
        }
        if let Some(exp_avg_factor) = exp_avg_factor {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationNormFwdExpAvgFactorDesc,
                exp_avg_factor.descriptor(),
            )?;
        }
        if let Some(input_running_mean) = input_running_mean {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationNormFwdInputRunningMeanDesc,
                input_running_mean.descriptor(),
            )?;
        }
        if let Some(input_running_var) = input_running_var {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationNormFwdInputRunningVarDesc,
                input_running_var.descriptor(),
            )?;
        }
        if let Some(output_running_mean) = output_running_mean {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationNormFwdOutputRunningMeanDesc,
                output_running_mean.descriptor(),
            )?;
        }
        if let Some(output_running_var) = output_running_var {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationNormFwdOutputRunningVarDesc,
                output_running_var.descriptor(),
            )?;
        }
        if !peer_stats.is_empty() {
            let peer_stats = peer_stats
                .iter()
                .map(|tensor| tensor.descriptor())
                .collect::<Vec<_>>();
            descriptor.set_attribute_descriptor_slice(
                BackendAttributeName::OperationNormFwdPeerStatDescs,
                &peer_stats,
            )?;
        }
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationNormFwdYDesc,
            y.descriptor(),
        )?;
        descriptor.finalize()?;

        Ok(Self { descriptor })
    }

    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}

#[derive(Debug)]
pub struct NormalizationBackwardOperation {
    descriptor: BackendDescriptor,
}

impl NormalizationBackwardOperation {
    pub fn create(
        norm: &NormalizationBackwardConfig,
        x: &Tensor,
        mean: Option<&Tensor>,
        inv_variance: &Tensor,
        dy: &Tensor,
        scale: &Tensor,
        epsilon: Option<&Tensor>,
        dscale: &Tensor,
        bias_gradient: Option<&Tensor>,
        peer_stats: &[&Tensor],
        dx: &Tensor,
    ) -> Result<Self> {
        let mut descriptor =
            BackendDescriptor::create(BackendDescriptorType::OperationNormBackward)?;
        descriptor.set_attribute_enum(
            BackendAttributeName::OperationNormBwdMode,
            BackendAttributeType::NormMode,
            norm.mode(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationNormBwdXDesc,
            x.descriptor(),
        )?;
        if let Some(mean) = mean {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationNormBwdMeanDesc,
                mean.descriptor(),
            )?;
        }
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationNormBwdInvVarianceDesc,
            inv_variance.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationNormBwdDyDesc,
            dy.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationNormBwdScaleDesc,
            scale.descriptor(),
        )?;
        if let Some(epsilon) = epsilon {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationNormBwdEpsilonDesc,
                epsilon.descriptor(),
            )?;
        }
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationNormBwdDScaleDesc,
            dscale.descriptor(),
        )?;
        if let Some(bias_gradient) = bias_gradient {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationNormBwdDBiasDesc,
                bias_gradient.descriptor(),
            )?;
        }
        if !peer_stats.is_empty() {
            let peer_stats = peer_stats
                .iter()
                .map(|tensor| tensor.descriptor())
                .collect::<Vec<_>>();
            descriptor.set_attribute_descriptor_slice(
                BackendAttributeName::OperationNormBwdPeerStatDescs,
                &peer_stats,
            )?;
        }
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationNormBwdDxDesc,
            dx.descriptor(),
        )?;
        descriptor.finalize()?;

        Ok(Self { descriptor })
    }

    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}
