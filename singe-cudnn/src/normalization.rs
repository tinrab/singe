use num_enum::{IntoPrimitive, TryFromPrimitive};
use serde::{Deserialize, Serialize};
use singe_cudnn_sys as sys;

use singe_core::{impl_enum_conversion, impl_enum_display};

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive, Serialize, Deserialize,
)]
/// Normalization mode for backend normalization forward and backward operations.
///
/// [`BackendNormalizationMode::Group`] is not yet supported.
/// Using it causes cuDNN to return [`crate::error::Status::InternalError`].
#[repr(u32)]
#[non_exhaustive]
pub enum BackendNormalizationMode {
    Layer = sys::cudnnBackendNormMode_t::CUDNN_LAYER_NORM as _,
    Instance = sys::cudnnBackendNormMode_t::CUDNN_INSTANCE_NORM as _,
    Batch = sys::cudnnBackendNormMode_t::CUDNN_BATCH_NORM as _,
    Group = sys::cudnnBackendNormMode_t::CUDNN_GROUP_NORM as _,
    Rms = sys::cudnnBackendNormMode_t::CUDNN_RMS_NORM as _,
    AdaLayerNorm = sys::cudnnBackendNormMode_t::CUDNN_ADA_LAYER_NORM as _,
}

impl_enum_conversion!(sys::cudnnBackendNormMode_t, BackendNormalizationMode);

impl_enum_display!(BackendNormalizationMode, {
    Self::Layer => "CUDNN_LAYER_NORM",
    Self::Instance => "CUDNN_INSTANCE_NORM",
    Self::Batch => "CUDNN_BATCH_NORM",
    Self::Group => "CUDNN_GROUP_NORM",
    Self::Rms => "CUDNN_RMS_NORM",
    Self::AdaLayerNorm => "CUDNN_ADA_LAYER_NORM",
});

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive, Serialize, Deserialize,
)]
/// Phase for backend normalization forward operations.
#[repr(u32)]
#[non_exhaustive]
pub enum BackendNormalizationForwardPhase {
    Inference = sys::cudnnBackendNormFwdPhase_t::CUDNN_NORM_FWD_INFERENCE as _,
    Training = sys::cudnnBackendNormFwdPhase_t::CUDNN_NORM_FWD_TRAINING as _,
}

impl_enum_conversion!(
    sys::cudnnBackendNormFwdPhase_t,
    BackendNormalizationForwardPhase
);

impl_enum_display!(BackendNormalizationForwardPhase, {
    Self::Inference => "CUDNN_NORM_FWD_INFERENCE",
    Self::Training => "CUDNN_NORM_FWD_TRAINING",
});
