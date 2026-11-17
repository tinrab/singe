use num_enum::{IntoPrimitive, TryFromPrimitive};

use serde::{Deserialize, Serialize};

use singe_cudnn_sys as sys;

use singe_core::impl_enum_display;

singe_core::native_enum! {
    #[derive(
        Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive, Serialize, Deserialize,
    )]
    /// Normalization mode for backend normalization forward and backward operations.
    ///
    /// [`BackendNormalizationMode::Group`] is not yet supported.
    /// Using it causes cuDNN to return [`Status::InternalError`](crate::error::Status::InternalError).
    #[repr(u32)]
    #[non_exhaustive]
    pub enum BackendNormalizationMode: sys::cudnnBackendNormMode_t {
        /// Layer normalization: normalizes over feature dims per sample.
        Layer = CUDNN_LAYER_NORM,
        /// Instance normalization: normalizes per instance per channel.
        Instance = CUDNN_INSTANCE_NORM,
        /// Batch normalization: normalizes across the batch dimension.
        Batch = CUDNN_BATCH_NORM,
        /// Group normalization, which cuDNN rejects with [`Status::InternalError`](crate::error::Status::InternalError).
        Group = CUDNN_GROUP_NORM,
        /// Root mean square normalization: normalizes by RMS of activations.
        Rms = CUDNN_RMS_NORM,
        /// Adaptive layer normalization with learned affine parameters.
        AdaLayerNorm = CUDNN_ADA_LAYER_NORM,
    }
}

impl_enum_display!(BackendNormalizationMode, {
    Self::Layer => "CUDNN_LAYER_NORM",
    Self::Instance => "CUDNN_INSTANCE_NORM",
    Self::Batch => "CUDNN_BATCH_NORM",
    Self::Group => "CUDNN_GROUP_NORM",
    Self::Rms => "CUDNN_RMS_NORM",
    Self::AdaLayerNorm => "CUDNN_ADA_LAYER_NORM",
});

singe_core::native_enum! {
    #[derive(
        Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive, Serialize, Deserialize,
    )]
    /// Phase for backend normalization forward operations.
    #[repr(u32)]
    #[non_exhaustive]
    pub enum BackendNormalizationForwardPhase: sys::cudnnBackendNormFwdPhase_t {
        /// Inference phase: uses pre-computed running statistics.
        Inference = CUDNN_NORM_FWD_INFERENCE,
        /// Training phase: computes batch statistics and updates running stats.
        Training = CUDNN_NORM_FWD_TRAINING,
    }
}

impl_enum_display!(BackendNormalizationForwardPhase, {
    Self::Inference => "CUDNN_NORM_FWD_INFERENCE",
    Self::Training => "CUDNN_NORM_FWD_TRAINING",
});
