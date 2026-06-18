use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{impl_enum_conversion, impl_enum_display};
use singe_cudnn_sys as sys;

/// Operation mode for a [`BackendDescriptorType::EngineHeur`](crate::descriptor::BackendDescriptorType::EngineHeur) descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum BackendHeuristicMode {
    Instant = sys::cudnnBackendHeurMode_t::CUDNN_HEUR_MODE_INSTANT as _,
    /// Can use the neural-net-based heuristics to improve generalization performance compared to [`BackendHeuristicMode::Instant`].
    ///
    /// When the neural net is used, CPU inference time increases by 10-100x
    /// compared to [`BackendHeuristicMode::Instant`].
    /// These neural net heuristics are not supported for any of the following cases:
    ///
    /// * 3-D convolutions.
    /// * Grouped convolutions with more than one group.
    /// * Dilated convolutions (any dilation for any spatial dimension larger than `1`).
    ///
    /// Further, the neural net is only enabled on x86 platforms when cuDNN is run on an A100 GPU.
    /// When the neural net is not supported, [`BackendHeuristicMode::B`] falls
    /// back to [`BackendHeuristicMode::Instant`].
    /// It also falls back to [`BackendHeuristicMode::Instant`] when its
    /// projected overhead would reduce overall network performance.
    B = sys::cudnnBackendHeurMode_t::CUDNN_HEUR_MODE_B as _,
    /// This heuristic mode is intended to be used for finding fallback options which provide functional support (without any expectation of providing optimal GPU performance).
    Fallback = sys::cudnnBackendHeurMode_t::CUDNN_HEUR_MODE_FALLBACK as _,
    A = sys::cudnnBackendHeurMode_t::CUDNN_HEUR_MODE_A as _,
}

impl_enum_conversion!(sys::cudnnBackendHeurMode_t, BackendHeuristicMode);

impl_enum_display!(BackendHeuristicMode, {
    Self::Instant => "CUDNN_HEUR_MODE_INSTANT",
    Self::B => "CUDNN_HEUR_MODE_B",
    Self::Fallback => "CUDNN_HEUR_MODE_FALLBACK",
    Self::A => "CUDNN_HEUR_MODE_A",
});
