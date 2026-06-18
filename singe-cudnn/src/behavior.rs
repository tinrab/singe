use num_enum::{IntoPrimitive, TryFromPrimitive};
use serde::{Deserialize, Serialize};
use singe_cudnn_sys as sys;

use singe_core::{impl_enum_conversion, impl_enum_display};

/// Queryable behavior note reported by a finalized
/// [`BackendDescriptorType::Engine`](crate::descriptor::BackendDescriptorType::Engine)
/// through
/// [`BackendDescriptor::attribute_enum_slice`](crate::descriptor::BackendDescriptor::attribute_enum_slice).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive, Serialize, Deserialize,
)]
#[repr(u32)]
#[non_exhaustive]
pub enum BackendBehaviorNote {
    RuntimeCompilation =
        sys::cudnnBackendBehaviorNote_t::CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION as _,
    RequiresFilterInt8x32Reorder =
        sys::cudnnBackendBehaviorNote_t::CUDNN_BEHAVIOR_NOTE_REQUIRES_FILTER_INT8x32_REORDER as _,
    RequiresBiasInt8x32Reorder =
        sys::cudnnBackendBehaviorNote_t::CUDNN_BEHAVIOR_NOTE_REQUIRES_BIAS_INT8x32_REORDER as _,
    SupportsCudaGraphNativeApi =
        sys::cudnnBackendBehaviorNote_t::CUDNN_BEHAVIOR_NOTE_SUPPORTS_CUDA_GRAPH_NATIVE_API as _,
    CublasLtDependency =
        sys::cudnnBackendBehaviorNote_t::CUDNN_BEHAVIOR_NOTE_CUBLASLT_DEPENDENCY as _,
}

impl_enum_conversion!(sys::cudnnBackendBehaviorNote_t, BackendBehaviorNote);

impl_enum_display!(BackendBehaviorNote, {
    Self::RuntimeCompilation => "CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION",
    Self::RequiresFilterInt8x32Reorder => "CUDNN_BEHAVIOR_NOTE_REQUIRES_FILTER_INT8x32_REORDER",
    Self::RequiresBiasInt8x32Reorder => "CUDNN_BEHAVIOR_NOTE_REQUIRES_BIAS_INT8x32_REORDER",
    Self::SupportsCudaGraphNativeApi => "CUDNN_BEHAVIOR_NOTE_SUPPORTS_CUDA_GRAPH_NATIVE_API",
    Self::CublasLtDependency => "CUDNN_BEHAVIOR_NOTE_CUBLASLT_DEPENDENCY",
});

/// Queryable numerical property reported by a finalized
/// [`BackendDescriptorType::Engine`](crate::descriptor::BackendDescriptorType::Engine)
/// through
/// [`BackendDescriptor::attribute_enum_slice`](crate::descriptor::BackendDescriptor::attribute_enum_slice).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive, Serialize, Deserialize,
)]
#[repr(u32)]
#[non_exhaustive]
pub enum BackendNumericalNote {
    TensorCore = sys::cudnnBackendNumericalNote_t::CUDNN_NUMERICAL_NOTE_TENSOR_CORE as _,
    DownConvertInputs =
        sys::cudnnBackendNumericalNote_t::CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS as _,
    ReducedPrecisionReduction =
        sys::cudnnBackendNumericalNote_t::CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION as _,
    Fft = sys::cudnnBackendNumericalNote_t::CUDNN_NUMERICAL_NOTE_FFT as _,
    Nondeterministic = sys::cudnnBackendNumericalNote_t::CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC as _,
    Winograd = sys::cudnnBackendNumericalNote_t::CUDNN_NUMERICAL_NOTE_WINOGRAD as _,
    WinogradTile4x4 = sys::cudnnBackendNumericalNote_t::CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_4x4 as _,
    WinogradTile6x6 = sys::cudnnBackendNumericalNote_t::CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_6x6 as _,
    WinogradTile13x13 =
        sys::cudnnBackendNumericalNote_t::CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_13x13 as _,
    StrictNanProp = sys::cudnnBackendNumericalNote_t::CUDNN_NUMERICAL_NOTE_STRICT_NAN_PROP as _,
}

impl_enum_conversion!(sys::cudnnBackendNumericalNote_t, BackendNumericalNote);

impl_enum_display!(BackendNumericalNote, {
    Self::TensorCore => "CUDNN_NUMERICAL_NOTE_TENSOR_CORE",
    Self::DownConvertInputs => "CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS",
    Self::ReducedPrecisionReduction => "CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION",
    Self::Fft => "CUDNN_NUMERICAL_NOTE_FFT",
    Self::Nondeterministic => "CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC",
    Self::Winograd => "CUDNN_NUMERICAL_NOTE_WINOGRAD",
    Self::WinogradTile4x4 => "CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_4x4",
    Self::WinogradTile6x6 => "CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_6x6",
    Self::WinogradTile13x13 => "CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_13x13",
    Self::StrictNanProp => "CUDNN_NUMERICAL_NOTE_STRICT_NAN_PROP",
});
