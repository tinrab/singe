use singe_cudnn_sys as sys;

use crate::metadata::backend_metadata_enum;

backend_metadata_enum! {
    /// Queryable behavior note reported by a finalized
    /// [`BackendDescriptorType::Engine`](crate::backend::descriptor::BackendDescriptorType::Engine)
    /// through
    /// [`BackendDescriptor::attribute_enum_slice`](crate::backend::descriptor::BackendDescriptor::attribute_enum_slice).
    pub enum BackendBehaviorNote: sys::cudnnBackendBehaviorNote_t {
        RuntimeCompilation => CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION => "Engine requires runtime compilation (NVRTC).",
        RequiresFilterInt8x32Reorder => CUDNN_BEHAVIOR_NOTE_REQUIRES_FILTER_INT8x32_REORDER => "Engine requires INT8x32-reordered filter tensors.",
        RequiresBiasInt8x32Reorder => CUDNN_BEHAVIOR_NOTE_REQUIRES_BIAS_INT8x32_REORDER => "Engine requires INT8x32-reordered bias tensors.",
        SupportsCudaGraphNativeApi => CUDNN_BEHAVIOR_NOTE_SUPPORTS_CUDA_GRAPH_NATIVE_API => "Engine supports native CUDA graph capture.",
        CublasLtDependency => CUDNN_BEHAVIOR_NOTE_CUBLASLT_DEPENDENCY => "Engine depends on the `cuBLASLt` library.",
    }
}

backend_metadata_enum! {
    /// Queryable numerical property reported by a finalized
    /// [`BackendDescriptorType::Engine`](crate::backend::descriptor::BackendDescriptorType::Engine)
    /// through
    /// [`BackendDescriptor::attribute_enum_slice`](crate::backend::descriptor::BackendDescriptor::attribute_enum_slice).
    pub enum BackendNumericalNote: sys::cudnnBackendNumericalNote_t {
        TensorCore => CUDNN_NUMERICAL_NOTE_TENSOR_CORE => "Engine uses Tensor Core hardware acceleration.",
        DownConvertInputs => CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS => "Engine down-converts inputs to lower precision for compute.",
        ReducedPrecisionReduction => CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION => "Engine uses reduced-precision accumulation in reductions.",
        Fft => CUDNN_NUMERICAL_NOTE_FFT => "Engine uses FFT-based computation.",
        Nondeterministic => CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC => "Engine may produce nondeterministic results across runs.",
        Winograd => CUDNN_NUMERICAL_NOTE_WINOGRAD => "Engine uses Winograd transform.",
        WinogradTile4x4 => CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_4x4 => "Engine uses Winograd with 4x4 output tiles.",
        WinogradTile6x6 => CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_6x6 => "Engine uses Winograd with 6x6 output tiles.",
        WinogradTile13x13 => CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_13x13 => "Engine uses Winograd with 13x13 output tiles.",
        StrictNanProp => CUDNN_NUMERICAL_NOTE_STRICT_NAN_PROP => "Engine strictly propagates NaN values.",
    }
}
