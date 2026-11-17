//! Native status codes and checked cuDNN wrapper errors.

use std::{
    borrow::Cow, collections::TryReserveError, fmt, hash::Hash, io::Error as IoError,
    num::ParseIntError, path::PathBuf, result::Result as StdResult,
};

use serde_json::Error as JsonError;
use singe_core::LibraryVersion;
use singe_cuda::error::Error as CudaError;
use singe_cudnn_sys as sys;
use thiserror::Error;

use crate::{
    backend::{
        attribute::BackendAttributeName, descriptor::BackendDescriptorType, knob::BackendKnobType,
        tensor::BackendTensorUid,
    },
    data_type::DataType,
    diagnostic::last_error_string,
    frontend::error::Error as FrontendError,
};

singe_core::wrapper_status! {
    library = "cudnn",
    error = Error::Cudnn,
    raw = sys::cudnnStatus_t,
    success = CUDNN_STATUS_SUCCESS,
    absence = sys::take_absence,
    diagnostic = last_error_string,
    unknown = "UNKNOWN_CUDNN_STATUS",
    description = sys::cudnnGetErrorString,
    /// Status returned by cuDNN operations.
    pub enum Status: u32 {
        /// The operation was completed successfully.
        Success = CUDNN_STATUS_SUCCESS,
        /// The cuDNN library was not initialized properly.
        /// Handle creation failed, usually through a CUDA runtime or hardware setup error, or a call ran without a handle.
        NotInitialized = CUDNN_STATUS_NOT_INITIALIZED,
        /// Some cuDNN sub libraries have different versions, indicative of an installation issue.
        SublibraryVersionMismatch = CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH,
        /// The schema used for serialization is not what the current cuDNN library expects, thus the serialized artifact is no longer valid and needs to be re-serialized.
        SerializationVersionMismatch = CUDNN_STATUS_SERIALIZATION_VERSION_MISMATCH,
        /// This error code may be reported in the warning logging level as a reminder that some functionality is under deprecation and will be removed in the next major version update.
        Deprecated = CUDNN_STATUS_DEPRECATED,
        /// This is an error category code.
        /// An incorrect value or parameter was passed to the function.
        BadParam = CUDNN_STATUS_BAD_PARAM,
        /// The cuDNN API has unexpectedly received a null pointer from the user.
        BadParamNullPointer = CUDNN_STATUS_BAD_PARAM_NULL_POINTER,
        /// The cuDNN API has received a misaligned pointer from the user.
        BadParamMisalignedPointer = CUDNN_STATUS_BAD_PARAM_MISALIGNED_POINTER,
        /// The backend descriptor has not been finalized.
        BadParamNotFinalized = CUDNN_STATUS_BAD_PARAM_NOT_FINALIZED,
        /// The cuDNN API has received an out-of-bound value.
        BadParamOutOfBound = CUDNN_STATUS_BAD_PARAM_OUT_OF_BOUND,
        /// The cuDNN API has received a memory buffer with insufficient space.
        BadParamSizeInsufficient = CUDNN_STATUS_BAD_PARAM_SIZE_INSUFFICIENT,
        /// The cuDNN API has received an unexpected stream.
        BadParamStreamMismatch = CUDNN_STATUS_BAD_PARAM_STREAM_MISMATCH,
        /// The cuDNN API has received inconsistent tensor shapes.
        BadParamShapeMismatch = CUDNN_STATUS_BAD_PARAM_SHAPE_MISMATCH,
        /// The cuDNN API has received duplicated entries.
        BadParamDuplicatedEntries = CUDNN_STATUS_BAD_PARAM_DUPLICATED_ENTRIES,
        /// The cuDNN API has received an invalid or unsupported attribute type.
        BadParamAttributeType = CUDNN_STATUS_BAD_PARAM_ATTRIBUTE_TYPE,
        /// The cuDNN API has received an unexpected CUDA graph.
        BadParamCudaGraphMismatch = CUDNN_STATUS_BAD_PARAM_CUDA_GRAPH_MISMATCH,
        /// Wrong descriptor type.
        BadParamDescriptorType = CUDNN_STATUS_BAD_PARAM_DESCRIPTOR_TYPE,
        /// This is an error category code.
        /// An internal cuDNN operation failed.
        InternalError = CUDNN_STATUS_INTERNAL_ERROR,
        /// A runtime kernel has failed to be compiled.
        InternalErrorCompilationFailed = CUDNN_STATUS_INTERNAL_ERROR_COMPILATION_FAILED,
        /// An unexpected internal inconsistency has been detected.
        InternalErrorUnexpectedValue = CUDNN_STATUS_INTERNAL_ERROR_UNEXPECTED_VALUE,
        /// An internal host memory allocation failed inside the cuDNN library.
        InternalErrorHostAllocationFailed = CUDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED,
        /// Resource allocation failed inside the cuDNN library.
        InternalErrorDeviceAllocationFailed = CUDNN_STATUS_INTERNAL_ERROR_DEVICE_ALLOCATION_FAILED,
        /// Invalid kernel launch parameters are unexpectedly detected.
        InternalErrorBadLaunchParam = CUDNN_STATUS_INTERNAL_ERROR_BAD_LAUNCH_PARAM,
        /// Access to GPU memory space failed, which is usually caused by a failure to bind a texture.
        /// To correct, prior to the function call, unbind any previously bound textures.
        /// Otherwise, this may indicate an internal error/bug in the library.
        InternalErrorTextureCreationFailed = CUDNN_STATUS_INTERNAL_ERROR_TEXTURE_CREATION_FAILED,
        /// An invalid value.
        /// Deprecated by cuDNN in favor of [`Status::BadParam`].
        InvalidValue = CUDNN_STATUS_INVALID_VALUE,
        /// This is an error category code.
        /// The GPU program failed to execute.
        /// This is usually caused by a failure to launch a kernel on the GPU, which can be caused by another library that cuDNN depends on.
        ExecutionFailed = CUDNN_STATUS_EXECUTION_FAILED,
        /// The GPU program failed to execute due to an error reported by the CUDA driver.
        ExecutionFailedCudaDriver = CUDNN_STATUS_EXECUTION_FAILED_CUDA_DRIVER,
        /// The GPU program failed to execute due to an error reported by cuBLAS.
        ExecutionFailedCublas = CUDNN_STATUS_EXECUTION_FAILED_CUBLAS,
        /// The GPU program failed to execute due to an error reported by the CUDA runtime.
        ExecutionFailedCudart = CUDNN_STATUS_EXECUTION_FAILED_CUDART,
        /// The GPU program failed to execute due to an error reported by cuRAND.
        ExecutionFailedCurand = CUDNN_STATUS_EXECUTION_FAILED_CURAND,
        /// This is an error category code.
        /// The functionality requested is not currently supported by cuDNN.
        NotSupported = CUDNN_STATUS_NOT_SUPPORTED,
        /// cuDNN does not currently support such an operation graph pattern.
        NotSupportedGraphPattern = CUDNN_STATUS_NOT_SUPPORTED_GRAPH_PATTERN,
        /// cuDNN does not currently support the tensor shapes used in some specific operation or graph pattern.
        NotSupportedShape = CUDNN_STATUS_NOT_SUPPORTED_SHAPE,
        /// cuDNN does not currently support the tensor data type.
        NotSupportedDataType = CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE,
        /// cuDNN does not currently support the tensor layout.
        NotSupportedLayout = CUDNN_STATUS_NOT_SUPPORTED_LAYOUT,
        /// The requested functionality is not compatible with the current CUDA driver.
        NotSupportedIncompatibleCudaDriver = CUDNN_STATUS_NOT_SUPPORTED_INCOMPATIBLE_CUDA_DRIVER,
        /// The requested functionality is not compatible with the current CUDA runtime.
        NotSupportedIncompatibleCudart = CUDNN_STATUS_NOT_SUPPORTED_INCOMPATIBLE_CUDART,
        /// The function requires a feature absent from the current GPU device.
        NotSupportedArchitectureMismatch = CUDNN_STATUS_NOT_SUPPORTED_ARCH_MISMATCH,
        /// A runtime library required by cuDNN cannot be found in the predefined search paths.
        /// These libraries are `libcuda.so` (`nvcuda.dll`) and `libnvrtc.so` (`nvrtc64_&lt;Major Release Version>&lt;Minor Release Version>_0.dll` and `nvrtc-builtins64_&lt;Major Release Version>&lt;Minor Release Version>.dll`).
        NotSupportedRuntimePrerequisiteMissing = CUDNN_STATUS_NOT_SUPPORTED_RUNTIME_PREREQUISITE_MISSING,
        /// The functionality requested requires some license and an error was detected when trying to check the current licensing.
        /// This error can happen if the license is not present or is expired or if the environment variable `NVIDIA_LICENSE_FILE` is not set properly.
        LicenseError = CUDNN_STATUS_LICENSE_ERROR,
        /// The requested functionality is not available due to missing a sublibrary.
        NotSupportedSublibraryUnavailable = CUDNN_STATUS_NOT_SUPPORTED_SUBLIBRARY_UNAVAILABLE,
        /// The requested functionality is not available due to the insufficient shared memory size on the GPU.
        NotSupportedSharedMemoryInsufficient = CUDNN_STATUS_NOT_SUPPORTED_SHARED_MEMORY_INSUFFICIENT,
        /// The requested functionality is not available due to padding requirements.
        NotSupportedPadding = CUDNN_STATUS_NOT_SUPPORTED_PADDING,
        /// The requested functionality is not available because they lead to invalid kernel launch parameters.
        NotSupportedBadLaunchParam = CUDNN_STATUS_NOT_SUPPORTED_BAD_LAUNCH_PARAM,
        /// The requested functionality is not available because this engine does not support the native CUDA graph API.
        /// Engines that support it report [`BackendBehaviorNote::SupportsCudaGraphNativeApi`](crate::backend::behavior::BackendBehaviorNote::SupportsCudaGraphNativeApi).
        NotSupportedCudaGraphNativeApi = CUDNN_STATUS_NOT_SUPPORTED_CUDA_GRAPH_NATIVE_API,
        /// Invalid dynamic shape.
        NotSupportedInvalidDynamicShape = CUDNN_STATUS_NOT_SUPPORTED_INVALID_DYNAMIC_SHAPE,
        /// Some tasks in the user stream are not completed.
        RuntimeInProgress = CUDNN_STATUS_RUNTIME_IN_PROGRESS,
        /// Numerical overflow occurred during the GPU kernel execution.
        RuntimeFpOverflow = CUDNN_STATUS_RUNTIME_FP_OVERFLOW,
        /// Sub-library loading failed.
        SublibraryLoadingFailed = CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED,
    }
}

impl Status {
    /// Returns whether this is [`Status::NotSupported`] or one of its `NotSupported*` sub-statuses.
    #[must_use]
    pub const fn is_not_supported(self) -> bool {
        let raw = self.raw();
        raw >= sys::cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED.0 && raw < 4000
    }
}

/// Errors from cuDNN, CUDA, frontend graph processing, and checked wrapper validation.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    /// cuDNN or one of its symbols would not load.
    #[error(transparent)]
    Load(#[from] sys::LoadError),

    /// Frontend graph construction, preparation, planning, or execution failed.
    #[error(transparent)]
    Frontend(#[from] FrontendError),

    /// An indexed errata rule failed validation.
    #[error("invalid errata rule {index}")]
    ErrataInvalidRule {
        /// Index of the invalid errata rule.
        index: usize,
        /// Validation error raised for that rule.
        source: Box<Self>,
    },

    /// An errata rule has invalid syntax.
    #[error("invalid errata rule syntax: {reason}")]
    ErrataRuleSyntax {
        /// Description of the invalid syntax.
        reason: &'static str,
    },

    /// An errata knob choice does not use the required `k<number>=<value>` syntax.
    #[error("invalid knob choice `{choice}`, expected k<number>=<value>")]
    ErrataKnobSyntax {
        /// Invalid knob-choice text.
        choice: String,
    },

    /// An integer in an errata knob choice could not be parsed.
    #[error("invalid knob number in `{choice}`")]
    ErrataKnobNumber {
        /// Knob-choice text containing the invalid integer.
        choice: String,
        /// Integer parse failure.
        source: ParseIntError,
    },

    /// A tensor identifier could not be parsed as an integer.
    #[error("invalid tensor id integer")]
    TensorIdParse(#[from] ParseIntError),

    /// A tensor identifier is negative.
    #[error("tensor id {value} must be non-negative")]
    TensorIdInvalid {
        /// Negative tensor identifier supplied by the caller.
        value: i64,
    },

    /// An errata rule has an empty operation name.
    #[error("errata operation cannot be empty")]
    ErrataOperationEmpty,

    /// An engine index is negative.
    #[error("engine index {value} must be nonnegative")]
    EngineIndexInvalid {
        /// Negative engine index supplied by the caller.
        value: i64,
    },

    /// An errata version interval has an end that does not exceed its start.
    #[error("errata version end {end} must exceed version start {start}")]
    ErrataVersionRangeInvalid {
        /// Start of the invalid cuDNN version interval.
        start: LibraryVersion,
        /// End of the invalid cuDNN version interval.
        end: LibraryVersion,
    },

    /// An errata shape contains a negative dimension.
    #[error("errata shape dimension {dimension} at axis {axis} must be nonnegative")]
    ErrataShapeDimensionInvalid {
        /// Axis containing the invalid dimension.
        axis: usize,
        /// Negative dimension at that axis.
        dimension: i64,
    },

    /// Reading an errata file failed.
    #[error("failed to read errata file `{}`", path.display())]
    ErrataFileRead {
        /// Path of the errata file that could not be read.
        path: PathBuf,
        /// Underlying file-read error.
        source: IoError,
    },

    /// An SDPA forward operation supplied only one of its query and key/value sequence-length tensors.
    #[error("sdpa forward query and key/value sequence lengths must be supplied together")]
    SdpaSequenceLengthsIncomplete,

    /// An SDPA implementation does not support a requested capability.
    #[error("{operation} does not support {feature}{}", on_sm(*.sm_version))]
    SdpaUnsupported {
        /// Operation whose requirements failed validation.
        operation: &'static str,
        /// Requested feature that is unavailable or requires a newer runtime.
        feature: SdpaFeature,
        /// Target SM architecture version, or `None` when unavailable.
        sm_version: Option<i32>,
    },

    /// A CUDA operation failed.
    #[error(transparent)]
    Cuda(CudaError),

    /// cuDNN returned a failure status, with the diagnostic it recorded for the calling thread.
    #[error(
        "cudnn error ({code}): {}{}{diagnostic}",
        code.description(),
        diagnostic_separator(diagnostic)
    )]
    Cudnn {
        /// Native cuDNN status returned by the failed call.
        code: Status,
        /// cuDNN's last error message for the calling thread, empty when it recorded none or the error was built from a [`Status`].
        diagnostic: String,
    },

    /// JSON serialization or deserialization failed.
    #[error(transparent)]
    Serde(#[from] JsonError),

    /// A native resource handle is unexpectedly null.
    #[error("unexpected null handle")]
    NullHandle,

    /// A native enum value has no typed variant in this crate.
    #[error("unknown `{name}` value {value}")]
    UnknownEnumValue {
        /// Name of the typed enum.
        name: Cow<'static, str>,
        /// Unrecognized native value.
        value: i64,
    },

    /// An argument or checked conversion is outside the supported range.
    #[error("`{name}` is out of range")]
    OutOfRange {
        /// Name of the invalid argument, attribute, or related resource.
        name: Cow<'static, str>,
    },

    /// Host allocation for scalar storage failed.
    #[error("unable to allocate {bytes} bytes for scalar storage")]
    ScalarStorageAllocation {
        /// Requested host scalar-storage capacity in bytes.
        bytes: usize,
        /// Allocation reservation failure.
        #[source]
        source: TryReserveError,
    },

    /// An argument or native result has an unexpected length.
    #[error("`{name}` length mismatch (expected {expected}, got {actual})")]
    LengthMismatch {
        /// Name of the invalid argument, attribute, or related resource.
        name: Cow<'static, str>,
        /// Required length for the named value.
        expected: usize,
        /// Observed length for the named value.
        actual: usize,
    },

    /// A required list is empty.
    #[error("`{name}` is empty")]
    Empty {
        /// Name of the empty list.
        name: Cow<'static, str>,
    },

    /// A data shape has invalid dimensions or checked size arithmetic.
    #[error("invalid data shape")]
    InvalidDataShape,

    /// A data layout has invalid strides.
    #[error("invalid data strides")]
    InvalidDataStrides,

    /// An axis order omits or duplicates an axis.
    #[error("axis order must contain every axis exactly once")]
    InvalidAxisOrder,

    /// A tensor shape differs from the required shape.
    #[error("`{name}` shape mismatch (expected {expected:?}, got {actual:?})")]
    ShapeMismatch {
        /// Name of the invalid argument, attribute, or related resource.
        name: Cow<'static, str>,
        /// Required tensor extents.
        expected: Vec<i64>,
        /// Supplied or inferred tensor extents.
        actual: Vec<i64>,
    },

    /// An operation does not support the supplied data type.
    #[error("unsupported data type")]
    UnsupportedDataType,

    /// SDPA statistics use storage other than 32-bit floating point.
    #[error("sdpa statistics require float32 storage, got `{actual}`")]
    SdpaStatsDataType {
        /// Storage data type supplied for SDPA statistics.
        actual: DataType,
    },

    /// A scalar value and its tensor descriptor have different data types.
    #[error("scalar data type {actual} does not match tensor data type {expected}")]
    TensorScalarDataTypeMismatch {
        /// Data type required by the tensor descriptor.
        expected: DataType,
        /// Data type of the supplied scalar.
        actual: DataType,
    },

    /// A native tensor binding has the wrong storage data type.
    #[error("native tensor `{tensor_id}` requires `{expected}` storage, got `{actual}`")]
    BackendTensorBindingTypeMismatch {
        /// Identifier of the native tensor whose binding is invalid.
        tensor_id: BackendTensorUid,
        /// Storage type required by the native tensor descriptor.
        expected: DataType,
        /// Storage type of the supplied binding.
        actual: DataType,
    },

    /// A pointwise axis was supplied for an operation that does not generate indices.
    #[error("pointwise axis requires index generation")]
    PointwiseAxisRequiresIndexGeneration,

    /// A mutex was poisoned by a panic while it was held.
    #[error("`{name}` lock is poisoned")]
    LockPoisoned {
        /// Name of the locked resource.
        name: Cow<'static, str>,
    },

    /// No compatible native execution engine is available.
    #[error("no available engines")]
    NoAvailableEngines,

    /// Planning without a live context requires explicit target-device properties.
    #[error("deviceless planning requires device properties")]
    DevicelessRequiresDeviceProperties,

    /// Native CUDA graph population requires an empty destination graph.
    #[error("cuda graph provided to populate must be empty")]
    CudaGraphPopulateRequiresEmptyGraph,

    /// Input, filter, and output channel counts cannot form valid convolution groups.
    #[error(
        "convolution channels cannot form groups: input={input_channels}, filter={filter_channels}, output={output_channels}"
    )]
    ConvolutionGroupChannelsMismatch {
        /// Number of input channels.
        input_channels: i64,
        /// Number of input channels per convolution filter.
        filter_channels: i64,
        /// Number of output channels.
        output_channels: i64,
    },

    /// Related descriptors do not describe the same required operation or resource.
    #[error("descriptor mismatch for `{name}`")]
    DescriptorMismatch {
        /// Name of the invalid argument, attribute, or related resource.
        name: Cow<'static, str>,
    },

    /// A stream belongs to another CUDA context.
    #[error("stream belongs to a different cuda context")]
    StreamContextMismatch,

    /// A cuDNN object belongs to a different cuDNN context.
    #[error("`{name}` belongs to a different cudnn context")]
    ContextMismatch {
        /// Name of the object.
        name: Cow<'static, str>,
    },

    /// A buffer or workspace is smaller than the operation requires.
    #[error("`{name}` is too small (requires {required}, got {actual})")]
    BufferTooSmall {
        /// Name of the buffer.
        name: Cow<'static, str>,
        /// Required size in bytes.
        required: usize,
        /// Supplied size in bytes.
        actual: usize,
    },

    /// An operation requires a finalized backend descriptor.
    #[error("descriptor needs to be finalized")]
    DescriptorRequiredFinalized,

    /// A descriptor was finalized before an operation that expects it to be unfinished.
    #[error("descriptor already finalized")]
    DescriptorAlreadyFinalized,

    /// An attribute update attempted to modify a finalized descriptor.
    #[error("cannot modify finalized descriptor")]
    DescriptorCannotModifyFinalized,

    /// A descriptor is not an execution-plan descriptor.
    #[error("descriptor expected execution plan")]
    DescriptorExpectedExecutionPlan,

    /// A descriptor is not a variant-pack descriptor.
    #[error("descriptor expected variant pack")]
    DescriptorExpectedVariantPack,

    /// A backend descriptor has the wrong native descriptor type.
    #[error("invalid backend descriptor type (expected {expected}, got {actual})")]
    InvalidBackendDescriptorType {
        /// Native descriptor type required by the operation.
        expected: BackendDescriptorType,
        /// Native descriptor type of the supplied resource.
        actual: BackendDescriptorType,
    },

    /// A backend attribute returned an unexpected descriptor count.
    #[error("invalid backend descriptor count for `{name}`")]
    InvalidBackendDescriptorCount {
        /// Name of the invalid argument, attribute, or related resource.
        name: Cow<'static, str>,
    },

    /// A requested backend descriptor attribute is absent.
    #[error("attribute `{0}` not found")]
    DescriptorAttributeNotFound(BackendAttributeName),

    /// An engine configuration specifies the same knob more than once.
    #[error("duplicate engine knob choice `{knob_type}`")]
    DuplicateEngineKnobChoice {
        /// Engine knob specified more than once.
        knob_type: BackendKnobType,
    },

    /// A backend descriptor attribute has an invalid value.
    #[error("invalid attribute value for `{0}`: {1}")]
    DescriptorInvalidAttributeValue(BackendAttributeName, String),
}

/// Result of a cuDNN wrapper operation.
pub type Result<T> = StdResult<T, Error>;

impl Error {
    pub(crate) fn out_of_range(name: impl Into<Cow<'static, str>>) -> Self {
        Self::OutOfRange { name: name.into() }
    }

    /// The native status when this error comes from a cuDNN call.
    #[must_use]
    pub fn status(&self) -> Option<Status> {
        match self {
            Self::Cudnn { code, .. } => Some(*code),
            _ => None,
        }
    }
}

impl From<CudaError> for Error {
    fn from(error: CudaError) -> Self {
        match error {
            CudaError::StreamContextMismatch => Self::StreamContextMismatch,
            error => Self::Cuda(error),
        }
    }
}

impl From<singe_core::UnknownEnumValue> for Error {
    fn from(error: singe_core::UnknownEnumValue) -> Self {
        Self::UnknownEnumValue {
            name: error.name.into(),
            value: error.value,
        }
    }
}

/// SDPA capability rejected by [`Error::SdpaUnsupported`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SdpaFeature {
    /// Auxiliary attention gradients.
    AuxiliaryGradients,
    /// An attention-bias gradient.
    BiasGradient,
    /// Bias gradients, dropout, or attention with linear biases (`ALiBi`).
    BiasDropoutOrAlibi,
    /// Causal masking aligned to the bottom-right corner of the attention matrix.
    CausalBottomRight,
    /// Cumulative sequence-length tensors.
    CumulativeSequenceLengths,
    /// Caller-supplied dropout mask and scale tensors.
    CustomDropout,
    /// Attention dropout.
    Dropout,
    /// Dropout combined with generated softmax statistics.
    DropoutWithGeneratedStats,
    /// Dynamic-shape compilation rather than runtime shape overrides.
    DynamicShape,
    /// Internally generated dropout.
    GeneratedDropout,
    /// A bias tensor with a nonnumeric data type.
    NonNumericBias,
    /// Paged key/value caches.
    PagedCaches,
    /// Ragged attention outputs.
    RaggedOutputs,
    /// Ragged tensor storage.
    RaggedTensors,
    /// Dumping generated random numbers.
    RngDump,
    /// A sequence containing one token.
    SingleTokenSequence,
    /// The selected target architecture.
    TargetArchitecture,
    /// `ALiBi` slopes without causal top-left or right-aligned sliding-window masking.
    UnalignedAlibi,
}

impl fmt::Display for SdpaFeature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::AuxiliaryGradients => "auxiliary gradients",
            Self::BiasGradient => "bias gradients",
            Self::BiasDropoutOrAlibi => "bias gradients, dropout, or alibi",
            Self::CausalBottomRight => "causal bottom-right masking",
            Self::CumulativeSequenceLengths => "cumulative sequence lengths",
            Self::CustomDropout => "custom dropout mask and scale tensors",
            Self::Dropout => "dropout",
            Self::DropoutWithGeneratedStats => "dropout with generated stats",
            Self::DynamicShape => "dynamic-shape compilation; use runtime shape overrides",
            Self::GeneratedDropout => "generated dropout",
            Self::NonNumericBias => "a bias tensor with a non-numeric data type",
            Self::PagedCaches => "paged key/value caches",
            Self::RaggedOutputs => "ragged outputs",
            Self::RaggedTensors => "ragged tensors",
            Self::RngDump => "rng dump",
            Self::SingleTokenSequence => "sequence length one",
            Self::TargetArchitecture => "this target architecture",
            Self::UnalignedAlibi => {
                "alibi slopes without causal top-left or right-aligned sliding-window masking"
            }
        })
    }
}

// Separates a non-empty thread diagnostic from the status description.
fn diagnostic_separator(diagnostic: &str) -> &'static str {
    if diagnostic.is_empty() { "" } else { ": " }
}

// Names the target architecture in an SDPA error when it is known.
pub(crate) fn on_sm(sm_version: Option<i32>) -> String {
    sm_version.map_or_default(|sm_version| format!(" on SM {sm_version}"))
}

// Converts a raw cuDNN status into this crate's `Result`.
macro_rules! try_ffi {
    ($expr:expr) => {
        $crate::error::Error::check($expr)
    };
}

pub(crate) use try_ffi;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sdpa_unsupported_mentions_architecture_only_when_known() {
        let with_sm = Error::SdpaUnsupported {
            operation: "sdpa backward",
            feature: SdpaFeature::RaggedTensors,
            sm_version: Some(90),
        };
        assert_eq!(
            with_sm.to_string(),
            "sdpa backward does not support ragged tensors on SM 90"
        );
        let without_sm = Error::SdpaUnsupported {
            operation: "mxfp8 sdpa",
            feature: SdpaFeature::PagedCaches,
            sm_version: None,
        };
        assert_eq!(
            without_sm.to_string(),
            "mxfp8 sdpa does not support paged key/value caches"
        );
    }
}
