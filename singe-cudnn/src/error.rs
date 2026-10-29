use std::{
    borrow::Cow, collections::TryReserveError, hash::Hash, io::Error as IoError,
    num::ParseIntError, path::PathBuf, result::Result as StdResult,
};

use serde_json::Error as JsonError;
use singe_core::{static_str_from_c_ptr, status_enum};
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

status_enum! {
    /// Status returned by cuDNN operations.
    pub enum Status: u32 => sys::cudnnStatus_t, unknown = "UNKNOWN_CUDNN_STATUS" {
        /// The operation was completed successfully.
        Success = CUDNN_STATUS_SUCCESS,
        /// The cuDNN library was not initialized properly. This error is usually returned when a call to `cudnnCreate` fails or when `cudnnCreate` has not been called prior to calling another cuDNN routine. In the former case, it is usually due to an error in the CUDA Runtime API called by `cudnnCreate` or by an error in the hardware setup.
        NotInitialized = CUDNN_STATUS_NOT_INITIALIZED,
        /// Some cuDNN sub libraries have different versions, indicative of an installation issue.
        SublibraryVersionMismatch = CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH,
        /// The schema used for serialization is not what the current cuDNN library expects, thus the serialized artifact is no longer valid and needs to be re-serialized.
        SerializationVersionMismatch = CUDNN_STATUS_SERIALIZATION_VERSION_MISMATCH,
        /// This error code may be reported in the warning logging level as a reminder that some functionality is under deprecation and will be removed in the next major version update.
        Deprecated = CUDNN_STATUS_DEPRECATED,
        /// This is an error category code. An incorrect value or parameter was passed to the function.
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
        ///< Wrong descriptor type. @since cuDNN 9.6.0
        BadParamDescriptorType = CUDNN_STATUS_BAD_PARAM_DESCRIPTOR_TYPE,
        /// This is an error category code. An internal cuDNN operation failed.
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
        /// Access to GPU memory space failed, which is usually caused by a failure to bind a texture. To correct, prior to the function call, unbind any previously bound textures. Otherwise, this may indicate an internal error/bug in the library.
        InternalErrorTextureCreationFailed = CUDNN_STATUS_INTERNAL_ERROR_TEXTURE_CREATION_FAILED,
        /// please transition to CUDNN_STATUS_BAD_PARAM instead
        InvalidValue = CUDNN_STATUS_INVALID_VALUE,
        /// This is an error category code. The GPU program failed to execute. This is usually caused by a failure to launch a kernel on the GPU, which can be caused by another library that cuDNN depends on.
        ExecutionFailed = CUDNN_STATUS_EXECUTION_FAILED,
        /// The GPU program failed to execute due to an error reported by the CUDA driver.
        ExecutionFailedCudaDriver = CUDNN_STATUS_EXECUTION_FAILED_CUDA_DRIVER,
        /// The GPU program failed to execute due to an error reported by cuBLAS.
        ExecutionFailedCublas = CUDNN_STATUS_EXECUTION_FAILED_CUBLAS,
        /// The GPU program failed to execute due to an error reported by the CUDA runtime.
        ExecutionFailedCudart = CUDNN_STATUS_EXECUTION_FAILED_CUDART,
        /// The GPU program failed to execute due to an error reported by cuRAND.
        ExecutionFailedCurand = CUDNN_STATUS_EXECUTION_FAILED_CURAND,
        /// This is an error category code. The functionality requested is not currently supported by cuDNN.
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
        /// A runtime library required by cuDNN cannot be found in the predefined search paths. These libraries are `libcuda.so` (`nvcuda.dll`) and `libnvrtc.so` (`nvrtc64_&lt;Major Release Version>&lt;Minor Release Version>_0.dll` and `nvrtc-builtins64_&lt;Major Release Version>&lt;Minor Release Version>.dll`).
        NotSupportedRuntimePrerequisiteMissing = CUDNN_STATUS_NOT_SUPPORTED_RUNTIME_PREREQUISITE_MISSING,
        /// The functionality requested requires some license and an error was detected when trying to check the current licensing. This error can happen if the license is not present or is expired or if the environment variable `NVIDIA_LICENSE_FILE` is not set properly.
        LicenseError = CUDNN_STATUS_LICENSE_ERROR,
        /// The requested functionality is not available due to missing a sublibrary.
        NotSupportedSublibraryUnavailable = CUDNN_STATUS_NOT_SUPPORTED_SUBLIBRARY_UNAVAILABLE,
        /// The requested functionality is not available due to the insufficient shared memory size on the GPU.
        NotSupportedSharedMemoryInsufficient = CUDNN_STATUS_NOT_SUPPORTED_SHARED_MEMORY_INSUFFICIENT,
        /// The requested functionality is not available due to padding requirements.
        NotSupportedPadding = CUDNN_STATUS_NOT_SUPPORTED_PADDING,
        /// The requested functionality is not available because they lead to invalid kernel launch parameters.
        NotSupportedBadLaunchParam = CUDNN_STATUS_NOT_SUPPORTED_BAD_LAUNCH_PARAM,
        /// The requested functionality is not available because this particular engine does not support the native CUDA graph API. (The engines that do support that API have the behavior note `cudnnBackendBehaviorNote_t::CUDNN_BEHAVIOR_NOTE_SUPPORTS_CUDA_GRAPH_NATIVE_API`.)
        NotSupportedCudaGraphNativeApi = CUDNN_STATUS_NOT_SUPPORTED_CUDA_GRAPH_NATIVE_API,
        ///< Invalid dynamic shape. @since cuDNN 9.18.0
        NotSupportedInvalidDynamicShape = CUDNN_STATUS_NOT_SUPPORTED_INVALID_DYNAMIC_SHAPE,
        /// Some tasks in the user stream are not completed.
        RuntimeInProgress = CUDNN_STATUS_RUNTIME_IN_PROGRESS,
        /// Numerical overflow occurred during the GPU kernel execution.
        RuntimeFpOverflow = CUDNN_STATUS_RUNTIME_FP_OVERFLOW,
        ///< Sub-library loading failed. @since cuDNN 9.2.0
        SublibraryLoadingFailed = CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED,
    }
}

impl Status {
    /// Library description of this status.
    #[must_use]
    pub fn description(self) -> &'static str {
        let Ok(raw) = sys::cudnnStatus_t::try_from(self) else {
            return "unknown cudnn error";
        };
        // SAFETY: `cudnnGetErrorString` returns null or a static NUL-terminated string.
        unsafe { static_str_from_c_ptr(sys::cudnnGetErrorString(raw), "unknown cudnn error") }
    }

    /// Whether this is `CUDNN_STATUS_NOT_SUPPORTED` or one of its sub-statuses.
    #[must_use]
    pub const fn is_not_supported(self) -> bool {
        let raw = self.raw();
        raw >= sys::cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED as u32 && raw < 4000
    }
}

/// Errors from cuDNN, CUDA, frontend graph processing, and checked wrapper validation.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    /// Frontend graph construction, preparation, planning, or execution failed.
    #[error(transparent)]
    Frontend(#[from] FrontendError),

    /// An indexed errata rule failed validation.
    #[error("invalid errata rule {index}: {source}")]
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
    #[error("invalid knob number in `{choice}`: {source}")]
    ErrataKnobNumber {
        /// Knob-choice text containing the invalid integer.
        choice: String,
        /// Integer parse failure.
        source: ParseIntError,
    },

    /// A tensor identifier could not be parsed as an integer.
    #[error("invalid tensor id integer: {0}")]
    TensorIdParse(#[from] ParseIntError),

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
        start: u64,
        /// End of the invalid cuDNN version interval.
        end: u64,
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
    #[error("failed to read errata file `{}`: {source}", path.display())]
    ErrataFileRead {
        /// Path of the errata file that could not be read.
        path: PathBuf,
        /// Underlying file-read error.
        source: IoError,
    },

    /// An SDPA forward operation supplied only one of its query and key/value sequence-length tensors.
    #[error("sdpa forward query and key/value sequence lengths must be supplied together")]
    SdpaSequenceLengthsIncomplete,

    /// A CUDA operation failed.
    #[error("cuda error: {0}")]
    Cuda(#[from] CudaError),

    /// Native status, its description and any captured current-thread cuDNN diagnostic.
    #[error("cudnn error ({code}): {message}")]
    Cudnn {
        /// Native cuDNN status returned by the failed call.
        code: Status,
        /// Status description, optionally followed by a captured thread-local cuDNN diagnostic.
        message: String,
    },

    /// JSON serialization or deserialization failed.
    #[error("serde error: {0}")]
    Serde(#[from] JsonError),

    /// A native resource handle is unexpectedly null.
    #[error("unexpected null handle")]
    NullHandle,

    /// An argument or checked conversion is outside the supported range.
    #[error("`{name}` is out of range")]
    OutOfRange {
        /// Name of the invalid argument, attribute, or related resource.
        name: Cow<'static, str>,
    },

    /// Host allocation for scalar storage failed.
    #[error("unable to allocate {bytes} bytes for scalar storage: {source}")]
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
    #[error("`{name}` must not be empty")]
    EmptyList {
        /// Name of the invalid argument, attribute, or related resource.
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

    /// A native enum value is invalid for the requested wrapper type.
    #[error("invalid enum value for `{name}`: {value}")]
    InvalidEnumValue {
        /// Name of the invalid argument, attribute, or related resource.
        name: Cow<'static, str>,
        /// Invalid native integer enum value.
        value: u32,
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

    /// A native tensor binding has insufficient byte capacity.
    #[error("native tensor `{tensor_id}` requires {required} bytes, but binding has {actual}")]
    BackendTensorBindingCapacityMismatch {
        /// Identifier of the native tensor whose binding is invalid.
        tensor_id: BackendTensorUid,
        /// Required binding capacity in bytes.
        required: usize,
        /// Supplied binding capacity in bytes.
        actual: usize,
    },

    /// A native tensor binding does not satisfy its byte alignment.
    #[error("native tensor `{tensor_id}` requires {required}-byte binding alignment")]
    BackendTensorBindingAlignmentMismatch {
        /// Identifier of the native tensor whose binding is invalid.
        tensor_id: BackendTensorUid,
        /// Required device-address alignment in bytes.
        required: usize,
    },

    /// A pointwise axis was supplied for an operation that does not generate indices.
    #[error("pointwise axis requires index generation")]
    PointwiseAxisRequiresIndexGeneration,

    /// The execution-plan cache mutex is poisoned.
    #[error("execution plan cache lock poisoned")]
    ExecutionPlanCacheLockPoisoned,

    /// The heuristic-mode B mutex is poisoned.
    #[error("heuristic mode b lock poisoned")]
    EngineHeuristicsLockPoisoned,

    /// An operation requires an explicitly configured cuDNN stream.
    #[error("context stream is not set")]
    ContextStreamNotSet,

    /// No compatible native execution engine is available.
    #[error("no available engines")]
    NoAvailableEngines,

    /// Related descriptors do not describe the same required operation or resource.
    #[error("descriptor mismatch for `{name}`")]
    DescriptorMismatch {
        /// Name of the invalid argument, attribute, or related resource.
        name: Cow<'static, str>,
    },

    /// A stream belongs to another CUDA context.
    #[error("stream belongs to a different cuda context")]
    StreamContextMismatch,

    /// Resources that must share one cuDNN handle belong to different handles.
    #[error("cudnn contexts must match for `{name}`")]
    ContextMismatch {
        /// Name of the invalid argument, attribute, or related resource.
        name: Cow<'static, str>,
    },

    /// The supplied workspace has insufficient byte capacity.
    #[error("insufficient workspace size: {actual} bytes provided (required {required} bytes)")]
    InsufficientWorkspaceSize {
        /// Required workspace capacity in bytes.
        required: usize,
        /// Supplied workspace capacity in bytes.
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

/// Captures the current thread's native diagnostic before any later cuDNN call can replace it.
/// Consumes that diagnostic, unlike conversion from a plain [`Status`].
impl From<sys::cudnnStatus_t> for Error {
    fn from(status: sys::cudnnStatus_t) -> Self {
        let diagnostic = last_error_string();
        let code = Status::from(status);
        let mut message = code.description().to_owned();
        if !diagnostic.is_empty() {
            message.push_str(": ");
            message.push_str(&diagnostic);
        }
        Self::Cudnn { code, message }
    }
}

impl From<Status> for Error {
    fn from(code: Status) -> Self {
        Self::Cudnn {
            code,
            message: code.description().to_string(),
        }
    }
}

/// Converts a raw cuDNN status-returning expression into a crate [`Result`](crate::error::Result).
#[macro_export]
macro_rules! try_ffi {
    ($expr:expr) => {{
        let status = { $expr };
        if status != singe_cudnn_sys::cudnnStatus_t::CUDNN_STATUS_SUCCESS {
            Err($crate::error::Error::from(status))
        } else {
            Ok(())
        }
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unknown_status_is_preserved_as_typed_cudnn_status() {
        let error = Error::from(Status::Unknown(12345));

        match error {
            Error::Cudnn {
                code: Status::Unknown(code),
                message,
            } => {
                assert_eq!(code, 12345);
                assert_eq!(message, "unknown cudnn error");
            }
            error => panic!("unexpected error: {error:?}"),
        }
    }

    #[test]
    fn test_known_status_round_trips_to_raw_status() {
        assert_eq!(
            sys::cudnnStatus_t::try_from(Status::NotSupported).unwrap(),
            sys::cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED
        );
        assert_eq!(
            Status::try_from(sys::cudnnStatus_t::CUDNN_STATUS_RUNTIME_FP_OVERFLOW as u32).unwrap(),
            Status::RuntimeFpOverflow
        );
    }
}
