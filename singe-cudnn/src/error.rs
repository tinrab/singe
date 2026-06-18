use std::{ffi::CStr, fmt};

use singe_cuda::error::Error as CudaError;
use singe_cudnn_sys as sys;
use thiserror::Error;

use crate::{
    attribute::BackendAttributeName, data_type::DataType, descriptor::BackendDescriptorType,
    tensor::TensorId,
};

macro_rules! cudnn_statuses {
    ($($variant:ident => $raw:ident,)*) => {
        /// Status returned by cuDNN operations.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        #[non_exhaustive]
        pub enum Status {
            $($variant,)*
            Unknown(u32),
        }

        impl Status {
            pub fn description(self) -> String {
                match sys::cudnnStatus_t::try_from(self.raw()) {
                    Ok(status) => cudnn_error_string(status),
                    Err(_) => String::from("unknown cudnn error"),
                }
            }

            pub const fn raw(self) -> u32 {
                match self {
                    $(Self::$variant => sys::cudnnStatus_t::$raw as _,)*
                    Self::Unknown(code) => code,
                }
            }
        }

        impl TryFrom<u32> for Status {
            type Error = u32;

            fn try_from(code: u32) -> std::result::Result<Self, u32> {
                match code {
                    $(code if code == sys::cudnnStatus_t::$raw as u32 => Ok(Self::$variant),)*
                    code => Err(code),
                }
            }
        }

        impl From<sys::cudnnStatus_t> for Status {
            fn from(status: sys::cudnnStatus_t) -> Self {
                Self::try_from(status as u32).unwrap_or_else(Self::Unknown)
            }
        }

        impl TryFrom<Status> for sys::cudnnStatus_t {
            type Error = Status;

            fn try_from(status: Status) -> std::result::Result<Self, Status> {
                match status {
                    $(Status::$variant => Ok(Self::$raw),)*
                    Status::Unknown(_) => Err(status),
                }
            }
        }

        impl fmt::Display for Status {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                match self {
                    $(Self::$variant => f.write_str(stringify!($raw)),)*
                    Self::Unknown(code) => write!(f, "UNKNOWN_CUDNN_STATUS({code})"),
                }
            }
        }
    };
}

cudnn_statuses! {
    Success => CUDNN_STATUS_SUCCESS,
    NotInitialized => CUDNN_STATUS_NOT_INITIALIZED,
    SublibraryVersionMismatch => CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH,
    SerializationVersionMismatch => CUDNN_STATUS_SERIALIZATION_VERSION_MISMATCH,
    Deprecated => CUDNN_STATUS_DEPRECATED,
    BadParam => CUDNN_STATUS_BAD_PARAM,
    BadParamNullPointer => CUDNN_STATUS_BAD_PARAM_NULL_POINTER,
    BadParamMisalignedPointer => CUDNN_STATUS_BAD_PARAM_MISALIGNED_POINTER,
    BadParamNotFinalized => CUDNN_STATUS_BAD_PARAM_NOT_FINALIZED,
    BadParamOutOfBound => CUDNN_STATUS_BAD_PARAM_OUT_OF_BOUND,
    BadParamSizeInsufficient => CUDNN_STATUS_BAD_PARAM_SIZE_INSUFFICIENT,
    BadParamStreamMismatch => CUDNN_STATUS_BAD_PARAM_STREAM_MISMATCH,
    BadParamShapeMismatch => CUDNN_STATUS_BAD_PARAM_SHAPE_MISMATCH,
    BadParamDuplicatedEntries => CUDNN_STATUS_BAD_PARAM_DUPLICATED_ENTRIES,
    BadParamAttributeType => CUDNN_STATUS_BAD_PARAM_ATTRIBUTE_TYPE,
    BadParamCudaGraphMismatch => CUDNN_STATUS_BAD_PARAM_CUDA_GRAPH_MISMATCH,
    BadParamDescriptorType => CUDNN_STATUS_BAD_PARAM_DESCRIPTOR_TYPE,
    InternalError => CUDNN_STATUS_INTERNAL_ERROR,
    InternalErrorCompilationFailed => CUDNN_STATUS_INTERNAL_ERROR_COMPILATION_FAILED,
    InternalErrorUnexpectedValue => CUDNN_STATUS_INTERNAL_ERROR_UNEXPECTED_VALUE,
    InternalErrorHostAllocationFailed => CUDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED,
    InternalErrorDeviceAllocationFailed => CUDNN_STATUS_INTERNAL_ERROR_DEVICE_ALLOCATION_FAILED,
    InternalErrorBadLaunchParam => CUDNN_STATUS_INTERNAL_ERROR_BAD_LAUNCH_PARAM,
    InternalErrorTextureCreationFailed => CUDNN_STATUS_INTERNAL_ERROR_TEXTURE_CREATION_FAILED,
    InvalidValue => CUDNN_STATUS_INVALID_VALUE,
    ExecutionFailed => CUDNN_STATUS_EXECUTION_FAILED,
    ExecutionFailedCudaDriver => CUDNN_STATUS_EXECUTION_FAILED_CUDA_DRIVER,
    ExecutionFailedCublas => CUDNN_STATUS_EXECUTION_FAILED_CUBLAS,
    ExecutionFailedCudart => CUDNN_STATUS_EXECUTION_FAILED_CUDART,
    ExecutionFailedCurand => CUDNN_STATUS_EXECUTION_FAILED_CURAND,
    NotSupported => CUDNN_STATUS_NOT_SUPPORTED,
    NotSupportedGraphPattern => CUDNN_STATUS_NOT_SUPPORTED_GRAPH_PATTERN,
    NotSupportedShape => CUDNN_STATUS_NOT_SUPPORTED_SHAPE,
    NotSupportedDataType => CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE,
    NotSupportedLayout => CUDNN_STATUS_NOT_SUPPORTED_LAYOUT,
    NotSupportedIncompatibleCudaDriver => CUDNN_STATUS_NOT_SUPPORTED_INCOMPATIBLE_CUDA_DRIVER,
    NotSupportedIncompatibleCudart => CUDNN_STATUS_NOT_SUPPORTED_INCOMPATIBLE_CUDART,
    NotSupportedArchitectureMismatch => CUDNN_STATUS_NOT_SUPPORTED_ARCH_MISMATCH,
    NotSupportedRuntimePrerequisiteMissing => CUDNN_STATUS_NOT_SUPPORTED_RUNTIME_PREREQUISITE_MISSING,
    LicenseError => CUDNN_STATUS_LICENSE_ERROR,
    NotSupportedSublibraryUnavailable => CUDNN_STATUS_NOT_SUPPORTED_SUBLIBRARY_UNAVAILABLE,
    NotSupportedSharedMemoryInsufficient => CUDNN_STATUS_NOT_SUPPORTED_SHARED_MEMORY_INSUFFICIENT,
    NotSupportedPadding => CUDNN_STATUS_NOT_SUPPORTED_PADDING,
    NotSupportedBadLaunchParam => CUDNN_STATUS_NOT_SUPPORTED_BAD_LAUNCH_PARAM,
    NotSupportedCudaGraphNativeApi => CUDNN_STATUS_NOT_SUPPORTED_CUDA_GRAPH_NATIVE_API,
    NotSupportedInvalidDynamicShape => CUDNN_STATUS_NOT_SUPPORTED_INVALID_DYNAMIC_SHAPE,
    RuntimeInProgress => CUDNN_STATUS_RUNTIME_IN_PROGRESS,
    RuntimeFpOverflow => CUDNN_STATUS_RUNTIME_FP_OVERFLOW,
    SublibraryLoadingFailed => CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED,
}

fn cudnn_error_string(status: sys::cudnnStatus_t) -> String {
    unsafe {
        let ptr = sys::cudnnGetErrorString(status);
        if ptr.is_null() {
            String::from("unknown cudnn error")
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    }
}

#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    #[error("cuda error: {0}")]
    Cuda(#[from] CudaError),

    #[error("cudnn error ({code}): {message}")]
    Cudnn { code: Status, message: String },

    #[error("serde error: {0}")]
    Serde(#[from] serde_json::Error),

    #[error("unexpected null handle")]
    NullHandle,

    #[error("`{name}` is out of range")]
    OutOfRange { name: String },

    #[error("`{name}` length mismatch (expected {expected}, got {actual})")]
    LengthMismatch {
        name: String,
        expected: usize,
        actual: usize,
    },

    #[error("`{name}` must not be empty")]
    EmptyList { name: String },

    #[error("invalid data shape")]
    InvalidDataShape,

    #[error("invalid data strides")]
    InvalidDataStrides,

    #[error("unsupported data type")]
    UnsupportedDataType,

    #[error("invalid enum value for `{name}`: {value}")]
    InvalidEnumValue { name: String, value: u32 },

    #[error("frontend graph has no operations")]
    FrontendGraphEmpty,

    #[error("frontend tensor `{0}` was not found")]
    FrontendTensorNotFound(TensorId),

    #[error("frontend tensor `{0}` was not bound")]
    FrontendTensorNotBound(TensorId),

    #[error(
        "frontend binding replacement source tensor `{source_id}` was not bound while preparing target `{target_id}` with byte offset {byte_offset}"
    )]
    FrontendBindingReplacementSourceNotBound {
        source_id: TensorId,
        target_id: TensorId,
        byte_offset: i64,
    },

    #[error("frontend tensor ID `{0}` is duplicated")]
    FrontendTensorIdConflict(TensorId),

    #[error(
        "frontend tensor `{tensor_id}` has incompatible scalar type `{actual}` (expected `{expected}`)"
    )]
    FrontendScalarTypeMismatch {
        tensor_id: TensorId,
        expected: String,
        actual: String,
    },

    #[error(
        "frontend tensor `{tensor_id}` has incompatible binding type `{actual}` (expected `{expected}`)"
    )]
    FrontendBindingTypeMismatch {
        tensor_id: TensorId,
        expected: String,
        actual: String,
    },

    #[error(
        "frontend tensor `{tensor_id}` has incompatible data type `{actual}` for {operation} (expected `{expected}`)"
    )]
    FrontendTensorDataTypeMismatch {
        tensor_id: TensorId,
        operation: String,
        expected: DataType,
        actual: DataType,
    },

    #[error(
        "frontend tensor `{tensor_id}` has unsupported data type `{actual}` for {operation} (supported {supported:?})"
    )]
    FrontendTensorDataTypeUnsupported {
        tensor_id: TensorId,
        operation: String,
        supported: Vec<DataType>,
        actual: DataType,
    },

    #[error(
        "frontend tensor `{tensor_id}` has incompatible dimensions for {operation} (expected {expected:?}, got {actual:?})"
    )]
    FrontendTensorDimensionsMismatch {
        tensor_id: TensorId,
        operation: String,
        expected: Vec<i64>,
        actual: Vec<i64>,
    },

    #[error(
        "frontend tensor `{tensor_id}` has incompatible rank for {operation} (expected {expected}, got {actual})"
    )]
    FrontendTensorRankMismatch {
        tensor_id: TensorId,
        operation: String,
        expected: String,
        actual: usize,
    },

    #[error(
        "frontend tensor `{tensor_id}` has incompatible strides for {operation} (expected {expected:?}, got {actual:?})"
    )]
    FrontendTensorStridesMismatch {
        tensor_id: TensorId,
        operation: String,
        expected: Vec<i64>,
        actual: Vec<i64>,
    },

    #[error(
        "frontend tensor `{tensor_id}` has incompatible element count for {operation} (expected {expected}, got {actual})"
    )]
    FrontendTensorElementCountMismatch {
        tensor_id: TensorId,
        operation: String,
        expected: usize,
        actual: usize,
    },

    #[error(
        "frontend alias target `{target_id}` byte offset {byte_offset} is not aligned to element size {element_size} from source `{source_id}`"
    )]
    FrontendAliasByteOffsetUnaligned {
        source_id: TensorId,
        target_id: TensorId,
        byte_offset: i64,
        element_size: i64,
    },

    #[error(
        "frontend alias target `{target_id}` byte range [{start}, {end}) is outside source `{source_id}` byte length {source_byte_len}"
    )]
    FrontendAliasByteRangeOutOfBounds {
        source_id: TensorId,
        target_id: TensorId,
        start: usize,
        end: usize,
        source_byte_len: usize,
    },

    #[error("frontend alias source tensor `{source_id}` is not bindable: {reason}")]
    FrontendAliasSourceNotBindable { source_id: TensorId, reason: String },

    #[error("frontend tensor `{tensor_id}` ragged offset `{ragged_offset}` would create a cycle")]
    FrontendRaggedOffsetCycle {
        tensor_id: TensorId,
        ragged_offset: TensorId,
    },

    #[error(
        "frontend dynamic shape bounds are invalid (min dimensions {min_dimensions:?}, max dimensions {max_dimensions:?})"
    )]
    FrontendDynamicShapeBoundsInvalid {
        min_dimensions: Vec<i64>,
        max_dimensions: Vec<i64>,
    },

    #[error(
        "frontend fused scalar tensor `{tensor_id}` is used with conflicting ranks ({first_rank} and {second_rank})"
    )]
    FrontendFusedScalarRankConflict {
        tensor_id: TensorId,
        first_rank: i64,
        second_rank: i64,
    },

    #[error("frontend pointwise binary alpha2 must be 1.0 for graph support (got {alpha2})")]
    FrontendPointwiseBinaryAlpha2Unsupported { alpha2: f64 },

    #[error("frontend pointwise inputs cannot broadcast together (lhs {lhs:?}, rhs {rhs:?})")]
    FrontendPointwiseBroadcastShapeMismatch { lhs: Vec<i64>, rhs: Vec<i64> },

    #[error(
        "frontend matmul inputs `{lhs_id}` and `{rhs_id}` have incompatible dimensions for {operation} (lhs {lhs:?}, rhs {rhs:?})"
    )]
    FrontendMatmulShapeMismatch {
        lhs_id: TensorId,
        rhs_id: TensorId,
        operation: String,
        lhs: Vec<i64>,
        rhs: Vec<i64>,
    },

    #[error(
        "frontend pointwise {operation} output shape mismatch (expected {expected:?}, got {actual:?})"
    )]
    FrontendPointwiseOutputShapeMismatch {
        operation: String,
        expected: Vec<i64>,
        actual: Vec<i64>,
    },

    #[error("frontend sdpa q/k shapes are incompatible (q {q:?}, k {k:?})")]
    FrontendSdpaQKShapeMismatch { q: Vec<i64>, k: Vec<i64> },

    #[error("frontend sdpa scores/v shapes are incompatible (scores {scores:?}, v {v:?})")]
    FrontendSdpaScoresVShapeMismatch { scores: Vec<i64>, v: Vec<i64> },

    #[error("frontend runtime overrides require dynamic shape or override shape enabled graph")]
    FrontendRuntimeOverridesRequireShapeEnabledGraph,

    #[error("frontend cuda graph provided to populate must be empty")]
    FrontendCudaGraphPopulateRequiresEmptyGraph,

    #[error("frontend kernel cache requires dynamic shape enabled graph")]
    FrontendKernelCacheRequiresDynamicShape,

    #[error("frontend deviceless compilation requires device properties")]
    FrontendDevicelessRequiresDeviceProperties,

    #[error(
        "frontend feature `{feature}` requires cuDNN >= {min_version} (detected version {actual_version})"
    )]
    FrontendFeatureRequiresVersion {
        feature: String,
        min_version: String,
        actual_version: u64,
    },

    #[error("frontend feature `{feature}` is unavailable: {reason}")]
    FrontendFeatureUnavailable { feature: String, reason: String },

    #[error("frontend kernel cache lock poisoned")]
    FrontendKernelCacheLockPoisoned,

    #[error("frontend plan cache lock poisoned")]
    FrontendPlanCacheLockPoisoned,

    #[error("frontend operation output is missing")]
    FrontendOperationMissingOutput,

    #[error("frontend internal state `{name}` is missing")]
    FrontendInternalStateMissing { name: &'static str },

    #[error("frontend graph schema version mismatch: {actual} (expected: ${expected})")]
    FrontendGraphSchemaVersionMismatch { expected: String, actual: String },

    #[error("compiled graph has no plan choices")]
    FrontendGraphMissingPlanChoices,

    #[error("frontend compilation failed: {0}")]
    FrontendCompile(String),

    #[error("context stream is not set")]
    ContextStreamNotSet,

    #[error("no available engines")]
    NoAvailableEngines,

    #[error("descriptor mismatch for `{name}`")]
    DescriptorMismatch { name: String },

    #[error("stream belongs to a different cuda context")]
    StreamContextMismatch,

    #[error("cudnn contexts must match for `{name}`")]
    ContextMismatch { name: String },

    #[error("insufficient workspace size: {actual} bytes provided (required {required} bytes)")]
    InsufficientWorkspaceSize { required: usize, actual: usize },

    #[error("descriptor needs to be finalized")]
    DescriptorRequiredFinalized,

    #[error("descriptor already finalized")]
    DescriptorAlreadyFinalized,

    #[error("cannot modify finalized descriptor")]
    DescriptorCannotModifyFinalized,

    #[error("descriptor expected execution plan")]
    DescriptorExpectedExecutionPlan,

    #[error("descriptor expected variant pack")]
    DescriptorExpectedVariantPack,

    #[error("invalid backend descriptor type (expected {expected}, got {actual})")]
    InvalidBackendDescriptorType {
        expected: BackendDescriptorType,
        actual: BackendDescriptorType,
    },

    #[error("invalid backend descriptor count for `{name}`")]
    InvalidBackendDescriptorCount { name: String },

    #[error("attribute `{0}` not found")]
    DescriptorAttributeNotFound(BackendAttributeName),

    #[error("invalid attribute value for `{0}`: {1}")]
    DescriptorInvalidAttributeValue(BackendAttributeName, String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl From<sys::cudnnStatus_t> for Error {
    fn from(status: sys::cudnnStatus_t) -> Self {
        let code = Status::from(status);
        Self::from(code)
    }
}

impl From<Status> for Error {
    fn from(code: Status) -> Self {
        Self::Cudnn {
            code,
            message: code.description(),
        }
    }
}

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
    fn unknown_status_is_preserved_as_typed_cudnn_status() {
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
    fn known_status_round_trips_to_raw_status() {
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
