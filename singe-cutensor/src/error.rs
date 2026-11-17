use std::{borrow::Cow, ffi::NulError, result::Result as StdResult};

use singe_cuda::{data_type::DataType, error::Error as CudaError};
#[cfg(feature = "mp")]
use singe_nccl::error::Error as NcclError;
use thiserror::Error;

use crate::sys;

singe_core::wrapper_status! {
    library = "cutensor",
    error = Error::Cutensor,
    raw = sys::cutensorStatus_t,
    success = CUTENSOR_STATUS_SUCCESS,
    absence = sys::take_absence,
    unknown = "UNKNOWN_CUTENSOR_STATUS",
    description = sys::cutensorGetErrorString,
    /// cuTENSOR status returned by library calls.
    pub enum Status: u32 {
        /// The operation completed successfully.
        Success = CUTENSOR_STATUS_SUCCESS,
        /// The opaque data structure was not initialized.
        NotInitialized = CUTENSOR_STATUS_NOT_INITIALIZED,
        /// Resource allocation failed inside the cuTENSOR library.
        AllocFailed = CUTENSOR_STATUS_ALLOC_FAILED,
        /// An unsupported value or parameter was passed to the operation.
        InvalidValue = CUTENSOR_STATUS_INVALID_VALUE,
        /// The device is not ready, or the target architecture is not supported.
        ArchitectureMismatch = CUTENSOR_STATUS_ARCH_MISMATCH,
        /// An access to GPU memory space failed, which is usually caused by a failure to bind a texture.
        MappingError = CUTENSOR_STATUS_MAPPING_ERROR,
        /// The GPU program failed to execute.
        /// A kernel launch failure on the GPU is a common cause.
        ExecutionFailed = CUTENSOR_STATUS_EXECUTION_FAILED,
        /// An internal cuTENSOR error occurred.
        InternalError = CUTENSOR_STATUS_INTERNAL_ERROR,
        /// The requested operation is not supported.
        NotSupported = CUTENSOR_STATUS_NOT_SUPPORTED,
        /// The requested operation requires a license, and an error was detected when checking the current licensing.
        LicenseError = CUTENSOR_STATUS_LICENSE_ERROR,
        /// A call to cuBLAS did not succeed.
        CublasError = CUTENSOR_STATUS_CUBLAS_ERROR,
        /// An unknown CUDA error occurred.
        CudaError = CUTENSOR_STATUS_CUDA_ERROR,
        /// The provided workspace was insufficient.
        InsufficientWorkspace = CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE,
        /// Indicates that the driver version is insufficient.
        InsufficientDriver = CUTENSOR_STATUS_INSUFFICIENT_DRIVER,
        /// Indicates an error related to file I/O.
        IoError = CUTENSOR_STATUS_IO_ERROR,
    }
}

/// Errors from cuTENSOR, CUDA, NCCL, and checked wrapper validation.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    /// cuTENSOR or one of its symbols would not load.
    #[error(transparent)]
    Load(#[from] sys::LoadError),

    /// A lock was poisoned by a thread that panicked while holding it.
    #[error("`{name}` lock is poisoned")]
    LockPoisoned {
        /// What the lock guards.
        name: Cow<'static, str>,
    },

    /// A CUDA operation failed.
    #[error(transparent)]
    Cuda(#[from] CudaError),

    /// An NCCL operation used by multi-process execution failed.
    #[cfg(feature = "mp")]
    #[error(transparent)]
    Nccl(#[from] NcclError),

    /// A native cuTENSOR call returned an error status.
    #[error("cutensor error ({code}): {}", code.description())]
    Cutensor {
        /// Native status returned by the failed call.
        code: Status,
    },

    /// A string passed to a native API contains an interior nul byte.
    #[error("string contains interior nul byte")]
    InteriorNul,

    /// A native resource handle is unexpectedly null.
    #[error("unexpected null handle")]
    NullHandle,

    /// A dependent resource still retains ownership required to be unique.
    #[error("`{name}` is still shared")]
    OwnershipShared {
        /// Name of the shared resource.
        name: Cow<'static, str>,
    },

    /// A native library or CUDA runtime version is invalid.
    #[error("invalid `{name}` version")]
    InvalidVersion {
        /// Name of the invalid argument, resource, or operation.
        name: &'static str,
    },

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
        /// Name of the invalid argument, resource, or operation.
        name: Cow<'static, str>,
    },

    /// Related slices or pointer lists have different lengths.
    #[error("`{name}` length mismatch (expected {expected}, got {actual})")]
    LengthMismatch {
        /// Name of the invalid argument, resource, or operation.
        name: Cow<'static, str>,
        /// Required number of entries.
        expected: usize,
        /// Observed number of entries.
        actual: usize,
    },

    /// An attribute buffer has the wrong byte size.
    #[error("`{name}` size mismatch (expected {expected} bytes, got {actual})")]
    SizeMismatch {
        /// Name of the attribute.
        name: Cow<'static, str>,
        /// Required byte count.
        expected: usize,
        /// Observed byte count.
        actual: usize,
    },

    /// A tensor's mode-label count does not equal its rank.
    #[error("tensor mode length mismatch for tensor rank {rank} (got {mode_length})")]
    TensorModeMismatch {
        /// Required tensor rank or layout entry count.
        rank: u32,
        /// Number of supplied mode labels.
        mode_length: usize,
    },

    /// A tensor's stride count does not match the required layout.
    #[error("tensor stride length mismatch for tensor rank {rank} (got {stride_length})")]
    TensorStrideMismatch {
        /// Required tensor rank or layout entry count.
        rank: u32,
        /// Number of supplied strides.
        stride_length: usize,
    },

    /// Tensor descriptors that must agree describe different layouts.
    #[error("tensor descriptors must match for `{name}`")]
    DescriptorMismatch {
        /// Name of the invalid argument, resource, or operation.
        name: Cow<'static, str>,
    },

    /// Tensor mode labels that must agree do not match.
    #[error("tensor modes must match for `{name}`")]
    ModeMismatch {
        /// Name of the invalid argument, resource, or operation.
        name: Cow<'static, str>,
    },

    /// The tensor descriptor and sample allocation have different data types.
    #[error("tensor data type mismatch (descriptor is {descriptor} and memory is {memory})")]
    TensorMemoryDataTypeMismatch {
        /// Data type required by the descriptor.
        descriptor: DataType,
        /// Data type of the supplied memory or scalar.
        memory: DataType,
    },

    /// The operation's scalar type and supplied scalar representation differ.
    #[error("scalar data type mismatch (descriptor is {descriptor} and memory is {memory})")]
    ScalarDataTypeMismatch {
        /// Data type required by the descriptor.
        descriptor: DataType,
        /// Data type of the supplied memory or scalar.
        memory: DataType,
    },

    /// The operation does not support the supplied scalar data type.
    #[error("unsupported scalar data type {data_type} for `{name}`")]
    UnsupportedScalarDataType {
        /// Name of the invalid argument, resource, or operation.
        name: Cow<'static, str>,
        /// Unsupported scalar data type.
        data_type: DataType,
    },

    /// A buffer or workspace is smaller than the storage the operation requires.
    #[error("`{name}` is too small (requires {required}, got {actual})")]
    BufferTooSmall {
        /// Name of the undersized buffer.
        name: Cow<'static, str>,
        /// Required size in bytes.
        required: usize,
        /// Supplied size in bytes.
        actual: usize,
    },

    /// The stream belongs to another CUDA context.
    #[error("stream belongs to a different cuda context")]
    StreamContextMismatch,

    /// Resources belong to different cuTENSOR, cuTENSORMg, or cuTENSORMp handles.
    #[error("`{name}` belongs to a different cutensor context")]
    ContextMismatch {
        /// Name of the invalid argument, resource, or operation.
        name: Cow<'static, str>,
    },

    /// The plan was created for a different operation kind.
    #[error("plan operation kind mismatch (expected {expected}, got {actual})")]
    PlanOperationMismatch {
        /// Required operation kind.
        expected: Cow<'static, str>,
        /// Observed operation kind.
        actual: Cow<'static, str>,
    },

    /// A device does not participate in the multi-GPU context.
    #[error("device {device} does not participate in the cutensormg context")]
    MgDeviceNotInContext {
        /// Device ordinal absent from the multi-GPU context.
        device: i32,
    },

    /// Distributed memory places its partitions on other devices than its tensor descriptor.
    #[error("`{name}` memory devices differ from its cutensormg tensor descriptor")]
    MgDeviceMismatch {
        /// Name of the mismatched operand.
        name: Cow<'static, str>,
    },

    /// A tensor layout relies on a native default stride, so its storage span cannot be checked.
    #[error("cannot bound the storage span of `{name}` without both element and block strides")]
    MgUncheckedLayout {
        /// Name of the operand whose layout cannot be checked.
        name: Cow<'static, str>,
    },

    /// An operation does not support multi-GPU tensor storage on the host.
    #[error("cutensormg host tensor memory is not supported by `{name}`")]
    MgHostTensorMemoryUnsupported {
        /// Name of the invalid argument, resource, or operation.
        name: Cow<'static, str>,
    },
}

/// Result of a cuTENSOR wrapper operation.
pub type Result<T> = StdResult<T, Error>;

impl Error {
    pub(crate) fn out_of_range(name: impl Into<Cow<'static, str>>) -> Self {
        Self::OutOfRange { name: name.into() }
    }

    // Native workspace sizes are `u64`; a size beyond `usize` is reported as `usize::MAX`.
    #[cfg(feature = "mp")]
    pub(crate) fn workspace_too_small(
        name: impl Into<Cow<'static, str>>,
        required: u64,
        actual: u64,
    ) -> Self {
        Self::BufferTooSmall {
            name: name.into(),
            required: usize::try_from(required).unwrap_or(usize::MAX),
            actual: usize::try_from(actual).unwrap_or(usize::MAX),
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

impl From<NulError> for Error {
    fn from(_: NulError) -> Self {
        Self::InteriorNul
    }
}

// Converts a raw cuTENSOR status into this crate's `Result`.
macro_rules! try_ffi {
    ($expr:expr) => {
        $crate::error::Error::check($expr)
    };
}

pub(crate) use try_ffi;
