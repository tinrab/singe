use std::{
    ffi::{CStr, NulError},
    fmt,
};

use singe_cuda::{data_type::DataType, error::Error as CudaError};
#[cfg(feature = "mp")]
use singe_nccl::error::Error as NcclError;
use thiserror::Error;

use crate::{
    sys,
    types::{OperationDescriptorAttribute, PlanAttribute, PlanPreferenceAttribute},
};

/// cuTENSOR status returned by library calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Status {
    /// The operation completed successfully.
    Success,
    /// The opaque data structure was not initialized.
    NotInitialized,
    /// Resource allocation failed inside the cuTENSOR library.
    AllocFailed,
    /// An unsupported value or parameter was passed to the operation.
    InvalidValue,
    /// The device is not ready, or the target architecture is not supported.
    ArchitectureMismatch,
    /// An access to GPU memory space failed, which is usually caused by a failure to bind a texture.
    MappingError,
    /// The GPU program failed to execute.
    /// A kernel launch failure on the GPU is a common cause.
    ExecutionFailed,
    /// An internal cuTENSOR error occurred.
    InternalError,
    /// The requested operation is not supported.
    NotSupported,
    /// The requested operation requires a license, and an error was detected
    /// when checking the current licensing.
    LicenseError,
    /// A call to cuBLAS did not succeed.
    CublasError,
    /// An unknown CUDA error occurred.
    CudaError,
    /// The provided workspace was insufficient.
    InsufficientWorkspace,
    /// Indicates that the driver version is insufficient.
    InsufficientDriver,
    /// Indicates an error related to file I/O.
    IoError,
    /// A status code not known to this version of Singe.
    Unknown(u32),
}

impl Status {
    pub fn description(self) -> String {
        match sys::cutensorStatus_t::try_from(self.raw()) {
            Ok(status) => cutensor_error_string(status),
            Err(_) => String::from("unknown cutensor error"),
        }
    }

    pub const fn raw(self) -> u32 {
        match self {
            Self::Success => sys::cutensorStatus_t::CUTENSOR_STATUS_SUCCESS as _,
            Self::NotInitialized => sys::cutensorStatus_t::CUTENSOR_STATUS_NOT_INITIALIZED as _,
            Self::AllocFailed => sys::cutensorStatus_t::CUTENSOR_STATUS_ALLOC_FAILED as _,
            Self::InvalidValue => sys::cutensorStatus_t::CUTENSOR_STATUS_INVALID_VALUE as _,
            Self::ArchitectureMismatch => sys::cutensorStatus_t::CUTENSOR_STATUS_ARCH_MISMATCH as _,
            Self::MappingError => sys::cutensorStatus_t::CUTENSOR_STATUS_MAPPING_ERROR as _,
            Self::ExecutionFailed => sys::cutensorStatus_t::CUTENSOR_STATUS_EXECUTION_FAILED as _,
            Self::InternalError => sys::cutensorStatus_t::CUTENSOR_STATUS_INTERNAL_ERROR as _,
            Self::NotSupported => sys::cutensorStatus_t::CUTENSOR_STATUS_NOT_SUPPORTED as _,
            Self::LicenseError => sys::cutensorStatus_t::CUTENSOR_STATUS_LICENSE_ERROR as _,
            Self::CublasError => sys::cutensorStatus_t::CUTENSOR_STATUS_CUBLAS_ERROR as _,
            Self::CudaError => sys::cutensorStatus_t::CUTENSOR_STATUS_CUDA_ERROR as _,
            Self::InsufficientWorkspace => {
                sys::cutensorStatus_t::CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE as _
            }
            Self::InsufficientDriver => {
                sys::cutensorStatus_t::CUTENSOR_STATUS_INSUFFICIENT_DRIVER as _
            }
            Self::IoError => sys::cutensorStatus_t::CUTENSOR_STATUS_IO_ERROR as _,
            Self::Unknown(code) => code,
        }
    }
}

impl TryFrom<u32> for Status {
    type Error = u32;

    fn try_from(code: u32) -> std::result::Result<Self, u32> {
        match code {
            code if code == sys::cutensorStatus_t::CUTENSOR_STATUS_SUCCESS as u32 => {
                Ok(Self::Success)
            }
            code if code == sys::cutensorStatus_t::CUTENSOR_STATUS_NOT_INITIALIZED as u32 => {
                Ok(Self::NotInitialized)
            }
            code if code == sys::cutensorStatus_t::CUTENSOR_STATUS_ALLOC_FAILED as u32 => {
                Ok(Self::AllocFailed)
            }
            code if code == sys::cutensorStatus_t::CUTENSOR_STATUS_INVALID_VALUE as u32 => {
                Ok(Self::InvalidValue)
            }
            code if code == sys::cutensorStatus_t::CUTENSOR_STATUS_ARCH_MISMATCH as u32 => {
                Ok(Self::ArchitectureMismatch)
            }
            code if code == sys::cutensorStatus_t::CUTENSOR_STATUS_MAPPING_ERROR as u32 => {
                Ok(Self::MappingError)
            }
            code if code == sys::cutensorStatus_t::CUTENSOR_STATUS_EXECUTION_FAILED as u32 => {
                Ok(Self::ExecutionFailed)
            }
            code if code == sys::cutensorStatus_t::CUTENSOR_STATUS_INTERNAL_ERROR as u32 => {
                Ok(Self::InternalError)
            }
            code if code == sys::cutensorStatus_t::CUTENSOR_STATUS_NOT_SUPPORTED as u32 => {
                Ok(Self::NotSupported)
            }
            code if code == sys::cutensorStatus_t::CUTENSOR_STATUS_LICENSE_ERROR as u32 => {
                Ok(Self::LicenseError)
            }
            code if code == sys::cutensorStatus_t::CUTENSOR_STATUS_CUBLAS_ERROR as u32 => {
                Ok(Self::CublasError)
            }
            code if code == sys::cutensorStatus_t::CUTENSOR_STATUS_CUDA_ERROR as u32 => {
                Ok(Self::CudaError)
            }
            code if code
                == sys::cutensorStatus_t::CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE as u32 =>
            {
                Ok(Self::InsufficientWorkspace)
            }
            code if code == sys::cutensorStatus_t::CUTENSOR_STATUS_INSUFFICIENT_DRIVER as u32 => {
                Ok(Self::InsufficientDriver)
            }
            code if code == sys::cutensorStatus_t::CUTENSOR_STATUS_IO_ERROR as u32 => {
                Ok(Self::IoError)
            }
            code => Err(code),
        }
    }
}

impl From<sys::cutensorStatus_t> for Status {
    fn from(status: sys::cutensorStatus_t) -> Self {
        Self::try_from(status as u32).unwrap_or_else(Self::Unknown)
    }
}

impl TryFrom<Status> for sys::cutensorStatus_t {
    type Error = Status;

    fn try_from(status: Status) -> std::result::Result<Self, Status> {
        match status {
            Status::Success => Ok(Self::CUTENSOR_STATUS_SUCCESS),
            Status::NotInitialized => Ok(Self::CUTENSOR_STATUS_NOT_INITIALIZED),
            Status::AllocFailed => Ok(Self::CUTENSOR_STATUS_ALLOC_FAILED),
            Status::InvalidValue => Ok(Self::CUTENSOR_STATUS_INVALID_VALUE),
            Status::ArchitectureMismatch => Ok(Self::CUTENSOR_STATUS_ARCH_MISMATCH),
            Status::MappingError => Ok(Self::CUTENSOR_STATUS_MAPPING_ERROR),
            Status::ExecutionFailed => Ok(Self::CUTENSOR_STATUS_EXECUTION_FAILED),
            Status::InternalError => Ok(Self::CUTENSOR_STATUS_INTERNAL_ERROR),
            Status::NotSupported => Ok(Self::CUTENSOR_STATUS_NOT_SUPPORTED),
            Status::LicenseError => Ok(Self::CUTENSOR_STATUS_LICENSE_ERROR),
            Status::CublasError => Ok(Self::CUTENSOR_STATUS_CUBLAS_ERROR),
            Status::CudaError => Ok(Self::CUTENSOR_STATUS_CUDA_ERROR),
            Status::InsufficientWorkspace => Ok(Self::CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE),
            Status::InsufficientDriver => Ok(Self::CUTENSOR_STATUS_INSUFFICIENT_DRIVER),
            Status::IoError => Ok(Self::CUTENSOR_STATUS_IO_ERROR),
            Status::Unknown(_) => Err(status),
        }
    }
}

impl fmt::Display for Status {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Success => f.write_str("CUTENSOR_STATUS_SUCCESS"),
            Self::NotInitialized => f.write_str("CUTENSOR_STATUS_NOT_INITIALIZED"),
            Self::AllocFailed => f.write_str("CUTENSOR_STATUS_ALLOC_FAILED"),
            Self::InvalidValue => f.write_str("CUTENSOR_STATUS_INVALID_VALUE"),
            Self::ArchitectureMismatch => f.write_str("CUTENSOR_STATUS_ARCH_MISMATCH"),
            Self::MappingError => f.write_str("CUTENSOR_STATUS_MAPPING_ERROR"),
            Self::ExecutionFailed => f.write_str("CUTENSOR_STATUS_EXECUTION_FAILED"),
            Self::InternalError => f.write_str("CUTENSOR_STATUS_INTERNAL_ERROR"),
            Self::NotSupported => f.write_str("CUTENSOR_STATUS_NOT_SUPPORTED"),
            Self::LicenseError => f.write_str("CUTENSOR_STATUS_LICENSE_ERROR"),
            Self::CublasError => f.write_str("CUTENSOR_STATUS_CUBLAS_ERROR"),
            Self::CudaError => f.write_str("CUTENSOR_STATUS_CUDA_ERROR"),
            Self::InsufficientWorkspace => f.write_str("CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE"),
            Self::InsufficientDriver => f.write_str("CUTENSOR_STATUS_INSUFFICIENT_DRIVER"),
            Self::IoError => f.write_str("CUTENSOR_STATUS_IO_ERROR"),
            Self::Unknown(code) => write!(f, "UNKNOWN_CUTENSOR_STATUS({code})"),
        }
    }
}

fn cutensor_error_string(status: sys::cutensorStatus_t) -> String {
    unsafe {
        let ptr = sys::cutensorGetErrorString(status);
        if ptr.is_null() {
            String::from("unknown cutensor error")
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

    #[cfg(feature = "mp")]
    #[error("nccl error: {0}")]
    Nccl(#[from] NcclError),

    #[error("cutensor error ({code}): {message}")]
    Cutensor { code: Status, message: String },

    #[error("string contains interior nul byte")]
    InteriorNul,

    #[error("unexpected null handle")]
    NullHandle,

    #[error("handle is shared and cannot be consumed")]
    HandleShared,

    #[error("invalid `{name}` version")]
    InvalidVersion { name: &'static str },

    #[error("`{name}` is out of range")]
    OutOfRange { name: String },

    #[error("`{name}` length mismatch (expected {expected}, got {actual})")]
    LengthMismatch {
        name: String,
        expected: usize,
        actual: usize,
    },

    #[error("tensor mode length mismatch for tensor rank {rank} (got {mode_length})")]
    TensorModeMismatch { rank: u32, mode_length: usize },

    #[error("tensor stride length mismatch for tensor rank {rank} (got {stride_length})")]
    TensorStrideMismatch { rank: u32, stride_length: usize },

    #[error("tensor descriptors must match for `{name}`")]
    DescriptorMismatch { name: String },

    #[error("tensor modes must match for `{name}`")]
    ModeMismatch { name: String },

    #[error("tensor data type mismatch (descriptor is {descriptor} and memory is {memory})")]
    TensorMemoryDataTypeMismatch {
        descriptor: DataType,
        memory: DataType,
    },

    #[error("scalar data type mismatch (descriptor is {descriptor} and memory is {memory})")]
    ScalarDataTypeMismatch {
        descriptor: DataType,
        memory: DataType,
    },

    #[error("unsupported scalar data type {data_type} for `{name}`")]
    UnsupportedScalarDataType { name: String, data_type: DataType },

    #[error("`{name}` pointer count mismatch (expected {expected}, got {actual})")]
    PointerCountMismatch {
        name: String,
        expected: usize,
        actual: usize,
    },

    #[error("insufficient workspace size: {actual} bytes provided (required {required} bytes)")]
    InsufficientWorkspaceSize { required: u64, actual: u64 },

    #[error("workspace size is not aligned to {required_alignment} bytes (size is {size} bytes)")]
    WorkspaceMisaligned {
        required_alignment: u32,
        size: usize,
    },

    #[error("stream belongs to a different cuda context")]
    StreamContextMismatch,

    #[error("cutensor contexts must match for `{name}`")]
    ContextMismatch { name: String },

    #[error("plan operation kind mismatch (expected {expected}, got {actual})")]
    PlanOperationMismatch { expected: String, actual: String },

    #[error("cutensormg contexts must match for `{name}`")]
    MgContextMismatch { name: String },

    #[error("cutensormg host tensor memory is not supported by `{name}`")]
    MgHostTensorMemoryUnsupported { name: String },

    #[error("cutensormp contexts must match for `{name}`")]
    MpContextMismatch { name: String },

    #[error("invalid plan attribute size for {attr:?}: {actual} (expected {expected})")]
    PlanInvalidAttributeSize {
        attr: PlanAttribute,
        expected: usize,
        actual: usize,
    },

    #[error(
        "invalid operation descriptor attribute size for {attr:?}: {actual} (expected {expected})"
    )]
    OperationDescriptorInvalidAttributeSize {
        attr: OperationDescriptorAttribute,
        expected: usize,
        actual: usize,
    },

    #[error("invalid plan preference attribute size for {attr:?}: {actual} (expected {expected})")]
    PlanPreferenceInvalidAttributeSize {
        attr: PlanPreferenceAttribute,
        expected: usize,
        actual: usize,
    },
}

pub type Result<T> = std::result::Result<T, Error>;

impl From<sys::cutensorStatus_t> for Error {
    fn from(status: sys::cutensorStatus_t) -> Self {
        let code = Status::from(status);
        Self::from(code)
    }
}

impl From<Status> for Error {
    fn from(code: Status) -> Self {
        Self::Cutensor {
            code,
            message: code.description(),
        }
    }
}

impl From<NulError> for Error {
    fn from(_: NulError) -> Self {
        Self::InteriorNul
    }
}

#[macro_export]
macro_rules! try_ffi {
    ($expr:expr) => {{
        let status = { $expr };
        if status != singe_cutensor_sys::cutensorStatus_t::CUTENSOR_STATUS_SUCCESS {
            Err($crate::error::Error::from(status))
        } else {
            Ok(())
        }
    }};
}
