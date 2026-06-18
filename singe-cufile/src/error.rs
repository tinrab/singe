use std::{ffi::NulError, fmt, io};

use singe_cuda::error::Error as CudaError;
use singe_cufile_sys as sys;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Status {
    Success,
    DriverNotInitialized,
    DriverInvalidProperties,
    DriverUnsupportedLimit,
    DriverVersionMismatch,
    DriverVersionReadError,
    DriverClosing,
    PlatformNotSupported,
    IoNotSupported,
    DeviceNotSupported,
    NvfsDriverError,
    CudaDriverError,
    CudaPointerInvalid,
    CudaMemoryTypeInvalid,
    CudaPointerRangeError,
    CudaContextMismatch,
    InvalidMappingSize,
    InvalidMappingRange,
    InvalidFileType,
    InvalidFileOpenFlag,
    DirectIoNotSet,
    InvalidValue,
    MemoryAlreadyRegistered,
    MemoryNotRegistered,
    PermissionDenied,
    DriverAlreadyOpen,
    HandleNotRegistered,
    HandleAlreadyRegistered,
    DeviceNotFound,
    InternalError,
    GetNewFileDescriptorFailed,
    NvfsSetupError,
    IoDisabled,
    BatchSubmitFailed,
    GpuMemoryPinningFailed,
    BatchFull,
    AsyncNotSupported,
    InternalBatchSetupError,
    InternalBatchSubmitError,
    InternalBatchGetStatusError,
    InternalBatchCancelError,
    NoMemory,
    IoError,
    InternalBufferRegisterError,
    HashOperationError,
    InvalidContextError,
    NvfsInternalDriverError,
    BatchNoCompatError,
    IoMaxError,
    Unknown(u32),
}

impl Status {
    pub fn message(self) -> &'static str {
        match self {
            Self::Success => "cufile success",
            Self::DriverNotInitialized => "nvidia-fs driver is not loaded",
            Self::DriverInvalidProperties => "invalid property",
            Self::DriverUnsupportedLimit => "property range error",
            Self::DriverVersionMismatch => "nvidia-fs driver version mismatch",
            Self::DriverVersionReadError => "nvidia-fs driver version read error",
            Self::DriverClosing => "driver shutdown in progress",
            Self::PlatformNotSupported => "gpudirect storage not supported on current platform",
            Self::IoNotSupported => "gpudirect storage not supported on current file",
            Self::DeviceNotSupported => "gpudirect storage not supported on current gpu",
            Self::NvfsDriverError => "nvidia-fs driver ioctl error",
            Self::CudaDriverError => "cuda driver api error",
            Self::CudaPointerInvalid => "invalid device pointer",
            Self::CudaMemoryTypeInvalid => "invalid pointer memory type",
            Self::CudaPointerRangeError => "pointer range exceeds allocated address range",
            Self::CudaContextMismatch => "cuda context mismatch",
            Self::InvalidMappingSize => "access beyond maximum pinned size",
            Self::InvalidMappingRange => "access beyond mapped size",
            Self::InvalidFileType => "unsupported file type",
            Self::InvalidFileOpenFlag => "unsupported file open flags",
            Self::DirectIoNotSet => "fd direct io not set",
            Self::InvalidValue => "invalid arguments",
            Self::MemoryAlreadyRegistered => "device pointer already registered",
            Self::MemoryNotRegistered => "device pointer lookup failure",
            Self::PermissionDenied => "driver or file access error",
            Self::DriverAlreadyOpen => "driver is already open",
            Self::HandleNotRegistered => "file descriptor is not registered",
            Self::HandleAlreadyRegistered => "file descriptor is already registered",
            Self::DeviceNotFound => "gpu device not found",
            Self::InternalError => "internal error",
            Self::GetNewFileDescriptorFailed => "failed to obtain new file descriptor",
            Self::NvfsSetupError => "nvfs driver initialization error",
            Self::IoDisabled => "gpudirect storage disabled by config on current file",
            Self::BatchSubmitFailed => "failed to submit batch operation",
            Self::GpuMemoryPinningFailed => "failed to allocate pinned gpu memory",
            Self::BatchFull => "queue full for batch operation",
            Self::AsyncNotSupported => "cufile stream operation not supported",
            Self::InternalBatchSetupError => "batch setup internal error",
            Self::InternalBatchSubmitError => "batch submit internal error",
            Self::InternalBatchGetStatusError => "batch get status internal error",
            Self::InternalBatchCancelError => "batch cancel internal error",
            Self::NoMemory => "cufile no memory error",
            Self::IoError => "cufile io error",
            Self::InternalBufferRegisterError => "cufile buf registration error",
            Self::HashOperationError => "cufile hash operation error",
            Self::InvalidContextError => "cufile invalid context error",
            Self::NvfsInternalDriverError => "nvfs internal driver error",
            Self::BatchNoCompatError => "compat mode off error",
            Self::IoMaxError => "gpudirect storage max error",
            Self::Unknown(_) => "unknown cufile error",
        }
    }

    pub fn description(self) -> &'static str {
        self.message()
    }

    pub const fn raw(self) -> u32 {
        match self {
            Self::Success => sys::CUfileOpError::CU_FILE_SUCCESS as _,
            Self::DriverNotInitialized => sys::CUfileOpError::CU_FILE_DRIVER_NOT_INITIALIZED as _,
            Self::DriverInvalidProperties => sys::CUfileOpError::CU_FILE_DRIVER_INVALID_PROPS as _,
            Self::DriverUnsupportedLimit => {
                sys::CUfileOpError::CU_FILE_DRIVER_UNSUPPORTED_LIMIT as _
            }
            Self::DriverVersionMismatch => sys::CUfileOpError::CU_FILE_DRIVER_VERSION_MISMATCH as _,
            Self::DriverVersionReadError => {
                sys::CUfileOpError::CU_FILE_DRIVER_VERSION_READ_ERROR as _
            }
            Self::DriverClosing => sys::CUfileOpError::CU_FILE_DRIVER_CLOSING as _,
            Self::PlatformNotSupported => sys::CUfileOpError::CU_FILE_PLATFORM_NOT_SUPPORTED as _,
            Self::IoNotSupported => sys::CUfileOpError::CU_FILE_IO_NOT_SUPPORTED as _,
            Self::DeviceNotSupported => sys::CUfileOpError::CU_FILE_DEVICE_NOT_SUPPORTED as _,
            Self::NvfsDriverError => sys::CUfileOpError::CU_FILE_NVFS_DRIVER_ERROR as _,
            Self::CudaDriverError => sys::CUfileOpError::CU_FILE_CUDA_DRIVER_ERROR as _,
            Self::CudaPointerInvalid => sys::CUfileOpError::CU_FILE_CUDA_POINTER_INVALID as _,
            Self::CudaMemoryTypeInvalid => {
                sys::CUfileOpError::CU_FILE_CUDA_MEMORY_TYPE_INVALID as _
            }
            Self::CudaPointerRangeError => {
                sys::CUfileOpError::CU_FILE_CUDA_POINTER_RANGE_ERROR as _
            }
            Self::CudaContextMismatch => sys::CUfileOpError::CU_FILE_CUDA_CONTEXT_MISMATCH as _,
            Self::InvalidMappingSize => sys::CUfileOpError::CU_FILE_INVALID_MAPPING_SIZE as _,
            Self::InvalidMappingRange => sys::CUfileOpError::CU_FILE_INVALID_MAPPING_RANGE as _,
            Self::InvalidFileType => sys::CUfileOpError::CU_FILE_INVALID_FILE_TYPE as _,
            Self::InvalidFileOpenFlag => sys::CUfileOpError::CU_FILE_INVALID_FILE_OPEN_FLAG as _,
            Self::DirectIoNotSet => sys::CUfileOpError::CU_FILE_DIO_NOT_SET as _,
            Self::InvalidValue => sys::CUfileOpError::CU_FILE_INVALID_VALUE as _,
            Self::MemoryAlreadyRegistered => {
                sys::CUfileOpError::CU_FILE_MEMORY_ALREADY_REGISTERED as _
            }
            Self::MemoryNotRegistered => sys::CUfileOpError::CU_FILE_MEMORY_NOT_REGISTERED as _,
            Self::PermissionDenied => sys::CUfileOpError::CU_FILE_PERMISSION_DENIED as _,
            Self::DriverAlreadyOpen => sys::CUfileOpError::CU_FILE_DRIVER_ALREADY_OPEN as _,
            Self::HandleNotRegistered => sys::CUfileOpError::CU_FILE_HANDLE_NOT_REGISTERED as _,
            Self::HandleAlreadyRegistered => {
                sys::CUfileOpError::CU_FILE_HANDLE_ALREADY_REGISTERED as _
            }
            Self::DeviceNotFound => sys::CUfileOpError::CU_FILE_DEVICE_NOT_FOUND as _,
            Self::InternalError => sys::CUfileOpError::CU_FILE_INTERNAL_ERROR as _,
            Self::GetNewFileDescriptorFailed => sys::CUfileOpError::CU_FILE_GETNEWFD_FAILED as _,
            Self::NvfsSetupError => sys::CUfileOpError::CU_FILE_NVFS_SETUP_ERROR as _,
            Self::IoDisabled => sys::CUfileOpError::CU_FILE_IO_DISABLED as _,
            Self::BatchSubmitFailed => sys::CUfileOpError::CU_FILE_BATCH_SUBMIT_FAILED as _,
            Self::GpuMemoryPinningFailed => {
                sys::CUfileOpError::CU_FILE_GPU_MEMORY_PINNING_FAILED as _
            }
            Self::BatchFull => sys::CUfileOpError::CU_FILE_BATCH_FULL as _,
            Self::AsyncNotSupported => sys::CUfileOpError::CU_FILE_ASYNC_NOT_SUPPORTED as _,
            Self::InternalBatchSetupError => {
                sys::CUfileOpError::CU_FILE_INTERNAL_BATCH_SETUP_ERROR as _
            }
            Self::InternalBatchSubmitError => {
                sys::CUfileOpError::CU_FILE_INTERNAL_BATCH_SUBMIT_ERROR as _
            }
            Self::InternalBatchGetStatusError => {
                sys::CUfileOpError::CU_FILE_INTERNAL_BATCH_GETSTATUS_ERROR as _
            }
            Self::InternalBatchCancelError => {
                sys::CUfileOpError::CU_FILE_INTERNAL_BATCH_CANCEL_ERROR as _
            }
            Self::NoMemory => sys::CUfileOpError::CU_FILE_NOMEM_ERROR as _,
            Self::IoError => sys::CUfileOpError::CU_FILE_IO_ERROR as _,
            Self::InternalBufferRegisterError => {
                sys::CUfileOpError::CU_FILE_INTERNAL_BUF_REGISTER_ERROR as _
            }
            Self::HashOperationError => sys::CUfileOpError::CU_FILE_HASH_OPR_ERROR as _,
            Self::InvalidContextError => sys::CUfileOpError::CU_FILE_INVALID_CONTEXT_ERROR as _,
            Self::NvfsInternalDriverError => {
                sys::CUfileOpError::CU_FILE_NVFS_INTERNAL_DRIVER_ERROR as _
            }
            Self::BatchNoCompatError => sys::CUfileOpError::CU_FILE_BATCH_NOCOMPAT_ERROR as _,
            Self::IoMaxError => sys::CUfileOpError::CU_FILE_IO_MAX_ERROR as _,
            Self::Unknown(code) => code,
        }
    }
}

impl TryFrom<u32> for Status {
    type Error = u32;

    fn try_from(code: u32) -> std::result::Result<Self, Self::Error> {
        match code {
            x if x == sys::CUfileOpError::CU_FILE_SUCCESS as u32 => Ok(Self::Success),
            x if x == sys::CUfileOpError::CU_FILE_DRIVER_NOT_INITIALIZED as u32 => {
                Ok(Self::DriverNotInitialized)
            }
            x if x == sys::CUfileOpError::CU_FILE_DRIVER_INVALID_PROPS as u32 => {
                Ok(Self::DriverInvalidProperties)
            }
            x if x == sys::CUfileOpError::CU_FILE_DRIVER_UNSUPPORTED_LIMIT as u32 => {
                Ok(Self::DriverUnsupportedLimit)
            }
            x if x == sys::CUfileOpError::CU_FILE_DRIVER_VERSION_MISMATCH as u32 => {
                Ok(Self::DriverVersionMismatch)
            }
            x if x == sys::CUfileOpError::CU_FILE_DRIVER_VERSION_READ_ERROR as u32 => {
                Ok(Self::DriverVersionReadError)
            }
            x if x == sys::CUfileOpError::CU_FILE_DRIVER_CLOSING as u32 => Ok(Self::DriverClosing),
            x if x == sys::CUfileOpError::CU_FILE_PLATFORM_NOT_SUPPORTED as u32 => {
                Ok(Self::PlatformNotSupported)
            }
            x if x == sys::CUfileOpError::CU_FILE_IO_NOT_SUPPORTED as u32 => {
                Ok(Self::IoNotSupported)
            }
            x if x == sys::CUfileOpError::CU_FILE_DEVICE_NOT_SUPPORTED as u32 => {
                Ok(Self::DeviceNotSupported)
            }
            x if x == sys::CUfileOpError::CU_FILE_NVFS_DRIVER_ERROR as u32 => {
                Ok(Self::NvfsDriverError)
            }
            x if x == sys::CUfileOpError::CU_FILE_CUDA_DRIVER_ERROR as u32 => {
                Ok(Self::CudaDriverError)
            }
            x if x == sys::CUfileOpError::CU_FILE_CUDA_POINTER_INVALID as u32 => {
                Ok(Self::CudaPointerInvalid)
            }
            x if x == sys::CUfileOpError::CU_FILE_CUDA_MEMORY_TYPE_INVALID as u32 => {
                Ok(Self::CudaMemoryTypeInvalid)
            }
            x if x == sys::CUfileOpError::CU_FILE_CUDA_POINTER_RANGE_ERROR as u32 => {
                Ok(Self::CudaPointerRangeError)
            }
            x if x == sys::CUfileOpError::CU_FILE_CUDA_CONTEXT_MISMATCH as u32 => {
                Ok(Self::CudaContextMismatch)
            }
            x if x == sys::CUfileOpError::CU_FILE_INVALID_MAPPING_SIZE as u32 => {
                Ok(Self::InvalidMappingSize)
            }
            x if x == sys::CUfileOpError::CU_FILE_INVALID_MAPPING_RANGE as u32 => {
                Ok(Self::InvalidMappingRange)
            }
            x if x == sys::CUfileOpError::CU_FILE_INVALID_FILE_TYPE as u32 => {
                Ok(Self::InvalidFileType)
            }
            x if x == sys::CUfileOpError::CU_FILE_INVALID_FILE_OPEN_FLAG as u32 => {
                Ok(Self::InvalidFileOpenFlag)
            }
            x if x == sys::CUfileOpError::CU_FILE_DIO_NOT_SET as u32 => Ok(Self::DirectIoNotSet),
            x if x == sys::CUfileOpError::CU_FILE_INVALID_VALUE as u32 => Ok(Self::InvalidValue),
            x if x == sys::CUfileOpError::CU_FILE_MEMORY_ALREADY_REGISTERED as u32 => {
                Ok(Self::MemoryAlreadyRegistered)
            }
            x if x == sys::CUfileOpError::CU_FILE_MEMORY_NOT_REGISTERED as u32 => {
                Ok(Self::MemoryNotRegistered)
            }
            x if x == sys::CUfileOpError::CU_FILE_PERMISSION_DENIED as u32 => {
                Ok(Self::PermissionDenied)
            }
            x if x == sys::CUfileOpError::CU_FILE_DRIVER_ALREADY_OPEN as u32 => {
                Ok(Self::DriverAlreadyOpen)
            }
            x if x == sys::CUfileOpError::CU_FILE_HANDLE_NOT_REGISTERED as u32 => {
                Ok(Self::HandleNotRegistered)
            }
            x if x == sys::CUfileOpError::CU_FILE_HANDLE_ALREADY_REGISTERED as u32 => {
                Ok(Self::HandleAlreadyRegistered)
            }
            x if x == sys::CUfileOpError::CU_FILE_DEVICE_NOT_FOUND as u32 => {
                Ok(Self::DeviceNotFound)
            }
            x if x == sys::CUfileOpError::CU_FILE_INTERNAL_ERROR as u32 => Ok(Self::InternalError),
            x if x == sys::CUfileOpError::CU_FILE_GETNEWFD_FAILED as u32 => {
                Ok(Self::GetNewFileDescriptorFailed)
            }
            x if x == sys::CUfileOpError::CU_FILE_NVFS_SETUP_ERROR as u32 => {
                Ok(Self::NvfsSetupError)
            }
            x if x == sys::CUfileOpError::CU_FILE_IO_DISABLED as u32 => Ok(Self::IoDisabled),
            x if x == sys::CUfileOpError::CU_FILE_BATCH_SUBMIT_FAILED as u32 => {
                Ok(Self::BatchSubmitFailed)
            }
            x if x == sys::CUfileOpError::CU_FILE_GPU_MEMORY_PINNING_FAILED as u32 => {
                Ok(Self::GpuMemoryPinningFailed)
            }
            x if x == sys::CUfileOpError::CU_FILE_BATCH_FULL as u32 => Ok(Self::BatchFull),
            x if x == sys::CUfileOpError::CU_FILE_ASYNC_NOT_SUPPORTED as u32 => {
                Ok(Self::AsyncNotSupported)
            }
            x if x == sys::CUfileOpError::CU_FILE_INTERNAL_BATCH_SETUP_ERROR as u32 => {
                Ok(Self::InternalBatchSetupError)
            }
            x if x == sys::CUfileOpError::CU_FILE_INTERNAL_BATCH_SUBMIT_ERROR as u32 => {
                Ok(Self::InternalBatchSubmitError)
            }
            x if x == sys::CUfileOpError::CU_FILE_INTERNAL_BATCH_GETSTATUS_ERROR as u32 => {
                Ok(Self::InternalBatchGetStatusError)
            }
            x if x == sys::CUfileOpError::CU_FILE_INTERNAL_BATCH_CANCEL_ERROR as u32 => {
                Ok(Self::InternalBatchCancelError)
            }
            x if x == sys::CUfileOpError::CU_FILE_NOMEM_ERROR as u32 => Ok(Self::NoMemory),
            x if x == sys::CUfileOpError::CU_FILE_IO_ERROR as u32 => Ok(Self::IoError),
            x if x == sys::CUfileOpError::CU_FILE_INTERNAL_BUF_REGISTER_ERROR as u32 => {
                Ok(Self::InternalBufferRegisterError)
            }
            x if x == sys::CUfileOpError::CU_FILE_HASH_OPR_ERROR as u32 => {
                Ok(Self::HashOperationError)
            }
            x if x == sys::CUfileOpError::CU_FILE_INVALID_CONTEXT_ERROR as u32 => {
                Ok(Self::InvalidContextError)
            }
            x if x == sys::CUfileOpError::CU_FILE_NVFS_INTERNAL_DRIVER_ERROR as u32 => {
                Ok(Self::NvfsInternalDriverError)
            }
            x if x == sys::CUfileOpError::CU_FILE_BATCH_NOCOMPAT_ERROR as u32 => {
                Ok(Self::BatchNoCompatError)
            }
            x if x == sys::CUfileOpError::CU_FILE_IO_MAX_ERROR as u32 => Ok(Self::IoMaxError),
            code => Err(code),
        }
    }
}

impl TryFrom<Status> for sys::CUfileOpError {
    type Error = Status;

    fn try_from(status: Status) -> std::result::Result<Self, Self::Error> {
        match status {
            Status::Unknown(_) => Err(status),
            _ => Self::try_from(status.raw()).map_err(|_| status),
        }
    }
}

impl fmt::Display for Status {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unknown(code) => write!(f, "UNKNOWN_CUFILE_STATUS({code})"),
            _ => write!(f, "{:?}", sys::CUfileOpError::try_from(self.raw()).unwrap()),
        }
    }
}

#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    #[error("cuda error: {0}")]
    Cuda(#[from] CudaError),

    #[error("cufile error ({code}): {message}")]
    Cufile {
        code: Status,
        cuda_code: Option<u32>,
        message: String,
    },

    #[error("string contains interior nul byte")]
    InteriorNul,

    #[error("io error: {0}")]
    Io(#[from] io::Error),

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

    #[error("cufile io operation failed with code {0}")]
    OperationFailed(isize),

    #[error("request batch is too big")]
    RequestBatchTooBig,
}

pub type Result<T> = std::result::Result<T, Error>;

impl From<NulError> for Error {
    fn from(_: NulError) -> Self {
        Self::InteriorNul
    }
}

impl From<sys::CUfileError_t> for Error {
    fn from(status: sys::CUfileError_t) -> Self {
        let code =
            Status::try_from(status.err as u32).unwrap_or(Status::Unknown(status.err as u32));
        let cuda_code = if code == Status::CudaDriverError {
            Some(status.cu_err as u32)
        } else {
            None
        };

        Self::Cufile {
            code,
            cuda_code,
            message: code.message().into(),
        }
    }
}

impl From<Status> for Error {
    fn from(status: Status) -> Self {
        match sys::CUfileOpError::try_from(status) {
            Ok(err) => sys::CUfileError_t {
                err,
                ..Default::default()
            }
            .into(),
            Err(code) => Self::Cufile {
                code,
                cuda_code: None,
                message: code.message().into(),
            },
        }
    }
}

#[macro_export]
macro_rules! try_ffi {
    ($expr:expr) => {{
        let status = { $expr };
        if status.err != singe_cufile_sys::CUfileOpError::CU_FILE_SUCCESS {
            Err($crate::error::Error::from(status))
        } else {
            Ok(())
        }
    }};
}
