use std::{borrow::Cow, ffi::NulError, io, result::Result as StdResult};

use singe_core::status_enum;
use singe_cuda::error::{Error as CudaError, Status as CudaStatus};
use thiserror::Error;

use crate::{sys, types::BatchStatus};

status_enum! {
    /// A cuFile library operation status.
    ///
    /// [`Error::Cufile`] carries this status and, for [`Self::CudaDriverError`], an accompanying CUDA driver status.
    /// Data-path errors may instead be reported as operating-system I/O errors.
    pub enum Status: u32 => sys::CUfileOpError, unknown = "UNKNOWN_CUFILE_STATUS", unknown_description = "unknown cufile error" {
        /// The cufile is successful.
        Success = CU_FILE_SUCCESS => "cufile success",
        /// The nvidia-fs driver is not loaded.
        DriverNotInitialized = CU_FILE_DRIVER_NOT_INITIALIZED => "nvidia-fs driver is not loaded",
        /// An invalid property.
        DriverInvalidProperties = CU_FILE_DRIVER_INVALID_PROPS => "invalid property",
        /// A property range error.
        DriverUnsupportedLimit = CU_FILE_DRIVER_UNSUPPORTED_LIMIT => "property range error",
        /// An nvidia-fs driver version mismatch.
        DriverVersionMismatch = CU_FILE_DRIVER_VERSION_MISMATCH => "nvidia-fs driver version mismatch",
        /// An nvidia-fs driver version read error.
        DriverVersionReadError = CU_FILE_DRIVER_VERSION_READ_ERROR => "nvidia-fs driver version read error",
        /// Driver shutdown in progress.
        DriverClosing = CU_FILE_DRIVER_CLOSING => "driver shutdown in progress",
        /// GDS is not supported on the current platform.
        PlatformNotSupported = CU_FILE_PLATFORM_NOT_SUPPORTED => "gpudirect storage not supported on current platform",
        /// GDS is not supported on the current file.
        IoNotSupported = CU_FILE_IO_NOT_SUPPORTED => "gpudirect storage not supported on current file",
        /// GDS is not supported on the current GPU.
        DeviceNotSupported = CU_FILE_DEVICE_NOT_SUPPORTED => "gpudirect storage not supported on current gpu",
        /// An nvidia-fs driver ioctl error.
        NvfsDriverError = CU_FILE_NVFS_DRIVER_ERROR => "nvidia-fs driver ioctl error",
        /// A CUDA driver API error.
        /// [`Error::Cufile`] carries the CUDA driver status in its `cuda_code` field.
        CudaDriverError = CU_FILE_CUDA_DRIVER_ERROR => "cuda driver api error",
        /// An invalid device pointer.
        CudaPointerInvalid = CU_FILE_CUDA_POINTER_INVALID => "invalid device pointer",
        /// An invalid pointer memory type.
        CudaMemoryTypeInvalid = CU_FILE_CUDA_MEMORY_TYPE_INVALID => "invalid pointer memory type",
        /// The pointer range exceeds the allocated address range.
        CudaPointerRangeError = CU_FILE_CUDA_POINTER_RANGE_ERROR => "pointer range exceeds allocated address range",
        /// A CUDA context mismatch.
        CudaContextMismatch = CU_FILE_CUDA_CONTEXT_MISMATCH => "cuda context mismatch",
        /// Access beyond the maximum pinned memory size.
        InvalidMappingSize = CU_FILE_INVALID_MAPPING_SIZE => "access beyond maximum pinned size",
        /// Access beyond the mapped size.
        InvalidMappingRange = CU_FILE_INVALID_MAPPING_RANGE => "access beyond mapped size",
        /// An unsupported file type.
        InvalidFileType = CU_FILE_INVALID_FILE_TYPE => "unsupported file type",
        /// Unsupported file open flags.
        InvalidFileOpenFlag = CU_FILE_INVALID_FILE_OPEN_FLAG => "unsupported file open flags",
        /// The fd direct IO is not set.
        DirectIoNotSet = CU_FILE_DIO_NOT_SET => "fd direct io not set",
        /// Invalid API arguments.
        InvalidValue = CU_FILE_INVALID_VALUE => "invalid arguments",
        /// Device pointer is already registered.
        MemoryAlreadyRegistered = CU_FILE_MEMORY_ALREADY_REGISTERED => "device pointer already registered",
        /// A device pointer lookup failure has occurred.
        MemoryNotRegistered = CU_FILE_MEMORY_NOT_REGISTERED => "device pointer lookup failure",
        /// A driver or file access error.
        PermissionDenied = CU_FILE_PERMISSION_DENIED => "driver or file access error",
        /// The driver is already open.
        DriverAlreadyOpen = CU_FILE_DRIVER_ALREADY_OPEN => "driver is already open",
        /// The file descriptor is not registered.
        HandleNotRegistered = CU_FILE_HANDLE_NOT_REGISTERED => "file descriptor is not registered",
        /// The file descriptor is already registered.
        HandleAlreadyRegistered = CU_FILE_HANDLE_ALREADY_REGISTERED => "file descriptor is already registered",
        /// The GPU device cannot be not found.
        DeviceNotFound = CU_FILE_DEVICE_NOT_FOUND => "gpu device not found",
        /// An internal error has occurred.
        /// Refer to `cufile.log` for more details.
        InternalError = CU_FILE_INTERNAL_ERROR => "internal error",
        /// Failed to obtain a new file descriptor.
        GetNewFileDescriptorFailed = CU_FILE_GETNEWFD_FAILED => "failed to obtain new file descriptor",
        /// An NVFS driver initialization error has occurred.
        NvfsSetupError = CU_FILE_NVFS_SETUP_ERROR => "nvfs driver initialization error",
        /// GDS is disabled by config on the current file.
        IoDisabled = CU_FILE_IO_DISABLED => "gpudirect storage disabled by config on current file",
        /// Failed to submit a batch operation.
        BatchSubmitFailed = CU_FILE_BATCH_SUBMIT_FAILED => "failed to submit batch operation",
        /// Failed to allocate pinned GPU memory.
        GpuMemoryPinningFailed = CU_FILE_GPU_MEMORY_PINNING_FAILED => "failed to allocate pinned gpu memory",
        /// Queue full for batch operation.
        BatchFull = CU_FILE_BATCH_FULL => "queue full for batch operation",
        /// cuFile stream operation is not supported.
        AsyncNotSupported = CU_FILE_ASYNC_NOT_SUPPORTED => "cufile stream operation not supported",
        /// batch setup internal error - retry later
        InternalBatchSetupError = CU_FILE_INTERNAL_BATCH_SETUP_ERROR => "batch setup internal error",
        /// batch submit internal error - retry later
        InternalBatchSubmitError = CU_FILE_INTERNAL_BATCH_SUBMIT_ERROR => "batch submit internal error",
        /// batch get status internal error - retry later
        InternalBatchGetStatusError = CU_FILE_INTERNAL_BATCH_GETSTATUS_ERROR => "batch get status internal error",
        /// batch cancel internal error - retry later
        InternalBatchCancelError = CU_FILE_INTERNAL_BATCH_CANCEL_ERROR => "batch cancel internal error",
        /// cufile no memory error - retry later
        NoMemory = CU_FILE_NOMEM_ERROR => "cufile no memory error",
        /// cufile io error
        IoError = CU_FILE_IO_ERROR => "cufile io error",
        /// cufile buf registration error
        InternalBufferRegisterError = CU_FILE_INTERNAL_BUF_REGISTER_ERROR => "cufile buf registration error",
        /// cufile hash operation error
        HashOperationError = CU_FILE_HASH_OPR_ERROR => "cufile hash operation error",
        /// cufile invalid context error
        InvalidContextError = CU_FILE_INVALID_CONTEXT_ERROR => "cufile invalid context error",
        /// nvfs internal driver error
        NvfsInternalDriverError = CU_FILE_NVFS_INTERNAL_DRIVER_ERROR => "nvfs internal driver error",
        /// compat mode off error
        BatchNoCompatError = CU_FILE_BATCH_NOCOMPAT_ERROR => "compat mode off error",
        /// Upper marker of the cuFile operation error-code range.
        IoMaxError = CU_FILE_IO_MAX_ERROR => "gpudirect storage max error",
    }
}

/// An error from CUDA, cuFile, the operating system, or wrapper argument validation.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    /// cuFile or one of its symbols would not load.
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

    /// A cuFile operation returned a failure status.
    #[error("cufile error ({code}): {}", code.description())]
    Cufile {
        /// The cuFile status code.
        code: Status,
        /// The accompanying CUDA status when `code` is [`Status::CudaDriverError`].
        cuda_code: Option<CudaStatus>,
    },

    /// A string contains an interior NUL byte.
    #[error("string contains interior nul byte")]
    InteriorNul,

    /// An operating-system I/O operation failed.
    #[error(transparent)]
    Io(#[from] io::Error),

    /// A native handle or buffer address is null.
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

    /// An argument, computed size, or native result is outside its supported range.
    #[error("`{name}` is out of range")]
    OutOfRange {
        /// The argument or quantity outside its supported range.
        name: Cow<'static, str>,
    },

    /// A negative I/O result cannot be decoded as a cuFile or operating-system error.
    #[error("`{operation}` failed with {code}")]
    OperationFailed {
        /// Name of the cuFile operation.
        operation: Cow<'static, str>,
        /// Negative native result of the operation.
        code: i64,
    },

    /// The requested batch is empty at creation or exceeds the supported capacity.
    #[error("request batch is too big")]
    RequestBatchTooBig,

    /// The batch event has no successful byte count for its status.
    #[error("cufile batch operation ended with status {0}")]
    BatchOperationFailed(BatchStatus),
}

/// A cuFile wrapper result with [`enum@Error`] as its error type.
pub type Result<T> = StdResult<T, Error>;

impl Error {
    pub(crate) fn out_of_range(name: impl Into<Cow<'static, str>>) -> Self {
        Self::OutOfRange { name: name.into() }
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

impl From<sys::CUfileError_t> for Error {
    fn from(status: sys::CUfileError_t) -> Self {
        if let Some(absence) = sys::take_absence() {
            return Self::Load(absence.into_error());
        }

        let code = Status::from(status.err);
        let cuda_code = (code == Status::CudaDriverError).then(|| CudaStatus::from(status.cu_err));
        Self::Cufile { code, cuda_code }
    }
}

impl From<Status> for Error {
    fn from(code: Status) -> Self {
        Self::Cufile {
            code,
            cuda_code: None,
        }
    }
}

impl Error {
    pub(crate) fn check(status: sys::CUfileError_t) -> Result<()> {
        if status.err == sys::CUfileOpError::CU_FILE_SUCCESS {
            Ok(())
        } else {
            Err(Self::from(status))
        }
    }
}

// Converts a raw cuFile status into this crate's `Result`.
macro_rules! try_ffi {
    ($expr:expr) => {
        $crate::error::Error::check($expr)
    };
}

pub(crate) use try_ffi;
