use std::{borrow::Cow, ffi::NulError, io, result::Result as StdResult};

use singe_core::status_enum;
use singe_cuda::error::Error as CudaError;
use singe_cufile_sys as sys;
use thiserror::Error;

use crate::types::BatchStatus;

status_enum! {
    /// A cuFile library operation status.
    ///
    /// [`Error::Cufile`] carries this status and, for [`Self::CudaDriverError`],
    /// an accompanying CUDA driver status. Data-path errors may instead be
    /// reported as operating-system I/O errors.
    pub enum Status: u32 => sys::CUfileOpError, unknown = "UNKNOWN_CUFILE_STATUS" {
        /// The cufile is successful.
        Success = CU_FILE_SUCCESS,
        /// The nvidia-fs driver is not loaded.
        DriverNotInitialized = CU_FILE_DRIVER_NOT_INITIALIZED,
        /// An invalid property.
        DriverInvalidProperties = CU_FILE_DRIVER_INVALID_PROPS,
        /// A property range error.
        DriverUnsupportedLimit = CU_FILE_DRIVER_UNSUPPORTED_LIMIT,
        /// An nvidia-fs driver version mismatch.
        DriverVersionMismatch = CU_FILE_DRIVER_VERSION_MISMATCH,
        /// An nvidia-fs driver version read error.
        DriverVersionReadError = CU_FILE_DRIVER_VERSION_READ_ERROR,
        /// Driver shutdown in progress.
        DriverClosing = CU_FILE_DRIVER_CLOSING,
        /// GDS is not supported on the current platform.
        PlatformNotSupported = CU_FILE_PLATFORM_NOT_SUPPORTED,
        /// GDS is not supported on the current file.
        IoNotSupported = CU_FILE_IO_NOT_SUPPORTED,
        /// GDS is not supported on the current GPU.
        DeviceNotSupported = CU_FILE_DEVICE_NOT_SUPPORTED,
        /// An nvidia-fs driver ioctl error.
        NvfsDriverError = CU_FILE_NVFS_DRIVER_ERROR,
        /// A CUDA Driver API error. This error indicates a CUDA driver-api error. If this is set, a CUDA-specific error code is set in the cu_err field for cuFileError.
        CudaDriverError = CU_FILE_CUDA_DRIVER_ERROR,
        /// An invalid device pointer.
        CudaPointerInvalid = CU_FILE_CUDA_POINTER_INVALID,
        /// An invalid pointer memory type.
        CudaMemoryTypeInvalid = CU_FILE_CUDA_MEMORY_TYPE_INVALID,
        /// The pointer range exceeds the allocated address range.
        CudaPointerRangeError = CU_FILE_CUDA_POINTER_RANGE_ERROR,
        /// A CUDA context mismatch.
        CudaContextMismatch = CU_FILE_CUDA_CONTEXT_MISMATCH,
        /// Access beyond the maximum pinned memory size.
        InvalidMappingSize = CU_FILE_INVALID_MAPPING_SIZE,
        /// Access beyond the mapped size.
        InvalidMappingRange = CU_FILE_INVALID_MAPPING_RANGE,
        /// An unsupported file type.
        InvalidFileType = CU_FILE_INVALID_FILE_TYPE,
        /// Unsupported file open flags.
        InvalidFileOpenFlag = CU_FILE_INVALID_FILE_OPEN_FLAG,
        /// The fd direct IO is not set.
        DirectIoNotSet = CU_FILE_DIO_NOT_SET,
        /// Invalid API arguments.
        InvalidValue = CU_FILE_INVALID_VALUE,
        /// Device pointer is already registered.
        MemoryAlreadyRegistered = CU_FILE_MEMORY_ALREADY_REGISTERED,
        /// A device pointer lookup failure has occurred.
        MemoryNotRegistered = CU_FILE_MEMORY_NOT_REGISTERED,
        /// A driver or file access error.
        PermissionDenied = CU_FILE_PERMISSION_DENIED,
        /// The driver is already open.
        DriverAlreadyOpen = CU_FILE_DRIVER_ALREADY_OPEN,
        /// The file descriptor is not registered.
        HandleNotRegistered = CU_FILE_HANDLE_NOT_REGISTERED,
        /// The file descriptor is already registered.
        HandleAlreadyRegistered = CU_FILE_HANDLE_ALREADY_REGISTERED,
        /// The GPU device cannot be not found.
        DeviceNotFound = CU_FILE_DEVICE_NOT_FOUND,
        /// An internal error has occurred. Refer to `cufile.log` for more details.
        InternalError = CU_FILE_INTERNAL_ERROR,
        /// Failed to obtain a new file descriptor.
        GetNewFileDescriptorFailed = CU_FILE_GETNEWFD_FAILED,
        /// An NVFS driver initialization error has occurred.
        NvfsSetupError = CU_FILE_NVFS_SETUP_ERROR,
        /// GDS is disabled by config on the current file.
        IoDisabled = CU_FILE_IO_DISABLED,
        /// Failed to submit a batch operation.
        BatchSubmitFailed = CU_FILE_BATCH_SUBMIT_FAILED,
        /// Failed to allocate pinned GPU memory.
        GpuMemoryPinningFailed = CU_FILE_GPU_MEMORY_PINNING_FAILED,
        /// Queue full for batch operation.
        BatchFull = CU_FILE_BATCH_FULL,
        /// cuFile stream operation is not supported.
        AsyncNotSupported = CU_FILE_ASYNC_NOT_SUPPORTED,
        /// batch setup internal error - retry later
        InternalBatchSetupError = CU_FILE_INTERNAL_BATCH_SETUP_ERROR,
        /// batch submit internal error - retry later
        InternalBatchSubmitError = CU_FILE_INTERNAL_BATCH_SUBMIT_ERROR,
        /// batch get status internal error - retry later
        InternalBatchGetStatusError = CU_FILE_INTERNAL_BATCH_GETSTATUS_ERROR,
        /// batch cancel internal error - retry later
        InternalBatchCancelError = CU_FILE_INTERNAL_BATCH_CANCEL_ERROR,
        /// cufile no memory error - retry later
        NoMemory = CU_FILE_NOMEM_ERROR,
        /// cufile io error
        IoError = CU_FILE_IO_ERROR,
        /// cufile buf registration error
        InternalBufferRegisterError = CU_FILE_INTERNAL_BUF_REGISTER_ERROR,
        /// cufile hash operation error
        HashOperationError = CU_FILE_HASH_OPR_ERROR,
        /// cufile invalid context error
        InvalidContextError = CU_FILE_INVALID_CONTEXT_ERROR,
        /// nvfs internal driver error
        NvfsInternalDriverError = CU_FILE_NVFS_INTERNAL_DRIVER_ERROR,
        /// compat mode off error
        BatchNoCompatError = CU_FILE_BATCH_NOCOMPAT_ERROR,
        /// Upper marker of the cuFile operation error-code range.
        IoMaxError = CU_FILE_IO_MAX_ERROR,
    }
}

impl Status {
    /// Returns a description of the cuFile status.
    #[must_use]
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

    /// Returns the same status description as [`Self::message`].
    #[must_use]
    pub fn description(self) -> &'static str {
        self.message()
    }
}

/// An error from CUDA, cuFile, the operating system, or wrapper argument validation.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    /// A CUDA operation failed.
    #[error("cuda error: {0}")]
    Cuda(#[from] CudaError),

    /// A cuFile operation returned a failure status.
    #[error("cufile error ({code}): {message}")]
    Cufile {
        /// The cuFile status code.
        code: Status,
        /// The accompanying CUDA status when `code` is [`Status::CudaDriverError`].
        cuda_code: Option<u32>,
        /// A description of the cuFile status.
        message: String,
    },

    /// A string contains an interior NUL byte.
    #[error("string contains interior nul byte")]
    InteriorNul,

    /// An operating-system I/O operation failed.
    #[error("io error: {0}")]
    Io(#[from] io::Error),

    /// A native handle or buffer address is null.
    #[error("unexpected null handle")]
    NullHandle,

    /// An argument, computed size, or native result is outside its supported range.
    #[error("`{name}` is out of range")]
    OutOfRange {
        /// The argument or quantity outside its supported range.
        name: Cow<'static, str>,
    },

    /// An argument has a different number of elements than required.
    #[error("`{name}` length mismatch (expected {expected}, got {actual})")]
    LengthMismatch {
        /// The argument with the incorrect length.
        name: Cow<'static, str>,
        /// The required number of elements.
        expected: usize,
        /// The supplied number of elements.
        actual: usize,
    },

    /// A negative I/O result cannot be decoded as a cuFile or operating-system error.
    #[error("cufile io operation failed with code {0}")]
    OperationFailed(i64),

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

impl From<NulError> for Error {
    fn from(_: NulError) -> Self {
        Self::InteriorNul
    }
}

impl From<sys::CUfileError_t> for Error {
    fn from(status: sys::CUfileError_t) -> Self {
        let code = Status::from(status.err as u32);
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

/// Converts a raw cuFile status-returning expression into a crate [`Result`](crate::error::Result).
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
