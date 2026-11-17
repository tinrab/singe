#![expect(
    deprecated,
    reason = "wraps deprecated upstream API values that remain part of the supported surface"
)]

use std::{borrow::Cow, ffi::NulError, result::Result as StdResult};

use singe_nvml_sys as sys;
use thiserror::Error;

singe_core::wrapper_status! {
    library = "nvml",
    error = Error::Nvml,
    raw = sys::nvmlReturn_t,
    success = NVML_SUCCESS,
    absence = sys::take_absence,
    unknown = "UNKNOWN_NVML_STATUS",
    description = sys::nvmlErrorString,
    /// Return values for NVML API calls.
    pub enum Status: u32 {
        /// The operation was successful.
        Success = NVML_SUCCESS,
        /// NVML was not initialized.
        Uninitialized = NVML_ERROR_UNINITIALIZED,
        /// A supplied argument is invalid.
        InvalidArgument = NVML_ERROR_INVALID_ARGUMENT,
        /// The requested operation is not available on target device.
        NotSupported = NVML_ERROR_NOT_SUPPORTED,
        /// The current user does not have permission for operation.
        NoPermission = NVML_ERROR_NO_PERMISSION,
        /// NVML was already initialized; NVML now counts initializations instead.
        #[deprecated]
        AlreadyInitialized = NVML_ERROR_ALREADY_INITIALIZED,
        /// A query to find an object was unsuccessful.
        NotFound = NVML_ERROR_NOT_FOUND,
        /// An input argument is not large enough.
        InsufficientSize = NVML_ERROR_INSUFFICIENT_SIZE,
        /// A device's external power cables are not properly attached.
        InsufficientPower = NVML_ERROR_INSUFFICIENT_POWER,
        /// NVIDIA driver is not loaded.
        DriverNotLoaded = NVML_ERROR_DRIVER_NOT_LOADED,
        /// User provided timeout passed.
        Timeout = NVML_ERROR_TIMEOUT,
        /// NVIDIA Kernel detected an interrupt issue with a GPU.
        IrqIssue = NVML_ERROR_IRQ_ISSUE,
        /// NVML Shared Library couldn't be found or loaded.
        LibraryNotFound = NVML_ERROR_LIBRARY_NOT_FOUND,
        /// Local version of NVML doesn't implement this function.
        FunctionNotFound = NVML_ERROR_FUNCTION_NOT_FOUND,
        /// The infoROM is corrupted.
        CorruptedInforom = NVML_ERROR_CORRUPTED_INFOROM,
        /// The GPU has fallen off the bus or has otherwise become inaccessible.
        GpuIsLost = NVML_ERROR_GPU_IS_LOST,
        /// The GPU requires a reset before it can be used again.
        ResetRequired = NVML_ERROR_RESET_REQUIRED,
        /// The GPU control device has been blocked by the operating system/cgroups.
        OperatingSystem = NVML_ERROR_OPERATING_SYSTEM,
        /// RM detects a driver/library version mismatch.
        LibRmVersionMismatch = NVML_ERROR_LIB_RM_VERSION_MISMATCH,
        /// An operation cannot be performed because the GPU is currently in use.
        InUse = NVML_ERROR_IN_USE,
        /// Insufficient memory.
        Memory = NVML_ERROR_MEMORY,
        /// No data.
        NoData = NVML_ERROR_NO_DATA,
        /// The requested vGPU operation is not available on the target device because ECC is enabled.
        VgpuEccNotSupported = NVML_ERROR_VGPU_ECC_NOT_SUPPORTED,
        /// Ran out of critical resources, other than memory.
        InsufficientResources = NVML_ERROR_INSUFFICIENT_RESOURCES,
        /// The requested frequency is not supported.
        FreqNotSupported = NVML_ERROR_FREQ_NOT_SUPPORTED,
        /// The provided version is invalid/unsupported.
        ArgumentVersionMismatch = NVML_ERROR_ARGUMENT_VERSION_MISMATCH,
        /// The requested functionality has been deprecated.
        Deprecated = NVML_ERROR_DEPRECATED,
        /// The system is not ready for the request.
        NotReady = NVML_ERROR_NOT_READY,
        /// No GPUs were found.
        GpuNotFound = NVML_ERROR_GPU_NOT_FOUND,
        /// Resource not in correct state to perform requested operation.
        InvalidState = NVML_ERROR_INVALID_STATE,
        /// Reset not supported for given device/parameters.
        ResetTypeNotSupported = NVML_ERROR_RESET_TYPE_NOT_SUPPORTED,
        /// An internal driver error occurred.
        UnknownError = NVML_ERROR_UNKNOWN,
    }
}

/// NVML status failures and errors validating Rust inputs or native output data.
#[derive(Error, Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// NVML or one of its symbols would not load.
    #[error(transparent)]
    Load(#[from] sys::LoadError),

    /// NVML returned a non-success status.
    #[error("nvml error ({code}): {}", code.description())]
    Nvml {
        /// Native NVML status code.
        code: Status,
    },

    /// A string contains a NUL byte before its terminator.
    #[error("string contains interior nul byte")]
    InteriorNul,

    /// NVML returned a null handle where a live object was required.
    #[error("unexpected null handle")]
    NullHandle,

    /// A field sample has a recognized value type that does not match the requested Rust type.
    #[error("unexpected `{name}` field value type ({value})")]
    UnexpectedFieldValueType {
        /// Field or requested value whose representation did not match.
        name: Cow<'static, str>,
        /// Reported field-value representation.
        value: String,
    },

    /// A native enum value has no typed variant in this crate.
    #[error("unknown `{name}` value {value}")]
    UnknownEnumValue {
        /// Name of the typed enum.
        name: Cow<'static, str>,
        /// Unrecognized native value.
        value: i64,
    },

    /// A value does not fit the range or native representation required by the operation.
    #[error("`{name}` is out of range")]
    OutOfRange {
        /// Value whose required range or representation was exceeded.
        name: Cow<'static, str>,
    },

    /// A native query returned no entries where at least one was required.
    #[error("`{name}` is empty")]
    Empty {
        /// Query or output expected to contain at least one entry.
        name: Cow<'static, str>,
    },

    /// A native byte buffer is shorter than the required record layout.
    #[error("`{name}` is too small (requires {required}, got {actual})")]
    BufferTooSmall {
        /// Native record or output being decoded.
        name: Cow<'static, str>,
        /// Minimum byte length required by the record layout.
        required: usize,
        /// Byte length returned by NVML.
        actual: usize,
    },
}

/// Result of an NVML wrapper operation.
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

// Converts a raw NVML status into this crate's `Result`.
macro_rules! try_ffi {
    ($expr:expr) => {
        $crate::error::Error::check($expr)
    };
}

pub(crate) use try_ffi;
