#![expect(
    deprecated,
    reason = "wraps deprecated upstream API values that remain part of the supported surface"
)]

use std::{borrow::Cow, ffi::NulError, result::Result as StdResult};

use singe_core::{static_str_from_c_ptr, status_enum};
use singe_nvml_sys as sys;
use thiserror::Error;

status_enum! {
    /// Return values for NVML API calls.
    pub enum Status: u32 => sys::nvmlReturn_t, unknown = "UNKNOWN_NVML_STATUS" {
        /// The operation was successful.
        Success = NVML_SUCCESS,
        /// NVML was not first initialized with nvmlInit().
        Uninitialized = NVML_ERROR_UNINITIALIZED,
        /// A supplied argument is invalid.
        InvalidArgument = NVML_ERROR_INVALID_ARGUMENT,
        /// The requested operation is not available on target device.
        NotSupported = NVML_ERROR_NOT_SUPPORTED,
        /// The current user does not have permission for operation.
        NoPermission = NVML_ERROR_NO_PERMISSION,
        /// Multiple initializations are now allowed through ref counting.
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
        /// infoROM is corrupted.
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
        /// The requested vgpu operation is not available on target device, becasue ECC is enabled.
        VgpuEccNotSupported = NVML_ERROR_VGPU_ECC_NOT_SUPPORTED,
        /// Ran out of critical resources, other than memory.
        InsufficientResources = NVML_ERROR_INSUFFICIENT_RESOURCES,
        /// Ran out of critical resources, other than memory.
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

impl Status {
    /// Library description of this status.
    #[must_use]
    pub fn description(self) -> &'static str {
        let Ok(raw) = sys::nvmlReturn_t::try_from(self) else {
            return "unknown nvml error";
        };
        // SAFETY: `nvmlErrorString` returns null or a static NUL-terminated string.
        unsafe { static_str_from_c_ptr(sys::nvmlErrorString(raw), "unknown nvml error") }
    }
}

/// NVML status failures and errors validating Rust inputs or native output data.
#[derive(Error, Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// NVML returned a non-success status.
    #[error("nvml error ({code}): {message}")]
    Nvml {
        /// Native NVML status code.
        code: Status,
        /// NVML's description of the status.
        message: String,
    },

    /// A string contains a NUL byte before its terminator.
    #[error("string contains interior nul byte")]
    InteriorNul,

    /// NVML returned a null handle where a live object was required.
    #[error("unexpected null handle")]
    NullHandle,

    /// A field sample reports an unrecognized native value type.
    #[error("unknown field value type ({0})")]
    UnknownFieldValueType(
        /// Unrecognized native field-value type code.
        u32,
    ),

    /// A field sample has a recognized value type that does not match the requested Rust type.
    #[error("unexpected `{name}` field value type ({value})")]
    UnexpectedFieldValueType {
        /// Field or requested value whose representation did not match.
        name: Cow<'static, str>,
        /// Reported field-value representation.
        value: String,
    },

    /// Native output contains a value outside the supported typed enum.
    #[error("unknown `{name}` value ({value})")]
    UnknownEnumValue {
        /// Enum or output field being decoded.
        name: Cow<'static, str>,
        /// Unrecognized native integer value.
        value: u32,
    },

    /// A native signed value is negative where the wrapper requires a nonnegative value.
    #[error("negative `{name}` value ({value})")]
    NegativeValue {
        /// Value required to be nonnegative.
        name: Cow<'static, str>,
        /// Negative value returned by NVML.
        value: i64,
    },

    /// A value does not fit the range or native representation required by the operation.
    #[error("`{name}` is out of range")]
    OutOfRange {
        /// Value whose required range or representation was exceeded.
        name: Cow<'static, str>,
    },

    /// A native query returned no entries where at least one was required.
    #[error("`{name}` returned no values")]
    EmptyOutput {
        /// Query or output expected to contain at least one entry.
        name: Cow<'static, str>,
    },

    /// A native byte buffer is shorter than the required record layout.
    #[error("`{name}` output is too short (expected at least {expected} bytes, got {actual})")]
    ShortOutput {
        /// Native record or output being decoded.
        name: Cow<'static, str>,
        /// Minimum byte length required by the record layout.
        expected: usize,
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

impl From<NulError> for Error {
    fn from(_: NulError) -> Self {
        Self::InteriorNul
    }
}

impl From<sys::nvmlReturn_t> for Error {
    fn from(code: sys::nvmlReturn_t) -> Self {
        Self::from(Status::from(code))
    }
}

impl From<Status> for Error {
    fn from(status: Status) -> Self {
        Self::Nvml {
            code: status,
            message: status.description().to_string(),
        }
    }
}

/// Converts a raw NVML status-returning expression into a crate [`Result`](crate::error::Result).
#[macro_export]
macro_rules! try_ffi {
    ($expr:expr) => {{
        let status = { $expr };
        if status != singe_nvml_sys::nvmlReturn_t::NVML_SUCCESS {
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
    fn test_status_converts_from_raw_code() {
        let status = Status::from(sys::nvmlReturn_t::NVML_ERROR_INVALID_ARGUMENT);
        assert_eq!(status, Status::InvalidArgument);
        assert_eq!(
            status.raw(),
            sys::nvmlReturn_t::NVML_ERROR_INVALID_ARGUMENT as u32
        );
    }

    #[test]
    fn test_status_display_uses_c_name() {
        assert_eq!(
            Status::InvalidArgument.to_string(),
            "NVML_ERROR_INVALID_ARGUMENT"
        );
    }

    #[test]
    fn test_unknown_status_keeps_raw_code() {
        let status = Status::Unknown(9999);
        assert_eq!(status.raw(), 9999);
        assert_eq!(status.to_string(), "UNKNOWN_NVML_STATUS(9999)");
    }
}
