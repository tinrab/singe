//! nvJPEG 2000 status codes.

use singe_core::status_enum;
use singe_nvjpeg_sys::k2 as k2_sys;

status_enum! {
    /// Status codes returned by nvJPEG 2000 calls.
    pub enum Status: u32 => k2_sys::nvjpeg2kStatus_t, unknown = "UNKNOWN_NVJPEG2K_STATUS", unknown_description = "unknown nvjpeg2k error" {
        /// The call succeeded.
        ///
        /// Many calls are asynchronous, so some errors appear only after synchronizing.
        Success = NVJPEG2K_STATUS_SUCCESS => "success",
        /// The library handle was not initialized.
        NotInitialized = NVJPEG2K_STATUS_NOT_INITIALIZED => "not initialized",
        /// An argument is invalid, such as null input data or an invalid enum value.
        InvalidParameter = NVJPEG2K_STATUS_INVALID_PARAMETER => "invalid parameter",
        /// The JPEG 2000 data cannot be parsed, most likely because it is corrupt.
        BadJpeg = NVJPEG2K_STATUS_BAD_JPEG => "bad jpeg",
        /// nvJPEG 2000 does not support decoding this data.
        JpegNotSupported = NVJPEG2K_STATUS_JPEG_NOT_SUPPORTED => "jpeg not supported",
        /// A custom allocation or free callback returned a nonzero code.
        AllocatorFailure = NVJPEG2K_STATUS_ALLOCATOR_FAILURE => "allocator failure",
        /// Work on the device failed.
        ExecutionFailed = NVJPEG2K_STATUS_EXECUTION_FAILED => "execution failed",
        /// The device does not support the requested settings.
        ArchitectureMismatch = NVJPEG2K_STATUS_ARCH_MISMATCH => "architecture mismatch",
        /// An internal nvJPEG 2000 error occurred.
        InternalError = NVJPEG2K_STATUS_INTERNAL_ERROR => "internal error",
        /// The backend does not support the call.
        ImplementationNotSupported = NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED => "implementation not supported",
    }
}

/// Evaluates a native nvJPEG 2000 call and converts its status into a [`Result`](crate::error::Result).
#[macro_export]
macro_rules! try_ffi_k2 {
    ($expr:expr) => {{
        let status = { $expr };
        if status != singe_nvjpeg_sys::k2::nvjpeg2kStatus_t::NVJPEG2K_STATUS_SUCCESS {
            Err($crate::error::Error::from(status))
        } else {
            Ok(())
        }
    }};
}
