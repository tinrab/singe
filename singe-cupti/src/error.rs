#![expect(
    deprecated,
    reason = "wraps deprecated upstream API values that remain part of the supported surface"
)]

use std::{
    borrow::Cow,
    ffi::{CStr, NulError},
    ptr,
    result::Result as StdResult,
};

use singe_core::{static_str_from_c_ptr, status_enum};
use singe_cuda::error::Error as CudaError;
use singe_cupti_sys as sys;
use thiserror::Error;

/// Errors from CUDA, CUPTI, and checked profiling argument or result validation.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    /// A CUDA operation failed.
    #[error("cuda error: {0}")]
    Cuda(#[from] CudaError),

    /// CUPTI returned a non-success status.
    #[error("cupti error ({code}): {message}")]
    Cupti {
        /// Native CUPTI status returned by the failed call.
        code: Status,
        /// Description of the native CUPTI status.
        message: String,
    },

    /// A string passed to a native API contains an interior nul byte.
    #[error("string contains interior nul byte")]
    InteriorNul,

    /// A native handle, returned string, or required table pointer is unexpectedly null.
    #[error("unexpected null handle")]
    NullHandle,

    /// An argument or checked conversion is outside the supported range.
    #[error("`{name}` is out of range")]
    OutOfRange {
        /// Name of the invalid argument, conversion, attribute, or shared state.
        name: Cow<'static, str>,
    },

    /// An argument or native result has an unexpected length.
    #[error("unexpected length for `{name}`: expected {expected}, got {actual}")]
    LengthMismatch {
        /// Name of the invalid argument, conversion, attribute, or shared state.
        name: Cow<'static, str>,
        /// Required length in the units named by the failing operation.
        expected: usize,
        /// Observed length in the units named by the failing operation.
        actual: usize,
    },

    /// A decoded native enum value is unrecognized.
    #[error("unknown `{name}` value ({value})")]
    UnknownEnumValue {
        /// Name of the invalid argument, conversion, attribute, or shared state.
        name: Cow<'static, str>,
        /// Unrecognized native enum value.
        value: u32,
    },

    /// An attribute is unsupported by the typed accessor or does not match its supplied value type.
    #[error("invalid attribute: `{name}`")]
    InvalidAttribute {
        /// Name of the invalid argument, conversion, attribute, or shared state.
        name: Cow<'static, str>,
    },

    /// Shared profiling or callback state is protected by a poisoned mutex.
    #[error("`{name}` lock poisoned")]
    LockPoisoned {
        /// Name of the invalid argument, conversion, attribute, or shared state.
        name: Cow<'static, str>,
    },
}

/// Result of a CUPTI wrapper operation.
pub type Result<T> = StdResult<T, Error>;

impl Error {
    pub(crate) fn out_of_range(name: impl Into<Cow<'static, str>>) -> Self {
        Self::OutOfRange { name: name.into() }
    }
}

status_enum! {
    /// Result codes returned by CUPTI.
    /// Codes from newer CUPTI releases are preserved in [`Status::Unknown`].
    pub enum Status: u32 => sys::CUptiResult, unknown = "UNKNOWN_CUPTI_STATUS" {
        /// No error.
        Success = CUPTI_SUCCESS,
        /// One or more of the parameters is invalid.
        InvalidParameter = CUPTI_ERROR_INVALID_PARAMETER,
        /// The device does not correspond to a valid CUDA device.
        InvalidDevice = CUPTI_ERROR_INVALID_DEVICE,
        /// The context is NULL or not valid.
        InvalidContext = CUPTI_ERROR_INVALID_CONTEXT,
        /// The event domain id is invalid.
        InvalidEventDomainId = CUPTI_ERROR_INVALID_EVENT_DOMAIN_ID,
        /// The event id is invalid.
        InvalidEventId = CUPTI_ERROR_INVALID_EVENT_ID,
        /// The event name is invalid.
        InvalidEventName = CUPTI_ERROR_INVALID_EVENT_NAME,
        /// The current operation cannot be performed due to dependency on other factors.
        InvalidOperation = CUPTI_ERROR_INVALID_OPERATION,
        /// Unable to allocate enough memory to perform the requested operation.
        OutOfMemory = CUPTI_ERROR_OUT_OF_MEMORY,
        /// An error occurred on the performance monitoring hardware.
        Hardware = CUPTI_ERROR_HARDWARE,
        /// The output buffer size is not sufficient to return all requested data.
        ParameterSizeNotSufficient = CUPTI_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT,
        /// API is not implemented.
        ApiNotImplemented = CUPTI_ERROR_API_NOT_IMPLEMENTED,
        /// The maximum limit is reached.
        MaxLimitReached = CUPTI_ERROR_MAX_LIMIT_REACHED,
        /// The object is not yet ready to perform the requested operation.
        NotReady = CUPTI_ERROR_NOT_READY,
        /// The current operation is not compatible with the current state of the object.
        NotCompatible = CUPTI_ERROR_NOT_COMPATIBLE,
        /// CUPTI is unable to initialize its connection to the CUDA driver.
        NotInitialized = CUPTI_ERROR_NOT_INITIALIZED,
        /// The metric id is invalid.
        InvalidMetricId = CUPTI_ERROR_INVALID_METRIC_ID,
        /// The metric name is invalid.
        InvalidMetricName = CUPTI_ERROR_INVALID_METRIC_NAME,
        /// The queue is empty.
        QueueEmpty = CUPTI_ERROR_QUEUE_EMPTY,
        /// Invalid handle (internal?).
        InvalidHandle = CUPTI_ERROR_INVALID_HANDLE,
        /// Invalid stream.
        InvalidStream = CUPTI_ERROR_INVALID_STREAM,
        /// Invalid kind.
        InvalidKind = CUPTI_ERROR_INVALID_KIND,
        /// Invalid event value.
        InvalidEventValue = CUPTI_ERROR_INVALID_EVENT_VALUE,
        /// CUPTI is disabled due to conflicts with other enabled profilers.
        Disabled = CUPTI_ERROR_DISABLED,
        /// Invalid module.
        InvalidModule = CUPTI_ERROR_INVALID_MODULE,
        /// Invalid metric value.
        InvalidMetricValue = CUPTI_ERROR_INVALID_METRIC_VALUE,
        /// The performance monitoring hardware is in use by other client.
        HardwareBusy = CUPTI_ERROR_HARDWARE_BUSY,
        /// The attempted operation is not supported on the current system or device.
        NotSupported = CUPTI_ERROR_NOT_SUPPORTED,
        /// Unified memory profiling is not supported on the system.
        ///
        /// Potential reason could be unsupported OS or architecture.
        UnifiedMemoryProfilingNotSupported = CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED,
        /// Unified memory profiling is not supported on the device.
        UnifiedMemoryProfilingNotSupportedOnDevice = CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_ON_DEVICE,
        /// Unified memory profiling is not supported on a multi-GPU configuration without P2P support between any pair of devices.
        UnifiedMemoryProfilingNotSupportedOnNonPeerToPeerDevices = CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_ON_NON_P2P_DEVICES,
        /// Unified memory profiling is not supported under the Multi-Process Service (MPS) environment.
        ///
        /// CUDA 7.5 removes this restriction.
        UnifiedMemoryProfilingNotSupportedWithMultiProcessService = CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_WITH_MPS,
        /// In CUDA 9.0, devices with compute capability 7.0 don't support CDP tracing.
        CudaDynamicParallelismTracingNotSupported = CUPTI_ERROR_CDP_TRACING_NOT_SUPPORTED,
        /// Profiling on virtualized GPU is not supported.
        VirtualizedDeviceNotSupported = CUPTI_ERROR_VIRTUALIZED_DEVICE_NOT_SUPPORTED,
        /// Profiling results might be incorrect for CUDA applications compiled with nvcc version older than 9.0 for devices with compute capability 6.0 and 6.1.
        ///
        /// Profiling session will continue and CUPTI will notify it using this error code. User is advised to recompile the application code with nvcc version 9.0 or later. Ignore this warning if code is already compiled with the recommended nvcc version.
        CudaCompilerNotCompatible = CUPTI_ERROR_CUDA_COMPILER_NOT_COMPATIBLE,
        /// User doesn't have sufficient privileges which are required to start the profiling session.
        ///
        /// One possible reason for this may be that the NVIDIA driver or your system administrator may have restricted access to the NVIDIA GPU performance counters. To learn how to resolve this issue and find more information, please visit <https://developer.nvidia.com/`CUptiResult::CUPTI_ERROR_INSUFFICIENT_PRIVILEGES`>.
        InsufficientPrivileges = CUPTI_ERROR_INSUFFICIENT_PRIVILEGES,
        /// Legacy CUPTI Profiling API i.e.
        ///
        /// event API from the header cupti_events.h and metric API from the header cupti_metrics.h are not compatible with the Profiling API in the header cupti_profiler_target.h and Perfworks metrics API in the headers nvperf_host.h and nvperf_target.h.
        OldProfilerApiInitialized = CUPTI_ERROR_OLD_PROFILER_API_INITIALIZED,
        /// Missing definition of the OpenACC API routine in the linked OpenACC library.
        ///
        /// One possible reason is that OpenACC library is linked statically in the user application, which might not have the definition of all the OpenACC API routines needed for the OpenACC profiling, as compiler might ignore definitions for the functions not used in the application. This issue can be mitigated by linking the OpenACC library dynamically.
        OpenAccUndefinedRoutine = CUPTI_ERROR_OPENACC_UNDEFINED_ROUTINE,
        /// Legacy CUPTI Profiling API i.e.
        ///
        /// event API from the header cupti_events.h and metric API from the header cupti_metrics.h are not supported on devices with compute capability 7.5 and higher (i.e. Turing and later GPU architectures). These APIs were deprecated in CUDA 12.8 and removed in CUDA 13.0. These are replaced by the host profiling API in the header cupti_profiler_host.h and target profiling API in the header cupti_range_profiler.h which are supported on devices with compute capability 7.5 and higher (i.e. Turing and later GPU architectures). Further, the PC Sampling Activity API and the source/SASS level metrics from the header cupti_activity.h are removed in CUDA 13.0.
        LegacyProfilerNotSupported = CUPTI_ERROR_LEGACY_PROFILER_NOT_SUPPORTED,
        /// CUPTI doesn't allow multiple callback subscribers.
        ///
        /// Only a single subscriber can be registered at a time. Same error code is used when application is launched using NVIDIA tools like Nsight Systems, Nsight Compute, and cuda-gdb.
        MultipleSubscribersNotSupported = CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED,
        /// Profiling on virtualized GPU is not allowed by hypervisor.
        VirtualizedDeviceInsufficientPrivileges = CUPTI_ERROR_VIRTUALIZED_DEVICE_INSUFFICIENT_PRIVILEGES,
        /// Profiling and tracing are not allowed when confidential computing mode is enabled.
        ConfidentialComputingNotSupported = CUPTI_ERROR_CONFIDENTIAL_COMPUTING_NOT_SUPPORTED,
        /// CUPTI does not support NVIDIA Crypto Mining Processors (CMP).
        ///
        /// For more information, please visit <https://developer.nvidia.com/ERR_NVCMPGPU>.
        CmpDeviceNotSupported = CUPTI_ERROR_CMP_DEVICE_NOT_SUPPORTED,
        /// Profiling on Multi-instance GPU (MIG) is not supported.
        MigDeviceNotSupported = CUPTI_ERROR_MIG_DEVICE_NOT_SUPPORTED,
        /// Profiling on SLI device is not supported.
        SliDeviceNotSupported = CUPTI_ERROR_SLI_DEVICE_NOT_SUPPORTED,
        /// Profiling on WSL device is not supported.
        WslDeviceNotSupported = CUPTI_ERROR_WSL_DEVICE_NOT_SUPPORTED,
        /// For invalid or unsupported chip name passed to cuptiProfilerHostInitialize.
        InvalidChipName = CUPTI_ERROR_INVALID_CHIP_NAME,
        /// Hardware Event System (HES) trace is not supported on MPS.
        HesTraceNotSupportedOnMultiProcessService = CUPTI_ERROR_HES_TRACE_NOT_SUPPORTED_ON_MPS,
        /// An unknown internal error has occurred.
        UnknownError = CUPTI_ERROR_UNKNOWN,
        /// An unknown internal error has occurred.
        ForceInt = CUPTI_ERROR_FORCE_INT,
    }
}

impl From<sys::CUptiResult> for Error {
    fn from(code: sys::CUptiResult) -> Self {
        let status = Status::from(code);
        // SAFETY: `cuptiGetErrorMessage` writes a static NUL-terminated string on success; the pointer is checked before use.
        let message = unsafe {
            let mut message_ptr = ptr::null();
            let error_result = sys::cuptiGetErrorMessage(code, &raw mut message_ptr);
            if error_result == sys::CUptiResult::CUPTI_SUCCESS && !message_ptr.is_null() {
                CStr::from_ptr(message_ptr).to_string_lossy().into_owned()
            } else {
                format!("cupti error code {code:?}")
            }
        };

        Self::Cupti {
            code: status,
            message,
        }
    }
}

impl From<Status> for Error {
    fn from(status: Status) -> Self {
        match sys::CUptiResult::try_from(status) {
            Ok(code) => code.into(),
            Err(status) => Self::Cupti {
                code: status,
                message: format!("unknown cupti result code {}", status.raw()),
            },
        }
    }
}

impl Status {
    /// CUPTI's description of this status.
    #[must_use]
    pub fn description(self) -> &'static str {
        let Ok(raw) = sys::CUptiResult::try_from(self) else {
            return "unknown cupti error";
        };
        let mut message = ptr::null();
        // SAFETY: `message` is a valid output location, and CUPTI returns a static string.
        unsafe {
            if sys::cuptiGetErrorMessage(raw, &raw mut message) != sys::CUptiResult::CUPTI_SUCCESS {
                return "unknown cupti error";
            }
            static_str_from_c_ptr(message, "unknown cupti error")
        }
    }
}

impl From<NulError> for Error {
    fn from(_: NulError) -> Self {
        Self::InteriorNul
    }
}

/// Converts a raw CUPTI status-returning expression into a crate [`Result`](crate::error::Result).
#[macro_export]
macro_rules! try_ffi {
    ($expr:expr) => {{
        let status = { $expr };
        if status != singe_cupti_sys::CUptiResult::CUPTI_SUCCESS {
            Err($crate::error::Error::from(status))
        } else {
            Ok(())
        }
    }};
}
