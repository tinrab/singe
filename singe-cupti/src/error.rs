#![expect(
    deprecated,
    reason = "wraps deprecated upstream API values that remain part of the supported surface"
)]

use std::{
    borrow::Cow,
    ffi::{NulError, c_char},
    ptr,
    result::Result as StdResult,
};

use singe_cuda::error::Error as CudaError;
use singe_cupti_sys as sys;
use thiserror::Error;

singe_core::wrapper_status! {
    library = "cupti",
    error = Error::Cupti,
    raw = sys::CUptiResult,
    success = CUPTI_SUCCESS,
    absence = sys::take_absence,
    unknown = "UNKNOWN_CUPTI_STATUS",
    description = error_message,
    /// Result codes returned by CUPTI.
    /// Codes from newer CUPTI releases are preserved in [`Status::Unknown`].
    pub enum Status: u32 {
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
        /// Profiling session will continue and CUPTI will notify it using this error code.
        /// User is advised to recompile the application code with nvcc version 9.0 or later.
        /// Ignore this warning if code is already compiled with the recommended nvcc version.
        CudaCompilerNotCompatible = CUPTI_ERROR_CUDA_COMPILER_NOT_COMPATIBLE,
        /// User doesn't have sufficient privileges which are required to start the profiling session.
        ///
        /// One possible reason for this may be that the NVIDIA driver or your system administrator may have restricted access to the NVIDIA GPU performance counters.
        /// To learn how to resolve this issue and find more information, please visit <https://developer.nvidia.com/ERR_NVGPUCTRPERM>.
        InsufficientPrivileges = CUPTI_ERROR_INSUFFICIENT_PRIVILEGES,
        /// The legacy event and metric APIs are active in this process, and they cannot be used together with the range profiler and host metric APIs in [`crate::profiler`].
        OldProfilerApiInitialized = CUPTI_ERROR_OLD_PROFILER_API_INITIALIZED,
        /// Missing definition of the OpenACC API routine in the linked OpenACC library.
        ///
        /// One possible reason is that OpenACC library is linked statically in the user application, which might not have the definition of all the OpenACC API routines needed for the OpenACC profiling, as compiler might ignore definitions for the functions not used in the application.
        /// This issue can be mitigated by linking the OpenACC library dynamically.
        OpenAccUndefinedRoutine = CUPTI_ERROR_OPENACC_UNDEFINED_ROUTINE,
        /// The legacy event and metric APIs are not supported on devices with compute capability 7.5 and higher.
        ///
        /// These APIs were deprecated in CUDA 12.8 and removed in CUDA 13.0, replaced by the host metric and range profiler APIs in [`crate::profiler`].
        /// CUDA 13.0 also removed PC sampling activity and source-level and SASS-level metrics.
        LegacyProfilerNotSupported = CUPTI_ERROR_LEGACY_PROFILER_NOT_SUPPORTED,
        /// CUPTI doesn't allow multiple callback subscribers.
        ///
        /// Only a single subscriber can be registered at a time.
        /// Same error code is used when application is launched using NVIDIA tools like Nsight Systems, Nsight Compute, and cuda-gdb.
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
        /// The chip name passed to [`ProfilerHost::create_for_chip`](crate::profiler::ProfilerHost::create_for_chip) is invalid or unsupported.
        InvalidChipName = CUPTI_ERROR_INVALID_CHIP_NAME,
        /// Hardware Event System (HES) trace is not supported on MPS.
        HesTraceNotSupportedOnMultiProcessService = CUPTI_ERROR_HES_TRACE_NOT_SUPPORTED_ON_MPS,
        /// An unknown internal error has occurred.
        UnknownError = CUPTI_ERROR_UNKNOWN,
        /// An unknown internal error has occurred.
        ForceInt = CUPTI_ERROR_FORCE_INT,
    }
}

/// Errors from CUDA, CUPTI, and checked profiling argument or result validation.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    /// CUPTI or one of its symbols would not load.
    #[error(transparent)]
    Load(#[from] sys::LoadError),

    /// A CUDA operation failed.
    #[error(transparent)]
    Cuda(#[from] CudaError),

    /// CUPTI returned a failure status.
    #[error("cupti error ({code}): {}", code.description())]
    Cupti {
        /// Native CUPTI status returned by the failed call.
        code: Status,
    },

    /// A string passed to a native API contains an interior nul byte.
    #[error("string contains interior nul byte")]
    InteriorNul,

    /// A native handle, returned string, or required table pointer is unexpectedly null.
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

    /// An argument or checked conversion is outside the supported range.
    #[error("`{name}` is out of range")]
    OutOfRange {
        /// Name of the invalid argument, conversion, attribute, or shared state.
        name: Cow<'static, str>,
    },

    /// A native value has a byte size other than the one its type requires.
    #[error("`{name}` size mismatch (expected {expected} bytes, got {actual})")]
    SizeMismatch {
        /// Name of the value whose size is checked.
        name: Cow<'static, str>,
        /// Required size in bytes.
        expected: usize,
        /// Reported size in bytes.
        actual: usize,
    },

    /// An attribute is unsupported by the typed accessor or does not match its supplied value type.
    #[error("invalid attribute: `{name}`")]
    InvalidAttribute {
        /// Name of the invalid argument, conversion, attribute, or shared state.
        name: Cow<'static, str>,
    },

    /// CUPTI rejected a subscription because another subscriber is active.
    #[error("cupti subscriber `{existing}` is already active")]
    SubscriberActive {
        /// Name the active subscriber registered with CUPTI.
        existing: String,
    },

    /// Activity buffer callbacks are registered for the process or another subscriber, and CUPTI's buffer callbacks cannot tell the two apart.
    #[error("activity buffer callbacks are registered for another subscriber")]
    ActivityCallbacksInUse,

    /// Shared profiling or callback state is protected by a poisoned mutex.
    #[error("`{name}` lock is poisoned")]
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

// Adapts `cuptiGetErrorMessage`, which writes its string through an out-parameter, to the error-string shape `wrapper_status!` expects.
fn error_message(status: sys::CUptiResult) -> *const c_char {
    let mut message = ptr::null();
    // SAFETY: `message` is a valid output location, and CUPTI writes a static NUL-terminated string to it.
    let result = unsafe { sys::cuptiGetErrorMessage(status, &raw mut message) };
    if result == sys::CUptiResult::CUPTI_SUCCESS {
        message
    } else {
        // A missing CUPTI records an absence that belongs to no wrapper call, so it must not reach the next status conversion.
        let _ = sys::take_absence();
        ptr::null()
    }
}

// Converts a raw CUPTI status into this crate's `Result`.
macro_rules! try_ffi {
    ($expr:expr) => {
        $crate::error::Error::check($expr)
    };
}

pub(crate) use try_ffi;
