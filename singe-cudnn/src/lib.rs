//! Safe cuDNN graph/frontend, backend descriptor, and execution wrappers.
//!
//! This crate exposes cuDNN frontend graph construction, graph lowering,
//! execution plan selection, variant packs, tensor descriptors, operation
//! descriptors, and lower-level backend APIs. It is designed for
//! inference-oriented graph execution while still leaving access to cuDNN
//! descriptor and engine details when required.

#![allow(deprecated)]

mod attribute;
mod behavior;
pub mod context;
mod convolution;
mod ctc;
pub mod data_type;
mod descriptor;
mod device;
mod dropout;
mod engine;
pub mod error;
mod execution;
mod heuristic;
mod knob;
mod layout;
mod lrn;
pub mod math;
mod normalization;
mod pointwise;
mod reduction;
pub mod scalar;
mod softmax;
mod spatial_transformer;
mod tensor;

/// Low-level cuDNN backend descriptor, engine, and execution APIs.
pub mod backend {
    pub mod attribute {
        pub use crate::attribute::*;
    }
    pub mod behavior {
        pub use crate::behavior::*;
    }
    pub mod convolution {
        pub use crate::convolution::*;
    }
    pub mod descriptor {
        pub use crate::descriptor::*;
    }
    pub mod device {
        pub use crate::device::*;
    }
    pub mod engine {
        pub use crate::engine::*;
    }
    pub mod execution {
        pub use crate::execution::*;
    }
    pub mod heuristic {
        pub use crate::heuristic::*;
    }
    pub mod knob {
        pub use crate::knob::*;
    }
    pub mod layout {
        pub use crate::layout::*;
    }
    pub mod normalization {
        pub use crate::normalization::*;
    }
    pub mod pointwise {
        pub use crate::pointwise::*;
    }
    pub mod reduction {
        pub use crate::reduction::*;
    }
    pub mod tensor {
        pub use crate::tensor::*;
    }
}

/// Legacy cuDNN descriptor and operation APIs.
pub mod legacy {
    pub mod ctc {
        pub use crate::ctc::*;
    }
    pub mod dropout {
        pub use crate::dropout::*;
    }
    pub mod lrn {
        pub use crate::lrn::*;
    }
    pub mod softmax {
        pub use crate::softmax::*;
    }
    pub mod spatial_transformer {
        pub use crate::spatial_transformer::*;
    }
}

pub mod frontend;

pub(crate) mod utility;

#[cfg(feature = "testing")]
pub mod testing;

use std::ptr;

use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{CudaRuntimeVersion, LibraryVersion, impl_enum_conversion, string_from_c_chars};
use singe_cuda::types::LibraryProperty;
use singe_cudnn_sys as sys;

use crate::{
    context::Context,
    error::{Result, Status},
    utility::to_u64,
};

/// Remote kernel error query mode used by [`query_runtime_error`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum RuntimeErrorQueryMode {
    /// Read the error storage location regardless of the kernel completion status.
    RawCode = sys::cudnnErrQueryMode_t::CUDNN_ERRQUERY_RAWCODE as _,
    /// Report whether all tasks in the configured cuDNN stream have completed.
    /// If that is the case, report the remote kernel error code.
    NonBlocking = sys::cudnnErrQueryMode_t::CUDNN_ERRQUERY_NONBLOCKING as _,
    /// Wait for all tasks in the configured cuDNN stream before reporting the remote kernel error code.
    Blocking = sys::cudnnErrQueryMode_t::CUDNN_ERRQUERY_BLOCKING as _,
}

impl_enum_conversion!(sys::cudnnErrQueryMode_t, RuntimeErrorQueryMode);

/// Severity passed to the configured cuDNN logging callback.
/// The numerical values match the values accepted by the `CUDNN_LOGLEVEL_DBG` environment variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum Severity {
    Fatal = sys::cudnnSeverity_t::CUDNN_SEV_FATAL as _,
    Error = sys::cudnnSeverity_t::CUDNN_SEV_ERROR as _,
    Warning = sys::cudnnSeverity_t::CUDNN_SEV_WARNING as _,
    Info = sys::cudnnSeverity_t::CUDNN_SEV_INFO as _,
}

impl_enum_conversion!(sys::cudnnSeverity_t, Severity);

pub type DebugInfo = sys::cudnnDebug_t;
pub type Callback = sys::cudnnCallback_t;

/// Queries the internal state of cuDNN error reporting.
///
/// # Errors
///
/// Returns an error if cuDNN cannot query the callback state.
pub fn callback() -> Result<(u32, *mut (), Callback)> {
    let mut mask = 0u32;
    let mut user_data = ptr::null_mut();
    let mut callback = None;
    unsafe {
        try_ffi!(sys::cudnnGetCallback(
            &raw mut mask,
            &raw mut user_data,
            &raw mut callback,
        ))?;
    }
    Ok((mask, user_data as _, callback))
}

/// Sets the internal state of cuDNN error reporting.
///
/// # Safety
///
/// `callback`, if present, must follow the cuDNN callback ABI. `user_data`
/// must remain valid for any callback invocation that may use it.
///
/// # Errors
///
/// Returns an error if cuDNN rejects the callback state.
pub unsafe fn set_callback(mask: u32, user_data: *mut (), callback: Callback) -> Result<()> {
    unsafe { try_ffi!(sys::cudnnSetCallback(mask, user_data as _, callback)) }
}

/// Returns the version number of the cuDNN library.
/// It returns the version number reported by the linked cuDNN library.
/// This can identify the cuDNN library dynamically loaded by the current process.
/// This distinguishes cuDNN libraries when the same application may be linked against different cuDNN versions.
pub fn version() -> Result<LibraryVersion> {
    let version = to_u64(unsafe { sys::cudnnGetVersion() }, "version")?;
    Ok(LibraryVersion::new(
        version / 10000,
        (version % 10000) / 100,
        version % 100,
    ))
}

/// The same version of a given cuDNN library can be compiled against different CUDA toolkit versions.
/// Returns the CUDA toolkit version that the currently used cuDNN library was compiled against.
pub fn cudart_version() -> Result<CudaRuntimeVersion> {
    to_u64(unsafe { sys::cudnnGetCudartVersion() }, "cudart version").map(CudaRuntimeVersion::from)
}

/// Returns the maximum SM version that the cuDNN library is aware of and supports natively.
/// Any SM version higher than this would be supported in forward compatibility mode.
/// For more information about forward compatibility, refer to the hardware forward compatibility
/// section in the cuDNN frontend developer guide.
///
/// A value indicating the latest known SM number for the current version of the library.
/// For example, if NVIDIA Hopper (GH100) is the latest known SM that the library is aware of, the value returned would be `900`.
pub fn max_device_version() -> u64 {
    unsafe { sys::cudnnGetMaxDeviceVersion() }
}

/// Cross-library version checker.
/// Each sub-library has a version checker that checks whether its own version matches that of its dependencies.
///
/// # Errors
///
/// Returns an error if cuDNN reports inconsistent graph sublibrary versions.
pub fn graph_version_check() -> Result<()> {
    unsafe { try_ffi!(sys::cudnnGraphVersionCheck()) }
}

/// Cross-library version checker.
/// Each sublibrary has a version checker that checks whether its own version matches that of its dependencies.
///
/// # Errors
///
/// Returns an error if cuDNN reports inconsistent operations sublibrary versions.
pub fn ops_version_check() -> Result<()> {
    unsafe { try_ffi!(sys::cudnnOpsVersionCheck()) }
}

/// Cross-library version checker.
/// Each sublibrary has a version checker that checks whether its own version matches that of its dependencies.
///
/// # Errors
///
/// Returns an error if cuDNN reports inconsistent advanced sublibrary versions.
pub fn adv_version_check() -> Result<()> {
    unsafe { try_ffi!(sys::cudnnAdvVersionCheck()) }
}

/// Cross-library version checker.
/// Each sublibrary has a version checker that checks whether its own version matches that of its dependencies.
///
/// # Errors
///
/// Returns an error if cuDNN reports inconsistent CNN sublibrary versions.
pub fn cnn_version_check() -> Result<()> {
    unsafe { try_ffi!(sys::cudnnCnnVersionCheck()) }
}

/// Returns the requested cuDNN library property.
/// See [`LibraryProperty`] for supported properties.
///
/// # Errors
///
/// Returns an error if cuDNN cannot report the requested property.
pub fn library_property(property: LibraryProperty) -> Result<i32> {
    let mut value = 0;
    unsafe {
        try_ffi!(sys::cudnnGetProperty(property.into(), &raw mut value))?;
    }
    Ok(value)
}

/// Returns the last encountered cuDNN error message in the current thread as a string.
/// Inside the cuDNN library, the messages are stored in thread local buffers.
/// The error is cleared after this call retrieves it.
pub fn last_error_string() -> String {
    let mut buffer = vec![0_i8; 4096];
    unsafe {
        sys::cudnnGetLastErrorString(buffer.as_mut_ptr(), buffer.len() as _);
    }
    string_from_c_chars(&buffer)
}

/// cuDNN library functions perform extensive input argument checking before launching GPU kernels.
/// The last step is to verify that the GPU kernel actually started.
/// A kernel launch failure is reported as [`Status::ExecutionFailed`] by the corresponding wrapper call.
/// Typically, after a GPU kernel starts, no runtime checks are performed by the kernel itself - numerical results are simply written to output buffers.
///
/// When persistent spatial batch normalization is selected, the algorithm may encounter
/// numerical overflows where ordinary spatial batch normalization performs just fine, albeit at a
/// slower speed.
/// Invoke [`query_runtime_error`] to verify that numerical overflows did not occur during kernel execution.
/// Those issues are reported by the kernel that performs computations.
///
/// Use [`query_runtime_error`] in polling and blocking software control flows.
/// There are two polling modes ([`RuntimeErrorQueryMode::RawCode`] and [`RuntimeErrorQueryMode::NonBlocking`]) and one blocking mode [`RuntimeErrorQueryMode::Blocking`].
///
/// [`RuntimeErrorQueryMode::RawCode`] reads the error storage location regardless of the kernel completion status.
/// The kernel might not even start and the error storage (allocated per cuDNN handle) might be used by an earlier call.
///
/// [`RuntimeErrorQueryMode::NonBlocking`] checks whether all tasks in the configured cuDNN stream are completed.
/// If tasks in that stream are pending, [`query_runtime_error`] returns immediately with [`Status::RuntimeInProgress`].
/// Otherwise, it returns the remote kernel error code.
///
/// In blocking mode ([`RuntimeErrorQueryMode::Blocking`]), this waits for all tasks to drain from the configured cuDNN stream before reporting the remote kernel error code.
/// The blocking flavor can be further adjusted by calling [`Device::set_flags`](singe_cuda::device::Device::set_flags) with the [`ContextFlags::SCHEDULE_SPIN`](singe_cuda::context::ContextFlags::SCHEDULE_SPIN), [`ContextFlags::SCHEDULE_YIELD`](singe_cuda::context::ContextFlags::SCHEDULE_YIELD), or [`ContextFlags::SCHEDULE_BLOCKING_SYNC`](singe_cuda::context::ContextFlags::SCHEDULE_BLOCKING_SYNC) flag.
///
/// Do not use [`RuntimeErrorQueryMode::NonBlocking`] or [`RuntimeErrorQueryMode::Blocking`] if [`Context::set_stream`] is called between functions that report runtime kernel errors and [`query_runtime_error`].
///
/// The result can be `Ok`, [`Status::RuntimeInProgress`], or [`Status::RuntimeFpOverflow`].
/// The remote kernel error is automatically cleared by [`query_runtime_error`].
///
/// Use [`query_runtime_error`] after batch normalization calls that use
/// persistent spatial mode.
///
/// # Errors
///
/// Returns an error if cuDNN rejects the input arguments, if the stream
/// synchronization or query used to retrieve the remote error fails, or if the
/// device cannot access zero-copy memory to report kernel errors.
pub fn query_runtime_error(ctx: &Context, mode: RuntimeErrorQueryMode) -> Result<Status> {
    let mut status = sys::cudnnStatus_t::CUDNN_STATUS_SUCCESS;
    unsafe {
        try_ffi!(sys::cudnnQueryRuntimeError(
            ctx.as_raw(),
            &raw mut status,
            mode.into(),
            ptr::null_mut(),
        ))?;
    }

    Ok(Status::from(status))
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use super::*;

    #[test]
    fn it_works() -> Result<()> {
        assert_ne!(version()?, 0);
        Ok(())
    }
}
