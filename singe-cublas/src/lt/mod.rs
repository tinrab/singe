pub mod context;
pub mod descriptor;
pub mod matmul;
pub mod types;

pub(crate) mod utility;

use std::{ffi::CString, path::Path};

use singe_core::{CudaRuntimeVersion, LibraryVersion};
use singe_cublas_sys as sys;
use singe_cuda::types::LibraryProperty;

use crate::{
    error::{Error, Result},
    try_ffi,
    utility::{to_u64, to_usize},
};

pub type LoggerCallback = sys::cublasLtLoggerCallback_t;

/// Returns the cuBLASLt library version.
pub fn version() -> Result<LibraryVersion> {
    let version = to_u64(unsafe { sys::cublasLtGetVersion() }, "version")?;
    Ok(LibraryVersion::new(
        version / 10000,
        (version % 10000) / 100,
        version % 100,
    ))
}

/// Returns the CUDA Runtime library version used by cuBLASLt.
pub fn cudart_version() -> Result<CudaRuntimeVersion> {
    to_u64(unsafe { sys::cublasLtGetCudartVersion() }, "cudart version")
        .map(CudaRuntimeVersion::from)
}

/// Returns the requested cuBLASLt library property.
/// See [`LibraryProperty`] for supported properties.
///
/// # Errors
///
/// Returns an error if cuBLASLt cannot report the requested property.
pub fn library_property(property: LibraryProperty) -> Result<i32> {
    let mut value = 0;
    unsafe {
        try_ffi!(sys::cublasLtGetProperty(property.into(), &raw mut value))?;
    }
    Ok(value)
}

/// Returns the heuristics cache capacity.
///
/// # Errors
///
/// Returns an error if cuBLASLt cannot query the current capacity.
pub fn heuristics_cache_capacity() -> Result<usize> {
    let mut capacity = 0;
    unsafe {
        try_ffi!(sys::cublasLtHeuristicsCacheGetCapacity(&raw mut capacity))?;
    }
    to_usize(capacity, "heuristics cache capacity")
}

/// Sets the heuristics cache capacity.
/// Use a `capacity` of `0` to disable the heuristics cache.
///
/// This takes precedence over the `CUBLASLT_HEURISTICS_CACHE_CAPACITY` environment variable.
///
/// # Errors
///
/// Returns an error if `capacity` cannot be represented by the cuBLASLt integer type or if cuBLASLt
/// rejects the update.
pub fn set_heuristics_cache_capacity(capacity: usize) -> Result<()> {
    let capacity = capacity.try_into().map_err(|_| Error::OutOfRange {
        name: "heuristics cache capacity".into(),
    })?;
    unsafe {
        try_ffi!(sys::cublasLtHeuristicsCacheSetCapacity(capacity))?;
    }
    Ok(())
}

/// Instructs cuBLASLt not to use CPU instructions selected by `mask`.
///
/// This takes precedence over the `CUBLASLT_DISABLE_CPU_INSTRUCTIONS_MASK` environment variable.
pub fn disable_cpu_instructions_set_mask(mask: u32) {
    unsafe {
        sys::cublasLtDisableCpuInstructionsSetMask(mask);
    }
}

/// Experimental: sets the cuBLASLt logging callback.
///
/// # Errors
///
/// Returns an error if cuBLASLt rejects the callback.
pub fn set_logger_callback(callback: LoggerCallback) -> Result<()> {
    unsafe {
        try_ffi!(sys::cublasLtLoggerSetCallback(callback))?;
    }
    Ok(())
}

/// Experimental: opens `path` and uses it as the cuBLASLt logging output.
///
/// # Errors
///
/// Returns an error if `path` contains an interior NUL byte or if cuBLASLt cannot open the file.
pub fn logger_open_file(path: &Path) -> Result<()> {
    let path = CString::new(path.as_os_str().to_string_lossy().as_bytes())?;
    unsafe {
        try_ffi!(sys::cublasLtLoggerOpenFile(path.as_ptr()))?;
    }
    Ok(())
}

/// Experimental: sets the cuBLASLt logging level.
///
/// # Errors
///
/// Returns an error if `level` is not accepted by cuBLASLt.
pub fn set_logger_level(level: i32) -> Result<()> {
    unsafe {
        try_ffi!(sys::cublasLtLoggerSetLevel(level))?;
    }
    Ok(())
}

/// Experimental: sets the cuBLASLt logging mask.
///
/// # Errors
///
/// Returns an error if cuBLASLt rejects `mask`.
pub fn set_logger_mask(mask: i32) -> Result<()> {
    unsafe {
        try_ffi!(sys::cublasLtLoggerSetMask(mask))?;
    }
    Ok(())
}

/// Experimental: disables cuBLASLt logging for the current process.
///
/// # Errors
///
/// Returns an error if cuBLASLt cannot disable logging.
pub fn force_disable_logger() -> Result<()> {
    unsafe {
        try_ffi!(sys::cublasLtLoggerForceDisable())?;
    }
    Ok(())
}
