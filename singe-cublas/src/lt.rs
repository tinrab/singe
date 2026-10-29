/// cuBLASLt handle ownership and CUDA context management.
pub mod context;
/// Matrix layouts, transforms, and floating-point emulation descriptors.
pub mod descriptor;
pub mod logger;
/// Matrix multiplication, transforms, and algorithm heuristics.
pub mod matmul;
/// Typed cuBLASLt attributes, layouts, and algorithm settings.
pub mod types;
pub(crate) mod utility;

use singe_core::{CudaRuntimeVersion, LibraryVersion};
use singe_cublas_sys as sys;
use singe_cuda::types::LibraryProperty;

use crate::{
    error::{Error, Result},
    try_ffi,
    utility::{to_u64, to_usize},
};

/// Native cuBLASLt logging callback signature.
pub type LoggerCallback = sys::cublasLtLoggerCallback_t;

/// Returns the cuBLASLt library version.
///
/// # Errors
///
/// Returns an error if the native library version cannot be represented as `u64`.
pub fn version() -> Result<LibraryVersion> {
    // SAFETY: `cublasLtGetVersion` takes no arguments.
    let version = to_u64(unsafe { sys::cublasLtGetVersion() }, "version")?;
    Ok(LibraryVersion::new(
        version / 10000,
        (version % 10000) / 100,
        version % 100,
    ))
}

/// Returns the CUDA Runtime library version used by cuBLASLt.
///
/// # Errors
///
/// Returns an error if the native runtime version cannot be represented as `u64`.
pub fn cudart_version() -> Result<CudaRuntimeVersion> {
    // SAFETY: `cublasLtGetCudartVersion` takes no arguments.
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
    // SAFETY: `cublasLtGetProperty` receives valid output locations.
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
    // SAFETY: `cublasLtHeuristicsCacheGetCapacity` receives valid output locations.
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
    let capacity = capacity
        .try_into()
        .map_err(|_| Error::out_of_range("heuristics cache capacity"))?;
    // SAFETY: `cublasLtHeuristicsCacheSetCapacity` receives only plain values.
    unsafe {
        try_ffi!(sys::cublasLtHeuristicsCacheSetCapacity(capacity))?;
    }
    Ok(())
}

/// Instructs cuBLASLt not to use CPU instructions selected by `mask`.
///
/// This takes precedence over the `CUBLASLT_DISABLE_CPU_INSTRUCTIONS_MASK` environment variable.
pub fn disable_cpu_instructions_set_mask(mask: u32) {
    // SAFETY: `cublasLtDisableCpuInstructionsSetMask` receives only plain values.
    unsafe {
        sys::cublasLtDisableCpuInstructionsSetMask(mask);
    }
}

/// Experimental: disables cuBLASLt logging for the current process.
///
/// # Errors
///
/// Returns an error if cuBLASLt cannot disable logging.
pub fn force_disable_logger() -> Result<()> {
    // SAFETY: `cublasLtLoggerForceDisable` takes no arguments.
    unsafe {
        try_ffi!(sys::cublasLtLoggerForceDisable())?;
    }
    Ok(())
}
