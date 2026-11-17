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

use singe_core::{CudaVersion, LibraryVersion};
use singe_cublas_sys as sys;
use singe_cuda::types::LibraryProperty;

use crate::{
    error::{Error, Result, try_ffi},
    utility::{to_size_t, to_usize},
};

/// Returns the cuBLASLt library version.
///
/// # Errors
///
/// Returns an error if cuBLASLt cannot report its version properties or a property is negative.
pub fn version() -> Result<LibraryVersion> {
    LibraryVersion::from_properties(
        library_property(LibraryProperty::Major)?,
        library_property(LibraryProperty::Minor)?,
        library_property(LibraryProperty::Patch)?,
    )
    .ok_or_else(|| Error::out_of_range("cublaslt version"))
}

/// Returns the CUDA Runtime library version used by cuBLASLt.
///
/// # Errors
///
/// Returns an error if the native runtime version does not fit in `u32`.
pub fn cudart_version() -> Result<CudaVersion> {
    // SAFETY: `cublasLtGetCudartVersion` takes no arguments and only reports a linked version.
    CudaVersion::from_raw(unsafe { sys::cublasLtGetCudartVersion() })
        .ok_or_else(|| Error::out_of_range("cublaslt cudart version"))
}

/// Returns the requested cuBLASLt library property.
/// See [`LibraryProperty`] for supported properties.
///
/// # Errors
///
/// Returns an error if cuBLASLt cannot report the requested property.
pub fn library_property(property: LibraryProperty) -> Result<i32> {
    let mut value = 0;
    // SAFETY: `cublasLtGetProperty` writes one `i32` to `value`, a live local.
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
    // SAFETY: `cublasLtHeuristicsCacheGetCapacity` writes one `size_t` to `capacity`, a live local.
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
/// Returns an error if `capacity` cannot be represented by the cuBLASLt integer type or if cuBLASLt rejects the update.
pub fn set_heuristics_cache_capacity(capacity: usize) -> Result<()> {
    let capacity = to_size_t(capacity, "heuristics cache capacity")?;
    // SAFETY: `cublasLtHeuristicsCacheSetCapacity` takes the capacity by value and reads no caller memory.
    unsafe {
        try_ffi!(sys::cublasLtHeuristicsCacheSetCapacity(capacity))?;
    }
    Ok(())
}

/// Instructs cuBLASLt not to use CPU instructions selected by `mask`.
///
/// This takes precedence over the `CUBLASLT_DISABLE_CPU_INSTRUCTIONS_MASK` environment variable.
pub fn disable_cpu_instructions_set_mask(mask: u32) {
    // SAFETY: `cublasLtDisableCpuInstructionsSetMask` takes the mask by value and reads no caller memory.
    unsafe {
        sys::cublasLtDisableCpuInstructionsSetMask(mask);
    }
}
