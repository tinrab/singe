//! Safe cuRAND host API wrappers for CUDA random number generation.
//!
//! GPU generators retain their actual context and stream. Completed generation
//! returns initialized output views; public unsafe enqueues expose native work.
//! Separate CPU generators produce host values synchronously.

pub mod distribution;
pub mod error;
pub mod generator;
pub mod quasi;
/// Generator kinds, output orderings, and Sobol direction-vector sets.
pub mod types;

#[cfg(feature = "testing")]
/// CUDA context setup shared by cuRAND tests.
pub mod testing;

use singe_core::LibraryVersion;
use singe_cuda::types::LibraryProperty;
use singe_curand_sys as sys;

use crate::error::Result;

/// Returns the loaded cuRAND library version.
///
/// # Errors
///
/// Returns the cuRAND status if the version query fails.
pub fn version() -> Result<LibraryVersion> {
    let mut version = 0;
    // SAFETY: `curandGetVersion` receives valid output locations.
    unsafe {
        try_ffi!(sys::curandGetVersion(&raw mut version))?;
    }
    let version = version as u64;
    Ok(LibraryVersion::new(
        version / 1000,
        (version % 1000) / 100,
        version % 100,
    ))
}

/// Returns a cuRAND library property value.
///
/// See [`LibraryProperty`] for supported properties.
///
/// # Errors
///
/// Returns the cuRAND status if the property query fails.
pub fn library_property(property: LibraryProperty) -> Result<i32> {
    let mut value = 0;
    // SAFETY: `curandGetProperty` receives valid output locations.
    unsafe {
        try_ffi!(sys::curandGetProperty(property.into(), &raw mut value))?;
    }
    Ok(value)
}

mod utility;
