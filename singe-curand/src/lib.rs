//! Safe cuRAND host API wrappers for CUDA random number generation.
//!
//! This crate wraps generator creation, stream binding, seeding, ordering, and
//! distribution generation over the raw `singe-curand-sys` bindings.

pub mod distribution;
pub mod error;
pub mod generator;
pub mod quasi;
pub mod types;

#[cfg(feature = "testing")]
pub mod testing;

use singe_core::LibraryVersion;
use singe_cuda::types::LibraryProperty;
use singe_curand_sys as sys;

use crate::error::Result;

/// Returns the loaded cuRAND library version.
pub fn version() -> Result<LibraryVersion> {
    let mut version = 0;
    unsafe {
        try_ffi!(sys::curandGetVersion(&raw mut version))?;
    }
    let version = version as u64;
    Ok(LibraryVersion::new(
        version / 10000,
        (version % 10000) / 100,
        version % 100,
    ))
}

/// Returns a cuRAND library property value.
///
/// See [`LibraryProperty`] for supported properties.
pub fn library_property(property: LibraryProperty) -> Result<i32> {
    let mut value = 0;
    unsafe {
        try_ffi!(sys::curandGetProperty(property.into(), &raw mut value))?;
    }
    Ok(value)
}
