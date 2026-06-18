//! Safe cuSOLVER wrappers for dense linear algebra and solver operations.
//!
//! This crate covers cuSOLVER context management, dense matrix routines,
//! eigenvalue helpers, SVD helpers, iterative refinement solver APIs, and parameter objects.

pub mod context;
pub mod dense;
pub mod eigen;
pub mod error;
pub mod irs;
pub mod layout;
pub mod params;
pub mod svd;
pub mod types;

pub(crate) mod utility;

#[cfg(feature = "testing")]
pub mod testing;

use singe_core::LibraryVersion;
use singe_cuda::types::LibraryProperty;
use singe_cusolver_sys as sys;

use crate::error::Result;

/// Returns the cuSOLVER library version.
///
/// # Errors
///
/// Returns an error if cuSOLVER cannot report the loaded library version.
pub fn version() -> Result<LibraryVersion> {
    let mut version = 0;
    unsafe {
        try_ffi!(sys::cusolverGetVersion(&raw mut version))?;
    }
    let version = version as u64;
    Ok(LibraryVersion::new(
        version / 10000,
        (version % 10000) / 100,
        version % 100,
    ))
}

/// Returns the requested cuSOLVER library property.
/// See [`LibraryProperty`] for supported properties.
///
/// # Errors
///
/// Returns an error if cuSOLVER cannot report the requested property.
pub fn library_property(property: LibraryProperty) -> Result<i32> {
    let mut value = 0;
    unsafe {
        try_ffi!(sys::cusolverGetProperty(property.into(), &raw mut value))?;
    }
    Ok(value)
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use super::*;
    use crate::testing::setup_context_if_available;

    #[test]
    fn it_works() -> Result<()> {
        let Some(_ctx) = setup_context_if_available()? else {
            return Ok(());
        };
        let version = version()?;
        println!("cuSOLVER version: {version}");
        assert_ne!(version, 0);
        assert_ne!(library_property(LibraryProperty::Major)?, 0);
        Ok(())
    }
}
