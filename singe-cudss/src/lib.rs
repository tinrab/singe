//! cuDSS wrappers for direct sparse solver operations.
//!
//! This crate covers cuDSS context management, solver configuration/data
//! objects, matrix descriptors, execution phases, and basic logging controls.
//! Matrix views retain their actual CUDA context and distinguish initialized
//! inputs from write-only destinations. Raw execution is public and unsafe: the
//! caller retains all resources through completion and checks numerical status
//! before treating produced values as initialized.

/// Solver configuration and typed configuration parameters.
pub mod config;
/// cuDSS handles, CUDA context ownership, streams, and execution phases.
pub mod context;
/// Solver state and host or device queries of solver results.
pub mod data;
/// Native statuses and errors from the Rust wrappers.
pub mod error;
/// Process-wide cuDSS logging controls and callbacks.
pub mod logger;
/// Dense and sparse matrix descriptors that borrow device storage.
pub mod matrix;
pub mod memory;
/// Numeric representations, algorithm choices, flags, and parameter identifiers.
pub mod types;
mod utility;

#[cfg(feature = "testing")]
/// Serialized CUDA device setup shared by cuDSS tests.
pub mod testing;

use singe_core::LibraryVersion;
use singe_cuda::types::LibraryProperty;
use singe_cudss_sys as sys;

use crate::error::Result;

/// Returns the cuDSS library version reported by the loaded runtime.
///
/// # Errors
///
/// Returns an error if cuDSS cannot report one of the version properties.
pub fn version() -> Result<LibraryVersion> {
    let major = library_property(LibraryProperty::Major)?;
    let minor = library_property(LibraryProperty::Minor)?;
    let patch = library_property(LibraryProperty::Patch)?;
    Ok(LibraryVersion::new(major as _, minor as _, patch as _))
}

/// Returns a cuDSS runtime library property.
///
/// cuDSS supports the standard CUDA [`LibraryProperty`] values for major,
/// minor, and patch version components.
///
/// # Errors
///
/// Returns an error if cuDSS cannot report the requested property.
pub fn library_property(property: LibraryProperty) -> Result<i32> {
    let mut value = 0;
    // SAFETY: `cudssGetProperty` receives valid output locations.
    unsafe {
        try_ffi!(sys::cudssGetProperty(property.into(), &raw mut value))?;
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "testing")]
    use crate::testing::setup_context;

    #[test]
    fn test_it_works() -> Result<()> {
        let version = version()?;
        println!("cuDSS version: {version}");
        assert_ne!(version, 0);
        Ok(())
    }

    #[cfg(feature = "testing")]
    #[test]
    fn test_setup_context_creates_context() -> Result<()> {
        let _context = setup_context()?;
        Ok(())
    }
}
