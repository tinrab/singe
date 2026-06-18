//! Safe cuDSS wrappers for direct sparse solver operations.
//!
//! This crate covers cuDSS context management, solver configuration/data
//! objects, matrix descriptors, execution phases, and basic logging controls.

pub mod config;
pub mod context;
pub mod data;
pub mod error;
pub mod logger;
pub mod matrix;
pub mod memory;
pub mod types;

mod utility;

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
    unsafe {
        try_ffi!(sys::cudssGetProperty(property.into(), &raw mut value))?;
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() -> Result<()> {
        let version = version()?;
        println!("cuDSS version: {version}");
        assert_ne!(version, 0);
        Ok(())
    }
}
