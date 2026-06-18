//! Safe GPUDirect Storage cuFile wrappers.

pub mod batch;
pub mod buffer;
pub mod config;
pub mod driver;
pub mod error;
pub mod file;
pub mod stats;
pub mod stream;
pub mod types;

pub(crate) mod utility;

use std::path::Path;

use singe_core::{LibraryVersion, path_to_cstring};
use singe_cufile_sys as sys;

use crate::error::Result;

/// Returns the loaded cuFile library version.
///
/// The value uses NVIDIA's encoded integer version format.
///
/// # Errors
///
/// Returns an error if cuFile cannot report the loaded library version.
pub fn version() -> Result<LibraryVersion> {
    let mut version = 0;
    unsafe {
        try_ffi!(sys::cuFileGetVersion(&raw mut version))?;
    }
    let version = version as u64;
    Ok(LibraryVersion::new(
        version / 10000,
        (version % 10000) / 100,
        version % 100,
    ))
}

/// Exports the GPUDirect Storage PCIe topology to `path`.
///
/// The path is passed to cuFile as a C string, so interior NUL bytes are rejected.
/// File creation, overwriting, permissions, and the topology format are controlled
/// by the cuFile library.
///
/// # Errors
///
/// Returns an error if `path` contains an interior NUL byte or if cuFile cannot
/// write the topology file.
pub fn export_pcie_topology(path: impl AsRef<Path>) -> Result<()> {
    let path = path_to_cstring(path.as_ref())?;
    unsafe {
        try_ffi!(sys::cuFileExportPCIeTopology(path.as_ptr()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() -> Result<()> {
        let version = version()?;
        assert_ne!(version, 0);
        Ok(())
    }
}
