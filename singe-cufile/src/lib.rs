//! Safe `GPUDirect` Storage cuFile wrappers.

/// Batched file I/O requests, submission, and completion events.
pub mod batch;
pub mod buffer;
/// Process-wide cuFile configuration parameters and buffer-pool settings.
pub mod config;
/// Shared ownership of the cuFile driver session and its properties.
pub mod driver;
/// cuFile status codes, wrapper errors, and result types.
pub mod error;
/// Registered file handles and synchronous, vectored, and asynchronous I/O.
pub mod file;
/// cuFile operation counters, size histograms, and per-GPU statistics.
pub mod stats;
/// Stream registration and borrowed asynchronous file I/O.
pub mod stream;
/// Typed cuFile configuration selectors, flags, and operation states.
pub mod types;
pub(crate) mod utility;

/// CUDA context setup for cuFile tests.
#[cfg(feature = "testing")]
pub mod testing;

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
    // SAFETY: `cuFileGetVersion` receives valid output locations.
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

/// Exports the `GPUDirect` Storage `PCIe` topology to `path`.
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
    // SAFETY: `cuFileExportPCIeTopology` receives pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cuFileExportPCIeTopology(path.as_ptr()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_it_works() -> Result<()> {
        let version = version()?;
        assert_ne!(version, 0);
        Ok(())
    }
}
