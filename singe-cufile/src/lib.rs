//! Safe Rust wrappers for the NVIDIA cuFile, which moves data directly between files and GPU memory with `GPUDirect` Storage.
//!
//! This crate provides:
//!
//! - **Driver session:** [`Driver`](driver::Driver), a shared lease that opens cuFile on first use and closes it after the last lease, with driver properties and limits.
//! - **File I/O:** [`FileHandle`](file::FileHandle) reads and writes that wait for the transfer, vectored forms, stream-ordered `*_scoped` forms, and `unsafe` [`read_async`](file::FileHandle::read_async) and [`write_async`](file::FileHandle::write_async).
//! - **Registration:** registered device buffers and streams.
//! - **Batch I/O:** [`Batch`](batch::Batch) submits several requests at once and reports an event for each.
//! - **Configuration and statistics:** process-wide parameters, buffer-pool settings, operation counters, and per-GPU statistics.
//!
//! # Example
//!
//! Write 4096 bytes from GPU memory to a file, read them back into new GPU memory, and compare the two buffers:
//!
//! ```rust,no_run
//! use std::{fs::OpenOptions, os::unix::fs::OpenOptionsExt};
//!
//! use singe_cuda::{context::Context, memory::DeviceMemory};
//! use singe_cufile::{
//!     buffer::{IoSlice, IoSliceMut},
//!     driver::Driver,
//!     file::FileHandle,
//! };
//!
//! fn main() -> anyhow::Result<()> {
//!     let context = Context::create()?;
//!     let driver = Driver::create()?;
//!
//!     let file = OpenOptions::new()
//!         .read(true)
//!         .write(true)
//!         .create(true)
//!         .truncate(true)
//!         .custom_flags(libc::O_DIRECT)
//!         .open("cufile-example.bin")?;
//!     let handle = FileHandle::create(&driver, &file)?;
//!
//!     let source = DeviceMemory::<u8>::filled(&context, 4096, 0xab)?;
//!     let written = handle.write(IoSlice::create(&source, ..)?, 0)?;
//!
//!     let mut destination = DeviceMemory::<u8>::create_uninit(&context, written)?;
//!     let read = handle.read(IoSliceMut::create(&mut destination, ..)?, 0)?;
//!     // SAFETY: the completed read initialized exactly the first `read` bytes.
//!     let destination = unsafe { destination.assume_init_prefix(read)? };
//!     assert_eq!(destination.copy_to_host_vec()?, source.copy_to_host_vec()?);
//!     Ok(())
//! }
//! ```
//!
//! [`write`](file::FileHandle::write) and [`read`](file::FileHandle::read) wait for the transfer and return the number of bytes moved, which can be less than requested.
//! The second argument is the offset in the file.
//! [`IoSlice::create`](buffer::IoSlice::create) takes an initialized device or host buffer, and [`IoSliceMut::create`](buffer::IoSliceMut::create) also takes uninitialized device memory; `..` selects every byte.
//!
//! The direct path needs a file opened with `O_DIRECT` on a file system `GPUDirect` Storage supports, such as ext4 or XFS on local `NVMe` drives.
//! When its configuration allows it, cuFile's compatibility mode runs the same calls through host bounce buffers instead, and [`Driver::properties`](driver::Driver::properties) reports which modes are available.
//!
//! # Raw execution
//!
//! [`FileHandle::read_async`](file::FileHandle::read_async) and [`FileHandle::write_async`](file::FileHandle::write_async) check that the buffer and stream share a CUDA context, reject a capturing stream, order the stream after the buffer's producers, enqueue the I/O, and return an [`AsyncIo`](stream::AsyncIo) without waiting.
//! [`Batch::execute_async`](batch::Batch::execute_async) waits on the host for its buffers' producers, submits the requests, and returns a [`BatchSubmission`](batch::BatchSubmission).
//! Dropping or waiting on the returned operation completes the I/O before the buffer borrows end.
//! The caller must:
//!
//! - keep the buffers alive and free of conflicting access until the I/O completes, if it forgets the returned operation instead of dropping or waiting on it.
//!
//! The safe [`read`](file::FileHandle::read) and [`write`](file::FileHandle::write) wait for the I/O, and the `_scoped` forms drain the scope's stream before the borrowed buffers are released.

#[cfg(not(feature = "cufile_1_19"))]
compile_error!("singe-cufile requires a version feature, such as `cufile_1_19`");

/// Batched file I/O requests, submission, and completion events.
pub mod batch;

pub mod buffer;

/// Process-wide cuFile configuration parameters and buffer-pool settings.
pub mod config;

/// Shared ownership of the cuFile driver session and its properties.
pub mod driver;

/// cuFile status codes, wrapper errors, and result types.
pub mod error;

/// Registered file handles and synchronous, vectored, and stream-ordered I/O.
pub mod file;

/// cuFile operation counters, size histograms, and per-GPU statistics.
pub mod stats;

/// Stream registration and pending stream-ordered file I/O.
pub mod stream;

/// Typed cuFile configuration selectors, flags, and operation states.
pub mod types;

pub(crate) mod utility;

use std::path::Path;

pub use singe_core::LibraryVersion;
use singe_core::{PackedVersion, path_to_cstring};
pub use singe_cufile_sys as sys;

use crate::error::{Error, Result, try_ffi};

/// Returns the loaded cuFile library version.
///
/// cuFile reports only the major and minor components, so the patch component is zero.
///
/// # Errors
///
/// Returns an error if cuFile cannot report the loaded library version.
pub fn version() -> Result<LibraryVersion> {
    let mut version = 0;
    // SAFETY: `version` is a live local `c_int` that cuFile writes the version into.
    unsafe {
        try_ffi!(sys::cuFileGetVersion(&raw mut version))?;
    }

    // cuFile packs versions as `1000 * major + 10 * minor`.
    LibraryVersion::from_packed(version, PackedVersion::Major1000Minor10)
        .ok_or_else(|| Error::out_of_range("cufile version"))
}

/// Exports the `GPUDirect` Storage `PCIe` topology to `path`.
///
/// The path is passed to cuFile as a C string, so interior NUL bytes are rejected.
/// File creation, overwriting, permissions, and the topology format are controlled by the cuFile library.
///
/// # Errors
///
/// Returns an error if `path` contains an interior NUL byte or if cuFile cannot write the topology file.
pub fn export_pcie_topology(path: impl AsRef<Path>) -> Result<()> {
    let path = path_to_cstring(path.as_ref())?;
    // SAFETY: `path` is a NUL-terminated `CString` that outlives the call.
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
        assert!(
            version.major() >= 1,
            "cuFile major version {version} is implausible"
        );
        Ok(())
    }
}
