use std::{ffi::CString, path::Path, sync::Arc};

use singe_core::{checked_conversions, path_to_cstring};
use singe_cuda::stream::{BorrowedStream, Stream as CudaStream};
use singe_nvtiff_sys as sys;

use crate::error::{Error, Result};

checked_conversions!(Error {
    to_size_t -> sys::size_t,
    to_usize -> usize,
    to_u32 -> u32,
    to_i32 -> i32,
});

pub fn c_string(value: impl AsRef<str>) -> Result<CString> {
    Ok(CString::new(value.as_ref())?)
}

pub fn path_c_string(path: impl AsRef<Path>) -> Result<CString> {
    Ok(path_to_cstring(path.as_ref())?)
}

/// Borrows `stream` for a native owner that must not keep the stream wrapper alive.
///
/// Capture cleanup retains such owners on the stream itself, so retaining the wrapper would form a cycle.
pub fn cleanup_stream(stream: &CudaStream) -> BorrowedStream {
    // SAFETY: the owner's wrapper keeps `stream` alive until the owner is dropped,
    // either directly or by the stream's capture cleanup.
    unsafe { BorrowedStream::from_raw(stream.as_raw(), Arc::clone(stream.context())) }
}
