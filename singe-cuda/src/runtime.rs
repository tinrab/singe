//! CUDA runtime version queries.

use singe_core::CudaVersion;
use singe_cuda_sys::runtime;

use crate::{
    error::{Error, Result},
    try_ffi,
};

/// Returns the version of the loaded CUDA Runtime.
///
/// As of CUDA 12.0, this no longer initializes CUDA.
/// The purpose of this call is solely to return a compile-time constant stating the CUDA Toolkit version.
///
/// This may also return [`crate::error::Status::NotInitialized`], [`crate::error::Status::CallRequiresNewerDriver`], or [`crate::error::Status::NoDevice`] if the call tries to initialize internal CUDA Runtime state.
///
/// No CUDA function may be called from a [`Stream::add_callback`](crate::stream::Stream::add_callback) callback.
/// [`crate::error::Status::NotPermitted`] may be returned as a diagnostic in that case, but this is not guaranteed.
///
/// # Errors
///
/// Returns an error if CUDA Runtime cannot report its version, if a previous asynchronous launch failed, if CUDA Runtime reports initialization or driver/device availability errors, or if the version is negative.
pub fn version() -> Result<CudaVersion> {
    let mut version = 0;
    // SAFETY: `version` is a local that `cudaRuntimeGetVersion` writes through an exclusive pointer.
    unsafe {
        try_ffi!(runtime::cudaRuntimeGetVersion(&raw mut version))?;
    }
    CudaVersion::from_raw(version).ok_or_else(|| Error::out_of_range("cuda runtime version"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_it_works() {
        assert_ne!(version().unwrap().major(), 0);
    }
}
