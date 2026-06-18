use singe_cuda_sys::runtime;

use crate::{error::Result, try_ffi};

/// Returns the version number of the current CUDA Runtime instance.
/// The version is returned as `1000 * major + 10 * minor`.
/// For example, CUDA 9.2 would be represented by 9020.
///
/// As of CUDA 12.0, this no longer initializes CUDA.
/// The purpose of this call is solely to return a compile-time constant stating the CUDA Toolkit version in the above format.
///
/// This may also return [`crate::error::Status::NotInitialized`], [`crate::error::Status::CallRequiresNewerDriver`],
/// or [`crate::error::Status::NoDevice`] if the call tries to initialize internal CUDA Runtime state.
///
/// No CUDA function may be called from a [`Stream::add_callback`](crate::stream::Stream::add_callback)
/// callback. [`crate::error::Status::NotPermitted`] may be returned as a diagnostic in that case, but this is not guaranteed.
///
/// # Errors
///
/// Returns an error if CUDA Runtime cannot report its version, if a previous asynchronous launch
/// failed, or if CUDA Runtime reports initialization or driver/device availability errors.
pub fn version() -> Result<i32> {
    let mut version = 0;
    unsafe {
        try_ffi!(runtime::cudaRuntimeGetVersion(&raw mut version))?;
    }
    Ok(version)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        assert_ne!(version().unwrap(), 0);
    }
}
