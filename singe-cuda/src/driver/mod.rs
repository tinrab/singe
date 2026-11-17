//! CUDA driver version queries.

pub(crate) mod entry_points;

use singe_core::CudaVersion;
use singe_cuda_sys::driver;

use crate::{
    error::{Error, Result},
    try_ffi,
};

/// Returns the newest CUDA version the driver supports.
///
/// # Errors
///
/// Returns an error if CUDA cannot query the driver version, if a previous asynchronous launch reported an error, or if the version is negative.
pub fn version() -> Result<CudaVersion> {
    let mut version: i32 = 0;
    // SAFETY: `version` is a local that `cuDriverGetVersion` writes through an exclusive pointer.
    unsafe {
        try_ffi!(driver::cuDriverGetVersion(&raw mut version))?;
    }
    CudaVersion::from_raw(version).ok_or_else(|| Error::out_of_range("cuda driver version"))
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use super::*;

    #[test]
    fn test_it_works() {
        assert_ne!(version().unwrap().major(), 0);
    }
}
