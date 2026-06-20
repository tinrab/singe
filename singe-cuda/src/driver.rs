use singe_cuda_sys::driver;

use crate::{error::Result, try_ffi};

/// Returns the version of CUDA supported by the driver.
/// The version is returned as `1000 * major + 10 * minor`.
/// For example, CUDA 9.2 would be represented by 9020.
///
/// # Errors
///
/// Returns an error if CUDA cannot query the driver version or if a previous asynchronous launch
/// reported an error.
pub fn version() -> Result<i32> {
    let mut version: i32 = 0;
    unsafe {
        try_ffi!(driver::cuDriverGetVersion(&raw mut version))?;
    }
    Ok(version)
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        assert_ne!(version().unwrap(), 0);
    }
}
