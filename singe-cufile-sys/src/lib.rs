//! Raw FFI bindings for cuFile.
//!
//! Use this crate when direct NVIDIA cuFile ABI access is required.

#![allow(deprecated, warnings, unused_qualifications, clippy::all)]

use num_enum::{IntoPrimitive, TryFromPrimitive};

use singe_cuda_sys::driver::{CUresult, CUstream};

#[cfg(feature = "cufile_1_18")]
include!("sys_1180.rs");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let mut version = 0;
        let status = unsafe { cuFileGetVersion(&raw mut version) };
        assert_eq!(status.err, CUfileOpError::CU_FILE_SUCCESS);
        assert_ne!(version, 0);
    }
}
