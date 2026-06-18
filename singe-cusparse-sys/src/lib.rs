//! Raw FFI bindings for cuSPARSE.
//!
//! Prefer the safe `singe-cusparse` crate unless direct NVIDIA ABI access is required.

#![allow(deprecated, warnings, unused_qualifications, clippy::all)]

use num_enum::{IntoPrimitive, TryFromPrimitive};

use singe_cuda_sys::{
    library_types::{cudaDataType, cudaDataType_t, libraryPropertyType},
    runtime::cudaStream_t,
};

#[cfg(feature = "cusparse_12_7")]
include!("sys_12710.rs");

#[cfg(test)]
mod tests {
    use std::ptr;

    use super::*;

    #[test]
    fn it_works() {
        let mut version = 0;
        unsafe {
            let mut ctx = ptr::null_mut();
            assert_eq!(
                cusparseCreate(&mut ctx),
                cusparseStatus_t::CUSPARSE_STATUS_SUCCESS,
            );
            cusparseGetVersion(ctx, &mut version);
        }
        println!("cuSPARSE version: {}", version);
        assert_ne!(version, 0);
    }
}
