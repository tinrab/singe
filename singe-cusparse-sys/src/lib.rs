//! Raw FFI bindings for cuSPARSE.
//!
//! Prefer the safe `singe-cusparse` crate unless direct NVIDIA ABI access is required.

#![allow(
    deprecated,
    warnings,
    dead_code,
    non_camel_case_types,
    non_snake_case,
    unnecessary_transmutes,
    unused_imports,
    unused_qualifications,
    clippy::all,
    clippy::correctness,
    clippy::suspicious,
    clippy::complexity,
    clippy::perf,
    clippy::style,
    clippy::absolute_paths,
    clippy::pedantic,
    clippy::nursery,
    reason = "generated bindings preserve upstream ABI conventions and documentation"
)]

use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_cuda_sys::{
    library_types::{cudaDataType, cudaDataType_t, libraryPropertyType},
    runtime::cudaStream_t,
};

#[cfg(feature = "cusparse_12_8")]
include!("sys_12806.rs");

#[cfg(test)]
mod tests {
    use std::ptr;

    use super::*;

    #[test]
    fn test_it_works() {
        let mut version = 0;
        // SAFETY: the handle is created, used and destroyed within this block.
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
