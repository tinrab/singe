//! Raw FFI bindings for cuFile.
//!
//! Use this crate when direct NVIDIA cuFile ABI access is required.

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
use singe_cuda_sys::driver::{CUresult, CUstream};

#[cfg(feature = "cufile_1_19")]
include!("sys_1190.rs");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_it_works() {
        let mut version = 0;
        // SAFETY: the version query writes only into the local output locations.
        let status = unsafe { cuFileGetVersion(&raw mut version) };
        assert_eq!(status.err, CUfileOpError::CU_FILE_SUCCESS);
        assert_ne!(version, 0);
    }
}
