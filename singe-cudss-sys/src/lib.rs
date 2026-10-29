//! Raw FFI bindings for cuDSS.
//!
//! Prefer the safe `singe-cudss` crate unless direct NVIDIA ABI access is required.

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
    library_types::{cudaDataType_t, libraryPropertyType},
    runtime::cudaStream_t,
};

#[cfg(feature = "cudss_0_8")]
include!("sys_800.rs");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_it_works() {
        let mut minor = 0;
        // SAFETY: the version query writes only into the local output locations.
        unsafe {
            assert_eq!(
                cudssGetProperty(libraryPropertyType::MINOR_VERSION, &raw mut minor),
                cudssStatus_t::CUDSS_STATUS_SUCCESS,
            );
        }
        println!("cuDSS minor version: {}", minor);
        assert_ne!(minor, 0);
    }
}
