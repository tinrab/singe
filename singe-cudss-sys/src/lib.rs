//! Raw FFI bindings for cuDSS.
//!
//! Prefer the safe `singe-cudss` crate unless direct NVIDIA ABI access is required.

#![allow(deprecated, warnings, unused_qualifications, clippy::all)]

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
    fn it_works() {
        let mut minor = 0;
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
