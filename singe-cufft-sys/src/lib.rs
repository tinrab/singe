//! Raw FFI bindings for cuFFT.
//!
//! Prefer the safe `singe-cufft` crate unless direct NVIDIA ABI access is required.

#![allow(deprecated, warnings, unused_qualifications, clippy::all)]

use num_enum::{IntoPrimitive, TryFromPrimitive};

#[cfg(feature = "cufft_12_2")]
use singe_cuda_sys::{
    library_types::{cudaDataType, cudaDataType_t, libraryPropertyType},
    runtime::cudaStream_t,
};

#[cfg(feature = "cufft_12_2")]
include!("sys_12200.rs");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let mut version = 0;
        unsafe {
            assert_eq!(cufftGetVersion(&mut version), cufftResult::CUFFT_SUCCESS,);
        }
        println!("cuFFT version: {}", version);
        assert_ne!(version, 0);
    }
}
