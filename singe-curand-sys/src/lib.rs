//! Raw FFI bindings for cuRAND.
//!
//! Prefer the safe `singe-curand` crate unless direct NVIDIA ABI access is required.

#![allow(deprecated, warnings, unused_qualifications, clippy::all)]

use num_enum::{IntoPrimitive, TryFromPrimitive};

#[cfg(feature = "curand_13_3")]
use singe_cuda_sys::{
    library_types::libraryPropertyType,
    runtime::{cudaError_t, cudaStream_t},
};

#[cfg(feature = "curand_13_3")]
include!("sys_10403.rs");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let mut version = 0;
        unsafe {
            assert_eq!(
                curandGetVersion(&mut version),
                curandStatus_t::CURAND_STATUS_SUCCESS,
            );
        }
        println!("cuRAND version: {}", version);
        assert_ne!(version, 0);
    }
}
