//! Raw FFI bindings for cuSOLVER.
//!
//! Prefer the safe `singe-cusolver` crate unless direct NVIDIA ABI access is required.

#![allow(deprecated, warnings, unused_qualifications, clippy::all)]

use num_enum::{IntoPrimitive, TryFromPrimitive};

use singe_cuda_sys::{
    library_types::{
        cudaDataType, cudaDataType_t, cudaEmulationMantissaControl_t,
        cudaEmulationSpecialValuesSupport_t, cudaEmulationStrategy_t, libraryPropertyType,
    },
    runtime::cudaStream_t,
};

#[cfg(feature = "cusolver_12_2")]
include!("sys_12202.rs");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let mut version = 0;
        unsafe {
            assert_eq!(
                cusolverGetVersion(&mut version),
                cusolverStatus_t::CUSOLVER_STATUS_SUCCESS,
            );
        }
        println!("cuSOLVER version: {}", version);
        assert_ne!(version, 0);
    }
}
