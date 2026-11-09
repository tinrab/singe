//! Raw FFI bindings for cuSOLVER.
//!
//! Prefer the safe `singe-cusolver` crate unless direct NVIDIA ABI access is required.

#![allow(
    non_camel_case_types,
    non_snake_case,
    unnecessary_transmutes,
    unused_imports,
    unused_qualifications,
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
    library_types::{
        cudaDataType, cudaDataType_t, cudaEmulationMantissaControl_t,
        cudaEmulationSpecialValuesSupport_t, cudaEmulationStrategy_t, libraryPropertyType,
    },
    runtime::cudaStream_t,
};

#[cfg(feature = "cusolver_12_3")]
include!("sys_12304.rs");

pub use singe_loader::LoadError;

static LIBRARY: singe_loader::Library = singe_loader::Library::cusolver();

// Loads cuSOLVER, which each of its functions otherwise does on first call.
pub fn load() -> Result<(), LoadError> {
    LIBRARY.load()
}

#[cfg(feature = "cusolver_12_3")]
singe_loader::missing_status!(
    cusolverStatus_t,
    library: cusolverStatus_t::CUSOLVER_STATUS_NOT_INITIALIZED,
    symbol: cusolverStatus_t::CUSOLVER_STATUS_NOT_SUPPORTED,
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_it_works() {
        let mut version = 0;
        // SAFETY: the version query writes only into the local output locations.
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
