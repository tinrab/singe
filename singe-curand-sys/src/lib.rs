//! Raw FFI bindings for cuRAND.
//!
//! Prefer the safe `singe-curand` crate unless direct NVIDIA ABI access is required.

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
#[cfg(feature = "curand_13_4")]
use singe_cuda_sys::{
    library_types::libraryPropertyType,
    runtime::{cudaError_t, cudaStream_t},
};

#[cfg(feature = "curand_13_4")]
include!("sys_10404.rs");

pub use singe_loader::LoadError;

static LIBRARY: singe_loader::Library = singe_loader::Library::curand();

// Loads cuRAND, which each of its functions otherwise does on first call.
pub fn load() -> Result<(), LoadError> {
    LIBRARY.load()
}

#[cfg(feature = "curand_13_4")]
singe_loader::missing_status!(
    curandStatus_t,
    library: curandStatus_t::CURAND_STATUS_INITIALIZATION_FAILED,
    symbol: curandStatus_t::CURAND_STATUS_INTERNAL_ERROR,
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
                curandGetVersion(&mut version),
                curandStatus_t::CURAND_STATUS_SUCCESS,
            );
        }
        println!("cuRAND version: {}", version);
        assert_ne!(version, 0);
    }
}
