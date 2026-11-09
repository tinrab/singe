//! Raw FFI bindings for cuDSS.
//!
//! Prefer the safe `singe-cudss` crate unless direct NVIDIA ABI access is required.

#![allow(
    unpredictable_function_pointer_comparisons,
    non_upper_case_globals,
    deprecated,
    dead_code,
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
    library_types::{cudaDataType_t, libraryPropertyType},
    runtime::cudaStream_t,
};

#[cfg(feature = "cudss_0_8")]
include!("sys_800.rs");

pub use singe_loader::LoadError;

static LIBRARY: singe_loader::Library = singe_loader::Library::cudss();

// Loads cuDSS, which each of its functions otherwise does on first call.
pub fn load() -> Result<(), LoadError> {
    LIBRARY.load()
}

#[cfg(feature = "cudss_0_8")]
singe_loader::missing_status!(
    cudssStatus_t,
    library: cudssStatus_t::CUDSS_STATUS_NOT_INITIALIZED,
    symbol: cudssStatus_t::CUDSS_STATUS_NOT_SUPPORTED,
);

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
