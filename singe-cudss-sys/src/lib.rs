//! Low-level FFI bindings for cuDSS 0.8 (`cudss.h`).
//! The library loads at run time through [`singe-loader`](singe_loader), on the first call of one of its functions or up front with [`load`], so building links no NVIDIA library.
//! If the library or one of its symbols is missing, a function that returns a status returns a failure status, and [`take_absence`] reports what was missing.
//! Most users want the safe `singe-cudss` crate.
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
    missing_debug_implementations,
    unnameable_types,
    unreachable_pub,
    clippy::correctness,
    clippy::suspicious,
    clippy::complexity,
    clippy::perf,
    clippy::style,
    clippy::absolute_paths,
    clippy::pedantic,
    clippy::nursery,
    clippy::undocumented_unsafe_blocks,
    reason = "generated bindings preserve upstream ABI conventions and documentation"
)]

#[cfg(not(feature = "cudss_0_8"))]
compile_error!("singe-cudss-sys requires a version feature, such as `cudss_0_8`");

use singe_cuda_sys::{
    library_types::{cudaDataType_t, libraryPropertyType},
    runtime::cudaStream_t,
};

#[cfg(feature = "cudss_0_8")]
include!("sys_800.rs");

pub use singe_loader::{Absence, LoadError, take_absence};

static LIBRARY: singe_loader::Library = singe_loader::Library::cudss();

/// Loads cuDSS now, instead of on the first call of one of its functions.
///
/// # Errors
///
/// Returns an error if the library has no files on this platform or one of its files would not load.
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
