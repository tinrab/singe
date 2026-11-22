//! Low-level FFI bindings for cuDSS (`cudss.h`).
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

#[cfg(not(feature = "cudss_1_0"))]
compile_error!("singe-cudss-sys requires a version feature, such as `cudss_1_0`");

pub use singe_cuda_sys::library_types::cudaDataType_t;
use singe_cuda_sys::{library_types::libraryPropertyType, runtime::cudaStream_t};

#[cfg(feature = "cudss_1_0")]
include!("sys_10000.rs");

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

#[cfg(feature = "cudss_1_0")]
singe_loader::missing_status!(
    cudssStatus_t,
    library: cudssStatus_t::CUDSS_STATUS_NOT_INITIALIZED,
    symbol: cudssStatus_t::CUDSS_STATUS_NOT_SUPPORTED,
);

// Data-type values of `cudss_data_types.h` outside CUDA's `cudaDataType_t`, which the header defines as casts bindgen cannot translate.
#[cfg(feature = "cudss_1_0")]
pub const CUDSS_R_64F_MP2: cudaDataType_t = cudaDataType_t(62);

#[cfg(feature = "cudss_1_0")]
pub const CUDSS_DATA_TYPE_UNSET: cudaDataType_t = cudaDataType_t(63);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_it_works() {
        let mut major = 0;
        // SAFETY: the version query writes only into the local output locations.
        unsafe {
            assert_eq!(
                cudssGetProperty(libraryPropertyType::MAJOR_VERSION, &raw mut major),
                cudssStatus_t::CUDSS_STATUS_SUCCESS,
            );
        }
        println!("cuDSS major version: {}", major);
        assert_ne!(major, 0);
    }
}
