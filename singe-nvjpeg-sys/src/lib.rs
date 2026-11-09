//! Raw FFI bindings for nvJPEG and nvJPEG 2000.

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
    library_types::libraryPropertyType,
    runtime::{cudaError_t, cudaStream_t},
};

#[cfg(feature = "nvjpeg_13_2")]
include!("sys_130203.rs");

#[cfg(feature = "k2")]
/// Raw FFI bindings for nvJPEG 2000.
pub mod k2 {
    use num_enum::{IntoPrimitive, TryFromPrimitive};
    use singe_cuda_sys::{
        library_types::libraryPropertyType,
        runtime::{cudaError_t, cudaStream_t},
    };

    include!("k2_sys_1100.rs");
}

pub use singe_loader::LoadError;

static LIBRARY: singe_loader::Library = singe_loader::Library::nvjpeg();

// Loads nvJPEG, which each of its functions otherwise does on first call.
pub fn load() -> Result<(), LoadError> {
    LIBRARY.load()
}

static K2: singe_loader::Library = singe_loader::Library::nvjpeg2k();

// Loads nvJPEG2000, which each of its functions otherwise does on first call.
pub fn load_k2() -> Result<(), LoadError> {
    K2.load()
}

#[cfg(feature = "nvjpeg_13_2")]
singe_loader::missing_status!(
    nvjpegStatus_t,
    library: nvjpegStatus_t::NVJPEG_STATUS_NOT_INITIALIZED,
    symbol: nvjpegStatus_t::NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED,
);

#[cfg(feature = "k2")]
singe_loader::missing_status!(
    k2::nvjpeg2kStatus_t,
    library: k2::nvjpeg2kStatus_t::NVJPEG2K_STATUS_NOT_INITIALIZED,
    symbol: k2::nvjpeg2kStatus_t::NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED,
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
                nvjpegGetProperty(libraryPropertyType::MAJOR_VERSION, &raw mut version),
                nvjpegStatus_t::NVJPEG_STATUS_SUCCESS,
            );
        }
        assert_ne!(version, 0);
    }

    #[cfg(feature = "k2")]
    #[test]
    fn test_k2_works() {
        let mut version = 0;
        // SAFETY: the version query writes only into the local output locations.
        unsafe {
            assert_eq!(
                k2::nvjpeg2kGetProperty(libraryPropertyType::MAJOR_VERSION, &raw mut version),
                k2::nvjpeg2kStatus_t::NVJPEG2K_STATUS_SUCCESS,
            );
        }
    }
}
