//! Low-level FFI bindings for NVIDIA nvJPEG and nvJPEG2000.
//!
//! Most users want the safe crate, `singe-nvjpeg`.
//!
//! # Bindings
//!
//! | Binds                                       | Header       | Module     | Feature       |
//! | ------------------------------------------- | ------------ | ---------- | ------------- |
//! | nvJPEG, JPEG decoding and encoding          | `nvjpeg.h`   | crate root | `nvjpeg_13_2` |
//! | nvJPEG2000, JPEG 2000 decoding and encoding | `nvjpeg2k.h` | `k2`       | `k2`          |
//!
//! The `nvjpeg_13_2` feature is on by default, and `k2` is off.
//! The bindings are generated with bindgen from the nvJPEG 13.2 and nvJPEG2000 0.11 headers.
//!
//! # Loading
//!
//! Nothing links at build time.
//! [`singe_loader`] loads each library on the first call of one of its functions, or up front with [`load`] for nvJPEG and `load_k2` for nvJPEG2000.
//! When a library or one of its symbols is missing, a function that returns a status returns an error status, and [`take_absence`] reports what was missing.
//! A function without a status returns zero, null, or nothing instead.
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

#[cfg(not(feature = "nvjpeg_13_2"))]
compile_error!("singe-nvjpeg-sys requires a version feature, such as `nvjpeg_13_2`");

use singe_cuda_sys::{
    library_types::libraryPropertyType,
    runtime::{cudaError_t, cudaStream_t},
};

#[cfg(feature = "nvjpeg_13_2")]
include!("sys_130203.rs");

#[cfg(feature = "k2")]
/// Raw FFI bindings for nvJPEG 2000.
pub mod k2 {
    use singe_cuda_sys::{
        library_types::libraryPropertyType,
        runtime::{cudaError_t, cudaStream_t},
    };

    include!("k2_sys_1100.rs");
}

pub use singe_loader::{Absence, LoadError, take_absence};

static LIBRARY: singe_loader::Library = singe_loader::Library::nvjpeg();

/// Loads nvJPEG now, instead of on the first call of one of its functions.
///
/// # Errors
///
/// Returns an error if the library has no files on this platform or one of its files would not load.
pub fn load() -> Result<(), LoadError> {
    LIBRARY.load()
}

#[cfg(feature = "k2")]
static K2: singe_loader::Library = singe_loader::Library::nvjpeg2k();

/// Loads nvJPEG2000 now, instead of on the first call of one of its functions.
///
/// # Errors
///
/// Returns an error if the library has no files on this platform or one of its files would not load.
#[cfg(feature = "k2")]
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
