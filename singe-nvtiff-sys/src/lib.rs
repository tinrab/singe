//! Low-level FFI bindings for NVIDIA nvTIFF, the TIFF codec library.
//!
//! Most users want the safe crate, `singe-nvtiff`.
//!
//! # Bindings
//!
//! | Binds                              | Header     | Module     | Feature      |
//! | ---------------------------------- | ---------- | ---------- | ------------ |
//! | nvTIFF, TIFF decoding and encoding | `nvtiff.h` | crate root | `nvtiff_0_8` |
//!
//! The `nvtiff_0_8` feature is on by default.
//! The bindings are generated with bindgen from the nvTIFF 0.8 headers.
//!
//! # Loading
//!
//! Nothing links at build time.
//! [`singe_loader`] loads the library on the first call of one of its functions, or up front with [`load`].
//! When the library or one of its symbols is missing, a function that returns a status returns an error status, and [`take_absence`] reports what was missing.
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

#[cfg(not(feature = "nvtiff_0_8"))]
compile_error!("singe-nvtiff-sys requires a version feature, such as `nvtiff_0_8`");

use singe_cuda_sys::{
    library_types::libraryPropertyType,
    runtime::{cudaError_t, cudaStream_t},
};

include!("sys_800.rs");

// nvTIFF defines no zero variant, so the default header names a regular TIFF.
impl Default for nvtiffStreamHeader {
    fn default() -> Self {
        Self {
            variant: nvtiffVariant::NVTIFF_REGULAR_TIFF,
            first_ifd_offset: 0,
        }
    }
}

pub use singe_loader::{Absence, LoadError, take_absence};

static LIBRARY: singe_loader::Library = singe_loader::Library::nvtiff();

/// Loads nvTIFF now, instead of on the first call of one of its functions.
///
/// # Errors
///
/// Returns an error if the library has no files on this platform or one of its files would not load.
pub fn load() -> Result<(), LoadError> {
    LIBRARY.load()
}

singe_loader::missing_status!(
    nvtiffStatus_t,
    library: nvtiffStatus_t::NVTIFF_STATUS_NOT_INITIALIZED,
    symbol: nvtiffStatus_t::NVTIFF_STATUS_INTERNAL_ERROR,
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_it_works() {
        let mut major = 0;
        let mut minor = 0;
        let mut patch = 0;
        // SAFETY: the version query writes only into the local output locations.
        unsafe {
            assert_eq!(
                nvtiffGetProperty(libraryPropertyType::MAJOR_VERSION, &raw mut major),
                nvtiffStatus_t::NVTIFF_STATUS_SUCCESS,
            );
            assert_eq!(
                nvtiffGetProperty(libraryPropertyType::MINOR_VERSION, &raw mut minor),
                nvtiffStatus_t::NVTIFF_STATUS_SUCCESS,
            );
            assert_eq!(
                nvtiffGetProperty(libraryPropertyType::PATCH_LEVEL, &raw mut patch),
                nvtiffStatus_t::NVTIFF_STATUS_SUCCESS,
            );
        }
        assert_eq!((major, minor, patch), (0, 8, 0));
    }
}
