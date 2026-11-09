//! Low-level FFI bindings for the NVIDIA nvTIFF library.

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

include!("sys_800.rs");

// Zero is not a valid nvtiffVariant. Keep bindgen from generating a zeroed
// Default for this struct and provide a fully initialized TIFF header instead.
impl Default for nvtiffStreamHeader {
    fn default() -> Self {
        Self {
            variant: nvtiffVariant::NVTIFF_REGULAR_TIFF,
            first_ifd_offset: 0,
        }
    }
}

pub use singe_loader::LoadError;

static LIBRARY: singe_loader::Library = singe_loader::Library::nvtiff();

// Loads nvTIFF, which each of its functions otherwise does on first call.
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
