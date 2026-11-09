//! Raw FFI bindings for nvImageCodec.
//!
//! Prefer the safe `singe-nvimagecodec` crate unless direct NVIDIA ABI access is required.

#![allow(
    non_camel_case_types,
    non_snake_case,
    unnecessary_transmutes,
    unused_imports,
    unused_qualifications,
    unpredictable_function_pointer_comparisons,
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
use singe_cuda_sys::runtime::cudaStream_t;

#[cfg(feature = "nvimagecodec_0_9")]
include!("sys_900.rs");

pub use singe_loader::LoadError;

static LIBRARY: singe_loader::Library = singe_loader::Library::nvimagecodec();

// Loads nvImageCodec, which each of its functions otherwise does on first call.
pub fn load() -> Result<(), LoadError> {
    LIBRARY.load()
}

#[cfg(feature = "nvimagecodec_0_9")]
singe_loader::missing_status!(
    nvimgcodecStatus_t,
    library: nvimgcodecStatus_t::NVIMGCODEC_STATUS_NOT_INITIALIZED,
    symbol: nvimgcodecStatus_t::NVIMGCODEC_STATUS_INTERNAL_ERROR,
);

#[cfg(test)]
mod tests {
    use std::{mem, ptr};

    use super::*;

    #[test]
    fn test_it_works() {
        let mut properties = nvimgcodecProperties_t {
            struct_type: nvimgcodecStructureType_t::NVIMGCODEC_STRUCTURE_TYPE_PROPERTIES,
            struct_size: mem::size_of::<nvimgcodecProperties_t>() as size_t,
            struct_next: ptr::null_mut(),
            ..Default::default()
        };
        // SAFETY: the version query writes only into the local output locations.
        unsafe {
            assert_eq!(
                nvimgcodecGetProperties(&raw mut properties),
                nvimgcodecStatus_t::NVIMGCODEC_STATUS_SUCCESS,
            );
        }
        assert_ne!(properties.version, 0);
    }
}
