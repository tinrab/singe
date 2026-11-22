//! Low-level FFI bindings for nvImageCodec, the NVIDIA image codec library, generated from `nvimgcodec.h`.
//! [`singe-loader`](singe_loader) loads the library at run time, on the first call or through [`load`], so building links no NVIDIA library.
//! Most users want the safe `singe-nvimagecodec` crate.
#![allow(
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    unnecessary_transmutes,
    unused_imports,
    unused_qualifications,
    unpredictable_function_pointer_comparisons,
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

#[cfg(not(feature = "nvimagecodec_0_9"))]
compile_error!("singe-nvimagecodec-sys requires a version feature, such as `nvimagecodec_0_9`");

use singe_cuda_sys::runtime::cudaStream_t;

#[cfg(feature = "nvimagecodec_0_9")]
include!("sys_900.rs");

pub use singe_loader::{Absence, LoadError, take_absence};

static LIBRARY: singe_loader::Library = singe_loader::Library::nvimagecodec();

/// Loads nvImageCodec now, instead of on the first call of one of its functions.
///
/// # Errors
///
/// Returns an error if the library has no files on this platform or one of its files would not load.
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
