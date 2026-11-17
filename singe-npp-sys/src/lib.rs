//! Low-level FFI bindings for NPP 13.2, the NVIDIA Performance Primitives, generated from `npp.h`.
//! NPP is split into several libraries, such as `nppc` for core functions, `nppi*` for image processing, and `npps` for signal processing.
//! [`singe-loader`](singe_loader) loads each library at run time, on the first call or through [`load`], so building links no NVIDIA library.
//! Most users want the safe `singe-npp` crate.
#![allow(
    deprecated,
    dead_code,
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
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

#[cfg(not(feature = "npp_13_2"))]
compile_error!("singe-npp-sys requires a version feature, such as `npp_13_2`");

use singe_cuda_sys::runtime::cudaStream_t;

#[cfg(feature = "npp_13_2")]
include!("sys_13200.rs");

pub use singe_loader::{Absence, LoadError, take_absence};

static LIBRARY: singe_loader::Library = singe_loader::Library::npp();

/// Loads NPP now, instead of on the first call of one of its functions.
///
/// # Errors
///
/// Returns an error if the library has no files on this platform or one of its files would not load.
pub fn load() -> Result<(), LoadError> {
    LIBRARY.load()
}

#[cfg(feature = "npp_13_2")]
singe_loader::missing_status!(
    NppStatus,
    library: NppStatus::NPP_NOT_IMPLEMENTED_ERROR,
    symbol: NppStatus::NPP_NOT_SUPPORTED_MODE_ERROR,
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_it_works() {
        // SAFETY: the version query writes only into the local output locations.
        let version = unsafe { nppGetLibVersion() };
        assert!(!version.is_null());
        // SAFETY: the pointer was checked for null and refers to NPP's static version struct.
        let version = unsafe { *version };

        assert_eq!(version.major, NPP_VERSION_MAJOR as i32);
        assert_eq!(version.minor, NPP_VERSION_MINOR as i32);
        assert_eq!(NPP_VERSION_BUILD, NPP_VER_BUILD);
    }
}
