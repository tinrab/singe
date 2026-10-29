//! Raw FFI bindings for NPP.
//!
//! Prefer the safe `singe-npp` crate unless direct NVIDIA ABI access is required.

#![allow(
    non_camel_case_types,
    non_snake_case,
    unnecessary_transmutes,
    unused_imports,
    unused_qualifications,
    clippy::all,
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

#[cfg(feature = "npp_13_2")]
include!("sys_13200.rs");

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
