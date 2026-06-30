//! Raw FFI bindings for NPP.
//!
//! Prefer the safe `singe-npp` crate unless direct NVIDIA ABI access is required.

#![allow(deprecated, warnings, unused_qualifications, clippy::all)]

use num_enum::{IntoPrimitive, TryFromPrimitive};

use singe_cuda_sys::runtime::cudaStream_t;

#[cfg(feature = "npp_13_1")]
include!("sys_13102.rs");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let version = unsafe { nppGetLibVersion() };
        assert!(!version.is_null());
        let version = unsafe { *version };

        assert_eq!(version.major, NPP_VERSION_MAJOR as i32);
        assert_eq!(version.minor, NPP_VERSION_MINOR as i32);
        assert_eq!(NPP_VERSION_BUILD, NPP_VER_BUILD);
    }
}
