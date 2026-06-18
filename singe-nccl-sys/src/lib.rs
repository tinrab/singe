//! Raw FFI bindings for NCCL.
//!
//! Prefer the safe `singe-nccl` crate unless direct NVIDIA ABI access is required.

#![allow(deprecated, warnings, unused_qualifications, clippy::all)]

use num_enum::{IntoPrimitive, TryFromPrimitive};

#[cfg(feature = "nccl_2_28")]
use singe_cuda_sys::runtime::cudaStream_t;

#[cfg(feature = "nccl_2_28")]
include!("sys_22807.rs");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let mut version = 0;
        unsafe {
            assert_eq!(ncclGetVersion(&raw mut version), ncclResult_t::ncclSuccess);
        }
        println!("NCCL version: {version}");
        assert_ne!(version, 0);
    }
}
