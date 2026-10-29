//! Raw FFI bindings for NCCL.
//!
//! Prefer the safe `singe-nccl` crate unless direct NVIDIA ABI access is required.

#![allow(
    deprecated,
    warnings,
    dead_code,
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

/// Include directory containing the device headers used to generate these bindings.
pub const DEVICE_INCLUDE_PATH: Option<&str> = option_env!("NCCL_INCLUDE_DIR");

/// CUDA toolkit headers needed by the NCCL device headers.
pub const CUDA_INCLUDE_PATH: Option<&str> = option_env!("NCCL_CUDA_INCLUDE_DIR");

/// Separate CCCL include directory when the toolkit uses that layout.
pub const CCCL_INCLUDE_PATH: Option<&str> = option_env!("NCCL_CCCL_INCLUDE_DIR");

use num_enum::{IntoPrimitive, TryFromPrimitive};
#[cfg(feature = "nccl_2_32")]
use singe_cuda_sys::runtime::{cudaEvent_t, cudaStream_t};

#[cfg(feature = "nccl_2_32")]
include!("sys_23203.rs");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_it_works() {
        let mut version = 0;
        // SAFETY: the version query writes only into the local output locations.
        unsafe {
            assert_eq!(ncclGetVersion(&raw mut version), ncclResult_t::ncclSuccess);
        }
        println!("NCCL version: {version}");
        assert_ne!(version, 0);
    }
}
