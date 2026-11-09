//! Raw FFI bindings for NCCL.
//!
//! Prefer the safe `singe-nccl` crate unless direct NVIDIA ABI access is required.

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

pub use singe_loader::LoadError;

// NCCL's dynamic operators are values outside the generated Rust enum's variants.
// This declaration passes the C enum's ABI as an integer, so no invalid Rust enum value is constructed.
#[cfg(feature = "nccl_2_32")]
pub mod dynamic {
    use super::{ncclComm_t, ncclResult_t};

    singe_loader::functions! { crate::LIBRARY;
        pub fn ncclRedOpDestroy(op: i32, comm: ncclComm_t) -> ncclResult_t;
    }
}

static LIBRARY: singe_loader::Library = singe_loader::Library::nccl();

// Loads NCCL, which each of its functions otherwise does on first call.
pub fn load() -> Result<(), LoadError> {
    LIBRARY.load()
}

#[cfg(feature = "nccl_2_32")]
singe_loader::missing_status!(
    ncclResult_t,
    library: ncclResult_t::ncclSystemError,
    symbol: ncclResult_t::ncclInternalError,
);

#[cfg(feature = "nccl_2_32")]
singe_loader::missing_status!(
    ncclTeam_t,
    library: ncclTeam_t::default(),
    symbol: ncclTeam_t::default(),
);

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
