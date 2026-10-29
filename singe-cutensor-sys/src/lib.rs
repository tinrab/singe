//! Raw FFI bindings for cuTENSOR and cuTENSORMg.
//!
//! Prefer the safe `singe-cutensor` crate unless direct NVIDIA ABI access is required.

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
use singe_cuda_sys::driver::CUstream as cudaStream_t;
pub use singe_cuda_sys::library_types::cudaDataType as cudaDataType_t;
use singe_nccl_sys::ncclComm_t;

#[cfg(feature = "cutensor_2_8")]
include!("sys_20800.rs");

#[cfg(all(feature = "cutensor_2_8", feature = "mg"))]
include!("mg_20800.rs");

#[cfg(all(feature = "cutensor_2_8", feature = "mp"))]
include!("mp_20800.rs");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_it_works() {
        // SAFETY: the version query writes only into the local output locations.
        let version = unsafe { cutensorGetVersion() };
        println!("cuTENSOR version: {}", version);
        assert_ne!(version, 0);
    }
}
