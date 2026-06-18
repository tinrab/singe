//! Raw FFI bindings for cuTENSOR and cuTENSORMg.
//!
//! Prefer the safe `singe-cutensor` crate unless direct NVIDIA ABI access is required.

#![allow(deprecated, warnings, unused_qualifications, clippy::all)]

use num_enum::{IntoPrimitive, TryFromPrimitive};

use singe_cuda_sys::driver::CUstream as cudaStream_t;
pub use singe_cuda_sys::library_types::cudaDataType as cudaDataType_t;
use singe_nccl_sys::ncclComm_t;

#[cfg(feature = "cutensor_2_6")]
include!("sys_20600.rs");

#[cfg(all(feature = "cutensor_2_6", feature = "mg"))]
include!("mg_20600.rs");

#[cfg(all(feature = "cutensor_2_6", feature = "mp"))]
include!("mp_20600.rs");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let version = unsafe { cutensorGetVersion() };
        println!("cuTENSOR version: {}", version);
        assert_ne!(version, 0);
    }
}
