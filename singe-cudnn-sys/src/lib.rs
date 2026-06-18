//! Raw FFI bindings for cuDNN.
//!
//! Prefer the safe `singe-cudnn` crate unless direct NVIDIA ABI access is required.

#![allow(deprecated, warnings, unused_qualifications, clippy::all)]

use num_enum::{IntoPrimitive, TryFromPrimitive};

#[cfg(feature = "cudnn_9_22")]
use singe_cuda_sys::{
    library_types::libraryPropertyType,
    runtime::{cudaGraph_t, cudaStream_t},
};

#[cfg(feature = "cudnn_9_22")]
include!("sys_92200.rs");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let version = unsafe { cudnnGetVersion() };
        println!("cuDNN Version: {}", version);
        assert_ne!(version, 0);
    }
}
