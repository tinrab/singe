//! Raw FFI bindings for cuDNN.
//!
//! Prefer the safe `singe-cudnn` crate unless direct NVIDIA ABI access is required.

#![allow(
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

use num_enum::{IntoPrimitive, TryFromPrimitive};
#[cfg(feature = "cudnn_9_27")]
use singe_cuda_sys::{
    library_types::libraryPropertyType,
    runtime::{cudaGraph_t, cudaStream_t},
};

#[cfg(feature = "cudnn_9_27")]
include!("sys_92700.rs");

pub use singe_loader::LoadError;

static LIBRARY: singe_loader::Library = singe_loader::Library::cudnn();

// Loads cuDNN, which each of its functions otherwise does on first call.
pub fn load() -> Result<(), LoadError> {
    LIBRARY.load()
}

#[cfg(feature = "cudnn_9_27")]
singe_loader::missing_status!(
    cudnnStatus_t,
    library: cudnnStatus_t::CUDNN_STATUS_NOT_INITIALIZED,
    symbol: cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED,
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_it_works() {
        // SAFETY: the version query writes only into the local output locations.
        let version = unsafe { cudnnGetVersion() };
        println!("cuDNN Version: {}", version);
        assert_ne!(version, 0);
    }
}
