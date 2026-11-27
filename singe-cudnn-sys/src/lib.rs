//! Low-level FFI bindings for NVIDIA cuDNN, the deep neural network library.
//!
//! Most users want the safe crate, `singe-cudnn`.
//!
//! # Bindings
//!
//! | Binds                                      | Header    | Module     | Feature      |
//! | ------------------------------------------ | --------- | ---------- | ------------ |
//! | cuDNN, its graph API and legacy operations | `cudnn.h` | crate root | `cudnn_9_27` |
//!
//! The `cudnn_9_27` feature is on by default.
//! The bindings are generated with bindgen from the cuDNN 9.27 headers.
//!
//! # Loading
//!
//! Nothing links at build time.
//! [`singe_loader`] loads the library on the first call of one of its functions, or up front with [`load`].
//! When the library or one of its symbols is missing, a function that returns a status returns an error status, and [`take_absence`] reports what was missing.
//! A function without a status returns zero, null, or nothing instead.
//! cuDNN loads its own sub-libraries.
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

#[cfg(not(feature = "cudnn_9_27"))]
compile_error!("singe-cudnn-sys requires a version feature, such as `cudnn_9_27`");

#[cfg(feature = "cudnn_9_27")]
use singe_cuda_sys::{
    library_types::libraryPropertyType,
    runtime::{cudaGraph_t, cudaStream_t},
};

#[cfg(feature = "cudnn_9_27")]
include!("sys_92700.rs");

pub use singe_loader::{Absence, LoadError, take_absence};

static LIBRARY: singe_loader::Library = singe_loader::Library::cudnn();

/// Loads cuDNN now, instead of on the first call of one of its functions.
///
/// # Errors
///
/// Returns an error if the library has no files on this platform or one of its files would not load.
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
