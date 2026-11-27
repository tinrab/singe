//! Low-level FFI bindings for NCCL, the NVIDIA Collective Communications Library.
//!
//! Most users want the safe crate, `singe-nccl`.
//!
//! # Bindings
//!
//! | Binds                                                            | Header               | Module     | Feature     |
//! | ---------------------------------------------------------------- | -------------------- | ---------- | ----------- |
//! | NCCL communicators, collectives, and point-to-point transfers    | `nccl.h`             | crate root | `nccl_2_32` |
//! | The host side of NCCL's device API, such as device communicators | `nccl_device/host.h` | crate root | `nccl_2_32` |
//!
//! The `nccl_2_32` feature is on by default.
//! The bindings are generated with bindgen from the NCCL 2.32 headers.
//!
//! # Loading
//!
//! Nothing links at build time.
//! [`singe_loader`] loads the library on the first call of one of its functions, or up front with [`load`].
//! When the library or one of its symbols is missing, a function that returns a status returns an error status, and [`take_absence`] reports what was missing.
//! A function without a status returns zero, null, or nothing instead.
//! [`provide`] serves NCCL from an in-process implementation instead, such as `singe-cuda-emulator`.
//! NCCL has no Windows release.
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

#[cfg(not(feature = "nccl_2_32"))]
compile_error!("singe-nccl-sys requires a version feature, such as `nccl_2_32`");

#[cfg(feature = "nccl_2_32")]
use singe_cuda_sys::runtime::{cudaEvent_t, cudaStream_t};

#[cfg(feature = "nccl_2_32")]
include!("sys_23203.rs");

pub use singe_loader::{Absence, LoadError, Provider, take_absence};

static LIBRARY: singe_loader::Library = singe_loader::Library::nccl();

/// Loads NCCL now, instead of on the first call of one of its functions.
///
/// # Errors
///
/// Returns an error if the library has no files on this platform or one of its files would not load.
pub fn load() -> Result<(), LoadError> {
    LIBRARY.load()
}

/// Serves NCCL's functions from `provider`, such as an emulator, instead of loading the library.
///
/// # Errors
///
/// Returns [`LoadError::AlreadyLoaded`] if an NCCL function was already called or NCCL was already provided.
pub fn provide(provider: Provider) -> Result<(), LoadError> {
    LIBRARY.provide(provider)
}

#[cfg(feature = "nccl_2_32")]
singe_loader::missing_status!(
    ncclResult_t,
    library: ncclResult_t::ncclSystemError,
    symbol: ncclResult_t::ncclInternalError,
);

// A missing team query records its absence, so the safe wrapper can tell the zero team apart from a real one.
#[cfg(feature = "nccl_2_32")]
impl singe_loader::Missing for ncclTeam_t {
    fn missing(absence: Absence) -> Self {
        singe_loader::record_absence(absence);
        Self::default()
    }
}

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
