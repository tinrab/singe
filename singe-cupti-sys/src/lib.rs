//! Low-level FFI bindings for NVIDIA CUPTI, the CUDA Profiling Tools Interface.
//!
//! Most users want the safe crate, `singe-cupti`.
//!
//! # Bindings
//!
//! | Binds                                      | Header                                                                       | Module     | Feature      |
//! | ------------------------------------------ | ---------------------------------------------------------------------------- | ---------- | ------------ |
//! | CUPTI callbacks and activity tracing       | `cupti.h`, `cupti_target.h`                                                  | crate root | `cupti_13_4` |
//! | CUPTI hardware metrics and range profiling | `cupti_profiler_host.h`, `cupti_profiler_target.h`, `cupti_range_profiler.h` | crate root | `cupti_13_4` |
//!
//! The `cupti_13_4` feature is on by default.
//! The bindings are generated with bindgen from the CUPTI 13.4 headers.
//!
//! # Loading
//!
//! Nothing links at build time.
//! [`singe_loader`] loads the library on the first call of one of its functions, or up front with [`load`].
//! When the library or one of its symbols is missing, a function that returns a status returns an error status, and [`take_absence`] reports what was missing.
//! A function without a status returns zero, null, or nothing instead.
#![allow(
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

#[cfg(not(feature = "cupti_13_4"))]
compile_error!("singe-cupti-sys requires a version feature, such as `cupti_13_4`");

#[cfg(feature = "cupti_13_4")]
include!("cupti_sys_130401.rs");

pub use singe_loader::{Absence, LoadError, take_absence};

static LIBRARY: singe_loader::Library = singe_loader::Library::cupti();

/// Loads CUPTI now, instead of on the first call of one of its functions.
///
/// # Errors
///
/// Returns an error if the library has no files on this platform or one of its files would not load.
pub fn load() -> Result<(), LoadError> {
    LIBRARY.load()
}

#[cfg(feature = "cupti_13_4")]
singe_loader::missing_status!(
    CUptiResult,
    library: CUptiResult::CUPTI_ERROR_NOT_INITIALIZED,
    symbol: CUptiResult::CUPTI_ERROR_API_NOT_IMPLEMENTED,
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_it_works() {
        assert_eq!(CUPTI_API_VERSION, 130401);
    }
}
