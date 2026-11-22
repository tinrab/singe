//! Low-level FFI bindings for CUPTI, the CUDA Profiling Tools Interface, generated from `cupti.h` and the profiler headers.
//! [`singe-loader`](singe_loader) loads `libcupti` from the CUDA Toolkit at run time, on the first call or through [`load`], so building links no NVIDIA library.
//! Most users want the safe `singe-cupti` crate.
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
