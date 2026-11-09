//! Raw FFI bindings for CUPTI.
//!
//! Prefer the safe `singe-cupti` crate unless direct NVIDIA ABI access is required.

#![allow(
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

#[cfg(feature = "cupti_13_4")]
include!("cupti_sys_130401.rs");

pub use singe_loader::LoadError;

static LIBRARY: singe_loader::Library = singe_loader::Library::cupti();

// Loads CUPTI, which each of its functions otherwise does on first call.
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
