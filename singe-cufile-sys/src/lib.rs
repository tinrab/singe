//! Raw FFI bindings for cuFile.
//!
//! Use this crate when direct NVIDIA cuFile ABI access is required.

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

use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_cuda_sys::driver::{CUresult, CUstream};

#[cfg(feature = "cufile_1_19")]
include!("sys_1190.rs");

pub use singe_loader::LoadError;

static LIBRARY: singe_loader::Library = singe_loader::Library::cufile();

// Loads cuFile, which each of its functions otherwise does on first call.
pub fn load() -> Result<(), LoadError> {
    LIBRARY.load()
}

#[cfg(feature = "cufile_1_19")]
impl singe_loader::Missing for CUfileError {
    fn missing(absence: singe_loader::Absence) -> Self {
        Self {
            err: match absence {
                singe_loader::Absence::Library => CUfileOpError::CU_FILE_DRIVER_NOT_INITIALIZED,
                singe_loader::Absence::Symbol => CUfileOpError::CU_FILE_PLATFORM_NOT_SUPPORTED,
            },
            cu_err: CUresult::CUDA_SUCCESS,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_it_works() {
        let mut version = 0;
        // SAFETY: the version query writes only into the local output locations.
        let status = unsafe { cuFileGetVersion(&raw mut version) };
        assert_eq!(status.err, CUfileOpError::CU_FILE_SUCCESS);
        assert_ne!(version, 0);
    }
}
