//! Low-level FFI bindings for NVIDIA cuFile, the `GPUDirect` Storage library.
//!
//! Most users want the safe crate, `singe-cufile`.
//!
//! # Bindings
//!
//! | Binds                                   | Header     | Module     | Feature       |
//! | --------------------------------------- | ---------- | ---------- | ------------- |
//! | cuFile, file I/O to and from GPU memory | `cufile.h` | crate root | `cufile_1_19` |
//!
//! The `cufile_1_19` feature is on by default.
//! The bindings are generated with bindgen from the cuFile 1.19 headers.
//!
//! # Loading
//!
//! Nothing links at build time.
//! [`singe_loader`] loads the library on the first call of one of its functions, or up front with [`load`].
//! When the library or one of its symbols is missing, a function that returns a status returns an error status, and [`take_absence`] reports what was missing.
//! A function without a status returns zero, null, or nothing instead.
//! cuFile has no Windows release.
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

#[cfg(not(feature = "cufile_1_19"))]
compile_error!("singe-cufile-sys requires a version feature, such as `cufile_1_19`");

use singe_cuda_sys::driver::{CUresult, CUstream};

#[cfg(feature = "cufile_1_19")]
include!("sys_1190.rs");

pub use singe_loader::{Absence, LoadError, take_absence};

static LIBRARY: singe_loader::Library = singe_loader::Library::cufile();

/// Loads cuFile now, instead of on the first call of one of its functions.
///
/// # Errors
///
/// Returns an error if the library has no files on this platform or one of its files would not load.
pub fn load() -> Result<(), LoadError> {
    LIBRARY.load()
}

#[cfg(feature = "cufile_1_19")]
impl singe_loader::Missing for CUfileError {
    fn missing(absence: singe_loader::Absence) -> Self {
        singe_loader::record_absence(absence);
        Self {
            err: match absence {
                singe_loader::Absence::Library(_) => CUfileOpError::CU_FILE_DRIVER_NOT_INITIALIZED,
                singe_loader::Absence::Symbol { .. } => {
                    CUfileOpError::CU_FILE_PLATFORM_NOT_SUPPORTED
                }
            },
            cu_err: CUresult::CUDA_SUCCESS,
        }
    }

    fn made(self) -> Self {
        singe_loader::clear_absence();
        self
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
