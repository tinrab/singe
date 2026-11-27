//! Low-level FFI bindings for NVIDIA cuSPARSE, the sparse linear algebra library.
//!
//! Most users want the safe crate, `singe-cusparse`.
//!
//! # Bindings
//!
//! | Binds                                         | Header       | Module     | Feature         |
//! | --------------------------------------------- | ------------ | ---------- | --------------- |
//! | cuSPARSE, sparse matrix and vector operations | `cusparse.h` | crate root | `cusparse_12_8` |
//!
//! The `cusparse_12_8` feature is on by default.
//! The bindings are generated with bindgen from the cuSPARSE 12.8 headers.
//!
//! # Loading
//!
//! Nothing links at build time.
//! [`singe_loader`] loads the library on the first call of one of its functions, or up front with [`load`].
//! When the library or one of its symbols is missing, a function that returns a status returns an error status, and [`take_absence`] reports what was missing.
//! A function without a status returns zero, null, or nothing instead.
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

#[cfg(not(feature = "cusparse_12_8"))]
compile_error!("singe-cusparse-sys requires a version feature, such as `cusparse_12_8`");

use singe_cuda_sys::{
    library_types::{cudaDataType, cudaDataType_t, libraryPropertyType},
    runtime::cudaStream_t,
};

#[cfg(feature = "cusparse_12_8")]
include!("sys_12806.rs");

pub use singe_loader::{Absence, LoadError, take_absence};

static LIBRARY: singe_loader::Library = singe_loader::Library::cusparse();

/// Loads cuSPARSE now, instead of on the first call of one of its functions.
///
/// # Errors
///
/// Returns an error if the library has no files on this platform or one of its files would not load.
pub fn load() -> Result<(), LoadError> {
    LIBRARY.load()
}

#[cfg(feature = "cusparse_12_8")]
singe_loader::missing_status!(
    cusparseStatus_t,
    library: cusparseStatus_t::CUSPARSE_STATUS_NOT_INITIALIZED,
    symbol: cusparseStatus_t::CUSPARSE_STATUS_NOT_SUPPORTED,
);

// The getters of a matrix descriptor's settings return them directly, so a missing one returns each setting's first value.
#[cfg(feature = "cusparse_12_8")]
singe_loader::missing_value!(
    cusparseMatrixType_t,
    cusparseMatrixType_t::CUSPARSE_MATRIX_TYPE_GENERAL
);

#[cfg(feature = "cusparse_12_8")]
singe_loader::missing_value!(
    cusparseFillMode_t,
    cusparseFillMode_t::CUSPARSE_FILL_MODE_LOWER
);

#[cfg(feature = "cusparse_12_8")]
singe_loader::missing_value!(
    cusparseDiagType_t,
    cusparseDiagType_t::CUSPARSE_DIAG_TYPE_NON_UNIT
);

#[cfg(feature = "cusparse_12_8")]
singe_loader::missing_value!(
    cusparseIndexBase_t,
    cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO
);

#[cfg(test)]
mod tests {
    use std::ptr;

    use super::*;

    #[test]
    fn test_it_works() {
        let mut version = 0;
        // SAFETY: the handle is created, used and destroyed within this block.
        unsafe {
            let mut ctx = ptr::null_mut();
            assert_eq!(
                cusparseCreate(&mut ctx),
                cusparseStatus_t::CUSPARSE_STATUS_SUCCESS,
            );
            cusparseGetVersion(ctx, &mut version);
        }
        println!("cuSPARSE version: {}", version);
        assert_ne!(version, 0);
    }
}
