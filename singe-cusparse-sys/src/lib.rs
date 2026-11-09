//! Raw FFI bindings for cuSPARSE.
//!
//! Prefer the safe `singe-cusparse` crate unless direct NVIDIA ABI access is required.

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
use singe_cuda_sys::{
    library_types::{cudaDataType, cudaDataType_t, libraryPropertyType},
    runtime::cudaStream_t,
};

#[cfg(feature = "cusparse_12_8")]
include!("sys_12806.rs");

pub use singe_loader::LoadError;

static LIBRARY: singe_loader::Library = singe_loader::Library::cusparse();

// Loads cuSPARSE, which each of its functions otherwise does on first call.
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
singe_loader::missing_status!(
    cusparseMatrixType_t,
    library: cusparseMatrixType_t::CUSPARSE_MATRIX_TYPE_GENERAL,
    symbol: cusparseMatrixType_t::CUSPARSE_MATRIX_TYPE_GENERAL,
);

#[cfg(feature = "cusparse_12_8")]
singe_loader::missing_status!(
    cusparseFillMode_t,
    library: cusparseFillMode_t::CUSPARSE_FILL_MODE_LOWER,
    symbol: cusparseFillMode_t::CUSPARSE_FILL_MODE_LOWER,
);

#[cfg(feature = "cusparse_12_8")]
singe_loader::missing_status!(
    cusparseDiagType_t,
    library: cusparseDiagType_t::CUSPARSE_DIAG_TYPE_NON_UNIT,
    symbol: cusparseDiagType_t::CUSPARSE_DIAG_TYPE_NON_UNIT,
);

#[cfg(feature = "cusparse_12_8")]
singe_loader::missing_status!(
    cusparseIndexBase_t,
    library: cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO,
    symbol: cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO,
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
