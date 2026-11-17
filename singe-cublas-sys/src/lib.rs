//! Low-level FFI bindings for cuBLAS 13.8 (`cublas_v2.h`), with optional bindings for cuBLASLt (`cublasLt.h`) and cuBLASXt (`cublasXt.h`).
//! Each library loads at run time through [`singe-loader`](singe_loader), on the first call of one of its functions or up front with [`load`] and [`load_lt`], so building links no NVIDIA library.
//! If the library or one of its symbols is missing, a function that returns a status returns a failure status, and [`take_absence`] reports what was missing.
//! Most users want the safe `singe-cublas` crate.
//!
//! # Features
//!
//! - `lt`: cuBLASLt, for matrix multiplication with fused epilogues and algorithm selection.
//! - `xt`: cuBLASXt, for BLAS level 3 routines split across several GPUs.
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

#[cfg(not(feature = "cublas_13_8"))]
compile_error!("singe-cublas-sys requires a version feature, such as `cublas_13_8`");

#[cfg(feature = "cublas_13_8")]
use singe_cuda_sys::{
    library_types::{
        cudaDataType, cudaDataType_t, cudaEmulationMantissaControl,
        cudaEmulationSpecialValuesSupport, libraryPropertyType,
    },
    runtime::cudaStream_t,
};

#[cfg(feature = "cublas_13_8")]
include!("sys_130800.rs");

#[cfg(all(feature = "cublas_13_8", feature = "lt"))]
mod lt_bindings {

    use crate::{
        cublasComputeType_t, cublasSetWorkspace_v2, cublasStatus_t, cudaDataType, cudaDataType_t,
        cudaStream_t, libraryPropertyType,
    };

    include!("sys_lt_130800.rs");
}

#[cfg(all(feature = "cublas_13_8", feature = "lt"))]
pub use self::lt_bindings::*;

#[cfg(all(feature = "cublas_13_8", feature = "xt"))]
mod xt_bindings {

    use crate::{
        cuComplex, cuDoubleComplex, cublasDiagType_t, cublasFillMode_t, cublasOperation_t,
        cublasSideMode_t, cublasStatus_t,
    };

    include!("sys_xt_130800.rs");
}

#[cfg(all(feature = "cublas_13_8", feature = "xt"))]
pub use self::xt_bindings::*;
#[cfg(feature = "cublas_13_8")]
pub use self::{
    cublasCreate_v2 as cublasCreate, cublasDestroy_v2 as cublasDestroy,
    cublasGetPointerMode_v2 as cublasGetPointerMode, cublasGetStream_v2 as cublasGetStream,
    cublasSetPointerMode_v2 as cublasSetPointerMode, cublasSetStream_v2 as cublasSetStream,
    cublasSetWorkspace_v2 as cublasSetWorkspace,
};

pub use singe_loader::{Absence, LoadError, take_absence};

static LIBRARY: singe_loader::Library = singe_loader::Library::cublas();

/// Loads cuBLAS now, instead of on the first call of one of its functions.
///
/// # Errors
///
/// Returns an error if the library has no files on this platform or one of its files would not load.
pub fn load() -> Result<(), LoadError> {
    LIBRARY.load()
}

static LT: singe_loader::Library = singe_loader::Library::cublas_lt();

/// Loads cuBLASLt now, instead of on the first call of one of its functions.
///
/// # Errors
///
/// Returns an error if the library has no files on this platform or one of its files would not load.
pub fn load_lt() -> Result<(), LoadError> {
    LT.load()
}

#[cfg(feature = "cublas_13_8")]
singe_loader::missing_status!(
    cublasStatus_t,
    library: cublasStatus_t::CUBLAS_STATUS_NOT_INITIALIZED,
    symbol: cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED,
);

#[cfg(test)]
mod tests {
    use std::ptr;

    use super::*;

    #[cfg(feature = "cublas_13_8")]
    #[test]
    fn test_it_works() {
        let mut version = 0;
        // SAFETY: the handle is created, used and destroyed within this block.
        unsafe {
            let mut ctx = ptr::null_mut();
            assert_eq!(
                cublasCreate_v2(&mut ctx),
                cublasStatus_t::CUBLAS_STATUS_SUCCESS,
            );
            cublasGetVersion_v2(ctx, &mut version);
        }
        println!("cuBLAS version: {}", version);
        assert_ne!(version, 0);
    }
}
