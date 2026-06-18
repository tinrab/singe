//! Safe cuBLAS and cuBLASLt wrappers for dense GPU linear algebra.
//!
//! This crate provides BLAS level routines, cuBLAS context management, cuBLASLt
//! matmul descriptors, algorithm selection, layouts, preferences, epilogues, and
//! execution helpers over the raw `singe-cublas-sys` bindings.

pub mod blas;
pub mod context;
pub mod error;
pub mod gemm;
pub mod lt;
pub mod memory;
pub mod scalar;
pub mod types;

pub(crate) mod utility;

#[cfg(feature = "testing")]
pub mod testing;

use singe_cublas_sys as sys;
use singe_cuda::types::LibraryProperty;

use crate::{error::Result, utility::to_u64};

/// Returns the CUDA runtime version used by cuBLAS.
///
/// This is the version of the CUDA runtime that cuBLAS was linked against at
/// build time.
pub fn cudart_version() -> Result<u64> {
    to_u64(unsafe { sys::cublasGetCudartVersion() }, "version")
}

/// Returns the requested cuBLAS library property.
/// See [`LibraryProperty`] for supported properties.
///
/// # Errors
///
/// Returns an error if cuBLAS cannot report the requested property.
pub fn library_property(property: LibraryProperty) -> Result<i32> {
    let mut value = 0;
    unsafe {
        try_ffi!(sys::cublasGetProperty(property.into(), &raw mut value))?;
    }
    Ok(value)
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use super::*;
    use crate::testing::setup_context;

    #[test]
    fn it_works() -> Result<()> {
        let ctx = setup_context()?;
        let version = ctx.version()?;
        println!("cuBLAS version: {version}");
        assert_ne!(version, 0);
        assert_ne!(cudart_version()?, 0);
        assert_ne!(library_property(LibraryProperty::Major)?, 0);
        Ok(())
    }
}
