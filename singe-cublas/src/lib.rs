//! Safe cuBLAS wrappers for dense GPU linear algebra.
//!
//! This crate provides BLAS level routines and cuBLAS context management over the raw `singe-cublas-sys` bindings.
//! Enable the `lt` feature for cuBLASLt APIs and the `xt` feature for cuBLASXt APIs.

pub mod blas;
/// cuBLAS handle ownership, CUDA contexts, streams, and execution settings.
pub mod context;
/// cuBLAS statuses, wrapper errors, and result types.
pub mod error;
/// Typed matrix views and general matrix multiplication.
pub mod gemm;
pub mod logger;
pub mod memory;
pub mod operation;
/// Borrowed host and device scalars with their cuBLAS pointer modes.
pub mod scalar;
/// Typed cuBLAS operations, algorithms, and execution settings.
pub mod types;

/// Descriptor-based matrix operations and algorithm selection with cuBLASLt.
#[cfg(feature = "lt")]
pub mod lt;
/// Multi-GPU and hybrid CPU/GPU matrix operations with cuBLASXt.
#[cfg(feature = "xt")]
pub mod xt;

pub(crate) mod utility;

/// Serialized GPU test setup for cuBLAS.
#[cfg(feature = "testing")]
pub mod testing;

use singe_core::{CudaRuntimeVersion, LibraryVersion};
use singe_cublas_sys as sys;
use singe_cuda::types::LibraryProperty;

use crate::{error::Result, utility::to_u64};

/// Returns the loaded cuBLAS library version.
///
/// # Errors
///
/// Returns an error if cuBLAS cannot report its version properties.
pub fn version() -> Result<LibraryVersion> {
    Ok(LibraryVersion::new(
        to_u64(library_property(LibraryProperty::Major)?, "major version")?,
        to_u64(library_property(LibraryProperty::Minor)?, "minor version")?,
        to_u64(library_property(LibraryProperty::Patch)?, "patch version")?,
    ))
}

/// Returns the CUDA runtime version used by cuBLAS.
///
/// This is the version of the CUDA runtime that cuBLAS was linked against at
/// build time.
///
/// # Errors
///
/// Returns an error if the native runtime version cannot be represented as `u64`.
pub fn cudart_version() -> Result<CudaRuntimeVersion> {
    Ok(CudaRuntimeVersion::from_raw(to_u64(
        // SAFETY: `cublasGetCudartVersion` takes no arguments and only reports a linked version.
        unsafe { sys::cublasGetCudartVersion() },
        "cudart version",
    )?))
}

/// Returns the requested cuBLAS library property.
/// See [`LibraryProperty`] for supported properties.
///
/// # Errors
///
/// Returns an error if cuBLAS cannot report the requested property.
pub fn library_property(property: LibraryProperty) -> Result<i32> {
    let mut value = 0;
    // SAFETY: `cublasGetProperty` receives valid output locations.
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
    fn test_it_works() -> Result<()> {
        let context = setup_context()?;
        let version = context.version()?;
        println!("cuBLAS version: {version}");
        assert_ne!(version, 0);
        assert_ne!(cudart_version()?, 0);
        assert_ne!(library_property(LibraryProperty::Major)?, 0);
        Ok(())
    }
}
