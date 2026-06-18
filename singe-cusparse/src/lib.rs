//! Safe cuSPARSE wrappers for sparse descriptors and sparse GPU operations.
//!
//! This crate wraps cuSPARSE contexts, sparse and dense vector descriptors,
//! sparse and dense matrix descriptors, SpMV, SpMM, SDDMM, SpGEMM, sparse
//! triangular solve, vector-vector operations, and scalar helpers.

pub mod context;
pub mod error;
pub mod matrix;
pub mod operation;
pub mod scalar;
pub mod sddmm;
pub mod spgemm;
pub mod spmm;
pub mod spmv;
pub mod spsm;
pub mod spsv;
pub mod spvv;
pub mod types;
pub mod vector;

pub(crate) mod utility;

#[cfg(feature = "testing")]
pub mod testing;

use singe_core::LibraryVersion;
use singe_cuda::types::LibraryProperty;
use singe_cusparse_sys as sys;

use crate::error::Result;

/// Returns the cuSPARSE library version as a packed NVIDIA integer.
///
/// # Errors
///
/// Returns an error if cuSPARSE cannot report one of the version properties.
pub fn version() -> Result<LibraryVersion> {
    let major = library_property(LibraryProperty::Major)?;
    let minor = library_property(LibraryProperty::Minor)?;
    let patch = library_property(LibraryProperty::Patch)?;
    Ok(LibraryVersion::new(major as _, minor as _, patch as _))
}

/// Returns the requested cuSPARSE library property.
/// See [`LibraryProperty`] for supported properties.
///
/// # Errors
///
/// Returns an error if cuSPARSE cannot report the requested property.
pub fn library_property(property: LibraryProperty) -> Result<i32> {
    let mut value = 0;
    unsafe {
        try_ffi!(sys::cusparseGetProperty(property.into(), &raw mut value))?;
    }
    Ok(value)
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use super::*;
    use crate::testing::setup_context;
    use singe_cuda::error::Error as CudaError;

    #[test]
    fn it_works() -> Result<()> {
        let ctx = match setup_context() {
            Ok(ctx) => ctx,
            Err(crate::error::Error::Cuda(CudaError::Cuda { message, .. }))
                if message == "CUDA driver is a stub library" =>
            {
                return Ok(());
            }
            Err(error) => return Err(error),
        };
        let context_version = ctx.version()?;
        let library_version = version()?;
        println!("cuSPARSE version: {context_version}");
        assert_ne!(context_version, 0);
        assert_ne!(library_version, 0);
        assert_ne!(library_property(LibraryProperty::Major)?, 0);
        Ok(())
    }
}
