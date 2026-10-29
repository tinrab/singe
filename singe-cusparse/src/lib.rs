//! cuSPARSE resource wrappers and public raw sparse GPU operations.
//!
//! This crate wraps cuSPARSE contexts, sparse and dense vector descriptors,
//! sparse and dense matrix descriptors, `SpMV`, `SpMM`, SDDMM, `SpGEMM`, sparse
//! triangular solve, vector-vector operations, and scalar helpers.
//!
//! # Raw execution
//!
//! Native operations are unsafe: descriptors do not establish initialized input
//! values, valid sparse indices, matching compute layouts, or queued ownership.
//! Order producers before reads, preserve initialized accumulators when beta is
//! nonzero, and retain every participant with the required access through
//! completion on success, error, and unwind, including captured graph replay.
//! Analysis and preprocessing state may retain workspaces and bindings across
//! later operations; keep them valid through the last dependent stage.
//! Scratch and custom operation code must match the queried native ABI.
//! Descriptor storage regions preserve actual allocation metadata and buffer
//! borrows, without promoting pending storage or claiming initialized padding.

/// cuSPARSE handle ownership, CUDA context association, streams, and scalar pointer modes.
pub mod context;
pub mod descriptor;
/// Typed cuSPARSE statuses and Rust wrapper errors.
pub mod error;
pub mod logger;
/// Sparse and dense matrix descriptors and storage-format conversions.
pub mod matrix;
/// Native analysis descriptors and custom sparse-matrix operation plans.
pub mod operation;
pub mod scalar;
/// Sampled dense matrix multiplication using a sparse output pattern.
pub mod sddmm;
/// Sparse-sparse matrix multiplication and reusable multiplication analysis.
pub mod spgemm;
/// Sparse-dense matrix multiplication and custom operator plans.
pub mod spmm;
/// Sparse matrix-vector multiplication and preprocessing.
pub mod spmv;
/// Sparse triangular solves with multiple right-hand sides.
pub mod spsm;
/// Sparse triangular solves with one vector right-hand side.
pub mod spsv;
/// Sparse-dense vector dot products.
pub mod spvv;
/// Typed matrix formats, index representations, algorithms, and operation settings.
pub mod types;
pub(crate) mod utility;
/// Sparse and dense vector descriptors and gather or scatter operations.
pub mod vector;

/// Serialized GPU test setup for cuSPARSE operations.
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
    // SAFETY: `cusparseGetProperty` receives valid output locations.
    unsafe {
        try_ffi!(sys::cusparseGetProperty(property.into(), &raw mut value))?;
    }
    Ok(value)
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use singe_cuda::error::Error as CudaError;

    use super::*;
    use crate::{error::Error, testing::setup_context};

    #[test]
    fn test_it_works() -> Result<()> {
        let context = match setup_context() {
            Ok(context) => context,
            Err(Error::Cuda(CudaError::Cuda { message, .. }))
                if message == "CUDA driver is a stub library" =>
            {
                return Ok(());
            }
            Err(error) => return Err(error),
        };
        let _context = context;
        let library_version = version()?;
        println!("cuSPARSE version: {library_version}");
        assert_ne!(library_version, 0);
        assert_ne!(library_property(LibraryProperty::Major)?, 0);
        Ok(())
    }
}
