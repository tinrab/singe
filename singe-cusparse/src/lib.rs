//! Safe Rust wrappers for cuSPARSE, NVIDIA's sparse linear algebra library.
//! A [`Context`](crate::context::Context) owns a cuSPARSE handle in a [`singe-cuda`](singe_cuda) context, and the descriptors in [`matrix`] and [`vector`] borrow device buffers.
//! Creating and configuring descriptors is safe; every operation and workspace-size query is `unsafe`, because a descriptor does not check that its sparse indices are in bounds or its values initialized.
//!
//! This crate provides:
//!
//! - **Sparse formats:** CSR, CSC, COO, BSR, blocked-ELL, and sliced-ELL matrix descriptors, dense descriptors, and conversions between them.
//! - **Products:** sparse matrix-vector ([`spmv`]), sparse-dense matrix-matrix ([`spmm`]), sparse-sparse ([`spgemm`]), and sampled dense-dense ([`sddmm`]) multiplication.
//! - **Custom operators:** [`SpMmOpPlan`](crate::spmm::SpMmOpPlan) compiles a sparse-dense product with operators supplied as LTO-IR.
//! - **Triangular solves:** [`spsv`] and [`spsm`] for one or several right-hand sides, with reusable analysis.
//! - **Vectors:** sparse and dense vector descriptors, with gather and scatter between them.
//!
//! # Example
//!
//! Compute `y = A * x` for a 3x3 CSR matrix and read `y` back:
//!
//! ```rust,no_run
//! use singe_cuda::{context::Context as CudaContext, memory::DeviceMemory, view::DeviceOutputView};
//! use singe_cusparse::{
//!     context::Context,
//!     matrix::SparseMatrixDescriptor,
//!     scalar::Scalar,
//!     spmv::{spmv_async, spmv_workspace_size},
//!     types::{IndexBase, Operation, SpMvAlgorithm},
//!     vector::DenseVectorDescriptor,
//! };
//!
//! fn main() -> anyhow::Result<()> {
//!     let cuda = CudaContext::create()?;
//!     let context = Context::create(&cuda)?;
//!
//!     // A = [1 0 2; 0 3 0; 4 0 5] in CSR, and x = [1, 2, 3].
//!     let mut row_offsets = DeviceMemory::from_slice(&cuda, &[0_i32, 2, 3, 5])?;
//!     let mut column_indices = DeviceMemory::from_slice(&cuda, &[0_i32, 2, 1, 0, 2])?;
//!     let mut values = DeviceMemory::from_slice(&cuda, &[1.0_f32, 2.0, 3.0, 4.0, 5.0])?;
//!     let mut x = DeviceMemory::from_slice(&cuda, &[1.0_f32, 2.0, 3.0])?;
//!     let mut y = DeviceMemory::<f32>::zeroed(&cuda, 3)?;
//!
//!     {
//!         let matrix = SparseMatrixDescriptor::create_csr(
//!             &context,
//!             3,
//!             3,
//!             5,
//!             Some(DeviceOutputView::from_output(&mut row_offsets)),
//!             Some(DeviceOutputView::from_output(&mut column_indices)),
//!             Some(DeviceOutputView::from_output(&mut values)),
//!             IndexBase::Zero,
//!         )?;
//!         let x_vector =
//!             DenseVectorDescriptor::create(&context, 3, DeviceOutputView::from_output(&mut x))?;
//!         let mut y_vector =
//!             DenseVectorDescriptor::create(&context, 3, DeviceOutputView::from_output(&mut y))?;
//!         let alpha = 1.0_f32;
//!         let beta = 0.0_f32;
//!
//!         // SAFETY: the descriptors cover the initialized device buffers above, and the CSR indices are in bounds.
//!         let workspace_size = unsafe {
//!             spmv_workspace_size::<f32>(
//!                 &context,
//!                 Operation::NonTranspose,
//!                 Scalar::host(&alpha),
//!                 &matrix,
//!                 &x_vector,
//!                 Scalar::host(&beta),
//!                 &mut y_vector,
//!                 SpMvAlgorithm::Default,
//!             )?
//!         };
//!         let mut workspace = DeviceMemory::<u8>::zeroed(&cuda, workspace_size.max(1))?;
//!         context.sync_scope(|| {
//!             // SAFETY: as above, and `sync_scope` waits for the stream before the borrowed buffers are released.
//!             unsafe {
//!                 spmv_async::<f32>(
//!                     &context,
//!                     Operation::NonTranspose,
//!                     Scalar::host(&alpha),
//!                     &matrix,
//!                     &x_vector,
//!                     Scalar::host(&beta),
//!                     &mut y_vector,
//!                     SpMvAlgorithm::Default,
//!                     Some(DeviceOutputView::from_output(&mut workspace)),
//!                 )
//!             }
//!         })?;
//!     }
//!
//!     assert_eq!(y.copy_to_host_vec()?, [7.0, 6.0, 19.0]);
//!     Ok(())
//! }
//! ```
//!
//! [`spmv_async`](crate::spmv::spmv_async) checks that the descriptors, scalars, and workspace belong to the handle's CUDA context, enqueues the product, and returns without waiting.
//! cuSPARSE operations have no waiting forms: [`Context::sync_scope`](crate::context::Context::sync_scope) runs a closure of `_async` calls, rejects stream capture, and waits for the handle's stream even when the closure fails.
//! The `_async` functions can also be called during stream capture, to record them into a CUDA graph.
//!
//! # Raw execution
//!
//! Operations are `unsafe fn <operation>_async`: each checks that its descriptors, scalars, and workspace belong to the handle's CUDA context, orders the handle's stream after their producers, enqueues the call, and returns without waiting.
//! They are allowed during stream capture.
//! The caller must:
//!
//! - keep every descriptor, scalar, workspace, and the storage they borrow alive and unmoved until the work completes, including after an error or unwind, and through every replay of a captured graph;
//! - initialize the inputs, including the accumulator when beta is nonzero, and give the operation exclusive access to its outputs;
//! - supply sparse indices in bounds for the descriptor shapes, and values, compute types, and custom operator code that match the native ABI;
//! - bind storage that covers every batch of a strided-batch descriptor, since descriptors check only the first batch;
//! - leave analysis and preprocessing workspaces unmodified through the last operation that depends on them.
//!
//! cuSPARSE has no safe waiting forms, because a descriptor does not establish that its sparse indices are in bounds or its values initialized; run operations inside [`Context::sync_scope`](crate::context::Context::sync_scope) to wait for them.

#[cfg(not(feature = "cusparse_12_8"))]
compile_error!("singe-cusparse requires a version feature, such as `cusparse_12_8`");

/// cuSPARSE handle ownership, CUDA context association, streams, and scalar pointer modes.
pub mod context;

pub mod descriptor;

/// Typed cuSPARSE statuses and Rust wrapper errors.
pub mod error;

pub mod logger;

/// Sparse and dense matrix descriptors and storage-format conversions.
pub mod matrix;

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

/// Typed matrix formats, index representations, algorithms, and operation settings.
pub mod types;

pub(crate) mod utility;

/// Sparse and dense vector descriptors and gather or scatter operations.
pub mod vector;

/// Serialized GPU test setup for cuSPARSE operations.
#[cfg(feature = "testing")]
pub mod testing;

pub use singe_core::LibraryVersion;
use singe_cuda::types::LibraryProperty;
pub use singe_cusparse_sys as sys;

use crate::error::{Error, Result, try_ffi};

/// Returns the cuSPARSE library version from its major, minor, and patch properties.
///
/// # Errors
///
/// Returns an error if cuSPARSE cannot report one of the version properties or a property is negative.
pub fn version() -> Result<LibraryVersion> {
    LibraryVersion::from_properties(
        library_property(LibraryProperty::Major)?,
        library_property(LibraryProperty::Minor)?,
        library_property(LibraryProperty::Patch)?,
    )
    .ok_or_else(|| Error::out_of_range("cusparse version"))
}

/// Returns the requested cuSPARSE library property.
/// See [`LibraryProperty`] for supported properties.
///
/// # Errors
///
/// Returns an error if cuSPARSE cannot report the requested property.
pub fn library_property(property: LibraryProperty) -> Result<i32> {
    let mut value = 0;
    // SAFETY: `property` is a valid enum value and `value` is a writable local.
    unsafe {
        try_ffi!(sys::cusparseGetProperty(property.into(), &raw mut value))?;
    }
    Ok(value)
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use super::*;
    use crate::testing::setup_context;

    #[test]
    fn test_it_works() -> Result<()> {
        let _test = setup_context()?;
        let library_version = version()?;
        println!("cuSPARSE version: {library_version}");
        assert_ne!(library_version.major(), 0);
        assert_ne!(library_property(LibraryProperty::Major)?, 0);
        Ok(())
    }
}
