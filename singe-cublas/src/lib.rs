//! Safe Rust wrappers for NVIDIA cuBLAS, cuBLASLt, and cuBLASXt: BLAS, configurable matrix multiplication, and BLAS level 3 across several GPUs.
//!
//! This crate provides:
//!
//! - **BLAS routines:** levels 1, 2, and 3 and the cuBLAS extensions in [`blas`], generic over [`f32`], [`f64`], [`Complex32`](singe_cuda::types::Complex32), and [`Complex64`](singe_cuda::types::Complex64).
//! - **Typed GEMM:** column-major matrix views and strided batches described by a [`GemmShape`](crate::gemm::GemmShape), plus a [`Gemm`](crate::operation::Gemm) operation that owns its buffers and can be recorded into a CUDA graph.
//! - **Scalars:** [`Scalar`](crate::scalar::Scalar) and [`ScalarMut`](crate::scalar::ScalarMut) hold host or device values and set the cuBLAS pointer mode to match.
//! - **cuBLASLt:** matmul descriptors, matrix layouts, algorithm heuristics, fused epilogues, and scaled FP8 and FP4 inputs, behind the `lt` feature.
//! - **cuBLASXt:** level 3 routines on host memory, split across several GPUs, behind the `xt` feature.
//!
//! # Example
//!
//! Multiply two column-major 2x2 matrices and read the product back:
//!
//! ```rust,no_run
//! use singe_cublas::{blas::level3::gemm, context::Context, scalar::Scalar, types::Operation};
//! use singe_cuda::{context::Context as CudaContext, memory::DeviceMemory};
//!
//! fn main() -> anyhow::Result<()> {
//!     let cuda = CudaContext::create()?;
//!     let context = Context::create(&cuda)?;
//!
//!     let a = DeviceMemory::from_slice(&cuda, &[1.0_f64, 3.0, 2.0, 4.0])?;
//!     let b = DeviceMemory::from_slice(&cuda, &[5.0_f64, 7.0, 6.0, 8.0])?;
//!     let mut c = DeviceMemory::<f64>::zeroed(&cuda, 4)?;
//!
//!     gemm(
//!         &context,
//!         Operation::NonTranspose,
//!         Operation::NonTranspose,
//!         2,
//!         2,
//!         2,
//!         Scalar::host(&1.0),
//!         &a,
//!         2,
//!         &b,
//!         2,
//!         Scalar::host(&0.0),
//!         &mut c,
//!         2,
//!     )?;
//!
//!     assert_eq!(c.copy_to_host_vec()?, [19.0, 43.0, 22.0, 50.0]);
//!     Ok(())
//! }
//! ```
//!
//! A [`Context`](crate::context::Context) owns a cuBLAS handle in a [`singe-cuda`](singe_cuda) context, and the BLAS routines take it with their device buffers.
//!
//! [`gemm`](crate::blas::level3::gemm) checks the shape, the leading dimensions, and the CUDA context of every operand, then waits for the handle's stream before it returns.
//! Each routine also has an `unsafe` `<operation>_async` form that only enqueues the work and can be captured into a CUDA graph; its caller keeps the operands alive until the work finishes.
//!
//! # Features
//!
//! - `lt`: the `lt` module, for cuBLASLt.
//! - `xt`: the `xt` module, for cuBLASXt.
//!
//! # Raw execution
//!
//! Each operation has an `unsafe fn <operation>_async` that checks shapes, leading dimensions, increments, data types, and the CUDA context of every operand and device scalar, orders the handle's stream after the producers of that storage, enqueues the call, and returns without waiting.
//! It is allowed during stream capture.
//! The caller must:
//!
//! - keep every operand, device scalar, and pointer array alive and unmoved until the work completes, including after an error or unwind, and through every replay of a captured graph;
//! - give the operation exclusive access to its outputs, and keep other work from writing its inputs until it completes;
//! - for device arrays of matrix pointers, make every pointer address a matrix of the declared type and shape in the handle's context, with outputs that overlap neither each other nor the inputs.
//!
//! Host scalars are read, and host results written, before the call returns.
//! The safe `fn <operation>` runs the `_async` form inside [`Context::sync_scope`](crate::context::Context::sync_scope), which rejects capture and waits for the stream even when enqueuing fails.
//! The copies in [`memory`] take a stream per call and also have `_scoped` forms over a [`StreamScope`](singe_cuda::stream::StreamScope).
//! cuBLASLt's `matmul_async` and `matrix_transform_async` have no safe forms, because a layout descriptor does not establish that its operands hold every element it describes.

#[cfg(not(feature = "cublas_13_8"))]
compile_error!("singe-cublas requires a version feature, such as `cublas_13_8`");

pub mod blas;

/// cuBLAS handle ownership, CUDA contexts, streams, and execution settings.
pub mod context;

/// cuBLAS statuses, wrapper errors, and result types.
pub mod error;

/// GEMM over typed matrix views.
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

pub use singe_core::{CudaVersion, LibraryVersion};
pub use singe_cublas_sys as sys;
use singe_cuda::types::LibraryProperty;

use crate::error::{Error, Result, try_ffi};

/// Returns the loaded cuBLAS library version.
///
/// # Errors
///
/// Returns an error if cuBLAS cannot report its version properties or a property is negative.
pub fn version() -> Result<LibraryVersion> {
    LibraryVersion::from_properties(
        library_property(LibraryProperty::Major)?,
        library_property(LibraryProperty::Minor)?,
        library_property(LibraryProperty::Patch)?,
    )
    .ok_or_else(|| Error::out_of_range("cublas version"))
}

/// Returns the CUDA runtime version used by cuBLAS.
///
/// This is the version of the CUDA runtime that cuBLAS was linked against at build time.
///
/// # Errors
///
/// Returns an error if the native runtime version does not fit in `u32`.
pub fn cudart_version() -> Result<CudaVersion> {
    // SAFETY: `cublasGetCudartVersion` takes no arguments and only reports a linked version.
    CudaVersion::from_raw(unsafe { sys::cublasGetCudartVersion() })
        .ok_or_else(|| Error::out_of_range("cublas cudart version"))
}

/// Returns the requested cuBLAS library property.
/// See [`LibraryProperty`] for supported properties.
///
/// # Errors
///
/// Returns an error if cuBLAS cannot report the requested property.
pub fn library_property(property: LibraryProperty) -> Result<i32> {
    let mut value = 0;
    // SAFETY: `cublasGetProperty` writes one `i32` to `value`, a live local.
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
        let _test = setup_context()?;
        let version = version()?;
        println!("cuBLAS version: {version}");
        assert_ne!(version.major(), 0);
        assert_ne!(cudart_version()?.major(), 0);
        assert_ne!(library_property(LibraryProperty::Major)?, 0);
        Ok(())
    }
}
