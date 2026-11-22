//! Safe Rust wrappers for NVIDIA cuDSS, the CUDA direct sparse solver library.
//!
//! This crate provides:
//!
//! - **Matrices:** CSR and dense descriptors over borrowed device storage, including uniform and non-uniform batches, with read-only inputs kept apart from write-only outputs.
//! - **Solver settings:** [`Config`](crate::config::Config) with typed parameters for the algorithms, hybrid CPU and GPU execution and memory modes, and the devices of a multi-GPU solve.
//! - **Solver state:** typed queries such as [`info`](crate::solver_state::SolverState::info), [`lu_nnz`](crate::solver_state::SolverState::lu_nnz), [`memory_estimates`](crate::solver_state::SolverState::memory_estimates), and [`schur_shape`](crate::solver_state::SolverState::schur_shape).
//! - **Multi-GPU and multi-node solves:** handles over several GPUs, plus the communication and threading layers for multi-node multi-GPU (MGMN) and multithreaded modes.
//! - **Memory handlers:** custom device allocators for the memory cuDSS allocates.
//!
//! # Example
//!
//! Solve a 5x5 symmetric positive-definite system and check the solution:
//!
//! ```rust,no_run
//! use singe_cuda::{
//!     context::Context as CudaContext, memory::DeviceMemory, stream::StreamOptions,
//!     view::DeviceOutputView,
//! };
//! use singe_cudss::{
//!     config::Config,
//!     context::Context,
//!     matrix::{CsrMatrixDescriptor, DenseMatrixDescriptor, Matrix},
//!     solver_state::SolverState,
//!     types::{IndexBase, MatrixType, MatrixViewType, Phase},
//! };
//!
//! fn main() -> anyhow::Result<()> {
//!     let cuda = CudaContext::create()?;
//!     let stream = cuda.create_stream(StreamOptions::default())?;
//!     let mut context = Context::create(&cuda)?;
//!     context.set_stream(Some(&stream))?;
//!
//!     // The upper triangle of the matrix in CSR, and the right-hand side.
//!     let row_offsets = DeviceMemory::from_slice(&cuda, &[0_i32, 2, 4, 6, 7, 8])?;
//!     let column_indices = DeviceMemory::from_slice(&cuda, &[0_i32, 2, 1, 2, 2, 4, 3, 4])?;
//!     let values = DeviceMemory::from_slice(&cuda, &[4.0_f64, 1.0, 3.0, 2.0, 5.0, 1.0, 1.0, 2.0])?;
//!     let rhs = DeviceMemory::from_slice(&cuda, &[7.0_f64, 12.0, 25.0, 4.0, 13.0])?;
//!     let mut solution = DeviceMemory::<f64>::zeroed(&cuda, 5)?;
//!
//!     let config = Config::create()?;
//!     let mut state = SolverState::create(&context)?;
//!     {
//!         let a = Matrix::create_csr(
//!             CsrMatrixDescriptor {
//!                 rows: 5,
//!                 columns: 5,
//!                 nnz: 8,
//!                 matrix_type: MatrixType::Spd,
//!                 view_type: MatrixViewType::Upper,
//!                 index_base: IndexBase::Zero,
//!             },
//!             row_offsets.view(),
//!             None,
//!             column_indices.view(),
//!             values.view(),
//!         )?;
//!         let b = Matrix::create_dense(DenseMatrixDescriptor::new(5, 1, 5), rhs.view())?;
//!         let mut x = Matrix::create_dense_output(
//!             DenseMatrixDescriptor::new(5, 1, 5),
//!             DeviceOutputView::from_output(&mut solution),
//!         )?;
//!
//!         context.sync_scope(|| {
//!             // SAFETY: the buffers were written synchronously before this call, the CSR indices are valid,
//!             // and `sync_scope` waits for the stream before the matrices and their buffers are released.
//!             unsafe {
//!                 context.execute_async(Phase::ANALYSIS, &config, &mut state, &a, &mut x, &b)?;
//!                 context.execute_async(Phase::FACTORIZATION, &config, &mut state, &a, &mut x, &b)?;
//!                 context.execute_async(Phase::SOLVE, &config, &mut state, &a, &mut x, &b)
//!             }
//!         })?;
//!     }
//!     assert_eq!(state.info()?, 0);
//!
//!     let x = solution.copy_to_host_vec()?;
//!     for (value, expected) in x.iter().zip([1.0, 2.0, 3.0, 4.0, 5.0]) {
//!         assert!((value - expected).abs() < 1.0e-12);
//!     }
//!     Ok(())
//! }
//! ```
//!
//! A [`Context`](crate::context::Context) owns a cuDSS handle in a [`singe-cuda`](singe_cuda) context and runs the analysis, factorization, and solve phases on matrices that borrow device buffers.
//! [`SolverState`](crate::solver_state::SolverState) carries the analysis, the factors, and the results from one phase to the next.
//!
//! [`execute_async`](crate::context::Context::execute_async) checks that the solver state and the matrices belong to the context, enqueues the phase on the context's stream, and returns without waiting.
//! [`Context::sync_scope`](crate::context::Context::sync_scope) runs a closure of [`execute_async`](crate::context::Context::execute_async) calls, rejects stream capture, and waits for the installed stream even when the closure fails; it does not wait for the per-device streams of a multi-GPU handle.
//! After the solve, [`SolverState::info`](crate::solver_state::SolverState::info) returns zero if it succeeded; check it before reading the solution.
//!
//! # Raw execution
//!
//! [`Context::execute_async`](context::Context::execute_async) checks that the solver state belongs to the context and the matrices to its CUDA context, orders the installed stream after the producers of pointer tables and of buffers bound through typed device views, enqueues the phase, and returns without waiting.
//! The caller must:
//!
//! - order the producers of every other buffer before the call: pointer-table entries, host buffers, solver data parameters, and buffers of descriptors reclaimed with [`Matrix::from_raw`](matrix::Matrix::from_raw);
//! - keep the context, configuration, solver state, matrix descriptors, their buffers and pointer tables, and the selected streams alive and unchanged until the work completes, and through every replay of a captured graph;
//! - initialize every input, and give the solution exclusive access that overlaps no input;
//! - wait for completion and check [`SolverState::info`](solver_state::SolverState::info) before reading produced values.
//!
//! [`Context::sync_scope`](context::Context::sync_scope) runs `execute_async` calls and waits for the installed stream.

#[cfg(not(feature = "cudss_0_8"))]
compile_error!("singe-cudss requires a version feature, such as `cudss_0_8`");

/// Solver configuration and typed configuration parameters.
pub mod config;

/// cuDSS handles, CUDA context ownership, streams, and execution phases.
pub mod context;

/// Native statuses and errors from the Rust wrappers.
pub mod error;

/// Process-wide cuDSS logging controls and callbacks.
pub mod logger;

/// Dense and sparse matrix descriptors that borrow device storage.
pub mod matrix;

pub mod memory;

/// Solver state and host or device queries of solver results.
pub mod solver_state;

/// Numeric representations, algorithm choices, flags, and parameter identifiers.
pub mod types;

mod utility;

pub use singe_core::LibraryVersion;
use singe_cuda::types::LibraryProperty;
pub use singe_cudss_sys as sys;

use crate::error::{Error, Result, try_ffi};

/// Returns the cuDSS library version reported by the loaded runtime.
///
/// # Errors
///
/// Returns an error if cuDSS cannot report one of the version properties or a version component is negative.
pub fn version() -> Result<LibraryVersion> {
    LibraryVersion::from_properties(
        library_property(LibraryProperty::Major)?,
        library_property(LibraryProperty::Minor)?,
        library_property(LibraryProperty::Patch)?,
    )
    .ok_or_else(|| Error::out_of_range("cudss version"))
}

/// Returns a cuDSS runtime library property.
///
/// cuDSS supports the standard CUDA [`LibraryProperty`] values for major, minor, and patch version components.
///
/// # Errors
///
/// Returns an error if cuDSS cannot report the requested property.
pub fn library_property(property: LibraryProperty) -> Result<i32> {
    let mut value = 0;
    // SAFETY: `value` is a live local `i32` that cuDSS writes the property into.
    unsafe {
        try_ffi!(sys::cudssGetProperty(property.into(), &raw mut value))?;
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_it_works() -> Result<()> {
        let version = version()?;
        println!("cuDSS version: {version}");
        assert_ne!(version, LibraryVersion::new(0, 0, 0));
        Ok(())
    }
}
