//! Safe Rust wrappers for cuSOLVER's dense API: matrix factorizations, linear solves, eigenvalue problems, and singular value decompositions on the GPU.
//! Each operation is a typed description, such as [`dense::Getrf`], that reports its workspace sizes and runs with a bindings struct of borrowed matrices, outputs, and workspaces.
//!
//! This crate provides:
//!
//! - **Dense factorizations and solves:** Cholesky, LU, QR, and symmetric indefinite factorizations with their solves, plus triangular inversion and tridiagonal and bidiagonal reductions, in [`dense`].
//! - **Eigenvalue solvers:** symmetric, generalized symmetric, and general problems, such as [`Syevd`](crate::eigen::Syevd), [`Syevj`](crate::eigen::Syevj), [`Sygvd`](crate::eigen::Sygvd), and [`Geev`](crate::eigen::Geev), including batched forms.
//! - **Singular value decompositions:** [`Gesvd`](crate::svd::Gesvd), [`Gesvdj`](crate::svd::Gesvdj), [`Gesvdp`](crate::svd::Gesvdp), [`Gesvdr`](crate::svd::Gesvdr), and [`GesvdaStridedBatched`](crate::svd::GesvdaStridedBatched).
//! - **Iterative refinement:** mixed-precision linear and least-squares solvers in [`irs`], with their residual history.
//! - **Layouts:** borrowed column-major matrices, vectors, and host and device workspaces in [`layout`].
//!
//! # Example
//!
//! LU-factorize a 3x3 matrix and solve `A * x = b`:
//!
//! ```rust,no_run
//! use std::mem::MaybeUninit;
//!
//! use singe_cuda::{context::Context as CudaContext, memory::DeviceMemory, view::DeviceOutputView};
//! use singe_cusolver::{
//!     context::Context,
//!     dense::{Getrf, GetrfBindings, Getrs, GetrsBindings},
//!     layout::{ByteWorkspaceMut, MatrixLayout, MatrixMut},
//!     params::Params,
//!     types::Operation,
//! };
//!
//! const N: usize = 3;
//!
//! fn main() -> anyhow::Result<()> {
//!     let cuda = CudaContext::create()?;
//!     let context = Context::create(&cuda)?;
//!     let params = Params::create()?;
//!
//!     // Column-major storage of
//!     //     | 1 2 3 |
//!     // A = | 4 5 6 |
//!     //     | 7 8 10|
//!     let mut a = DeviceMemory::from_slice(&cuda, &[1.0_f64, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 10.0])?;
//!     let mut b = DeviceMemory::from_slice(&cuda, &[1.0_f64, 2.0, 3.0])?;
//!     let mut pivots = DeviceMemory::<i64>::zeroed(&cuda, N)?;
//!     let mut info = DeviceMemory::<i32>::zeroed(&cuda, 1)?;
//!
//!     let getrf = Getrf::<f64>::new(N, N);
//!     let sizes = getrf.workspace_size(&context, &params, MatrixLayout::new(N))?;
//!     let mut device_workspace = DeviceMemory::<u8>::zeroed(&cuda, sizes.device_bytes)?;
//!     let mut host_workspace = vec![MaybeUninit::<u8>::uninit(); sizes.host_bytes];
//!
//!     getrf.execute(
//!         &context,
//!         GetrfBindings {
//!             a: MatrixMut::new(&mut a, N),
//!             pivots: Some(DeviceOutputView::from_output(&mut pivots)),
//!             workspace: ByteWorkspaceMut::new(&mut device_workspace, &mut host_workspace),
//!             dev_info: DeviceOutputView::from_output(&mut info),
//!         },
//!         &params,
//!     )?;
//!     assert_eq!(info.copy_to_host_vec()?, [0]);
//!
//!     let factor = MatrixMut::new(&mut a, N);
//!     Getrs::<f64>::new(Operation::NonTranspose, N, 1).execute(
//!         &context,
//!         GetrsBindings {
//!             factor: factor.as_ref(),
//!             pivots: pivots.view(),
//!             rhs: MatrixMut::new(&mut b, N),
//!             dev_info: DeviceOutputView::from_output(&mut info),
//!         },
//!         &params,
//!     )?;
//!     assert_eq!(info.copy_to_host_vec()?, [0]);
//!
//!     let x = b.copy_to_host_vec()?;
//!     let expected = [-1.0 / 3.0, 2.0 / 3.0, 0.0];
//!     assert!(x.iter().zip(expected).all(|(x, e)| (x - e).abs() < 1.0e-9));
//!     Ok(())
//! }
//! ```
//!
//! [`execute`](crate::dense::Getrf::execute) waits for the context's stream, even when enqueuing fails.
//! Most operations also have an `unsafe` [`execute_async`](crate::dense::Getrf::execute_async) that only enqueues the work and can be captured into a CUDA graph; the eigenvalue solves that read back a count have none.
//! A finished operation can still have failed numerically, so check the [`dev_info`](crate::dense::GetrfBindings::dev_info) it writes before using its other outputs.
//!
//! # Raw execution
//!
//! The `unsafe` `*_async` functions and [`execute_async`](crate::dense::Getrf::execute_async) methods check contexts, shapes, capacities, and workspace sizes, order the context's stream after the producers of their device operands, enqueue the call, and return without waiting.
//! They are allowed during stream capture.
//! The caller must:
//!
//! - keep the context, parameter objects, operands, and device and host workspaces alive until the work completes, including after an error or unwind, and through every replay of a captured graph;
//! - initialize every input;
//! - give the operation exclusive access to every range it writes, overlapping no other operand;
//! - keep every pointee of a pointer table alive and unused by other work until the work completes, even past the lifetime its constructor names.
//!
//! Outputs hold a valid result only after the work completes and the status it writes reports success.
//! The safe [`execute`](crate::dense::Getrf::execute) methods reject stream capture and wait for completion, even on failure.
//!
//! # Workspaces
//!
//! Generic-API operations take byte workspaces, sized by [`workspace_size`](crate::dense::Getrf::workspace_size) queries that return bytes.
//! Legacy routines take workspaces typed by the element `T`, because cuSOLVER counts their workspace in elements, and their [`workspace_len`](crate::eigen::Syevj::workspace_len) queries return that element count.

#[cfg(not(feature = "cusolver_12_3"))]
compile_error!("singe-cusolver requires a version feature, such as `cusolver_12_3`");

pub mod context;
pub mod data_type;
pub mod dense;
pub mod eigen;
pub mod error;
pub mod irs;
pub mod layout;
pub mod logger;
pub mod params;
pub mod svd;
pub mod types;

pub(crate) mod utility;

#[cfg(feature = "testing")]
pub mod testing;

pub use singe_core::LibraryVersion;
use singe_cuda::types::LibraryProperty;
pub use singe_cusolver_sys as sys;

use crate::error::{Error, Result, try_ffi};

/// Returns the cuSOLVER library version from its major, minor, and patch properties.
///
/// # Errors
///
/// Returns an error if cuSOLVER cannot report one of the version properties or a property is negative.
pub fn version() -> Result<LibraryVersion> {
    LibraryVersion::from_properties(
        library_property(LibraryProperty::Major)?,
        library_property(LibraryProperty::Minor)?,
        library_property(LibraryProperty::Patch)?,
    )
    .ok_or_else(|| Error::out_of_range("cusolver version"))
}

/// Returns the requested cuSOLVER library property.
///
/// # Errors
///
/// Returns an error if cuSOLVER cannot report the requested property.
pub fn library_property(property: LibraryProperty) -> Result<i32> {
    let mut value = 0;
    // SAFETY: the call writes one `i32` to the local `value` and reads only the property enum.
    unsafe {
        try_ffi!(sys::cusolverGetProperty(property.into(), &raw mut value))?;
    }
    Ok(value)
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use singe_core::PackedVersion;

    use super::*;
    use crate::testing::setup_context;

    #[test]
    fn test_it_works() -> Result<()> {
        let _test = setup_context()?;
        let version = version()?;
        let mut packed = 0;
        // SAFETY: the call writes one `i32` to the local `packed`.
        unsafe {
            try_ffi!(sys::cusolverGetVersion(&raw mut packed))?;
        }
        assert_eq!(
            LibraryVersion::from_packed(packed, PackedVersion::Major1000Minor100),
            Some(version)
        );
        Ok(())
    }
}
