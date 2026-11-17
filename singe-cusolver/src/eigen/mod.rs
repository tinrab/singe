//! Eigenvalue operations.
//!
//! Workspace queries accept layout metadata.
//! Execution accepts initialized in-place matrices and write-only result and scratch views, including pending storage.
//! Each operation has a waiting `execute` and, except for the selected solves that read back a count, an unsafe `execute_async` that follows the crate's [raw execution contract](crate#raw-execution).
//! Except for selected operations returning a count, numerical status is left in `dev_info` for the caller to inspect before promoting produced values.
//! The generic native 64-bit entry points stay public as `x*_async` functions.

mod info;
mod raw;
pub(crate) mod scalar;
mod validation;

use std::{mem::MaybeUninit, ops::Range};

use singe_cuda::view::DeviceOutputView;
use validation::{check_completed_status, selected_count};

pub use self::{
    info::SyevjInfo,
    raw::{
        xgeev_async, xgeev_workspace_size, xsyev_batched_async, xsyev_batched_workspace_size,
        xsyevd_async, xsyevd_workspace_size, xsyevdx_async, xsyevdx_workspace_size,
    },
};
use crate::{
    context::Context,
    data_type::{SolverElement, SolverReal},
    error::Result,
    layout::{ByteWorkspaceMut, MatrixLayout, MatrixMut, MatrixOutput, WorkspaceSizes},
    params::Params,
    types::{EigenMode, EigenType, FillMode},
};

/// Selection of ascending symmetric or generalized eigenvalues.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EigenSelection<R> {
    /// The entire spectrum.
    All,
    /// Values in the interval (`lower_exclusive`, `upper_inclusive`].
    ByValue {
        /// Excluded lower endpoint of the eigenvalue interval.
        lower_exclusive: R,
        /// Included upper endpoint of the eigenvalue interval.
        upper_inclusive: R,
    },
    /// A zero-based, half-open range of positions in the ascending spectrum.
    ByIndex(Range<usize>),
}

/// In-place symmetric matrix, real eigenvalue output, byte scratch, and status.
/// Batched operations use packed lda*n matrices, n values, and one status per matrix.
#[derive(Debug)]
pub struct SyevdBindings<'a, T: SolverElement> {
    /// Initialized input matrix overwritten by the eigensolver.
    /// In vector mode, the leading result columns contain eigenvectors.
    /// Batched execution packs consecutive matrices at `leading_dimension * n` elements.
    pub a: MatrixMut<'a, T>,
    /// Real eigenvalue output capacity for `n` values per matrix.
    /// Selected solves produce only the returned prefix.
    pub eigenvalues: DeviceOutputView<'a, T::Real>,
    /// Exclusive device and host scratch capacities, both measured in bytes.
    pub workspace: ByteWorkspaceMut<'a>,
    /// Device output capacity for one solver status per matrix in batched execution.
    /// Inspect completed statuses before using numerical results.
    pub dev_info: DeviceOutputView<'a, i32>,
}

/// In-place symmetric matrix and Jacobi outputs with scratch measured in T elements.
#[derive(Debug)]
pub struct SyevjBindings<'a, T: SolverElement> {
    /// Initialized input matrix overwritten by the eigensolver.
    /// In vector mode, the leading result columns contain eigenvectors.
    /// Batched execution packs consecutive matrices at `leading_dimension * n` elements.
    pub a: MatrixMut<'a, T>,
    /// Real eigenvalue output capacity for `n` values per matrix.
    /// Selected solves produce only the returned prefix.
    pub eigenvalues: DeviceOutputView<'a, T::Real>,
    /// Exclusive device scratch capacity measured in elements of `T`.
    pub workspace: DeviceOutputView<'a, T>,
    /// Device output capacity for one solver status per matrix in batched execution.
    /// Inspect completed statuses before using numerical results.
    pub dev_info: DeviceOutputView<'a, i32>,
}

/// Initialized generalized matrices with pending eigenvalues, scratch, and status.
/// B must be positive definite; its selected triangle is overwritten with a factor.
#[derive(Debug)]
pub struct SygvBindings<'a, T: SolverElement> {
    /// Initialized input matrix overwritten by the eigensolver.
    /// In vector mode, the leading result columns contain eigenvectors.
    pub a: MatrixMut<'a, T>,
    /// Initialized positive-definite generalized-problem matrix; the selected triangle is overwritten with its Cholesky factor.
    pub b: MatrixMut<'a, T>,
    /// Real eigenvalue output capacity for `n` values per matrix.
    /// Selected solves produce only the returned prefix.
    pub eigenvalues: DeviceOutputView<'a, T::Real>,
    /// Exclusive device scratch capacity measured in elements of `T`.
    pub workspace: DeviceOutputView<'a, T>,
    /// Device output capacity for one solver status.
    /// Inspect completed statuses before using numerical results.
    pub dev_info: DeviceOutputView<'a, i32>,
}

/// General eigenvalues use complex storage matching the input precision.
///
/// Right eigenvectors use the native input type.
/// For real A, conjugate pairs are encoded by adjacent real and imaginary columns: `v[j] + i*v[j+1]` and `v[j] - i*v[j+1]`.
/// For complex A, each column is a complex eigenvector.
/// Matrix padding and unused allocation tails remain outside the produced range.
#[derive(Debug)]
pub struct GeevBindings<'a, T: SolverElement> {
    /// Initialized input matrix overwritten by the eigensolver.
    pub a: MatrixMut<'a, T>,
    /// Device output capacity for `n` complex eigenvalues matching the input precision.
    pub eigenvalues: DeviceOutputView<'a, T::Complex>,
    /// Optional output for `n` right eigenvectors; `None` disables vector computation.
    ///
    /// For real inputs, adjacent real columns encode complex-conjugate pairs as described on [`GeevBindings`].
    pub right_vectors: Option<MatrixOutput<'a, T>>,
    /// Exclusive device and host scratch capacities, both measured in bytes.
    pub workspace: ByteWorkspaceMut<'a>,
    /// Device output capacity for one solver status.
    /// Inspect completed statuses before using numerical results.
    pub dev_info: DeviceOutputView<'a, i32>,
}

/// Symmetric or Hermitian eigendecomposition using the native 64-bit interface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Syevd {
    /// Whether to compute eigenvalues alone or also eigenvectors.
    pub mode: EigenMode,
    /// Input triangle containing the symmetric or Hermitian matrix data.
    pub fill_mode: FillMode,
    /// Order of each square input matrix.
    pub n: usize,
}

impl Syevd {
    /// Records settings for a symmetric or Hermitian eigendecomposition without validating dimensions or selection bounds.
    #[must_use]
    pub fn new(mode: EigenMode, fill_mode: FillMode, n: usize) -> Self {
        Self { mode, fill_mode, n }
    }

    /// Queries workspace capacity without allocating matrices or outputs.
    ///
    /// Returns device and host capacities in bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a matrix dimension or leading dimension is invalid, checked storage-size arithmetic or a native integer conversion fails, or cuSOLVER rejects the configuration or cannot report workspace requirements.
    pub fn workspace_size<T: SolverElement>(
        &self,
        context: &Context,
        params: &Params,
        a: MatrixLayout,
    ) -> Result<WorkspaceSizes> {
        xsyevd_workspace_size::<T>(context, params, self.mode, self.fill_mode, self.n, a)
    }

    /// Enqueues this operation on the context's stream without waiting for it; [`Self::execute`] is the waiting form.
    ///
    /// # Safety
    ///
    /// Follow the [raw execution contract](crate#raw-execution).
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA binding or dependency ordering fails, a buffer belongs to another CUDA context, matrix, output, or status capacity is invalid, workspace is insufficient, checked arithmetic or native conversions fail, or cuSOLVER rejects or fails to enqueue the operation.
    pub unsafe fn execute_async<T: SolverElement>(
        &self,
        context: &Context,
        bindings: SyevdBindings<'_, T>,
        params: &Params,
    ) -> Result<()> {
        // SAFETY: this method's caller accepted the raw execution contract for `bindings`; the callee checks their capacities and contexts and orders the operation after their producers.
        unsafe {
            xsyevd_async::<T>(
                context,
                params,
                self.mode,
                self.fill_mode,
                self.n,
                bindings.a,
                bindings.eigenvalues,
                bindings.workspace,
                bindings.dev_info,
            )
        }
    }

    /// Completes borrowed matrix, output, workspace, and parameter use.
    /// Inspect `dev_info` before promoting pending result storage.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active or invalidated, CUDA binding, dependency ordering, or completion fails, a buffer belongs to another CUDA context, matrix, output, or status capacity is invalid, workspace is insufficient, checked arithmetic or native conversions fail, or cuSOLVER rejects or fails to enqueue the operation.
    pub fn execute<T: SolverElement>(
        &self,
        context: &Context,
        bindings: SyevdBindings<'_, T>,
        params: &Params,
    ) -> Result<()> {
        // SAFETY: `sync_scope` rejects capture and waits for the work before the borrowed bindings are released, on every path.
        context.sync_scope(|| unsafe { self.execute_async(context, bindings, params) })
    }
}

/// Packed batched symmetric or Hermitian eigendecomposition with 64-bit metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SyevBatched {
    /// Whether to compute eigenvalues alone or also eigenvectors.
    pub mode: EigenMode,
    /// Input triangle containing the symmetric or Hermitian matrix data.
    pub fill_mode: FillMode,
    /// Order of each square input matrix.
    pub n: usize,
    /// Number of packed matrices in the batch.
    pub batch_count: usize,
}

impl SyevBatched {
    /// Records settings for packed batched symmetric or Hermitian eigendecomposition without validating dimensions or selection bounds.
    #[must_use]
    pub fn new(mode: EigenMode, fill_mode: FillMode, n: usize, batch_count: usize) -> Self {
        Self {
            mode,
            fill_mode,
            n,
            batch_count,
        }
    }

    /// Queries workspace capacity without allocating matrices or outputs.
    ///
    /// Returns device and host capacities in bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a matrix dimension or leading dimension is invalid, checked storage-size arithmetic or a native integer conversion fails, or cuSOLVER rejects the configuration or cannot report workspace requirements.
    /// The batch count must be nonzero and fit the native limits.
    pub fn workspace_size<T: SolverElement>(
        &self,
        context: &Context,
        params: &Params,
        a: MatrixLayout,
    ) -> Result<WorkspaceSizes> {
        xsyev_batched_workspace_size::<T>(
            context,
            params,
            self.mode,
            self.fill_mode,
            self.n,
            a,
            self.batch_count,
        )
    }

    /// Enqueues this operation on the context's stream without waiting for it; [`Self::execute`] is the waiting form.
    ///
    /// # Safety
    ///
    /// Follow the [raw execution contract](crate#raw-execution).
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA binding or dependency ordering fails, a buffer belongs to another CUDA context, matrix, output, or status capacity is invalid, workspace is insufficient, checked arithmetic or native conversions fail, or cuSOLVER rejects or fails to enqueue the operation.
    pub unsafe fn execute_async<T: SolverElement>(
        &self,
        context: &Context,
        bindings: SyevdBindings<'_, T>,
        params: &Params,
    ) -> Result<()> {
        // SAFETY: this method's caller accepted the raw execution contract for `bindings`; the callee checks their capacities and contexts and orders the operation after their producers.
        unsafe {
            xsyev_batched_async::<T>(
                context,
                params,
                self.mode,
                self.fill_mode,
                self.n,
                bindings.a,
                bindings.eigenvalues,
                bindings.workspace,
                bindings.dev_info,
                self.batch_count,
            )
        }
    }

    /// Completes borrowed matrix, output, workspace, and parameter use.
    /// Inspect `dev_info` before promoting pending result storage.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active or invalidated, CUDA binding, dependency ordering, or completion fails, a buffer belongs to another CUDA context, matrix, output, or status capacity is invalid, workspace is insufficient, checked arithmetic or native conversions fail, or cuSOLVER rejects or fails to enqueue the operation.
    pub fn execute<T: SolverElement>(
        &self,
        context: &Context,
        bindings: SyevdBindings<'_, T>,
        params: &Params,
    ) -> Result<()> {
        // SAFETY: `sync_scope` rejects capture and waits for the work before the borrowed bindings are released, on every path.
        context.sync_scope(|| unsafe { self.execute_async(context, bindings, params) })
    }
}

/// Selected symmetric eigenvalues and optional vectors using the native 64-bit interface.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Syevdx<R: SolverReal> {
    /// Whether to compute eigenvalues alone or also eigenvectors.
    pub mode: EigenMode,
    /// Input triangle containing the symmetric or Hermitian matrix data.
    pub fill_mode: FillMode,
    /// Subset of the ascending spectrum to compute; validated during workspace queries and execution.
    pub selection: EigenSelection<R>,
    /// Order of each square input matrix.
    pub n: usize,
}

impl<R: SolverReal> Syevdx<R> {
    /// Records settings for a selected symmetric or Hermitian eigendecomposition without validating dimensions or selection bounds.
    pub fn new(
        mode: EigenMode,
        fill_mode: FillMode,
        selection: EigenSelection<R>,
        n: usize,
    ) -> Self {
        Self {
            mode,
            fill_mode,
            selection,
            n,
        }
    }

    /// Queries workspace capacity without allocating matrices or outputs.
    ///
    /// Returns device and host capacities in bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a matrix dimension or leading dimension is invalid, checked storage-size arithmetic or a native integer conversion fails, or cuSOLVER rejects the configuration or cannot report workspace requirements.
    /// Also returns an error if the selected value or index interval is invalid.
    pub fn workspace_size<T: SolverElement<Real = R>>(
        &self,
        context: &Context,
        params: &Params,
        a: MatrixLayout,
    ) -> Result<WorkspaceSizes> {
        xsyevdx_workspace_size::<T>(
            context,
            params,
            self.mode,
            self.fill_mode,
            &self.selection,
            self.n,
            a,
        )
    }

    /// Completes borrowed matrix, output, workspace, and parameter use.
    /// Returns the execution count after completion and a zero device status.
    /// Eigenvalue storage must have n elements; only the returned prefix is produced.
    /// In vector mode, only that many leading columns of A contain eigenvectors.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active or invalidated, CUDA binding, dependency ordering, or completion fails, a buffer belongs to another CUDA context, matrix, output, or status capacity is invalid, workspace is insufficient, checked arithmetic or native conversions fail, or cuSOLVER rejects or fails to enqueue the operation.
    /// Also returns an error if the selection is invalid, completed device status is nonzero, status readback fails, or the returned eigenvalue count is invalid.
    pub fn execute<T: SolverElement<Real = R>>(
        &self,
        context: &Context,
        mut bindings: SyevdBindings<'_, T>,
        params: &Params,
    ) -> Result<usize> {
        let mut count = MaybeUninit::uninit();
        context.sync_scope(|| {
            // SAFETY: `sync_scope` rejects capture and waits for completion on every path, so the bindings and `count` outlive the queued work.
            unsafe {
                xsyevdx_async::<T>(
                    context,
                    params,
                    self.mode,
                    self.fill_mode,
                    &self.selection,
                    self.n,
                    bindings.a,
                    bindings.eigenvalues,
                    bindings.workspace,
                    bindings.dev_info.reborrow(),
                    &mut count,
                )
            }
        })?;
        check_completed_status("syevdx", &bindings.dev_info)?;
        // SAFETY: `check_completed_status` saw a zero status after completion, and a successful solve writes `count`.
        selected_count(unsafe { count.assume_init() }, self.n)
    }
}

/// Symmetric or Hermitian Jacobi eigendecomposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Syevj {
    /// Whether to compute eigenvalues alone or also eigenvectors.
    pub mode: EigenMode,
    /// Input triangle containing the symmetric or Hermitian matrix data.
    pub fill_mode: FillMode,
    /// Order of each square input matrix.
    pub n: usize,
}

impl Syevj {
    /// Records settings for a Jacobi eigendecomposition without validating dimensions or selection bounds.
    #[must_use]
    pub fn new(mode: EigenMode, fill_mode: FillMode, n: usize) -> Self {
        Self { mode, fill_mode, n }
    }

    /// Queries workspace capacity without allocating matrices or outputs.
    ///
    /// Returns device scratch capacity in elements of `T`.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a matrix dimension or leading dimension is invalid, checked storage-size arithmetic or a native integer conversion fails, or cuSOLVER rejects the configuration or cannot report workspace requirements.
    pub fn workspace_len<T: SolverElement>(
        &self,
        context: &Context,
        a: MatrixLayout,
        params: &SyevjInfo,
    ) -> Result<usize> {
        (T::JACOBI_QUERY)(context, self.mode, self.fill_mode, self.n, a, params)
    }

    /// Enqueues this operation on the context's stream without waiting for it; [`Self::execute`] is the waiting form.
    ///
    /// # Safety
    ///
    /// Follow the [raw execution contract](crate#raw-execution).
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA binding or dependency ordering fails, a buffer belongs to another CUDA context, matrix, output, or status capacity is invalid, workspace is insufficient, checked arithmetic or native conversions fail, or cuSOLVER rejects or fails to enqueue the operation.
    pub unsafe fn execute_async<T: SolverElement>(
        &self,
        context: &Context,
        bindings: SyevjBindings<'_, T>,
        params: &mut SyevjInfo,
    ) -> Result<()> {
        // SAFETY: this method's caller accepted the raw execution contract for `bindings`; the callee checks their capacities and contexts and orders the operation after their producers.
        unsafe {
            (T::JACOBI)(
                context,
                self.mode,
                self.fill_mode,
                self.n,
                bindings.a,
                bindings.eigenvalues,
                bindings.workspace,
                bindings.dev_info,
                params,
            )
        }
    }

    /// Completes borrowed matrix, output, workspace, and parameter use.
    /// Inspect `dev_info` before promoting pending result storage.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active or invalidated, CUDA binding, dependency ordering, or completion fails, a buffer belongs to another CUDA context, matrix, output, or status capacity is invalid, workspace is insufficient, checked arithmetic or native conversions fail, or cuSOLVER rejects or fails to enqueue the operation.
    pub fn execute<T: SolverElement>(
        &self,
        context: &Context,
        bindings: SyevjBindings<'_, T>,
        params: &mut SyevjInfo,
    ) -> Result<()> {
        // SAFETY: `sync_scope` rejects capture and waits for the work before the borrowed bindings are released, on every path.
        context.sync_scope(|| unsafe { self.execute_async(context, bindings, params) })
    }
}

/// Packed batched Jacobi eigendecomposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SyevjBatched {
    /// Whether to compute eigenvalues alone or also eigenvectors.
    pub mode: EigenMode,
    /// Input triangle containing the symmetric or Hermitian matrix data.
    pub fill_mode: FillMode,
    /// Order of each square input matrix.
    pub n: usize,
    /// Number of packed matrices in the batch.
    pub batch_count: usize,
}

impl SyevjBatched {
    /// Records settings for packed batched Jacobi eigendecomposition without validating dimensions or selection bounds.
    #[must_use]
    pub fn new(mode: EigenMode, fill_mode: FillMode, n: usize, batch_count: usize) -> Self {
        Self {
            mode,
            fill_mode,
            n,
            batch_count,
        }
    }

    /// Queries workspace capacity without allocating matrices or outputs.
    ///
    /// Returns device scratch capacity in elements of `T`.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a matrix dimension or leading dimension is invalid, checked storage-size arithmetic or a native integer conversion fails, or cuSOLVER rejects the configuration or cannot report workspace requirements.
    /// The batch count must be nonzero and fit the native limits.
    pub fn workspace_len<T: SolverElement>(
        &self,
        context: &Context,
        a: MatrixLayout,
        params: &SyevjInfo,
    ) -> Result<usize> {
        (T::BATCHED_QUERY)(
            context,
            self.mode,
            self.fill_mode,
            self.n,
            a,
            params,
            self.batch_count,
        )
    }

    /// Enqueues this operation on the context's stream without waiting for it; [`Self::execute`] is the waiting form.
    ///
    /// # Safety
    ///
    /// Follow the [raw execution contract](crate#raw-execution).
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA binding or dependency ordering fails, a buffer belongs to another CUDA context, matrix, output, or status capacity is invalid, workspace is insufficient, checked arithmetic or native conversions fail, or cuSOLVER rejects or fails to enqueue the operation.
    pub unsafe fn execute_async<T: SolverElement>(
        &self,
        context: &Context,
        bindings: SyevjBindings<'_, T>,
        params: &mut SyevjInfo,
    ) -> Result<()> {
        // SAFETY: this method's caller accepted the raw execution contract for `bindings`; the callee checks their capacities and contexts and orders the operation after their producers.
        unsafe {
            (T::BATCHED)(
                context,
                self.mode,
                self.fill_mode,
                self.n,
                bindings.a,
                bindings.eigenvalues,
                bindings.workspace,
                bindings.dev_info,
                params,
                self.batch_count,
            )
        }
    }

    /// Completes borrowed matrix, output, workspace, and parameter use.
    /// Inspect `dev_info` before promoting pending result storage.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active or invalidated, CUDA binding, dependency ordering, or completion fails, a buffer belongs to another CUDA context, matrix, output, or status capacity is invalid, workspace is insufficient, checked arithmetic or native conversions fail, or cuSOLVER rejects or fails to enqueue the operation.
    pub fn execute<T: SolverElement>(
        &self,
        context: &Context,
        bindings: SyevjBindings<'_, T>,
        params: &mut SyevjInfo,
    ) -> Result<()> {
        // SAFETY: `sync_scope` rejects capture and waits for the work before the borrowed bindings are released, on every path.
        context.sync_scope(|| unsafe { self.execute_async(context, bindings, params) })
    }
}

/// Generalized symmetric-definite Jacobi eigendecomposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sygvj {
    /// Generalized eigenvalue equation relating the two input matrices.
    pub eig_type: EigenType,
    /// Whether to compute eigenvalues alone or also eigenvectors.
    pub mode: EigenMode,
    /// Input triangle containing the symmetric or Hermitian matrix data.
    pub fill_mode: FillMode,
    /// Order of each square input matrix.
    pub n: usize,
}

impl Sygvj {
    /// Records settings for a generalized Jacobi eigendecomposition without validating dimensions or selection bounds.
    #[must_use]
    pub fn new(eig_type: EigenType, mode: EigenMode, fill_mode: FillMode, n: usize) -> Self {
        Self {
            eig_type,
            mode,
            fill_mode,
            n,
        }
    }

    /// Queries workspace capacity without allocating matrices or outputs.
    ///
    /// Returns device scratch capacity in elements of `T`.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a matrix dimension or leading dimension is invalid, checked storage-size arithmetic or a native integer conversion fails, or cuSOLVER rejects the configuration or cannot report workspace requirements.
    pub fn workspace_len<T: SolverElement>(
        &self,
        context: &Context,
        a: MatrixLayout,
        b: MatrixLayout,
        params: &SyevjInfo,
    ) -> Result<usize> {
        (T::GENERALIZED_JACOBI_QUERY)(
            context,
            self.eig_type,
            self.mode,
            self.fill_mode,
            self.n,
            a,
            b,
            params,
        )
    }

    /// Enqueues this operation on the context's stream without waiting for it; [`Self::execute`] is the waiting form.
    ///
    /// # Safety
    ///
    /// Follow the [raw execution contract](crate#raw-execution).
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA binding or dependency ordering fails, a buffer belongs to another CUDA context, matrix, output, or status capacity is invalid, workspace is insufficient, checked arithmetic or native conversions fail, or cuSOLVER rejects or fails to enqueue the operation.
    pub unsafe fn execute_async<T: SolverElement>(
        &self,
        context: &Context,
        bindings: SygvBindings<'_, T>,
        params: &mut SyevjInfo,
    ) -> Result<()> {
        // SAFETY: this method's caller accepted the raw execution contract for `bindings`; the callee checks their capacities and contexts and orders the operation after their producers.
        unsafe {
            (T::GENERALIZED_JACOBI)(
                context,
                self.eig_type,
                self.mode,
                self.fill_mode,
                self.n,
                bindings.a,
                bindings.b,
                bindings.eigenvalues,
                bindings.workspace,
                bindings.dev_info,
                params,
            )
        }
    }

    /// Completes borrowed matrix, output, workspace, and parameter use.
    /// Inspect `dev_info` before promoting pending result storage.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active or invalidated, CUDA binding, dependency ordering, or completion fails, a buffer belongs to another CUDA context, matrix, output, or status capacity is invalid, workspace is insufficient, checked arithmetic or native conversions fail, or cuSOLVER rejects or fails to enqueue the operation.
    pub fn execute<T: SolverElement>(
        &self,
        context: &Context,
        bindings: SygvBindings<'_, T>,
        params: &mut SyevjInfo,
    ) -> Result<()> {
        // SAFETY: `sync_scope` rejects capture and waits for the work before the borrowed bindings are released, on every path.
        context.sync_scope(|| unsafe { self.execute_async(context, bindings, params) })
    }
}

/// Generalized symmetric-definite divide-and-conquer eigendecomposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sygvd {
    /// Generalized eigenvalue equation relating the two input matrices.
    pub eig_type: EigenType,
    /// Whether to compute eigenvalues alone or also eigenvectors.
    pub mode: EigenMode,
    /// Input triangle containing the symmetric or Hermitian matrix data.
    pub fill_mode: FillMode,
    /// Order of each square input matrix.
    pub n: usize,
}

impl Sygvd {
    /// Records settings for a generalized divide-and-conquer eigendecomposition without validating dimensions or selection bounds.
    #[must_use]
    pub fn new(eig_type: EigenType, mode: EigenMode, fill_mode: FillMode, n: usize) -> Self {
        Self {
            eig_type,
            mode,
            fill_mode,
            n,
        }
    }

    /// Queries workspace capacity without allocating matrices or outputs.
    ///
    /// Returns device scratch capacity in elements of `T`.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a matrix dimension or leading dimension is invalid, checked storage-size arithmetic or a native integer conversion fails, or cuSOLVER rejects the configuration or cannot report workspace requirements.
    pub fn workspace_len<T: SolverElement>(
        &self,
        context: &Context,
        a: MatrixLayout,
        b: MatrixLayout,
    ) -> Result<usize> {
        (T::GENERALIZED_DIVIDE_QUERY)(
            context,
            self.eig_type,
            self.mode,
            self.fill_mode,
            self.n,
            a,
            b,
        )
    }

    /// Enqueues this operation on the context's stream without waiting for it; [`Self::execute`] is the waiting form.
    ///
    /// # Safety
    ///
    /// Follow the [raw execution contract](crate#raw-execution).
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA binding or dependency ordering fails, a buffer belongs to another CUDA context, matrix, output, or status capacity is invalid, workspace is insufficient, checked arithmetic or native conversions fail, or cuSOLVER rejects or fails to enqueue the operation.
    pub unsafe fn execute_async<T: SolverElement>(
        &self,
        context: &Context,
        bindings: SygvBindings<'_, T>,
    ) -> Result<()> {
        // SAFETY: this method's caller accepted the raw execution contract for `bindings`; the callee checks their capacities and contexts and orders the operation after their producers.
        unsafe {
            (T::GENERALIZED_DIVIDE)(
                context,
                self.eig_type,
                self.mode,
                self.fill_mode,
                self.n,
                bindings.a,
                bindings.b,
                bindings.eigenvalues,
                bindings.workspace,
                bindings.dev_info,
            )
        }
    }

    /// Completes borrowed matrix, output, workspace, and parameter use.
    /// Inspect `dev_info` before promoting pending result storage.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active or invalidated, CUDA binding, dependency ordering, or completion fails, a buffer belongs to another CUDA context, matrix, output, or status capacity is invalid, workspace is insufficient, checked arithmetic or native conversions fail, or cuSOLVER rejects or fails to enqueue the operation.
    pub fn execute<T: SolverElement>(
        &self,
        context: &Context,
        bindings: SygvBindings<'_, T>,
    ) -> Result<()> {
        // SAFETY: `sync_scope` rejects capture and waits for the work before the borrowed bindings are released, on every path.
        context.sync_scope(|| unsafe { self.execute_async(context, bindings) })
    }
}

/// Selected generalized symmetric-definite eigenvalues and optional vectors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sygvdx<R: SolverReal> {
    /// Generalized eigenvalue equation relating the two input matrices.
    pub eig_type: EigenType,
    /// Whether to compute eigenvalues alone or also eigenvectors.
    pub mode: EigenMode,
    /// Input triangle containing the symmetric or Hermitian matrix data.
    pub fill_mode: FillMode,
    /// Subset of the ascending spectrum to compute; validated during workspace queries and execution.
    pub selection: EigenSelection<R>,
    /// Order of each square input matrix.
    pub n: usize,
}

impl<R: SolverReal> Sygvdx<R> {
    /// Records settings for a selected generalized eigendecomposition without validating dimensions or selection bounds.
    pub fn new(
        eig_type: EigenType,
        mode: EigenMode,
        fill_mode: FillMode,
        selection: EigenSelection<R>,
        n: usize,
    ) -> Self {
        Self {
            eig_type,
            mode,
            fill_mode,
            selection,
            n,
        }
    }

    /// Queries workspace capacity without allocating matrices or outputs.
    ///
    /// Returns device scratch capacity in elements of `T`.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a matrix dimension or leading dimension is invalid, checked storage-size arithmetic or a native integer conversion fails, or cuSOLVER rejects the configuration or cannot report workspace requirements.
    /// Also returns an error if the selected value or index interval is invalid.
    pub fn workspace_len<T: SolverElement<Real = R>>(
        &self,
        context: &Context,
        a: MatrixLayout,
        b: MatrixLayout,
    ) -> Result<usize> {
        (T::GENERALIZED_SELECTED_QUERY)(
            context,
            self.eig_type,
            self.mode,
            self.fill_mode,
            &self.selection,
            self.n,
            a,
            b,
        )
    }

    /// Completes borrowed matrix, output, workspace, and parameter use.
    /// Returns the execution count after completion and a zero device status.
    /// Eigenvalue storage must have n elements; only the returned prefix is produced.
    /// In vector mode, only that many leading columns of A contain eigenvectors.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active or invalidated, CUDA binding, dependency ordering, or completion fails, a buffer belongs to another CUDA context, matrix, output, or status capacity is invalid, workspace is insufficient, checked arithmetic or native conversions fail, or cuSOLVER rejects or fails to enqueue the operation.
    /// Also returns an error if the selection is invalid, completed device status is nonzero, status readback fails, or the returned eigenvalue count is invalid.
    pub fn execute<T: SolverElement<Real = R>>(
        &self,
        context: &Context,
        mut bindings: SygvBindings<'_, T>,
    ) -> Result<usize> {
        let mut count = MaybeUninit::uninit();
        context.sync_scope(|| {
            // SAFETY: `sync_scope` rejects capture and waits for completion on every path, so the bindings and `count` outlive the queued work.
            unsafe {
                (T::GENERALIZED_SELECTED)(
                    context,
                    self.eig_type,
                    self.mode,
                    self.fill_mode,
                    &self.selection,
                    self.n,
                    bindings.a,
                    bindings.b,
                    bindings.eigenvalues,
                    bindings.workspace,
                    bindings.dev_info.reborrow(),
                    &mut count,
                )
            }
        })?;
        check_completed_status("sygvdx", &bindings.dev_info)?;
        // SAFETY: `check_completed_status` saw a zero status after completion, and a successful solve writes `count`.
        selected_count(unsafe { count.assume_init() }, self.n)
    }
}

/// General nonsymmetric eigenvalues and optional native right eigenvectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Geev {
    /// Order of each square input matrix.
    pub n: usize,
}

impl Geev {
    /// Records settings for a general nonsymmetric eigendecomposition without validating dimensions or selection bounds.
    #[must_use]
    pub fn new(n: usize) -> Self {
        Self { n }
    }

    /// Queries workspace capacity without allocating matrices or outputs.
    ///
    /// Returns device and host capacities in bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a matrix dimension or leading dimension is invalid, checked storage-size arithmetic or a native integer conversion fails, or cuSOLVER rejects the configuration or cannot report workspace requirements.
    pub fn workspace_size<T: SolverElement>(
        &self,
        context: &Context,
        params: &Params,
        a: MatrixLayout,
        right_vectors: Option<MatrixLayout>,
    ) -> Result<WorkspaceSizes> {
        (T::GENERAL_QUERY)(context, params, self.n, a, right_vectors)
    }

    /// Enqueues this operation on the context's stream without waiting for it; [`Self::execute`] is the waiting form.
    ///
    /// # Safety
    ///
    /// Follow the [raw execution contract](crate#raw-execution).
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA binding or dependency ordering fails, a buffer belongs to another CUDA context, matrix, output, or status capacity is invalid, workspace is insufficient, checked arithmetic or native conversions fail, or cuSOLVER rejects or fails to enqueue the operation.
    pub unsafe fn execute_async<T: SolverElement>(
        &self,
        context: &Context,
        bindings: GeevBindings<'_, T>,
        params: &Params,
    ) -> Result<()> {
        // SAFETY: this method's caller accepted the raw execution contract for `bindings`; the callee checks their capacities and contexts and orders the operation after their producers.
        unsafe {
            (T::GENERAL)(
                context,
                params,
                self.n,
                bindings.a,
                bindings.right_vectors,
                bindings.eigenvalues,
                bindings.workspace,
                bindings.dev_info,
            )
        }
    }

    /// Completes borrowed matrix, output, workspace, and parameter use.
    /// Inspect `dev_info` before promoting pending result storage.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active or invalidated, CUDA binding, dependency ordering, or completion fails, a buffer belongs to another CUDA context, matrix, output, or status capacity is invalid, workspace is insufficient, checked arithmetic or native conversions fail, or cuSOLVER rejects or fails to enqueue the operation.
    pub fn execute<T: SolverElement>(
        &self,
        context: &Context,
        bindings: GeevBindings<'_, T>,
        params: &Params,
    ) -> Result<()> {
        // SAFETY: `sync_scope` rejects capture and waits for the work before the borrowed bindings are released, on every path.
        context.sync_scope(|| unsafe { self.execute_async(context, bindings, params) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{eigen::validation::native_selection, types::EigenRange};

    #[test]
    fn test_eigen_selection_all_maps_cleanly() {
        let (range, lower, upper, first, last) =
            native_selection::<f32>(&EigenSelection::All, 8).unwrap();
        assert_eq!(range, EigenRange::All);
        assert_eq!((*lower, *upper, first, last), (0.0, 0.0, 0, 0));
    }

    #[test]
    fn test_eigen_selection_index_maps_cleanly() {
        let (range, _, _, first, last) =
            native_selection::<f64>(&EigenSelection::ByIndex(2..5), 8).unwrap();
        assert_eq!(range, EigenRange::Index);
        assert_eq!((first, last), (3, 5));
    }
}
