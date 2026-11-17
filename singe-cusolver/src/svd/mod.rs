//! Singular value decompositions and singular-vector output modes.

mod generic;
mod info;
mod legacy;
mod raw;
pub(crate) mod scalar;
mod validation;

use std::mem::MaybeUninit;

use singe_cuda::{memory::DeviceMemory, view::DeviceOutputView};

pub use self::{
    generic::{
        SvdLayout, xgesvd_async, xgesvd_workspace_size, xgesvdp_async, xgesvdp_workspace_size,
        xgesvdr_async, xgesvdr_workspace_size,
    },
    info::GesvdjInfo,
    legacy::StridedSvdLayout,
};
use crate::{
    context::Context,
    data_type::SolverElement,
    error::{Error, Result},
    layout::{
        ByteWorkspaceMut, MatrixMut, MatrixOutput, StridedBatchedMatrixOutput,
        StridedBatchedMatrixRef, StridedBatchedVectorOutput, StridedMatrixLayout, WorkspaceSizes,
    },
    params::Params,
    types::{EigenMode, SvdMode, SvdVectorColumns, TruncatedSvdMode},
};

/// Classic singular value decomposition with independently selected left and right vector modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Gesvd {
    /// Requested left singular-vector output mode.
    pub job_u: SvdMode,
    /// Requested output mode for the conjugate transpose of the right singular-vector matrix.
    pub job_vt: SvdMode,
    /// Number of rows in each input matrix.
    pub rows: usize,
    /// Number of columns in each input matrix.
    pub columns: usize,
}

impl Gesvd {
    /// Records SVD dimensions and output modes without validating native limits.
    #[must_use]
    pub const fn new(job_u: SvdMode, job_vt: SvdMode, rows: usize, columns: usize) -> Self {
        Self {
            job_u,
            job_vt,
            rows,
            columns,
        }
    }

    /// Queries scratch sizes using layout metadata before allocating matrices.
    ///
    /// Returns device and host scratch capacities in bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions, requested output layouts, or singular-vector modes are invalid, checked layout arithmetic or a native integer conversion fails, CUDA context binding fails, or cuSOLVER rejects the settings or cannot report workspace requirements.
    pub fn workspace_size<T: SolverElement>(
        self,
        context: &Context,
        params: &Params,
        layout: SvdLayout,
    ) -> Result<WorkspaceSizes> {
        xgesvd_workspace_size::<T>(
            context,
            params,
            self.job_u,
            self.job_vt,
            self.rows,
            self.columns,
            layout,
        )
    }

    /// Enqueues this SVD on the context's stream without waiting for it; [`Self::execute`] is the waiting form.
    ///
    /// # Safety
    ///
    /// Follow the [raw execution contract](crate#raw-execution) and [`xgesvd_async`]'s produced-range contract.
    ///
    /// # Errors
    ///
    /// Returns an error if a buffer belongs to another CUDA context, dimensions, modes, output or status capacities, or workspace sizes are invalid, checked arithmetic or native integer conversion fails, CUDA binding fails, or cuSOLVER rejects or fails to enqueue the operation.
    pub unsafe fn execute_async<T: SolverElement>(
        self,
        context: &Context,
        mut bindings: GesvdBindings<'_, T>,
        params: &Params,
    ) -> Result<()> {
        // SAFETY: this method's caller accepted the raw execution contract for `bindings`; the callee checks their capacities and contexts and orders the operation after their producers.
        unsafe {
            xgesvd_async(
                context,
                params,
                self.job_u,
                self.job_vt,
                self.rows,
                self.columns,
                bindings.a,
                &mut bindings.singular_values,
                bindings.left_vectors,
                bindings.right_vectors_transposed,
                bindings.workspace,
                &mut bindings.dev_info,
            )
        }
    }

    /// Computes this SVD and waits for completion before returning.
    /// Inspect the produced status before using numerical outputs.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active or invalidated, a buffer belongs to another CUDA context, dimensions, modes, output or status capacities, or workspace sizes are invalid, checked arithmetic or native integer conversion fails, CUDA binding or completion fails, or cuSOLVER rejects or fails to enqueue the operation.
    pub fn execute<T: SolverElement>(
        self,
        context: &Context,
        bindings: GesvdBindings<'_, T>,
        params: &Params,
    ) -> Result<()> {
        // SAFETY: `sync_scope` rejects capture and waits for the work before the borrowed bindings are released, on every path.
        context.sync_scope(|| unsafe { self.execute_async(context, bindings, params) })
    }
}

/// Borrowed input, numerical output capacity, scratch, and status for [`Gesvd::execute`].
///
/// Only logical output ranges become results after successful completion; padding, stride gaps, and scratch remain outside those ranges.
#[derive(Debug)]
pub struct GesvdBindings<'a, T: SolverElement> {
    /// Initialized input matrix, whose logical contents may be overwritten by the decomposition.
    pub a: MatrixMut<'a, T>,
    /// Real output capacity for `min(rows, columns)` singular values per matrix.
    pub singular_values: DeviceOutputView<'a, T::Real>,
    /// Optional output for left singular vectors U; full or economy column counts follow the operation settings.
    pub left_vectors: Option<MatrixOutput<'a, T>>,
    /// Optional output for the conjugate transpose of the right singular-vector matrix, as selected by `job_vt`.
    pub right_vectors_transposed: Option<MatrixOutput<'a, T>>,
    /// Exclusive device and host scratch capacities, both measured in bytes.
    pub workspace: ByteWorkspaceMut<'a>,
    /// Device output capacity for one solver status.
    /// Inspect completed status before using numerical outputs.
    pub dev_info: DeviceOutputView<'a, i32>,
}

/// Polar-decomposition-based SVD with full or economy singular vectors returned as U and V.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Gesvdp {
    /// Whether to compute singular values alone or also singular vectors.
    pub mode: EigenMode,
    /// Whether vector output has full square factors or only `min(rows, columns)` columns.
    pub vector_columns: SvdVectorColumns,
    /// Number of rows in each input matrix.
    pub rows: usize,
    /// Number of columns in each input matrix.
    pub columns: usize,
}

impl Gesvdp {
    /// Records SVD dimensions and output modes without validating native limits.
    #[must_use]
    pub const fn new(
        mode: EigenMode,
        vector_columns: SvdVectorColumns,
        rows: usize,
        columns: usize,
    ) -> Self {
        Self {
            mode,
            vector_columns,
            rows,
            columns,
        }
    }

    /// Queries scratch sizes using layout metadata before allocating matrices.
    ///
    /// Returns device and host scratch capacities in bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions, requested output layouts, or singular-vector modes are invalid, checked layout arithmetic or a native integer conversion fails, CUDA context binding fails, or cuSOLVER rejects the settings or cannot report workspace requirements.
    pub fn workspace_size<T: SolverElement>(
        self,
        context: &Context,
        params: &Params,
        layout: SvdLayout,
    ) -> Result<WorkspaceSizes> {
        xgesvdp_workspace_size::<T>(
            context,
            params,
            self.mode,
            self.vector_columns,
            self.rows,
            self.columns,
            layout,
        )
    }

    /// Enqueues this SVD on the context's stream without waiting for it; [`Self::execute`] is the waiting form.
    ///
    /// # Safety
    ///
    /// Follow the [raw execution contract](crate#raw-execution) and [`xgesvdp_async`]'s produced-range contract.
    ///
    /// # Errors
    ///
    /// Returns an error if a buffer belongs to another CUDA context, dimensions, modes, output or status capacities, or workspace sizes are invalid, checked arithmetic or native integer conversion fails, CUDA binding fails, or cuSOLVER rejects or fails to enqueue the operation.
    pub unsafe fn execute_async<T: SolverElement>(
        self,
        context: &Context,
        mut bindings: GesvdpBindings<'_, T>,
        params: &Params,
    ) -> Result<()> {
        // SAFETY: this method's caller accepted the raw execution contract for `bindings`; the callee checks their capacities and contexts and orders the operation after their producers.
        unsafe {
            xgesvdp_async(
                context,
                params,
                self.mode,
                self.vector_columns,
                self.rows,
                self.columns,
                bindings.a,
                &mut bindings.singular_values,
                bindings.left_vectors,
                bindings.right_vectors,
                bindings.workspace,
                &mut bindings.dev_info,
                bindings.residual,
            )
        }
    }

    /// Computes this SVD and waits for completion before returning.
    /// Inspect the produced status before using numerical outputs.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active or invalidated, a buffer belongs to another CUDA context, dimensions, modes, output or status capacities, or workspace sizes are invalid, checked arithmetic or native integer conversion fails, CUDA binding or completion fails, or cuSOLVER rejects or fails to enqueue the operation.
    pub fn execute<T: SolverElement>(
        self,
        context: &Context,
        bindings: GesvdpBindings<'_, T>,
        params: &Params,
    ) -> Result<()> {
        // SAFETY: `sync_scope` rejects capture and waits for the work before the borrowed bindings are released, on every path.
        context.sync_scope(|| unsafe { self.execute_async(context, bindings, params) })
    }
}

/// Borrowed input, numerical output capacity, scratch, and status for [`Gesvdp::execute`].
///
/// Only logical output ranges become results after successful completion; padding, stride gaps, and scratch remain outside those ranges.
#[derive(Debug)]
pub struct GesvdpBindings<'a, T: SolverElement> {
    /// Initialized input matrix, whose logical contents may be overwritten by the decomposition.
    pub a: MatrixMut<'a, T>,
    /// Real output capacity for `min(rows, columns)` singular values per matrix.
    pub singular_values: DeviceOutputView<'a, T::Real>,
    /// Optional output for left singular vectors U; full or economy column counts follow the operation settings.
    pub left_vectors: Option<MatrixOutput<'a, T>>,
    /// Optional output for right singular vectors V, without conjugate transposition; full or economy column counts follow the operation settings.
    pub right_vectors: Option<MatrixOutput<'a, T>>,
    /// Exclusive device and host scratch capacities, both measured in bytes.
    pub workspace: ByteWorkspaceMut<'a>,
    /// Device output capacity for one solver status.
    /// Inspect completed status before using numerical outputs.
    pub dev_info: DeviceOutputView<'a, i32>,
    /// Optional host output for the native singular-value perturbation magnitude, indicating SVD accuracy.
    /// Retain the destination through completion.
    pub residual: Option<&'a mut f64>,
}

/// Randomized truncated SVD with configurable target rank, oversampling, and power iterations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Gesvdr {
    /// Whether to compute the truncated left singular vectors.
    pub left_vectors: TruncatedSvdMode,
    /// Whether to compute the truncated right singular vectors.
    pub right_vectors: TruncatedSvdMode,
    /// Number of rows in each input matrix.
    pub rows: usize,
    /// Number of columns in each input matrix.
    pub columns: usize,
    /// Number of leading singular values and requested singular-vector columns to compute.
    pub rank: usize,
    /// Number of additional random sample vectors beyond the target rank.
    pub oversamples: usize,
    /// Number of power iterations used by randomized SVD.
    pub iterations: usize,
}

impl Gesvdr {
    /// Records SVD dimensions and output modes without validating native limits.
    #[must_use]
    pub const fn new(
        left_vectors: TruncatedSvdMode,
        right_vectors: TruncatedSvdMode,
        rows: usize,
        columns: usize,
        rank: usize,
        oversamples: usize,
        iterations: usize,
    ) -> Self {
        Self {
            left_vectors,
            right_vectors,
            rows,
            columns,
            rank,
            oversamples,
            iterations,
        }
    }

    /// Queries scratch sizes using layout metadata before allocating matrices.
    ///
    /// Returns device and host scratch capacities in bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions, requested output layouts, or singular-vector modes are invalid, checked layout arithmetic or a native integer conversion fails, CUDA context binding fails, or cuSOLVER rejects the settings or cannot report workspace requirements.
    /// Target rank must be positive, and rank plus oversampling must be smaller than `min(rows, columns)`.
    pub fn workspace_size<T: SolverElement>(
        self,
        context: &Context,
        params: &Params,
        layout: SvdLayout,
    ) -> Result<WorkspaceSizes> {
        xgesvdr_workspace_size::<T>(
            context,
            params,
            self.left_vectors,
            self.right_vectors,
            self.rows,
            self.columns,
            self.rank,
            self.oversamples,
            self.iterations,
            layout,
        )
    }

    /// Enqueues this SVD on the context's stream without waiting for it; [`Self::execute`] is the waiting form.
    ///
    /// # Safety
    ///
    /// Follow the [raw execution contract](crate#raw-execution) and [`xgesvdr_async`]'s produced-range contract.
    ///
    /// # Errors
    ///
    /// Returns an error if a buffer belongs to another CUDA context, dimensions, modes, output or status capacities, or workspace sizes are invalid, checked arithmetic or native integer conversion fails, CUDA binding fails, or cuSOLVER rejects or fails to enqueue the operation.
    pub unsafe fn execute_async<T: SolverElement>(
        self,
        context: &Context,
        mut bindings: GesvdrBindings<'_, T>,
        params: &Params,
    ) -> Result<()> {
        // SAFETY: this method's caller accepted the raw execution contract for `bindings`; the callee checks their capacities and contexts and orders the operation after their producers.
        unsafe {
            xgesvdr_async(
                context,
                params,
                self.left_vectors,
                self.right_vectors,
                self.rows,
                self.columns,
                self.rank,
                self.oversamples,
                self.iterations,
                bindings.a,
                &mut bindings.singular_values,
                bindings.left_vectors,
                bindings.right_vectors,
                bindings.workspace,
                &mut bindings.dev_info,
            )
        }
    }

    /// Computes this SVD and waits for completion before returning.
    /// Inspect the produced status before using numerical outputs.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active or invalidated, a buffer belongs to another CUDA context, dimensions, modes, output or status capacities, or workspace sizes are invalid, checked arithmetic or native integer conversion fails, CUDA binding or completion fails, or cuSOLVER rejects or fails to enqueue the operation.
    pub fn execute<T: SolverElement>(
        self,
        context: &Context,
        bindings: GesvdrBindings<'_, T>,
        params: &Params,
    ) -> Result<()> {
        // SAFETY: `sync_scope` rejects capture and waits for the work before the borrowed bindings are released, on every path.
        context.sync_scope(|| unsafe { self.execute_async(context, bindings, params) })
    }
}

/// Borrowed input, numerical output capacity, scratch, and status for [`Gesvdr::execute`].
///
/// Only logical output ranges become results after successful completion; padding, stride gaps, and scratch remain outside those ranges.
#[derive(Debug)]
pub struct GesvdrBindings<'a, T: SolverElement> {
    /// Initialized input matrix, whose logical contents may be overwritten by the decomposition.
    pub a: MatrixMut<'a, T>,
    /// Real output capacity for `rank` leading singular values.
    pub singular_values: DeviceOutputView<'a, T::Real>,
    /// Optional output for left singular vectors U; only `rank` columns are produced per matrix.
    pub left_vectors: Option<MatrixOutput<'a, T>>,
    /// Optional output for right singular vectors V, without conjugate transposition; only `rank` columns are produced per matrix.
    pub right_vectors: Option<MatrixOutput<'a, T>>,
    /// Exclusive device and host scratch capacities, both measured in bytes.
    pub workspace: ByteWorkspaceMut<'a>,
    /// Device output capacity for one solver status.
    /// Inspect completed status before using numerical outputs.
    pub dev_info: DeviceOutputView<'a, i32>,
}

/// Jacobi SVD with full or economy singular vectors, returned as U and V.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Gesvdj {
    /// Whether to compute singular values alone or also singular vectors.
    pub mode: EigenMode,
    /// Whether vector output has full square factors or only `min(rows, columns)` columns.
    pub vector_columns: SvdVectorColumns,
    /// Number of rows in each input matrix.
    pub rows: usize,
    /// Number of columns in each input matrix.
    pub columns: usize,
}

impl Gesvdj {
    /// Records SVD dimensions and output modes without validating native limits.
    #[must_use]
    pub const fn new(
        mode: EigenMode,
        vector_columns: SvdVectorColumns,
        rows: usize,
        columns: usize,
    ) -> Self {
        Self {
            mode,
            vector_columns,
            rows,
            columns,
        }
    }

    /// Queries scratch length in T elements before allocating matrices or outputs.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions, requested output layouts, or singular-vector modes are invalid, checked layout arithmetic or a native integer conversion fails, CUDA context binding fails, or cuSOLVER rejects the settings or cannot report workspace requirements.
    pub fn workspace_len<T: SolverElement>(
        self,
        context: &Context,
        layout: SvdLayout,
        params: &GesvdjInfo,
    ) -> Result<usize> {
        (T::SVD_JACOBI_QUERY)(
            context,
            self.mode,
            self.vector_columns.is_economy(),
            self.rows,
            self.columns,
            layout,
            params,
        )
    }

    /// Enqueues this SVD on the context's stream without waiting for it; [`Self::execute`] is the waiting form.
    ///
    /// # Safety
    ///
    /// Follow the [raw execution contract](crate#raw-execution); `params` is also borrowed by the queued work through completion.
    ///
    /// # Errors
    ///
    /// Returns an error if a buffer belongs to another CUDA context, dimensions, modes, output or status capacities, or workspace sizes are invalid, checked arithmetic or native integer conversion fails, CUDA binding fails, or cuSOLVER rejects or fails to enqueue the operation.
    pub unsafe fn execute_async<T: SolverElement>(
        self,
        context: &Context,
        bindings: GesvdjBindings<'_, T>,
        params: &mut GesvdjInfo,
    ) -> Result<()> {
        // SAFETY: this method's caller accepted the raw execution contract for `bindings`; the callee checks their capacities and contexts and orders the operation after their producers.
        unsafe {
            (T::SVD_JACOBI)(
                context,
                self.mode,
                self.vector_columns.is_economy(),
                self.rows,
                self.columns,
                bindings.a,
                bindings.singular_values,
                bindings.left_vectors,
                bindings.right_vectors,
                bindings.workspace,
                bindings.dev_info,
                params,
            )
        }
    }

    /// Computes this SVD and waits for completion before returning.
    /// Check the produced status before promoting results.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions, modes, output or status capacities, or workspace sizes are invalid, checked arithmetic or native integer conversion fails, CUDA binding fails, or cuSOLVER rejects or fails to enqueue the operation.
    /// Also returns an error if stream capture is active or invalidated, a buffer belongs to another CUDA context, or dependency ordering or stream completion fails.
    pub fn execute<T: SolverElement>(
        self,
        context: &Context,
        bindings: GesvdjBindings<'_, T>,
        params: &mut GesvdjInfo,
    ) -> Result<()> {
        // SAFETY: `sync_scope` rejects capture and waits for the work before the borrowed bindings and `params` are released, on every path.
        context.sync_scope(|| unsafe { self.execute_async(context, bindings, params) })
    }
}

/// Initialized input and pending result storage for Jacobi SVD.
#[derive(Debug)]
pub struct GesvdjBindings<'a, T: SolverElement> {
    /// Initialized input matrix, whose logical contents may be overwritten by the decomposition.
    pub a: MatrixMut<'a, T>,
    /// Real output capacity for `min(rows, columns)` singular values per matrix.
    /// Batched Jacobi packs consecutive value prefixes.
    pub singular_values: DeviceOutputView<'a, T::Real>,
    /// Optional output for left singular vectors U; full or economy column counts follow the operation settings.
    pub left_vectors: Option<MatrixOutput<'a, T>>,
    /// Optional output for right singular vectors V, without conjugate transposition; full or economy column counts follow the operation settings.
    pub right_vectors: Option<MatrixOutput<'a, T>>,
    /// Exclusive device scratch capacity measured in elements of `T`.
    pub workspace: DeviceOutputView<'a, T>,
    /// Device output capacity for one solver status per matrix.
    /// Inspect completed status before using numerical outputs.
    pub dev_info: DeviceOutputView<'a, i32>,
}

/// Packed batched Jacobi SVD with full square singular-vector matrices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GesvdjBatched {
    /// Whether to compute singular values alone or also singular vectors.
    pub mode: EigenMode,
    /// Number of rows in each input matrix.
    pub rows: usize,
    /// Number of columns in each input matrix.
    pub columns: usize,
    /// Number of input matrices in the batch.
    pub batch_size: usize,
}

impl GesvdjBatched {
    /// Records SVD dimensions and output modes without validating native limits.
    #[must_use]
    pub const fn new(mode: EigenMode, rows: usize, columns: usize, batch_size: usize) -> Self {
        Self {
            mode,
            rows,
            columns,
            batch_size,
        }
    }

    /// Queries device scratch capacity in elements of `T` using batch layout metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions, requested output layouts, or singular-vector modes are invalid, checked layout arithmetic or a native integer conversion fails, CUDA context binding fails, or cuSOLVER rejects the settings or cannot report workspace requirements.
    /// The batch must be nonempty, with at most 32 rows and 32 columns per matrix.
    pub fn workspace_len<T: SolverElement>(
        self,
        context: &Context,
        layout: SvdLayout,
        params: &GesvdjInfo,
    ) -> Result<usize> {
        (T::SVD_BATCHED_QUERY)(
            context,
            self.mode,
            self.rows,
            self.columns,
            layout,
            params,
            self.batch_size,
        )
    }

    /// Completes production of each compact value prefix, requested vectors, and status.
    /// Values-only execution needs one native vector buffer as temporary storage.
    /// If the required buffer is absent, this method retains full U storage for tall matrices or full V storage for wide matrices through completion.
    /// Its contents are not exposed as initialized results.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions, modes, output or status capacities, or workspace sizes are invalid, checked arithmetic or native integer conversion fails, CUDA binding fails, or cuSOLVER rejects or fails to enqueue the operation.
    /// Also returns an error if stream capture is active or invalidated, a buffer belongs to another CUDA context, or dependency ordering or stream completion fails.
    /// Private temporary allocation can also fail.
    pub fn execute<T: SolverElement>(
        self,
        context: &Context,
        bindings: GesvdjBindings<'_, T>,
        params: &mut GesvdjInfo,
    ) -> Result<()> {
        let mut native_vectors: Option<DeviceMemory<MaybeUninit<T>>> = None;
        context.sync_scope(|| {
            let left_scratch = self.rows >= self.columns;
            let (left, right) = if self.mode == EigenMode::NoVector
                && if left_scratch {
                    bindings.left_vectors.is_none()
                } else {
                    bindings.right_vectors.is_none()
                } {
                // The native batched implementation uses the larger vector buffer even when no vectors are requested.
                // Keep its owner outside the closure so errors and unwinding drain before destruction.
                let side = self.rows.max(self.columns);
                let elements = side
                    .checked_mul(side)
                    .and_then(|size| size.checked_mul(self.batch_size))
                    .ok_or(Error::InvalidMatrixShape)?;
                let storage = native_vectors.insert(DeviceMemory::<T>::create_uninit(
                    context.cuda_context(),
                    elements,
                )?);
                let output = Some(MatrixOutput::new(storage, side));
                if left_scratch {
                    (output, bindings.right_vectors)
                } else {
                    (bindings.left_vectors, output)
                }
            } else {
                (bindings.left_vectors, bindings.right_vectors)
            };
            // SAFETY: the scope waits for completion before releasing every binding, the private vector storage, and the parameter borrow; the callee orders the operation after its operands' producers.
            unsafe {
                (T::SVD_BATCHED)(
                    context,
                    self.mode,
                    self.rows,
                    self.columns,
                    bindings.a,
                    bindings.singular_values,
                    left,
                    right,
                    bindings.workspace,
                    bindings.dev_info,
                    params,
                    self.batch_size,
                )
            }
        })
    }
}

/// Approximate SVD of equally shaped tall matrices with explicit batch strides.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GesvdaStridedBatched {
    /// Whether to compute singular values alone or also singular vectors.
    pub mode: EigenMode,
    /// Number of leading singular values and requested singular-vector columns to compute.
    pub rank: usize,
    /// Number of rows in each input matrix.
    pub rows: usize,
    /// Number of columns in each input matrix.
    pub columns: usize,
    /// Number of input matrices in the batch.
    pub batch_size: usize,
    /// Whether to produce one host residual per matrix; defaults to `false`.
    ///
    /// Values-only execution uses private vector storage when residuals are requested.
    pub compute_residual: bool,
}

impl GesvdaStridedBatched {
    /// Records SVD dimensions and output modes without validating native limits.
    /// Residual output is initially disabled; set [`Self::compute_residual`] to request it.
    #[must_use]
    pub const fn new(
        mode: EigenMode,
        rank: usize,
        rows: usize,
        columns: usize,
        batch_size: usize,
    ) -> Self {
        Self {
            mode,
            rank,
            rows,
            columns,
            batch_size,
            compute_residual: false,
        }
    }

    /// Queries device scratch capacity in elements of `T` using batch layout metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions, requested output layouts, or singular-vector modes are invalid, checked layout arithmetic or a native integer conversion fails, CUDA context binding fails, or cuSOLVER rejects the settings or cannot report workspace requirements.
    /// Matrices must be tall or square, rank must be in `1..=columns`, and batch strides must cover each logical storage footprint.
    pub fn workspace_len<T: SolverElement>(
        self,
        context: &Context,
        mut layout: StridedSvdLayout,
    ) -> Result<usize> {
        let mode = if self.compute_residual {
            EigenMode::Vector
        } else {
            self.mode
        };
        if self.compute_residual && self.mode == EigenMode::NoVector {
            let vectors = |rows: usize| -> Result<StridedMatrixLayout> {
                let stride = rows
                    .checked_mul(self.rank)
                    .ok_or(Error::InvalidMatrixShape)?;
                Ok(StridedMatrixLayout::new(rows, stride))
            };
            if layout.left_vectors.is_none() {
                layout.left_vectors = Some(vectors(self.rows)?);
            }
            if layout.right_vectors.is_none() {
                layout.right_vectors = Some(vectors(self.columns)?);
            }
        }
        (T::SVD_APPROXIMATE_QUERY)(
            context,
            mode,
            self.rank,
            self.rows,
            self.columns,
            layout,
            self.batch_size,
        )
    }

    /// Completes borrowed device and host outputs before returning.
    /// Only rank singular values and rank vector columns are produced per matrix.
    /// Inspect each completed status before promoting its results or residual.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions, modes, output or status capacities, or workspace sizes are invalid, checked arithmetic or native integer conversion fails, CUDA binding fails, or cuSOLVER rejects or fails to enqueue the operation.
    /// Also returns an error if stream capture is active or invalidated, a buffer belongs to another CUDA context, or dependency ordering or stream completion fails.
    /// Private temporary allocation can also fail.
    /// Residual storage must be present exactly when residual computation is enabled.
    pub fn execute<T: SolverElement>(
        self,
        context: &Context,
        bindings: GesvdaStridedBatchedBindings<'_, T>,
    ) -> Result<()> {
        let mut temporary_left: Option<DeviceMemory<MaybeUninit<T>>> = None;
        let mut temporary_right: Option<DeviceMemory<MaybeUninit<T>>> = None;
        context.sync_scope(|| {
            if self.compute_residual != bindings.residual.is_some() {
                return Err(Error::InvalidSvdMode);
            }

            let mode = if self.compute_residual {
                EigenMode::Vector
            } else {
                self.mode
            };
            let private_vectors = self.compute_residual && self.mode == EigenMode::NoVector;
            let left = if private_vectors && bindings.left_vectors.is_none() {
                let stride = self
                    .rows
                    .checked_mul(self.rank)
                    .ok_or(Error::InvalidMatrixShape)?;
                let elements = stride
                    .checked_mul(self.batch_size)
                    .ok_or(Error::InvalidMatrixShape)?;
                let storage = temporary_left.insert(DeviceMemory::<T>::create_uninit(
                    context.cuda_context(),
                    elements,
                )?);
                Some(StridedBatchedMatrixOutput::new(storage, self.rows, stride))
            } else {
                bindings.left_vectors
            };
            let right = if private_vectors && bindings.right_vectors.is_none() {
                let stride = self
                    .columns
                    .checked_mul(self.rank)
                    .ok_or(Error::InvalidMatrixShape)?;
                let elements = stride
                    .checked_mul(self.batch_size)
                    .ok_or(Error::InvalidMatrixShape)?;
                let storage = temporary_right.insert(DeviceMemory::<T>::create_uninit(
                    context.cuda_context(),
                    elements,
                )?);
                Some(StridedBatchedMatrixOutput::new(
                    storage,
                    self.columns,
                    stride,
                ))
            } else {
                bindings.right_vectors
            };
            // SAFETY: the scope waits for completion before releasing the device bindings, private vector storage, and host residuals; the callee orders the operation after its operands' producers.
            unsafe {
                (T::SVD_APPROXIMATE)(
                    context,
                    mode,
                    self.rank,
                    self.rows,
                    self.columns,
                    bindings.a,
                    bindings.singular_values,
                    left,
                    right,
                    bindings.workspace,
                    bindings.dev_info,
                    bindings.residual,
                    self.batch_size,
                )
            }
        })
    }
}

/// Borrowed input, numerical output capacity, scratch, and status for [`GesvdaStridedBatched::execute`].
///
/// Only logical output ranges become results after successful completion; padding, stride gaps, and scratch remain outside those ranges.
#[derive(Debug)]
pub struct GesvdaStridedBatchedBindings<'a, T: SolverElement> {
    /// Initialized strided batched input matrices; each logical matrix is read by the approximate solver.
    pub a: StridedBatchedMatrixRef<'a, T>,
    /// Real output capacity for `rank` singular values per matrix; stride gaps are not produced.
    pub singular_values: StridedBatchedVectorOutput<'a, T::Real>,
    /// Optional output for left singular vectors U; only `rank` columns are produced per matrix.
    pub left_vectors: Option<StridedBatchedMatrixOutput<'a, T>>,
    /// Optional output for right singular vectors V, without conjugate transposition; only `rank` columns are produced per matrix.
    pub right_vectors: Option<StridedBatchedMatrixOutput<'a, T>>,
    /// Exclusive device scratch capacity measured in elements of `T`.
    pub workspace: DeviceOutputView<'a, T>,
    /// Device output capacity for one solver status per matrix.
    /// Inspect completed status before using numerical outputs.
    pub dev_info: DeviceOutputView<'a, i32>,
    /// Optional host output for one approximation residual per matrix.
    /// Must be present exactly when residual computation is enabled.
    pub residual: Option<&'a mut [MaybeUninit<f64>]>,
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use std::mem::MaybeUninit;

    use singe_core::assert_close;
    use singe_cuda::{memory::DeviceMemory, stream::StreamOptions};

    use super::*;
    use crate::{params::Params, testing::setup_context};

    #[test]
    fn test_gesvd_descriptor_returns_expected_singular_values() -> Result<()> {
        let mut test = setup_context()?;
        let context = test.context_mut();
        let params = Params::create()?;
        let operation = Gesvd::new(SvdMode::None, SvdMode::None, 2, 2);
        // Query before allocating input, outputs, or scratch.
        let workspace_sizes =
            operation.workspace_size::<f32>(context, &params, SvdLayout::new(2))?;
        let mut a = DeviceMemory::from_slice(context.cuda_context(), &[3.0_f32, 0.0, 0.0, 2.0])?;
        let mut s = DeviceMemory::<f32>::create_uninit(context.cuda_context(), 2)?;
        let mut workspace = DeviceMemory::<u8>::create_uninit(
            context.cuda_context(),
            workspace_sizes.device_bytes,
        )?;
        let mut host_workspace = vec![MaybeUninit::<u8>::uninit(); workspace_sizes.host_bytes];
        let mut dev_info = DeviceMemory::<i32>::create_uninit(context.cuda_context(), 1)?;
        let stream = context
            .cuda_context()
            .create_stream(StreamOptions::default())?;
        context.set_stream(Some(&stream))?;
        operation.execute(
            context,
            GesvdBindings {
                a: MatrixMut::new(&mut a, 2),
                singular_values: s.output(),
                left_vectors: None,
                right_vectors_transposed: None,
                workspace: ByteWorkspaceMut::new(&mut workspace, &mut host_workspace),
                dev_info: dev_info.output(),
            },
            &params,
        )?;
        // SAFETY: the solve completed without error, which writes the status element.
        let dev_info = unsafe { dev_info.assume_init() };
        assert_eq!(dev_info.copy_to_host_vec()?, vec![0]);
        // SAFETY: the status is zero, so both singular values were written.
        let s = unsafe { s.assume_init() };
        assert_close!(&s.copy_to_host_vec()?, &[3.0, 2.0], 1.0e-5);
        Ok(())
    }

    #[test]
    fn test_xgesvd_returns_expected_singular_values() -> Result<()> {
        let mut test = setup_context()?;
        let context = test.context_mut();
        let params = Params::create()?;
        // Query before allocating input, outputs, or scratch.
        let workspace_sizes = xgesvd_workspace_size::<f32>(
            context,
            &params,
            SvdMode::None,
            SvdMode::None,
            2,
            2,
            SvdLayout::new(2),
        )?;
        let mut a = DeviceMemory::from_slice(context.cuda_context(), &[3.0_f32, 0.0, 0.0, 2.0])?;
        let mut s = DeviceMemory::<f32>::create_uninit(context.cuda_context(), 2)?;
        let mut workspace = DeviceMemory::<u8>::create_uninit(
            context.cuda_context(),
            workspace_sizes.device_bytes,
        )?;
        let mut host_workspace = vec![MaybeUninit::<u8>::uninit(); workspace_sizes.host_bytes];
        let mut dev_info = DeviceMemory::<i32>::create_uninit(context.cuda_context(), 1)?;
        let stream = context
            .cuda_context()
            .create_stream(StreamOptions::default())?;
        context.set_stream(Some(&stream))?;
        stream.sync_scope(|_| {
            // SAFETY: the buffers are disjoint and declared outside `sync_scope`, which waits for the solve before they can drop.
            Ok(unsafe {
                xgesvd_async::<f32>(
                    context,
                    &params,
                    SvdMode::None,
                    SvdMode::None,
                    2,
                    2,
                    MatrixMut::new(&mut a, 2),
                    &mut s,
                    None,
                    None,
                    ByteWorkspaceMut::new(&mut workspace, &mut host_workspace),
                    &mut dev_info,
                )
            })
        })??;
        // SAFETY: the solve completed without error, which writes the status element.
        let dev_info = unsafe { dev_info.assume_init() };
        assert_eq!(dev_info.copy_to_host_vec()?, vec![0]);
        // SAFETY: the status is zero, so both singular values were written.
        let s = unsafe { s.assume_init() };
        assert_close!(&s.copy_to_host_vec()?, &[3.0, 2.0], 1.0e-5);
        Ok(())
    }
}
