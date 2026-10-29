mod generic;
mod info;
mod legacy;
pub mod raw;
mod scalar;
mod validation;

use std::mem::MaybeUninit;

pub use generic::{
    SvdLayout, xgesvd, xgesvd_buffer_size, xgesvdp, xgesvdp_buffer_size, xgesvdr,
    xgesvdr_buffer_size,
};
pub use info::GesvdjInfo;
pub use legacy::StridedSvdLayout;
pub use scalar::SvdScalar;
use singe_cuda::{memory::DeviceMemory, view::DeviceOutputView};

use crate::{
    context::Context,
    data_type::SolverElement,
    error::{Error, Result},
    layout::{
        ByteWorkspaceMut, MatrixMut, MatrixOutput, StridedBatchedMatrixOutput,
        StridedBatchedMatrixRef, StridedBatchedVectorOutput, StridedMatrixLayout, WorkspaceSizes,
    },
    params::Params,
    types::{EigenMode, SvdMode, TruncatedSvdMode},
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
        xgesvd_buffer_size::<T>(
            context,
            params,
            self.job_u,
            self.job_vt,
            self.rows,
            self.columns,
            layout,
        )
    }

    /// Queues this SVD with initialized input and write-only output bindings.
    ///
    /// # Safety
    ///
    /// Follow [`xgesvd`]'s context, ordering, aliasing, retention, and produced-range contract.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions, modes, output or status capacities, or workspace sizes are invalid, checked arithmetic or native integer conversion fails, CUDA binding fails, or cuSOLVER rejects or fails to enqueue the operation.
    pub unsafe fn execute<T: SolverElement>(
        self,
        context: &Context,
        params: &Params,
        mut bindings: GesvdBindings<'_, T>,
    ) -> Result<()> {
        // SAFETY: the caller upholds the raw operation's contract for these bindings.
        unsafe {
            xgesvd(
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
    /// Device output capacity for one solver status. Inspect completed status before using numerical outputs.
    pub dev_info: DeviceOutputView<'a, i32>,
}

/// Polar-decomposition-based SVD with full or economy singular vectors returned as U and V.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Gesvdp {
    /// Whether to compute singular values alone or also singular vectors.
    pub mode: EigenMode,
    /// Whether vector output contains only `min(rows, columns)` columns instead of full square factors.
    pub economy: bool,
    /// Number of rows in each input matrix.
    pub rows: usize,
    /// Number of columns in each input matrix.
    pub columns: usize,
}

impl Gesvdp {
    /// Records SVD dimensions and output modes without validating native limits.
    #[must_use]
    pub const fn new(mode: EigenMode, economy: bool, rows: usize, columns: usize) -> Self {
        Self {
            mode,
            economy,
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
        xgesvdp_buffer_size::<T>(
            context,
            params,
            self.mode,
            self.economy,
            self.rows,
            self.columns,
            layout,
        )
    }

    /// Queues this SVD with initialized input and write-only output bindings.
    ///
    /// # Safety
    ///
    /// Follow [`xgesvdp`]'s context, ordering, aliasing, retention, and produced-range contract.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions, modes, output or status capacities, or workspace sizes are invalid, checked arithmetic or native integer conversion fails, CUDA binding fails, or cuSOLVER rejects or fails to enqueue the operation.
    pub unsafe fn execute<T: SolverElement>(
        self,
        context: &Context,
        params: &Params,
        mut bindings: GesvdpBindings<'_, T>,
    ) -> Result<()> {
        // SAFETY: the caller upholds the raw operation's contract for these bindings.
        unsafe {
            xgesvdp(
                context,
                params,
                self.mode,
                self.economy,
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
    /// Device output capacity for one solver status. Inspect completed status before using numerical outputs.
    pub dev_info: DeviceOutputView<'a, i32>,
    /// Optional host output for the native singular-value perturbation magnitude, indicating SVD accuracy. Retain the destination through completion.
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
    /// Returns an error if dimensions, requested output layouts, or singular-vector modes are invalid, checked layout arithmetic or a native integer conversion fails, CUDA context binding fails, or cuSOLVER rejects the settings or cannot report workspace requirements. Target rank must be positive, and rank plus oversampling must be smaller than `min(rows, columns)`.
    pub fn workspace_size<T: SolverElement>(
        self,
        context: &Context,
        params: &Params,
        layout: SvdLayout,
    ) -> Result<WorkspaceSizes> {
        xgesvdr_buffer_size::<T>(
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

    /// Queues this SVD with initialized input and write-only output bindings.
    ///
    /// # Safety
    ///
    /// Follow [`xgesvdr`]'s context, ordering, aliasing, retention, and produced-range contract.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions, modes, output or status capacities, or workspace sizes are invalid, checked arithmetic or native integer conversion fails, CUDA binding fails, or cuSOLVER rejects or fails to enqueue the operation.
    pub unsafe fn execute<T: SolverElement>(
        self,
        context: &Context,
        params: &Params,
        mut bindings: GesvdrBindings<'_, T>,
    ) -> Result<()> {
        // SAFETY: the caller upholds the raw operation's contract for these bindings.
        unsafe {
            xgesvdr(
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
    /// Device output capacity for one solver status. Inspect completed status before using numerical outputs.
    pub dev_info: DeviceOutputView<'a, i32>,
}

/// Jacobi SVD with full or economy singular vectors, returned as U and V.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Gesvdj {
    /// Whether to compute singular values alone or also singular vectors.
    pub mode: EigenMode,
    /// Whether vector output contains only `min(rows, columns)` columns instead of full square factors.
    pub economy: bool,
    /// Number of rows in each input matrix.
    pub rows: usize,
    /// Number of columns in each input matrix.
    pub columns: usize,
}

impl Gesvdj {
    /// Records SVD dimensions and output modes without validating native limits.
    #[must_use]
    pub const fn new(mode: EigenMode, economy: bool, rows: usize, columns: usize) -> Self {
        Self {
            mode,
            economy,
            rows,
            columns,
        }
    }

    /// Queries scratch length in T elements before allocating matrices or outputs.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions, requested output layouts, or singular-vector modes are invalid, checked layout arithmetic or a native integer conversion fails, CUDA context binding fails, or cuSOLVER rejects the settings or cannot report workspace requirements.
    pub fn workspace_size<T: SvdScalar>(
        self,
        context: &Context,
        layout: SvdLayout,
        params: &GesvdjInfo,
    ) -> Result<usize> {
        (T::JACOBI_QUERY)(
            context,
            self.mode,
            self.economy,
            self.rows,
            self.columns,
            layout,
            params,
        )
    }

    /// Completes borrowed execution. Check the produced status before promoting results.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions, modes, output or status capacities, or workspace sizes are invalid, checked arithmetic or native integer conversion fails, CUDA binding fails, or cuSOLVER rejects or fails to enqueue the operation. Also returns an error if stream capture is active or invalidated, a buffer belongs to another CUDA context, or dependency ordering or stream completion fails.
    pub fn execute<T: SvdScalar>(
        self,
        context: &Context,
        bindings: GesvdjBindings<'_, T>,
        params: &mut GesvdjInfo,
    ) -> Result<()> {
        context.sync_scope(|| {
            context.prepare_region(&bindings.a.data)?;
            context.prepare_region(&bindings.singular_values)?;
            context.prepare_region(&bindings.workspace)?;
            context.prepare_region(&bindings.dev_info)?;
            for matrix in [
                bindings.left_vectors.as_ref(),
                bindings.right_vectors.as_ref(),
            ]
            .into_iter()
            .flatten()
            {
                context.prepare_region(&matrix.data)?;
            }
            // SAFETY: prepared disjoint bindings and exclusive parameters survive this completed call.
            unsafe {
                (T::JACOBI)(
                    context,
                    self.mode,
                    self.economy,
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
        })
    }
}

/// Initialized input and pending result storage for Jacobi SVD.
#[derive(Debug)]
pub struct GesvdjBindings<'a, T: SvdScalar> {
    /// Initialized input matrix, whose logical contents may be overwritten by the decomposition.
    pub a: MatrixMut<'a, T>,
    /// Real output capacity for `min(rows, columns)` singular values per matrix. Batched Jacobi packs consecutive value prefixes.
    pub singular_values: DeviceOutputView<'a, T::Real>,
    /// Optional output for left singular vectors U; full or economy column counts follow the operation settings.
    pub left_vectors: Option<MatrixOutput<'a, T>>,
    /// Optional output for right singular vectors V, without conjugate transposition; full or economy column counts follow the operation settings.
    pub right_vectors: Option<MatrixOutput<'a, T>>,
    /// Exclusive device scratch capacity measured in elements of `T`.
    pub workspace: DeviceOutputView<'a, T>,
    /// Device output capacity for one solver status per matrix. Inspect completed status before using numerical outputs.
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
    /// Returns an error if dimensions, requested output layouts, or singular-vector modes are invalid, checked layout arithmetic or a native integer conversion fails, CUDA context binding fails, or cuSOLVER rejects the settings or cannot report workspace requirements. The batch must be nonempty, with at most 32 rows and 32 columns per matrix.
    pub fn workspace_size<T: SvdScalar>(
        self,
        context: &Context,
        layout: SvdLayout,
        params: &GesvdjInfo,
    ) -> Result<usize> {
        (T::BATCHED_QUERY)(
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
    /// If the required buffer is absent, this method retains full U storage for
    /// tall matrices or full V storage for wide matrices through completion.
    /// Its contents are not exposed as initialized results.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions, modes, output or status capacities, or workspace sizes are invalid, checked arithmetic or native integer conversion fails, CUDA binding fails, or cuSOLVER rejects or fails to enqueue the operation. Also returns an error if stream capture is active or invalidated, a buffer belongs to another CUDA context, or dependency ordering or stream completion fails. Private temporary allocation can also fail.
    pub fn execute<T: SvdScalar>(
        self,
        context: &Context,
        bindings: GesvdjBindings<'_, T>,
        params: &mut GesvdjInfo,
    ) -> Result<()> {
        let mut native_vectors: Option<DeviceMemory<MaybeUninit<T>>> = None;
        context.sync_scope(|| {
            context.prepare_region(&bindings.a.data)?;
            context.prepare_region(&bindings.singular_values)?;
            context.prepare_region(&bindings.workspace)?;
            context.prepare_region(&bindings.dev_info)?;
            for matrix in [
                bindings.left_vectors.as_ref(),
                bindings.right_vectors.as_ref(),
            ]
            .into_iter()
            .flatten()
            {
                context.prepare_region(&matrix.data)?;
            }
            let left_scratch = self.rows >= self.columns;
            let (left, right) = if self.mode == EigenMode::NoVector
                && if left_scratch {
                    bindings.left_vectors.is_none()
                } else {
                    bindings.right_vectors.is_none()
                } {
                // The native batched implementation uses the larger vector buffer
                // even when no vectors are requested. Keep its owner outside
                // the closure so errors and unwinding drain before destruction.
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
            // SAFETY: every binding and exclusive parameter borrow survives completion.
            unsafe {
                (T::BATCHED)(
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
    /// Records SVD dimensions and output modes without validating native limits. Residual output is initially disabled.
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

    /// Requests one host residual per matrix. Values-only results use private
    /// vector storage because the native values-only path does not produce residuals.
    #[must_use]
    pub const fn with_residual(mut self) -> Self {
        self.compute_residual = true;
        self
    }

    /// Queries device scratch capacity in elements of `T` using batch layout metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions, requested output layouts, or singular-vector modes are invalid, checked layout arithmetic or a native integer conversion fails, CUDA context binding fails, or cuSOLVER rejects the settings or cannot report workspace requirements. Matrices must be tall or square, rank must be in `1..=columns`, and batch strides must cover each logical storage footprint.
    pub fn workspace_size<T: SvdScalar>(
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
        (T::APPROXIMATE_QUERY)(
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
    /// Returns an error if dimensions, modes, output or status capacities, or workspace sizes are invalid, checked arithmetic or native integer conversion fails, CUDA binding fails, or cuSOLVER rejects or fails to enqueue the operation. Also returns an error if stream capture is active or invalidated, a buffer belongs to another CUDA context, or dependency ordering or stream completion fails. Private temporary allocation can also fail. Residual storage must be present exactly when residual computation is enabled.
    pub fn execute<T: SvdScalar>(
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
            context.prepare_region(&bindings.a.data)?;
            context.prepare_region(&bindings.singular_values.data)?;
            context.prepare_region(&bindings.workspace)?;
            context.prepare_region(&bindings.dev_info)?;
            for matrix in [
                bindings.left_vectors.as_ref(),
                bindings.right_vectors.as_ref(),
            ]
            .into_iter()
            .flatten()
            {
                context.prepare_region(&matrix.data)?;
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
            // SAFETY: prepared device regions and host residuals remain borrowed through completion.
            unsafe {
                (T::APPROXIMATE)(
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
pub struct GesvdaStridedBatchedBindings<'a, T: SvdScalar> {
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
    /// Device output capacity for one solver status per matrix. Inspect completed status before using numerical outputs.
    pub dev_info: DeviceOutputView<'a, i32>,
    /// Optional host output for one approximation residual per matrix. Must be present exactly when residual computation is enabled.
    pub residual: Option<&'a mut [MaybeUninit<f64>]>,
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use std::mem::MaybeUninit;

    use singe_core::assert_close;
    use singe_cuda::memory::DeviceMemory;

    use super::*;
    use crate::{params::Params, testing::setup_context_if_available};

    #[test]
    fn test_gesvd_descriptor_returns_expected_singular_values() -> Result<()> {
        let Some(mut context) = setup_context_if_available()? else {
            return Ok(());
        };
        let params = Params::create()?;
        let operation = Gesvd::new(SvdMode::None, SvdMode::None, 2, 2);
        // Query before allocating input, outputs, or scratch.
        let workspace_sizes =
            operation.workspace_size::<f32>(&context, &params, SvdLayout::new(2))?;
        let mut a = DeviceMemory::from_slice(context.cuda_context(), &[3.0_f32, 0.0, 0.0, 2.0])?;
        let mut s = DeviceMemory::<f32>::create_uninit(context.cuda_context(), 2)?;
        let mut workspace = DeviceMemory::<u8>::create_uninit(
            context.cuda_context(),
            workspace_sizes.device_bytes,
        )?;
        let mut host_workspace = vec![MaybeUninit::<u8>::uninit(); workspace_sizes.host_bytes];
        let mut dev_info = DeviceMemory::<i32>::create_uninit(context.cuda_context(), 1)?;
        let stream = context.cuda_context().create_stream()?;
        context.set_stream(Some(&stream))?;
        stream.sync_scope(|_| {
            // SAFETY: disjoint buffers, host scratch, and handles survive the draining scope.
            Ok(unsafe {
                operation.execute(
                    &context,
                    &params,
                    GesvdBindings {
                        a: MatrixMut::new(&mut a, 2),
                        singular_values: s.output(),
                        left_vectors: None,
                        right_vectors_transposed: None,
                        workspace: ByteWorkspaceMut::new(&mut workspace, &mut host_workspace),
                        dev_info: dev_info.output(),
                    },
                )
            })
        })??;
        // SAFETY: successful completion writes the first status element.
        let dev_info = unsafe { dev_info.assume_init() };
        assert_eq!(dev_info.copy_to_host_vec()?, vec![0]);
        // SAFETY: a zero status confirms the compact singular-value range was produced.
        let s = unsafe { s.assume_init() };
        assert_close!(&s.copy_to_host_vec()?, &[3.0, 2.0], 1.0e-5);
        Ok(())
    }

    #[test]
    fn test_gesvdp_descriptor_preserves_operation_shape() {
        let operation = Gesvdp::new(EigenMode::Vector, true, 32, 16);

        assert_eq!(operation.mode, EigenMode::Vector);
        assert!(operation.economy);
        assert_eq!(operation.rows, 32);
        assert_eq!(operation.columns, 16);
    }

    #[test]
    fn test_gesvdr_descriptor_preserves_operation_shape() {
        let operation = Gesvdr::new(
            TruncatedSvdMode::Some,
            TruncatedSvdMode::None,
            128,
            64,
            12,
            24,
            3,
        );

        assert_eq!(operation.left_vectors, TruncatedSvdMode::Some);
        assert_eq!(operation.right_vectors, TruncatedSvdMode::None);
        assert_eq!(operation.rows, 128);
        assert_eq!(operation.columns, 64);
        assert_eq!(operation.rank, 12);
        assert_eq!(operation.oversamples, 24);
        assert_eq!(operation.iterations, 3);
    }

    #[test]
    fn test_xgesvd_returns_expected_singular_values() -> Result<()> {
        let Some(mut context) = setup_context_if_available()? else {
            return Ok(());
        };
        let params = Params::create()?;
        // Query before allocating input, outputs, or scratch.
        let workspace_sizes = xgesvd_buffer_size::<f32>(
            &context,
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
        let stream = context.cuda_context().create_stream()?;
        context.set_stream(Some(&stream))?;
        stream.sync_scope(|_| {
            // SAFETY: disjoint buffers, host scratch, and handles survive the draining scope.
            Ok(unsafe {
                xgesvd::<f32>(
                    &context,
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
        // SAFETY: successful completion writes the first status element.
        let dev_info = unsafe { dev_info.assume_init() };
        assert_eq!(dev_info.copy_to_host_vec()?, vec![0]);
        // SAFETY: a zero status confirms the compact singular-value range was produced.
        let s = unsafe { s.assume_init() };
        assert_close!(&s.copy_to_host_vec()?, &[3.0, 2.0], 1.0e-5);
        Ok(())
    }
}
