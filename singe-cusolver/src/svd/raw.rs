//! Direct native Jacobi and approximate SVD entry points.
//!
//! Raw execution requires inputs, outputs, scratch, parameter objects, and the
//! context to survive completion and every captured replay. Order inputs before
//! use and exclude concurrent access to writes and mutable Jacobi parameters.
//! Successful enqueue does not establish initialization or a valid numerical result.
//! Inspect completed status before promoting the documented logical results.
//! Leading-dimension padding, stride gaps, allocation tails, and scratch are not results.
//!
//! Jacobi SVD produces min(m,n) singular values. Full vectors have m columns in
//! U and n columns in V; economy vectors both have min(m,n) columns. Batched
//! Jacobi always produces full vectors and one status per matrix.
//! Approximate SVD produces rank values and rank columns of U and V per matrix,
//! one status per matrix, and one host residual per matrix when requested with
//! Vector mode. `NoVector` mode leaves host residuals unwritten.
//! Both families return V, rather than its conjugate transpose.
//!
//! The linked cuSOLVER 12.3.4 batched Jacobi values-only implementation requires
//! full U storage when m >= n, or full V storage when m < n, as temporary storage.
//! A null pointer for that buffer causes a device fault even in `NoVector` mode.
//! Those contents are scratch.
#![expect(
    clippy::many_single_char_names,
    reason = "BLAS and factorization operand names follow the standard mathematical API"
)]
use std::{mem::MaybeUninit, ptr};

use singe_cuda::{
    types::{Complex32, Complex64},
    view::DeviceOutputView,
};

use crate::{
    context::Context,
    error::{Error, Result},
    layout::{
        MatrixMut, MatrixOutput, StridedBatchedMatrixOutput, StridedBatchedMatrixRef,
        StridedBatchedVectorOutput,
    },
    svd::{
        GesvdjInfo, StridedSvdLayout, SvdLayout,
        legacy::{
            approximate_layout, jacobi_columns, jacobi_layout, output_parts, require_matrix,
            require_strided, strided_output_parts, validate_output,
        },
    },
    sys, try_ffi,
    types::EigenMode,
    utility::{to_i32, to_i64, to_usize},
};

/// Queries native scratch length in f32 elements without allocating buffers.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a required vector layout is missing or a batched matrix exceeds the native size limit, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn sgesvdj_buffer_size(
    context: &Context,
    mode: EigenMode,
    economy: bool,
    m: usize,
    n: usize,
    layout: SvdLayout,
    params: &GesvdjInfo,
) -> Result<usize> {
    let (lda, ldu, ldv) = jacobi_layout(m, n, mode, economy, layout, 1, false)?;
    context.bind()?;
    let mut lwork = 0;
    // SAFETY: buffer-size entry points inspect layout metadata, not matrix contents.
    unsafe {
        try_ffi!(sys::cusolverDnSgesvdj_bufferSize(
            context.as_raw(),
            mode.into(),
            i32::from(economy),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            ptr::null(),
            lda,
            ptr::null(),
            ptr::null(),
            ldu,
            ptr::null(),
            ldv,
            &raw mut lwork,
            params.as_raw(),
        ))?;
    }
    to_usize(lwork, "lwork")
}

/// Use the matching buffer-size helper to calculate the sizes needed for pre-allocated workspace.
///
/// The S and D data types are real valued single and double precision, respectively.
///
/// The C and Z data types are complex valued single and double precision, respectively.
///
/// Computes the singular value decomposition (SVD) of an $m \times n$ matrix `A` and the corresponding left and/or right singular vectors.
/// The SVD is written as $A = U \Sigma V^{H}$, where `Σ` is an $m \times n$ matrix which is zero except for its `min(m,n)` diagonal elements, `U` is an $m \times m$ unitary matrix, and `V` is an $n \times n$ unitary matrix.
/// The diagonal elements of `Σ` are the singular values of `A`; they are real and non-negative, and are returned in descending order.
/// The first `min(m,n)` columns of `U` and `V` are the left and right singular vectors of `A`.
///
/// `gesvdj` computes the same decomposition as `gesvd`.
/// The difference is that `gesvd` uses a QR algorithm and `gesvdj` uses the Jacobi method.
/// The Jacobi method gives GPUs better parallelism on small and medium-size matrices.
/// Callers can configure `gesvdj` to target a chosen accuracy.
///
/// `gesvdj` iteratively generates a sequence of unitary matrices that transform `A` toward $A = U(S + E)V^{H}$, where `S` is diagonal and the diagonal of `E` is zero.
///
/// During the iterations, the Frobenius norm of `E` decreases monotonically.
/// As `E` goes down to zero, `S` is the set of singular values.
/// In practice, the Jacobi method stops when the off-diagonal residual is below the configured tolerance `eps`.
/// If the real residual norm is computed, it differs from ${\\|{E}\\|}\_{F}$ by roundoff errors of order $N = max(m, n)$ while still meeting the standard SVD accuracy expectation.
///
/// $O(N)$ is typically $N$, but the constant depends on the number of sweeps, which gives an upper roundoff error bound of $sweeps \cdot N$.
///
/// `gesvdj` has two parameters to control the accuracy.
/// The first parameter is the tolerance (`eps`).
/// The default value is machine accuracy, but [`GesvdjInfo::set_tolerance`] can set an a priori tolerance.
/// The maximum-sweep parameter is the maximum number of sweeps, which controls the number of Jacobi iterations.
/// The default value is 100, but [`GesvdjInfo::set_max_sweeps`] can set a different bound.
/// Experiments show that 15 sweeps are enough to converge to machine accuracy.
/// `gesvdj` stops when either the tolerance or the maximum number of sweeps is reached.
///
/// The Jacobi method has quadratic convergence, so the accuracy is not proportional to the number of sweeps.
/// To guarantee a target accuracy, configure only the tolerance.
///
/// Provide workspace through `workspace`.
/// Use the corresponding `*_buffer_size` helper to query the required workspace length.
/// The workspace size in bytes is `size_of::<T>() * lwork`.
///
/// If the reported `info` value is `-i`, the `i`th parameter is invalid.
/// If `info == min(m, n) + 1`, `gesvdj` did not converge within the given tolerance and maximum sweep count.
///
/// If the tolerance is too small, `gesvdj` may not converge.
/// Use a tolerance no smaller than machine accuracy.
///
/// - `gesvdj` supports any combination of `m` and `n`.
///
/// - Returns `V`, not $V^{H}$.
///   This is different from `gesvd`.
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions, leading dimensions, or vector-computation mode are
/// invalid, or if cuSOLVER reports an internal failure.
/// # Safety
///
/// Follow the [raw SVD contracts](crate::svd::raw).
pub unsafe fn sgesvdj(
    context: &Context,
    mode: EigenMode,
    economy: bool,
    m: usize,
    n: usize,
    mut a: MatrixMut<'_, f32>,
    mut s: DeviceOutputView<'_, f32>,
    u: Option<MatrixOutput<'_, f32>>,
    v: Option<MatrixOutput<'_, f32>>,
    mut workspace: DeviceOutputView<'_, f32>,
    mut dev_info: DeviceOutputView<'_, i32>,
    params: &mut GesvdjInfo,
) -> Result<()> {
    let layout = SvdLayout {
        input_leading_dimension: a.leading_dimension,
        left_leading_dimension: u.as_ref().map(|matrix| matrix.leading_dimension),
        right_leading_dimension: v.as_ref().map(|matrix| matrix.leading_dimension),
    };
    jacobi_layout(m, n, mode, economy, layout, 1, false)?;
    require_matrix(
        m,
        n.checked_mul(1).ok_or(Error::InvalidMatrixShape)?,
        a.leading_dimension,
        a.data.len(),
    )?;
    validate_output(
        &s,
        m.min(n).checked_mul(1).ok_or(Error::InvalidVectorShape)?,
    )?;
    validate_output(&dev_info, 1)?;
    context.bind()?;
    context.validate_region(&a.data)?;
    context.validate_region(&s)?;
    context.validate_region(&workspace)?;
    context.validate_region(&dev_info)?;
    for matrix in [u.as_ref(), v.as_ref()].into_iter().flatten() {
        context.validate_region(&matrix.data)?;
    }
    let lwork = sgesvdj_buffer_size(context, mode, economy, m, n, layout, params)?;
    if workspace.len() < lwork {
        return Err(Error::InsufficientWorkspaceSize {
            required: lwork,
            actual: workspace.len(),
        });
    }
    let (left, right) = jacobi_columns(m, n, economy);
    let (u, ldu) = output_parts(u, m, left, 1, mode)?;
    let (v, ldv) = output_parts(v, n, right, 1, mode)?;
    // SAFETY: native ABI and extents are checked; retention and ordering belong to the caller.
    unsafe {
        try_ffi!(sys::cusolverDnSgesvdj(
            context.as_raw(),
            mode.into(),
            i32::from(economy),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            a.data.as_mut_ptr().cast(),
            to_i32(a.leading_dimension, "lda")?,
            s.as_mut_ptr(),
            u.cast(),
            ldu,
            v.cast(),
            ldv,
            workspace.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr(),
            params.as_raw(),
        ))?;
    }
    Ok(())
}

/// Queries native scratch length in f32 elements without allocating buffers.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a required vector layout is missing or a batched matrix exceeds the native size limit, the batch count or packed storage extent is invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn sgesvdj_batched_buffer_size(
    context: &Context,
    mode: EigenMode,
    m: usize,
    n: usize,
    layout: SvdLayout,
    params: &GesvdjInfo,
    batch_size: usize,
) -> Result<usize> {
    let (lda, ldu, ldv) = jacobi_layout(m, n, mode, false, layout, batch_size, true)?;
    context.bind()?;
    let mut lwork = 0;
    // SAFETY: buffer-size entry points inspect layout metadata, not matrix contents.
    unsafe {
        try_ffi!(sys::cusolverDnSgesvdjBatched_bufferSize(
            context.as_raw(),
            mode.into(),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            ptr::null(),
            lda,
            ptr::null(),
            ptr::null(),
            ldu,
            ptr::null(),
            ldv,
            &raw mut lwork,
            params.as_raw(),
            to_i32(batch_size, "batch_size")?,
        ))?;
    }
    to_usize(lwork, "lwork")
}

/// Use the matching buffer-size helper to calculate the sizes needed for pre-allocated workspace.
///
/// The S and D data types are real valued single and double precision, respectively.
///
/// The C and Z data types are complex valued single and double precision, respectively.
///
/// Computes singular values and singular vectors of a sequence of general $m \times n$ matrices
///
/// where $\Sigma\_{j}$ is a real $m \times n$ diagonal matrix which is zero except for its `min(m,n)` diagonal elements. $U\_{j}$ (left singular vectors) is an $m \times m$ unitary matrix and $V\_{j}$ (right singular vectors) is a $n \times n$ unitary matrix.
/// The diagonal elements of $\Sigma\_{j}$ are the singular values of $A\_{j}$ in either descending order or non-sorting order.
///
/// `gesvdjBatched` performs `gesvdj` on each matrix.
/// It requires that all matrices are of the same size `m,n` no greater than 32 and are packed contiguously,
///
/// Each matrix is column-major with leading dimension `lda`, so the formula for random access is $A\_{k}\operatorname{(i,j)} = {A\lbrack\ i\ +\ lda\cdot j\ +\ lda\cdot n\cdot k\rbrack}$.
///
/// The `S` parameter also contains the singular values of each matrix contiguously,
///
/// The formula for random access of `S` is $S\_{k}\operatorname{(j)} = {S\lbrack\ j\ +\ min(m,n)\cdot k\rbrack}$.
///
/// Except for tolerance and maximum sweeps, `gesvdjBatched` can either sort the singular values in descending order (default) or choose as-is (without sorting) with [`GesvdjInfo::set_sort_eigenvalues`].
/// If several tiny matrices are packed into diagonal blocks of one matrix, the non-sorting option can separate the singular values of those tiny matrices.
///
/// `gesvdjBatched` cannot report residual and executed sweeps through [`GesvdjInfo::residual`] and [`GesvdjInfo::executed_sweeps`].
/// Calling either accessor returns [`crate::error::Status::NotSupported`].
/// Compute the residual explicitly when needed.
///
/// Provide workspace through `workspace`.
/// Use the corresponding `*_buffer_size` helper to query the required workspace length.
/// The workspace size in bytes is `size_of::<T>() * lwork`.
///
/// `dev_info` has one entry per batch item.
/// If the call returns [`crate::error::Status::InvalidValue`], `dev_info[0] == -i` indicates that the `i`th parameter is invalid.
/// Otherwise, `dev_info[i] == min(m, n) + 1` indicates that `gesvdjBatched` did not converge on the `i`th matrix within the given tolerance and maximum sweep count.
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions, leading dimensions, vector-computation mode, or batch
/// size are invalid, or if cuSOLVER reports an internal failure.
/// # Safety
///
/// Follow the [raw SVD contracts](crate::svd::raw).
pub unsafe fn sgesvdj_batched(
    context: &Context,
    mode: EigenMode,
    m: usize,
    n: usize,
    mut a: MatrixMut<'_, f32>,
    mut s: DeviceOutputView<'_, f32>,
    u: Option<MatrixOutput<'_, f32>>,
    v: Option<MatrixOutput<'_, f32>>,
    mut workspace: DeviceOutputView<'_, f32>,
    mut dev_info: DeviceOutputView<'_, i32>,
    params: &mut GesvdjInfo,
    batch_size: usize,
) -> Result<()> {
    if mode == EigenMode::NoVector && if m >= n { u.is_none() } else { v.is_none() } {
        return Err(Error::InvalidMatrixShape);
    }

    let layout = SvdLayout {
        input_leading_dimension: a.leading_dimension,
        left_leading_dimension: u.as_ref().map(|matrix| matrix.leading_dimension),
        right_leading_dimension: v.as_ref().map(|matrix| matrix.leading_dimension),
    };
    jacobi_layout(m, n, mode, false, layout, batch_size, true)?;
    require_matrix(
        m,
        n.checked_mul(batch_size).ok_or(Error::InvalidMatrixShape)?,
        a.leading_dimension,
        a.data.len(),
    )?;
    validate_output(
        &s,
        m.min(n)
            .checked_mul(batch_size)
            .ok_or(Error::InvalidVectorShape)?,
    )?;
    validate_output(&dev_info, batch_size)?;
    context.bind()?;
    context.validate_region(&a.data)?;
    context.validate_region(&s)?;
    context.validate_region(&workspace)?;
    context.validate_region(&dev_info)?;
    for matrix in [u.as_ref(), v.as_ref()].into_iter().flatten() {
        context.validate_region(&matrix.data)?;
    }
    let lwork = sgesvdj_batched_buffer_size(context, mode, m, n, layout, params, batch_size)?;
    if workspace.len() < lwork {
        return Err(Error::InsufficientWorkspaceSize {
            required: lwork,
            actual: workspace.len(),
        });
    }
    let (left, right) = jacobi_columns(m, n, false);
    let (u, ldu) = output_parts(u, m, left, batch_size, mode)?;
    let (v, ldv) = output_parts(v, n, right, batch_size, mode)?;
    // SAFETY: native ABI and extents are checked; retention and ordering belong to the caller.
    unsafe {
        try_ffi!(sys::cusolverDnSgesvdjBatched(
            context.as_raw(),
            mode.into(),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            a.data.as_mut_ptr().cast(),
            to_i32(a.leading_dimension, "lda")?,
            s.as_mut_ptr(),
            u.cast(),
            ldu,
            v.cast(),
            ldv,
            workspace.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr(),
            params.as_raw(),
            to_i32(batch_size, "batch_size")?,
        ))?;
    }
    Ok(())
}

/// Queries approximate SVD scratch length in f32 elements from batch layouts.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, rank or a matrix or singular-value stride is invalid, the batch count or packed storage extent is invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn sgesvda_strided_batched_buffer_size(
    context: &Context,
    mode: EigenMode,
    rank: usize,
    m: usize,
    n: usize,
    layout: StridedSvdLayout,
    batch_size: usize,
) -> Result<usize> {
    let (lda, ldu, su, ldv, sv) = approximate_layout(rank, m, n, mode, layout, batch_size)?;
    context.bind()?;
    let mut lwork = 0;
    // SAFETY: this query uses dimensions, datatype, and strides, not matrix contents.
    unsafe {
        try_ffi!(sys::cusolverDnSgesvdaStridedBatched_bufferSize(
            context.as_raw(),
            mode.into(),
            to_i32(rank, "rank")?,
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            ptr::null(),
            lda,
            to_i64(layout.input.stride, "stride_a")?,
            ptr::null(),
            to_i64(layout.singular_value_stride, "stride_s")?,
            ptr::null(),
            ldu,
            su,
            ptr::null(),
            ldv,
            sv,
            &raw mut lwork,
            to_i32(batch_size, "batch_size")?
        ))?;
    }
    to_usize(lwork, "lwork")
}

/// Use the matching buffer-size helper to calculate the sizes needed for pre-allocated workspace.
///
/// The S and D data types are real valued single and double precision, respectively.
///
/// The C and Z data types are complex valued single and double precision, respectively.
///
/// `gesvda` (`a` stands for approximate) approximates the singular value decomposition of a tall skinny $m \times n$ matrix `A` and the corresponding left and right singular vectors.
/// The economy form of SVD is written as $A = U \Sigma V^{H}$, where `Σ` is
/// an $n \times n$ matrix.
/// `U` is an $m \times n$ unitary matrix, and `V` is an $n \times n$ unitary matrix.
/// The diagonal elements of `Σ` are the singular values of `A`; they are real and non-negative, and are returned in descending order.
/// `U` and `V` are the left and right singular vectors of `A`.
///
/// `gesvda` computes eigenvalues of $A^{T}A$, or $A^{H}A$ if `A` is
/// complex, to approximate singular values and singular vectors.
/// It generates matrices `U` and `V` and transforms matrix `A` to
/// $A = U(S + E)V^{H}$, where `S` is diagonal and `E` depends on rounding
/// errors.
/// To certain conditions, `U`, `V` and `S` approximate singular values and singular vectors up to machine zero of single precision.
/// In general, `V` is unitary, `S` is more accurate than `U`.
/// If singular value is far from zero, then left singular vector `U` is accurate.
/// In other words, the accuracy of singular values and left singular vectors depend on the distance between singular value and zero.
/// Since computing $A^{T}A$ or $A^{H}A$ can greatly amplify errors, use
/// `gesvda` only with well-conditioned data.
///
/// `rank` controls how many singular values and singular vectors are computed in `S`, `U`, and `V`.
///
/// `residual`, when requested, receives one Frobenius residual norm per matrix.
/// When `rank == n`, it measures how well `A` is approximated by the computed SVD.
/// Otherwise, it reports in the Frobenius norm sense how far `U` is from unitary.
///
/// `gesvdaStridedBatched` performs `gesvda` on each matrix.
/// It requires that all matrices are of the same size `m,n` and are packed contiguously,
///
/// Each matrix is column-major with leading dimension `lda`, so the formula for random access is $A\_{k}\operatorname{(i,j)} = {A\lbrack\ i\ +\ lda\cdot j\ +\ `stride_a\cdot` k\rbrack}$.
/// Similarly, the formula for random access of `S` is $S\_{k}\operatorname{(j)} = {S\lbrack\ j\ +\ `stride_s\cdot` k\rbrack}$, the formula for random access of `U` is $U\_{k}\operatorname{(i,j)} = {U\lbrack\ i\ +\ ldu\cdot j\ +\ `stride_u\cdot` k\rbrack}$ and the formula for random access of `V` is $V\_{k}\operatorname{(i,j)} = {V\lbrack\ i\ +\ ldv\cdot j\ +\ `stride_v\cdot` k\rbrack}$.
///
/// Provide workspace through `workspace`.
/// Use the corresponding `*_buffer_size` helper to query the required workspace length.
/// The workspace size in bytes is `size_of::<T>() * lwork`.
///
/// `dev_info` has one entry per batch item.
/// If the call returns [`crate::error::Status::InvalidValue`], `dev_info[0] == -i` indicates that the `i`th parameter is invalid.
/// Otherwise, `dev_info[i] == min(m, n) + 1` indicates that `gesvdaStridedBatched` did not converge on the `i`th matrix.
/// If `0 < dev_info[i] < min(m, n) + 1`, `gesvdaStridedBatched` could not compute an SVD of the `i`th matrix fully; the leading singular values `S_i[k]`, `0 <= k <= dev_info[i] - 1`, and corresponding singular vectors may still be useful.
/// In this case, if `residual` is requested, it is reported as if `rank` was set to `dev_info[i] - 1`.
///
/// The problem size is limited by `batch_size * stride{A/S/U/V} <= INT32_MAX` primarily due to the current implementation constraints.
///
/// - Returns `V`, not $V^{H}$.
///   This is different from `gesvd`.
///
/// - Only supports `m >= n`.
///
/// - Prefer an FP64 data type, such as `DgesvdaStridedBatched` or `ZgesvdaStridedBatched`.
///
/// - If singular values and singular vectors are known to be accurate, for example when the required singular value is far from zero, performance can be improved by passing `None` for `residual`, with no residual norm computation.
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions, leading dimensions, vector-computation mode, strides, or
/// batch size are invalid, or if cuSOLVER reports an internal failure.
/// # Safety
///
/// Follow the [raw SVD contracts](crate::svd::raw), retaining the host residual array too.
pub unsafe fn sgesvda_strided_batched(
    context: &Context,
    mode: EigenMode,
    rank: usize,
    m: usize,
    n: usize,
    a: StridedBatchedMatrixRef<'_, f32>,
    mut s: StridedBatchedVectorOutput<'_, f32>,
    u: Option<StridedBatchedMatrixOutput<'_, f32>>,
    v: Option<StridedBatchedMatrixOutput<'_, f32>>,
    mut workspace: DeviceOutputView<'_, f32>,
    mut dev_info: DeviceOutputView<'_, i32>,
    residual: Option<&mut [MaybeUninit<f64>]>,
    batch_size: usize,
) -> Result<()> {
    let layout = StridedSvdLayout {
        input: a.layout(),
        singular_value_stride: s.stride,
        left_vectors: u.as_ref().map(StridedBatchedMatrixOutput::layout),
        right_vectors: v.as_ref().map(StridedBatchedMatrixOutput::layout),
    };
    approximate_layout(rank, m, n, mode, layout, batch_size)?;
    require_strided(a.data.len(), a.stride, batch_size)?;
    require_strided(s.data.len(), s.stride, batch_size)?;
    validate_output(&dev_info, batch_size)?;
    if residual
        .as_ref()
        .is_some_and(|values| values.len() < batch_size)
    {
        return Err(Error::InvalidVectorShape);
    }
    context.bind()?;
    context.validate_region(&a.data)?;
    context.validate_region(&s.data)?;
    context.validate_region(&workspace)?;
    context.validate_region(&dev_info)?;
    for matrix in [u.as_ref(), v.as_ref()].into_iter().flatten() {
        context.validate_region(&matrix.data)?;
    }
    let lwork = sgesvda_strided_batched_buffer_size(context, mode, rank, m, n, layout, batch_size)?;
    if workspace.len() < lwork {
        return Err(Error::InsufficientWorkspaceSize {
            required: lwork,
            actual: workspace.len(),
        });
    }
    let (u, ldu, su) = strided_output_parts(u, m, rank, batch_size, mode)?;
    let (v, ldv, sv) = strided_output_parts(v, n, rank, batch_size, mode)?;
    // SAFETY: the caller retains and orders all device and host participants through completion.
    unsafe {
        try_ffi!(sys::cusolverDnSgesvdaStridedBatched(
            context.as_raw(),
            mode.into(),
            to_i32(rank, "rank")?,
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            a.data.as_ptr().cast(),
            to_i32(a.leading_dimension, "lda")?,
            to_i64(a.stride, "stride_a")?,
            s.data.as_mut_ptr(),
            to_i64(s.stride, "stride_s")?,
            u.cast(),
            ldu,
            su,
            v.cast(),
            ldv,
            sv,
            workspace.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr(),
            residual.map_or(ptr::null_mut(), |values| values.as_mut_ptr().cast()),
            to_i32(batch_size, "batch_size")?
        ))?;
    }
    Ok(())
}

/// Queries native scratch length in f64 elements without allocating buffers.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a required vector layout is missing or a batched matrix exceeds the native size limit, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn dgesvdj_buffer_size(
    context: &Context,
    mode: EigenMode,
    economy: bool,
    m: usize,
    n: usize,
    layout: SvdLayout,
    params: &GesvdjInfo,
) -> Result<usize> {
    let (lda, ldu, ldv) = jacobi_layout(m, n, mode, economy, layout, 1, false)?;
    context.bind()?;
    let mut lwork = 0;
    // SAFETY: buffer-size entry points inspect layout metadata, not matrix contents.
    unsafe {
        try_ffi!(sys::cusolverDnDgesvdj_bufferSize(
            context.as_raw(),
            mode.into(),
            i32::from(economy),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            ptr::null(),
            lda,
            ptr::null(),
            ptr::null(),
            ldu,
            ptr::null(),
            ldv,
            &raw mut lwork,
            params.as_raw(),
        ))?;
    }
    to_usize(lwork, "lwork")
}

/// Use the matching buffer-size helper to calculate the sizes needed for pre-allocated workspace.
///
/// The S and D data types are real valued single and double precision, respectively.
///
/// The C and Z data types are complex valued single and double precision, respectively.
///
/// Computes the singular value decomposition (SVD) of an $m \times n$ matrix `A` and the corresponding left and/or right singular vectors.
/// The SVD is written as $A = U \Sigma V^{H}$, where `Σ` is an $m \times n$ matrix which is zero except for its `min(m,n)` diagonal elements, `U` is an $m \times m$ unitary matrix, and `V` is an $n \times n$ unitary matrix.
/// The diagonal elements of `Σ` are the singular values of `A`; they are real and non-negative, and are returned in descending order.
/// The first `min(m,n)` columns of `U` and `V` are the left and right singular vectors of `A`.
///
/// `gesvdj` computes the same decomposition as `gesvd`.
/// The difference is that `gesvd` uses a QR algorithm and `gesvdj` uses the Jacobi method.
/// The Jacobi method gives GPUs better parallelism on small and medium-size matrices.
/// Callers can configure `gesvdj` to target a chosen accuracy.
///
/// `gesvdj` iteratively generates a sequence of unitary matrices that transform `A` toward $A = U(S + E)V^{H}$, where `S` is diagonal and the diagonal of `E` is zero.
///
/// During the iterations, the Frobenius norm of `E` decreases monotonically.
/// As `E` goes down to zero, `S` is the set of singular values.
/// In practice, the Jacobi method stops when the off-diagonal residual is below the configured tolerance `eps`.
/// If the real residual norm is computed, it differs from ${\\|{E}\\|}\_{F}$ by roundoff errors of order $N = max(m, n)$ while still meeting the standard SVD accuracy expectation.
///
/// $O(N)$ is typically $N$, but the constant depends on the number of sweeps, which gives an upper roundoff error bound of $sweeps \cdot N$.
///
/// `gesvdj` has two parameters to control the accuracy.
/// The first parameter is the tolerance (`eps`).
/// The default value is machine accuracy, but [`GesvdjInfo::set_tolerance`] can set an a priori tolerance.
/// The maximum-sweep parameter is the maximum number of sweeps, which controls the number of Jacobi iterations.
/// The default value is 100, but [`GesvdjInfo::set_max_sweeps`] can set a different bound.
/// Experiments show that 15 sweeps are enough to converge to machine accuracy.
/// `gesvdj` stops when either the tolerance or the maximum number of sweeps is reached.
///
/// The Jacobi method has quadratic convergence, so the accuracy is not proportional to the number of sweeps.
/// To guarantee a target accuracy, configure only the tolerance.
///
/// Provide workspace through `workspace`.
/// Use the corresponding `*_buffer_size` helper to query the required workspace length.
/// The workspace size in bytes is `size_of::<T>() * lwork`.
///
/// If the reported `info` value is `-i`, the `i`th parameter is invalid.
/// If `info == min(m, n) + 1`, `gesvdj` did not converge within the given tolerance and maximum sweep count.
///
/// If the tolerance is too small, `gesvdj` may not converge.
/// Use a tolerance no smaller than machine accuracy.
///
/// - `gesvdj` supports any combination of `m` and `n`.
///
/// - Returns `V`, not $V^{H}$.
///   This is different from `gesvd`.
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions, leading dimensions, or vector-computation mode are
/// invalid, or if cuSOLVER reports an internal failure.
/// # Safety
///
/// Follow the [raw SVD contracts](crate::svd::raw).
pub unsafe fn dgesvdj(
    context: &Context,
    mode: EigenMode,
    economy: bool,
    m: usize,
    n: usize,
    mut a: MatrixMut<'_, f64>,
    mut s: DeviceOutputView<'_, f64>,
    u: Option<MatrixOutput<'_, f64>>,
    v: Option<MatrixOutput<'_, f64>>,
    mut workspace: DeviceOutputView<'_, f64>,
    mut dev_info: DeviceOutputView<'_, i32>,
    params: &mut GesvdjInfo,
) -> Result<()> {
    let layout = SvdLayout {
        input_leading_dimension: a.leading_dimension,
        left_leading_dimension: u.as_ref().map(|matrix| matrix.leading_dimension),
        right_leading_dimension: v.as_ref().map(|matrix| matrix.leading_dimension),
    };
    jacobi_layout(m, n, mode, economy, layout, 1, false)?;
    require_matrix(
        m,
        n.checked_mul(1).ok_or(Error::InvalidMatrixShape)?,
        a.leading_dimension,
        a.data.len(),
    )?;
    validate_output(
        &s,
        m.min(n).checked_mul(1).ok_or(Error::InvalidVectorShape)?,
    )?;
    validate_output(&dev_info, 1)?;
    context.bind()?;
    context.validate_region(&a.data)?;
    context.validate_region(&s)?;
    context.validate_region(&workspace)?;
    context.validate_region(&dev_info)?;
    for matrix in [u.as_ref(), v.as_ref()].into_iter().flatten() {
        context.validate_region(&matrix.data)?;
    }
    let lwork = dgesvdj_buffer_size(context, mode, economy, m, n, layout, params)?;
    if workspace.len() < lwork {
        return Err(Error::InsufficientWorkspaceSize {
            required: lwork,
            actual: workspace.len(),
        });
    }
    let (left, right) = jacobi_columns(m, n, economy);
    let (u, ldu) = output_parts(u, m, left, 1, mode)?;
    let (v, ldv) = output_parts(v, n, right, 1, mode)?;
    // SAFETY: native ABI and extents are checked; retention and ordering belong to the caller.
    unsafe {
        try_ffi!(sys::cusolverDnDgesvdj(
            context.as_raw(),
            mode.into(),
            i32::from(economy),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            a.data.as_mut_ptr().cast(),
            to_i32(a.leading_dimension, "lda")?,
            s.as_mut_ptr(),
            u.cast(),
            ldu,
            v.cast(),
            ldv,
            workspace.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr(),
            params.as_raw(),
        ))?;
    }
    Ok(())
}

/// Queries native scratch length in f64 elements without allocating buffers.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a required vector layout is missing or a batched matrix exceeds the native size limit, the batch count or packed storage extent is invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn dgesvdj_batched_buffer_size(
    context: &Context,
    mode: EigenMode,
    m: usize,
    n: usize,
    layout: SvdLayout,
    params: &GesvdjInfo,
    batch_size: usize,
) -> Result<usize> {
    let (lda, ldu, ldv) = jacobi_layout(m, n, mode, false, layout, batch_size, true)?;
    context.bind()?;
    let mut lwork = 0;
    // SAFETY: buffer-size entry points inspect layout metadata, not matrix contents.
    unsafe {
        try_ffi!(sys::cusolverDnDgesvdjBatched_bufferSize(
            context.as_raw(),
            mode.into(),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            ptr::null(),
            lda,
            ptr::null(),
            ptr::null(),
            ldu,
            ptr::null(),
            ldv,
            &raw mut lwork,
            params.as_raw(),
            to_i32(batch_size, "batch_size")?,
        ))?;
    }
    to_usize(lwork, "lwork")
}

/// Use the matching buffer-size helper to calculate the sizes needed for pre-allocated workspace.
///
/// The S and D data types are real valued single and double precision, respectively.
///
/// The C and Z data types are complex valued single and double precision, respectively.
///
/// Computes singular values and singular vectors of a sequence of general $m \times n$ matrices
///
/// where $\Sigma\_{j}$ is a real $m \times n$ diagonal matrix which is zero except for its `min(m,n)` diagonal elements. $U\_{j}$ (left singular vectors) is an $m \times m$ unitary matrix and $V\_{j}$ (right singular vectors) is a $n \times n$ unitary matrix.
/// The diagonal elements of $\Sigma\_{j}$ are the singular values of $A\_{j}$ in either descending order or non-sorting order.
///
/// `gesvdjBatched` performs `gesvdj` on each matrix.
/// It requires that all matrices are of the same size `m,n` no greater than 32 and are packed contiguously,
///
/// Each matrix is column-major with leading dimension `lda`, so the formula for random access is $A\_{k}\operatorname{(i,j)} = {A\lbrack\ i\ +\ lda\cdot j\ +\ lda\cdot n\cdot k\rbrack}$.
///
/// The `S` parameter also contains the singular values of each matrix contiguously,
///
/// The formula for random access of `S` is $S\_{k}\operatorname{(j)} = {S\lbrack\ j\ +\ min(m,n)\cdot k\rbrack}$.
///
/// Except for tolerance and maximum sweeps, `gesvdjBatched` can either sort the singular values in descending order (default) or choose as-is (without sorting) with [`GesvdjInfo::set_sort_eigenvalues`].
/// If several tiny matrices are packed into diagonal blocks of one matrix, the non-sorting option can separate the singular values of those tiny matrices.
///
/// `gesvdjBatched` cannot report residual and executed sweeps through [`GesvdjInfo::residual`] and [`GesvdjInfo::executed_sweeps`].
/// Calling either accessor returns [`crate::error::Status::NotSupported`].
/// Compute the residual explicitly when needed.
///
/// Provide workspace through `workspace`.
/// Use the corresponding `*_buffer_size` helper to query the required workspace length.
/// The workspace size in bytes is `size_of::<T>() * lwork`.
///
/// `dev_info` has one entry per batch item.
/// If the call returns [`crate::error::Status::InvalidValue`], `dev_info[0] == -i` indicates that the `i`th parameter is invalid.
/// Otherwise, `dev_info[i] == min(m, n) + 1` indicates that `gesvdjBatched` did not converge on the `i`th matrix within the given tolerance and maximum sweep count.
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions, leading dimensions, vector-computation mode, or batch
/// size are invalid, or if cuSOLVER reports an internal failure.
/// # Safety
///
/// Follow the [raw SVD contracts](crate::svd::raw).
pub unsafe fn dgesvdj_batched(
    context: &Context,
    mode: EigenMode,
    m: usize,
    n: usize,
    mut a: MatrixMut<'_, f64>,
    mut s: DeviceOutputView<'_, f64>,
    u: Option<MatrixOutput<'_, f64>>,
    v: Option<MatrixOutput<'_, f64>>,
    mut workspace: DeviceOutputView<'_, f64>,
    mut dev_info: DeviceOutputView<'_, i32>,
    params: &mut GesvdjInfo,
    batch_size: usize,
) -> Result<()> {
    if mode == EigenMode::NoVector && if m >= n { u.is_none() } else { v.is_none() } {
        return Err(Error::InvalidMatrixShape);
    }

    let layout = SvdLayout {
        input_leading_dimension: a.leading_dimension,
        left_leading_dimension: u.as_ref().map(|matrix| matrix.leading_dimension),
        right_leading_dimension: v.as_ref().map(|matrix| matrix.leading_dimension),
    };
    jacobi_layout(m, n, mode, false, layout, batch_size, true)?;
    require_matrix(
        m,
        n.checked_mul(batch_size).ok_or(Error::InvalidMatrixShape)?,
        a.leading_dimension,
        a.data.len(),
    )?;
    validate_output(
        &s,
        m.min(n)
            .checked_mul(batch_size)
            .ok_or(Error::InvalidVectorShape)?,
    )?;
    validate_output(&dev_info, batch_size)?;
    context.bind()?;
    context.validate_region(&a.data)?;
    context.validate_region(&s)?;
    context.validate_region(&workspace)?;
    context.validate_region(&dev_info)?;
    for matrix in [u.as_ref(), v.as_ref()].into_iter().flatten() {
        context.validate_region(&matrix.data)?;
    }
    let lwork = dgesvdj_batched_buffer_size(context, mode, m, n, layout, params, batch_size)?;
    if workspace.len() < lwork {
        return Err(Error::InsufficientWorkspaceSize {
            required: lwork,
            actual: workspace.len(),
        });
    }
    let (left, right) = jacobi_columns(m, n, false);
    let (u, ldu) = output_parts(u, m, left, batch_size, mode)?;
    let (v, ldv) = output_parts(v, n, right, batch_size, mode)?;
    // SAFETY: native ABI and extents are checked; retention and ordering belong to the caller.
    unsafe {
        try_ffi!(sys::cusolverDnDgesvdjBatched(
            context.as_raw(),
            mode.into(),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            a.data.as_mut_ptr().cast(),
            to_i32(a.leading_dimension, "lda")?,
            s.as_mut_ptr(),
            u.cast(),
            ldu,
            v.cast(),
            ldv,
            workspace.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr(),
            params.as_raw(),
            to_i32(batch_size, "batch_size")?,
        ))?;
    }
    Ok(())
}

/// Queries approximate SVD scratch length in f64 elements from batch layouts.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, rank or a matrix or singular-value stride is invalid, the batch count or packed storage extent is invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn dgesvda_strided_batched_buffer_size(
    context: &Context,
    mode: EigenMode,
    rank: usize,
    m: usize,
    n: usize,
    layout: StridedSvdLayout,
    batch_size: usize,
) -> Result<usize> {
    let (lda, ldu, su, ldv, sv) = approximate_layout(rank, m, n, mode, layout, batch_size)?;
    context.bind()?;
    let mut lwork = 0;
    // SAFETY: this query uses dimensions, datatype, and strides, not matrix contents.
    unsafe {
        try_ffi!(sys::cusolverDnDgesvdaStridedBatched_bufferSize(
            context.as_raw(),
            mode.into(),
            to_i32(rank, "rank")?,
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            ptr::null(),
            lda,
            to_i64(layout.input.stride, "stride_a")?,
            ptr::null(),
            to_i64(layout.singular_value_stride, "stride_s")?,
            ptr::null(),
            ldu,
            su,
            ptr::null(),
            ldv,
            sv,
            &raw mut lwork,
            to_i32(batch_size, "batch_size")?
        ))?;
    }
    to_usize(lwork, "lwork")
}

/// Use the matching buffer-size helper to calculate the sizes needed for pre-allocated workspace.
///
/// The S and D data types are real valued single and double precision, respectively.
///
/// The C and Z data types are complex valued single and double precision, respectively.
///
/// `gesvda` (`a` stands for approximate) approximates the singular value decomposition of a tall skinny $m \times n$ matrix `A` and the corresponding left and right singular vectors.
/// The economy form of SVD is written as $A = U \Sigma V^{H}$, where `Σ` is
/// an $n \times n$ matrix.
/// `U` is an $m \times n$ unitary matrix, and `V` is an $n \times n$ unitary matrix.
/// The diagonal elements of `Σ` are the singular values of `A`; they are real and non-negative, and are returned in descending order.
/// `U` and `V` are the left and right singular vectors of `A`.
///
/// `gesvda` computes eigenvalues of $A^{T}A$, or $A^{H}A$ if `A` is
/// complex, to approximate singular values and singular vectors.
/// It generates matrices `U` and `V` and transforms matrix `A` to
/// $A = U(S + E)V^{H}$, where `S` is diagonal and `E` depends on rounding
/// errors.
/// To certain conditions, `U`, `V` and `S` approximate singular values and singular vectors up to machine zero of single precision.
/// In general, `V` is unitary, `S` is more accurate than `U`.
/// If singular value is far from zero, then left singular vector `U` is accurate.
/// In other words, the accuracy of singular values and left singular vectors depend on the distance between singular value and zero.
/// Since computing $A^{T}A$ or $A^{H}A$ can greatly amplify errors, use
/// `gesvda` only with well-conditioned data.
///
/// `rank` controls how many singular values and singular vectors are computed in `S`, `U`, and `V`.
///
/// `residual`, when requested, receives one Frobenius residual norm per matrix.
/// When `rank == n`, it measures how well `A` is approximated by the computed SVD.
/// Otherwise, it reports in the Frobenius norm sense how far `U` is from unitary.
///
/// `gesvdaStridedBatched` performs `gesvda` on each matrix.
/// It requires that all matrices are of the same size `m,n` and are packed contiguously,
///
/// Each matrix is column-major with leading dimension `lda`, so the formula for random access is $A\_{k}\operatorname{(i,j)} = {A\lbrack\ i\ +\ lda\cdot j\ +\ `stride_a\cdot` k\rbrack}$.
/// Similarly, the formula for random access of `S` is $S\_{k}\operatorname{(j)} = {S\lbrack\ j\ +\ `stride_s\cdot` k\rbrack}$, the formula for random access of `U` is $U\_{k}\operatorname{(i,j)} = {U\lbrack\ i\ +\ ldu\cdot j\ +\ `stride_u\cdot` k\rbrack}$ and the formula for random access of `V` is $V\_{k}\operatorname{(i,j)} = {V\lbrack\ i\ +\ ldv\cdot j\ +\ `stride_v\cdot` k\rbrack}$.
///
/// Provide workspace through `workspace`.
/// Use the corresponding `*_buffer_size` helper to query the required workspace length.
/// The workspace size in bytes is `size_of::<T>() * lwork`.
///
/// `dev_info` has one entry per batch item.
/// If the call returns [`crate::error::Status::InvalidValue`], `dev_info[0] == -i` indicates that the `i`th parameter is invalid.
/// Otherwise, `dev_info[i] == min(m, n) + 1` indicates that `gesvdaStridedBatched` did not converge on the `i`th matrix.
/// If `0 < dev_info[i] < min(m, n) + 1`, `gesvdaStridedBatched` could not compute an SVD of the `i`th matrix fully; the leading singular values `S_i[k]`, `0 <= k <= dev_info[i] - 1`, and corresponding singular vectors may still be useful.
/// In this case, if `residual` is requested, it is reported as if `rank` was set to `dev_info[i] - 1`.
///
/// The problem size is limited by `batch_size * stride{A/S/U/V} <= INT32_MAX` primarily due to the current implementation constraints.
///
/// - Returns `V`, not $V^{H}$.
///   This is different from `gesvd`.
///
/// - Only supports `m >= n`.
///
/// - Prefer an FP64 data type, such as `DgesvdaStridedBatched` or `ZgesvdaStridedBatched`.
///
/// - If singular values and singular vectors are known to be accurate, for example when the required singular value is far from zero, performance can be improved by passing `None` for `residual`, with no residual norm computation.
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions, leading dimensions, vector-computation mode, strides, or
/// batch size are invalid, or if cuSOLVER reports an internal failure.
/// # Safety
///
/// Follow the [raw SVD contracts](crate::svd::raw), retaining the host residual array too.
pub unsafe fn dgesvda_strided_batched(
    context: &Context,
    mode: EigenMode,
    rank: usize,
    m: usize,
    n: usize,
    a: StridedBatchedMatrixRef<'_, f64>,
    mut s: StridedBatchedVectorOutput<'_, f64>,
    u: Option<StridedBatchedMatrixOutput<'_, f64>>,
    v: Option<StridedBatchedMatrixOutput<'_, f64>>,
    mut workspace: DeviceOutputView<'_, f64>,
    mut dev_info: DeviceOutputView<'_, i32>,
    residual: Option<&mut [MaybeUninit<f64>]>,
    batch_size: usize,
) -> Result<()> {
    let layout = StridedSvdLayout {
        input: a.layout(),
        singular_value_stride: s.stride,
        left_vectors: u.as_ref().map(StridedBatchedMatrixOutput::layout),
        right_vectors: v.as_ref().map(StridedBatchedMatrixOutput::layout),
    };
    approximate_layout(rank, m, n, mode, layout, batch_size)?;
    require_strided(a.data.len(), a.stride, batch_size)?;
    require_strided(s.data.len(), s.stride, batch_size)?;
    validate_output(&dev_info, batch_size)?;
    if residual
        .as_ref()
        .is_some_and(|values| values.len() < batch_size)
    {
        return Err(Error::InvalidVectorShape);
    }
    context.bind()?;
    context.validate_region(&a.data)?;
    context.validate_region(&s.data)?;
    context.validate_region(&workspace)?;
    context.validate_region(&dev_info)?;
    for matrix in [u.as_ref(), v.as_ref()].into_iter().flatten() {
        context.validate_region(&matrix.data)?;
    }
    let lwork = dgesvda_strided_batched_buffer_size(context, mode, rank, m, n, layout, batch_size)?;
    if workspace.len() < lwork {
        return Err(Error::InsufficientWorkspaceSize {
            required: lwork,
            actual: workspace.len(),
        });
    }
    let (u, ldu, su) = strided_output_parts(u, m, rank, batch_size, mode)?;
    let (v, ldv, sv) = strided_output_parts(v, n, rank, batch_size, mode)?;
    // SAFETY: the caller retains and orders all device and host participants through completion.
    unsafe {
        try_ffi!(sys::cusolverDnDgesvdaStridedBatched(
            context.as_raw(),
            mode.into(),
            to_i32(rank, "rank")?,
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            a.data.as_ptr().cast(),
            to_i32(a.leading_dimension, "lda")?,
            to_i64(a.stride, "stride_a")?,
            s.data.as_mut_ptr(),
            to_i64(s.stride, "stride_s")?,
            u.cast(),
            ldu,
            su,
            v.cast(),
            ldv,
            sv,
            workspace.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr(),
            residual.map_or(ptr::null_mut(), |values| values.as_mut_ptr().cast()),
            to_i32(batch_size, "batch_size")?
        ))?;
    }
    Ok(())
}

/// Queries native scratch length in Complex32 elements without allocating buffers.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a required vector layout is missing or a batched matrix exceeds the native size limit, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn cgesvdj_buffer_size(
    context: &Context,
    mode: EigenMode,
    economy: bool,
    m: usize,
    n: usize,
    layout: SvdLayout,
    params: &GesvdjInfo,
) -> Result<usize> {
    let (lda, ldu, ldv) = jacobi_layout(m, n, mode, economy, layout, 1, false)?;
    context.bind()?;
    let mut lwork = 0;
    // SAFETY: buffer-size entry points inspect layout metadata, not matrix contents.
    unsafe {
        try_ffi!(sys::cusolverDnCgesvdj_bufferSize(
            context.as_raw(),
            mode.into(),
            i32::from(economy),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            ptr::null(),
            lda,
            ptr::null(),
            ptr::null(),
            ldu,
            ptr::null(),
            ldv,
            &raw mut lwork,
            params.as_raw(),
        ))?;
    }
    to_usize(lwork, "lwork")
}

/// Use the matching buffer-size helper to calculate the sizes needed for pre-allocated workspace.
///
/// The S and D data types are real valued single and double precision, respectively.
///
/// The C and Z data types are complex valued single and double precision, respectively.
///
/// Computes the singular value decomposition (SVD) of an $m \times n$ matrix `A` and the corresponding left and/or right singular vectors.
/// The SVD is written as $A = U \Sigma V^{H}$, where `Σ` is an $m \times n$ matrix which is zero except for its `min(m,n)` diagonal elements, `U` is an $m \times m$ unitary matrix, and `V` is an $n \times n$ unitary matrix.
/// The diagonal elements of `Σ` are the singular values of `A`; they are real and non-negative, and are returned in descending order.
/// The first `min(m,n)` columns of `U` and `V` are the left and right singular vectors of `A`.
///
/// `gesvdj` computes the same decomposition as `gesvd`.
/// The difference is that `gesvd` uses a QR algorithm and `gesvdj` uses the Jacobi method.
/// The Jacobi method gives GPUs better parallelism on small and medium-size matrices.
/// Callers can configure `gesvdj` to target a chosen accuracy.
///
/// `gesvdj` iteratively generates a sequence of unitary matrices that transform `A` toward $A = U(S + E)V^{H}$, where `S` is diagonal and the diagonal of `E` is zero.
///
/// During the iterations, the Frobenius norm of `E` decreases monotonically.
/// As `E` goes down to zero, `S` is the set of singular values.
/// In practice, the Jacobi method stops when the off-diagonal residual is below the configured tolerance `eps`.
/// If the real residual norm is computed, it differs from ${\\|{E}\\|}\_{F}$ by roundoff errors of order $N = max(m, n)$ while still meeting the standard SVD accuracy expectation.
///
/// $O(N)$ is typically $N$, but the constant depends on the number of sweeps, which gives an upper roundoff error bound of $sweeps \cdot N$.
///
/// `gesvdj` has two parameters to control the accuracy.
/// The first parameter is the tolerance (`eps`).
/// The default value is machine accuracy, but [`GesvdjInfo::set_tolerance`] can set an a priori tolerance.
/// The maximum-sweep parameter is the maximum number of sweeps, which controls the number of Jacobi iterations.
/// The default value is 100, but [`GesvdjInfo::set_max_sweeps`] can set a different bound.
/// Experiments show that 15 sweeps are enough to converge to machine accuracy.
/// `gesvdj` stops when either the tolerance or the maximum number of sweeps is reached.
///
/// The Jacobi method has quadratic convergence, so the accuracy is not proportional to the number of sweeps.
/// To guarantee a target accuracy, configure only the tolerance.
///
/// Provide workspace through `workspace`.
/// Use the corresponding `*_buffer_size` helper to query the required workspace length.
/// The workspace size in bytes is `size_of::<T>() * lwork`.
///
/// If the reported `info` value is `-i`, the `i`th parameter is invalid.
/// If `info == min(m, n) + 1`, `gesvdj` did not converge within the given tolerance and maximum sweep count.
///
/// If the tolerance is too small, `gesvdj` may not converge.
/// Use a tolerance no smaller than machine accuracy.
///
/// - `gesvdj` supports any combination of `m` and `n`.
///
/// - Returns `V`, not $V^{H}$.
///   This is different from `gesvd`.
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions, leading dimensions, or vector-computation mode are
/// invalid, or if cuSOLVER reports an internal failure.
/// # Safety
///
/// Follow the [raw SVD contracts](crate::svd::raw).
pub unsafe fn cgesvdj(
    context: &Context,
    mode: EigenMode,
    economy: bool,
    m: usize,
    n: usize,
    mut a: MatrixMut<'_, Complex32>,
    mut s: DeviceOutputView<'_, f32>,
    u: Option<MatrixOutput<'_, Complex32>>,
    v: Option<MatrixOutput<'_, Complex32>>,
    mut workspace: DeviceOutputView<'_, Complex32>,
    mut dev_info: DeviceOutputView<'_, i32>,
    params: &mut GesvdjInfo,
) -> Result<()> {
    let layout = SvdLayout {
        input_leading_dimension: a.leading_dimension,
        left_leading_dimension: u.as_ref().map(|matrix| matrix.leading_dimension),
        right_leading_dimension: v.as_ref().map(|matrix| matrix.leading_dimension),
    };
    jacobi_layout(m, n, mode, economy, layout, 1, false)?;
    require_matrix(
        m,
        n.checked_mul(1).ok_or(Error::InvalidMatrixShape)?,
        a.leading_dimension,
        a.data.len(),
    )?;
    validate_output(
        &s,
        m.min(n).checked_mul(1).ok_or(Error::InvalidVectorShape)?,
    )?;
    validate_output(&dev_info, 1)?;
    context.bind()?;
    context.validate_region(&a.data)?;
    context.validate_region(&s)?;
    context.validate_region(&workspace)?;
    context.validate_region(&dev_info)?;
    for matrix in [u.as_ref(), v.as_ref()].into_iter().flatten() {
        context.validate_region(&matrix.data)?;
    }
    let lwork = cgesvdj_buffer_size(context, mode, economy, m, n, layout, params)?;
    if workspace.len() < lwork {
        return Err(Error::InsufficientWorkspaceSize {
            required: lwork,
            actual: workspace.len(),
        });
    }
    let (left, right) = jacobi_columns(m, n, economy);
    let (u, ldu) = output_parts(u, m, left, 1, mode)?;
    let (v, ldv) = output_parts(v, n, right, 1, mode)?;
    // SAFETY: native ABI and extents are checked; retention and ordering belong to the caller.
    unsafe {
        try_ffi!(sys::cusolverDnCgesvdj(
            context.as_raw(),
            mode.into(),
            i32::from(economy),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            a.data.as_mut_ptr().cast(),
            to_i32(a.leading_dimension, "lda")?,
            s.as_mut_ptr(),
            u.cast(),
            ldu,
            v.cast(),
            ldv,
            workspace.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr(),
            params.as_raw(),
        ))?;
    }
    Ok(())
}

/// Queries native scratch length in Complex32 elements without allocating buffers.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a required vector layout is missing or a batched matrix exceeds the native size limit, the batch count or packed storage extent is invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn cgesvdj_batched_buffer_size(
    context: &Context,
    mode: EigenMode,
    m: usize,
    n: usize,
    layout: SvdLayout,
    params: &GesvdjInfo,
    batch_size: usize,
) -> Result<usize> {
    let (lda, ldu, ldv) = jacobi_layout(m, n, mode, false, layout, batch_size, true)?;
    context.bind()?;
    let mut lwork = 0;
    // SAFETY: buffer-size entry points inspect layout metadata, not matrix contents.
    unsafe {
        try_ffi!(sys::cusolverDnCgesvdjBatched_bufferSize(
            context.as_raw(),
            mode.into(),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            ptr::null(),
            lda,
            ptr::null(),
            ptr::null(),
            ldu,
            ptr::null(),
            ldv,
            &raw mut lwork,
            params.as_raw(),
            to_i32(batch_size, "batch_size")?,
        ))?;
    }
    to_usize(lwork, "lwork")
}

/// Use the matching buffer-size helper to calculate the sizes needed for pre-allocated workspace.
///
/// The S and D data types are real valued single and double precision, respectively.
///
/// The C and Z data types are complex valued single and double precision, respectively.
///
/// Computes singular values and singular vectors of a sequence of general $m \times n$ matrices
///
/// where $\Sigma\_{j}$ is a real $m \times n$ diagonal matrix which is zero except for its `min(m,n)` diagonal elements. $U\_{j}$ (left singular vectors) is an $m \times m$ unitary matrix and $V\_{j}$ (right singular vectors) is a $n \times n$ unitary matrix.
/// The diagonal elements of $\Sigma\_{j}$ are the singular values of $A\_{j}$ in either descending order or non-sorting order.
///
/// `gesvdjBatched` performs `gesvdj` on each matrix.
/// It requires that all matrices are of the same size `m,n` no greater than 32 and are packed contiguously,
///
/// Each matrix is column-major with leading dimension `lda`, so the formula for random access is $A\_{k}\operatorname{(i,j)} = {A\lbrack\ i\ +\ lda\cdot j\ +\ lda\cdot n\cdot k\rbrack}$.
///
/// The `S` parameter also contains the singular values of each matrix contiguously,
///
/// The formula for random access of `S` is $S\_{k}\operatorname{(j)} = {S\lbrack\ j\ +\ min(m,n)\cdot k\rbrack}$.
///
/// Except for tolerance and maximum sweeps, `gesvdjBatched` can either sort the singular values in descending order (default) or choose as-is (without sorting) with [`GesvdjInfo::set_sort_eigenvalues`].
/// If several tiny matrices are packed into diagonal blocks of one matrix, the non-sorting option can separate the singular values of those tiny matrices.
///
/// `gesvdjBatched` cannot report residual and executed sweeps through [`GesvdjInfo::residual`] and [`GesvdjInfo::executed_sweeps`].
/// Calling either accessor returns [`crate::error::Status::NotSupported`].
/// Compute the residual explicitly when needed.
///
/// Provide workspace through `workspace`.
/// Use the corresponding `*_buffer_size` helper to query the required workspace length.
/// The workspace size in bytes is `size_of::<T>() * lwork`.
///
/// `dev_info` has one entry per batch item.
/// If the call returns [`crate::error::Status::InvalidValue`], `dev_info[0] == -i` indicates that the `i`th parameter is invalid.
/// Otherwise, `dev_info[i] == min(m, n) + 1` indicates that `gesvdjBatched` did not converge on the `i`th matrix within the given tolerance and maximum sweep count.
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions, leading dimensions, vector-computation mode, or batch
/// size are invalid, or if cuSOLVER reports an internal failure.
/// # Safety
///
/// Follow the [raw SVD contracts](crate::svd::raw).
pub unsafe fn cgesvdj_batched(
    context: &Context,
    mode: EigenMode,
    m: usize,
    n: usize,
    mut a: MatrixMut<'_, Complex32>,
    mut s: DeviceOutputView<'_, f32>,
    u: Option<MatrixOutput<'_, Complex32>>,
    v: Option<MatrixOutput<'_, Complex32>>,
    mut workspace: DeviceOutputView<'_, Complex32>,
    mut dev_info: DeviceOutputView<'_, i32>,
    params: &mut GesvdjInfo,
    batch_size: usize,
) -> Result<()> {
    if mode == EigenMode::NoVector && if m >= n { u.is_none() } else { v.is_none() } {
        return Err(Error::InvalidMatrixShape);
    }

    let layout = SvdLayout {
        input_leading_dimension: a.leading_dimension,
        left_leading_dimension: u.as_ref().map(|matrix| matrix.leading_dimension),
        right_leading_dimension: v.as_ref().map(|matrix| matrix.leading_dimension),
    };
    jacobi_layout(m, n, mode, false, layout, batch_size, true)?;
    require_matrix(
        m,
        n.checked_mul(batch_size).ok_or(Error::InvalidMatrixShape)?,
        a.leading_dimension,
        a.data.len(),
    )?;
    validate_output(
        &s,
        m.min(n)
            .checked_mul(batch_size)
            .ok_or(Error::InvalidVectorShape)?,
    )?;
    validate_output(&dev_info, batch_size)?;
    context.bind()?;
    context.validate_region(&a.data)?;
    context.validate_region(&s)?;
    context.validate_region(&workspace)?;
    context.validate_region(&dev_info)?;
    for matrix in [u.as_ref(), v.as_ref()].into_iter().flatten() {
        context.validate_region(&matrix.data)?;
    }
    let lwork = cgesvdj_batched_buffer_size(context, mode, m, n, layout, params, batch_size)?;
    if workspace.len() < lwork {
        return Err(Error::InsufficientWorkspaceSize {
            required: lwork,
            actual: workspace.len(),
        });
    }
    let (left, right) = jacobi_columns(m, n, false);
    let (u, ldu) = output_parts(u, m, left, batch_size, mode)?;
    let (v, ldv) = output_parts(v, n, right, batch_size, mode)?;
    // SAFETY: native ABI and extents are checked; retention and ordering belong to the caller.
    unsafe {
        try_ffi!(sys::cusolverDnCgesvdjBatched(
            context.as_raw(),
            mode.into(),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            a.data.as_mut_ptr().cast(),
            to_i32(a.leading_dimension, "lda")?,
            s.as_mut_ptr(),
            u.cast(),
            ldu,
            v.cast(),
            ldv,
            workspace.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr(),
            params.as_raw(),
            to_i32(batch_size, "batch_size")?,
        ))?;
    }
    Ok(())
}

/// Queries approximate SVD scratch length in Complex32 elements from batch layouts.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, rank or a matrix or singular-value stride is invalid, the batch count or packed storage extent is invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn cgesvda_strided_batched_buffer_size(
    context: &Context,
    mode: EigenMode,
    rank: usize,
    m: usize,
    n: usize,
    layout: StridedSvdLayout,
    batch_size: usize,
) -> Result<usize> {
    let (lda, ldu, su, ldv, sv) = approximate_layout(rank, m, n, mode, layout, batch_size)?;
    context.bind()?;
    let mut lwork = 0;
    // SAFETY: this query uses dimensions, datatype, and strides, not matrix contents.
    unsafe {
        try_ffi!(sys::cusolverDnCgesvdaStridedBatched_bufferSize(
            context.as_raw(),
            mode.into(),
            to_i32(rank, "rank")?,
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            ptr::null(),
            lda,
            to_i64(layout.input.stride, "stride_a")?,
            ptr::null(),
            to_i64(layout.singular_value_stride, "stride_s")?,
            ptr::null(),
            ldu,
            su,
            ptr::null(),
            ldv,
            sv,
            &raw mut lwork,
            to_i32(batch_size, "batch_size")?
        ))?;
    }
    to_usize(lwork, "lwork")
}

/// Use the matching buffer-size helper to calculate the sizes needed for pre-allocated workspace.
///
/// The S and D data types are real valued single and double precision, respectively.
///
/// The C and Z data types are complex valued single and double precision, respectively.
///
/// `gesvda` (`a` stands for approximate) approximates the singular value decomposition of a tall skinny $m \times n$ matrix `A` and the corresponding left and right singular vectors.
/// The economy form of SVD is written as $A = U \Sigma V^{H}$, where `Σ` is
/// an $n \times n$ matrix.
/// `U` is an $m \times n$ unitary matrix, and `V` is an $n \times n$ unitary matrix.
/// The diagonal elements of `Σ` are the singular values of `A`; they are real and non-negative, and are returned in descending order.
/// `U` and `V` are the left and right singular vectors of `A`.
///
/// `gesvda` computes eigenvalues of $A^{T}A$, or $A^{H}A$ if `A` is
/// complex, to approximate singular values and singular vectors.
/// It generates matrices `U` and `V` and transforms matrix `A` to
/// $A = U(S + E)V^{H}$, where `S` is diagonal and `E` depends on rounding
/// errors.
/// To certain conditions, `U`, `V` and `S` approximate singular values and singular vectors up to machine zero of single precision.
/// In general, `V` is unitary, `S` is more accurate than `U`.
/// If singular value is far from zero, then left singular vector `U` is accurate.
/// In other words, the accuracy of singular values and left singular vectors depend on the distance between singular value and zero.
/// Since computing $A^{T}A$ or $A^{H}A$ can greatly amplify errors, use
/// `gesvda` only with well-conditioned data.
///
/// `rank` controls how many singular values and singular vectors are computed in `S`, `U`, and `V`.
///
/// `residual`, when requested, receives one Frobenius residual norm per matrix.
/// When `rank == n`, it measures how well `A` is approximated by the computed SVD.
/// Otherwise, it reports in the Frobenius norm sense how far `U` is from unitary.
///
/// `gesvdaStridedBatched` performs `gesvda` on each matrix.
/// It requires that all matrices are of the same size `m,n` and are packed contiguously,
///
/// Each matrix is column-major with leading dimension `lda`, so the formula for random access is $A\_{k}\operatorname{(i,j)} = {A\lbrack\ i\ +\ lda\cdot j\ +\ `stride_a\cdot` k\rbrack}$.
/// Similarly, the formula for random access of `S` is $S\_{k}\operatorname{(j)} = {S\lbrack\ j\ +\ `stride_s\cdot` k\rbrack}$, the formula for random access of `U` is $U\_{k}\operatorname{(i,j)} = {U\lbrack\ i\ +\ ldu\cdot j\ +\ `stride_u\cdot` k\rbrack}$ and the formula for random access of `V` is $V\_{k}\operatorname{(i,j)} = {V\lbrack\ i\ +\ ldv\cdot j\ +\ `stride_v\cdot` k\rbrack}$.
///
/// Provide workspace through `workspace`.
/// Use the corresponding `*_buffer_size` helper to query the required workspace length.
/// The workspace size in bytes is `size_of::<T>() * lwork`.
///
/// `dev_info` has one entry per batch item.
/// If the call returns [`crate::error::Status::InvalidValue`], `dev_info[0] == -i` indicates that the `i`th parameter is invalid.
/// Otherwise, `dev_info[i] == min(m, n) + 1` indicates that `gesvdaStridedBatched` did not converge on the `i`th matrix.
/// If `0 < dev_info[i] < min(m, n) + 1`, `gesvdaStridedBatched` could not compute an SVD of the `i`th matrix fully; the leading singular values `S_i[k]`, `0 <= k <= dev_info[i] - 1`, and corresponding singular vectors may still be useful.
/// In this case, if `residual` is requested, it is reported as if `rank` was set to `dev_info[i] - 1`.
///
/// The problem size is limited by `batch_size * stride{A/S/U/V} <= INT32_MAX` primarily due to the current implementation constraints.
///
/// - Returns `V`, not $V^{H}$.
///   This is different from `gesvd`.
///
/// - Only supports `m >= n`.
///
/// - Prefer an FP64 data type, such as `DgesvdaStridedBatched` or `ZgesvdaStridedBatched`.
///
/// - If singular values and singular vectors are known to be accurate, for example when the required singular value is far from zero, performance can be improved by passing `None` for `residual`, with no residual norm computation.
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions, leading dimensions, vector-computation mode, strides, or
/// batch size are invalid, or if cuSOLVER reports an internal failure.
/// # Safety
///
/// Follow the [raw SVD contracts](crate::svd::raw), retaining the host residual array too.
pub unsafe fn cgesvda_strided_batched(
    context: &Context,
    mode: EigenMode,
    rank: usize,
    m: usize,
    n: usize,
    a: StridedBatchedMatrixRef<'_, Complex32>,
    mut s: StridedBatchedVectorOutput<'_, f32>,
    u: Option<StridedBatchedMatrixOutput<'_, Complex32>>,
    v: Option<StridedBatchedMatrixOutput<'_, Complex32>>,
    mut workspace: DeviceOutputView<'_, Complex32>,
    mut dev_info: DeviceOutputView<'_, i32>,
    residual: Option<&mut [MaybeUninit<f64>]>,
    batch_size: usize,
) -> Result<()> {
    let layout = StridedSvdLayout {
        input: a.layout(),
        singular_value_stride: s.stride,
        left_vectors: u.as_ref().map(StridedBatchedMatrixOutput::layout),
        right_vectors: v.as_ref().map(StridedBatchedMatrixOutput::layout),
    };
    approximate_layout(rank, m, n, mode, layout, batch_size)?;
    require_strided(a.data.len(), a.stride, batch_size)?;
    require_strided(s.data.len(), s.stride, batch_size)?;
    validate_output(&dev_info, batch_size)?;
    if residual
        .as_ref()
        .is_some_and(|values| values.len() < batch_size)
    {
        return Err(Error::InvalidVectorShape);
    }
    context.bind()?;
    context.validate_region(&a.data)?;
    context.validate_region(&s.data)?;
    context.validate_region(&workspace)?;
    context.validate_region(&dev_info)?;
    for matrix in [u.as_ref(), v.as_ref()].into_iter().flatten() {
        context.validate_region(&matrix.data)?;
    }
    let lwork = cgesvda_strided_batched_buffer_size(context, mode, rank, m, n, layout, batch_size)?;
    if workspace.len() < lwork {
        return Err(Error::InsufficientWorkspaceSize {
            required: lwork,
            actual: workspace.len(),
        });
    }
    let (u, ldu, su) = strided_output_parts(u, m, rank, batch_size, mode)?;
    let (v, ldv, sv) = strided_output_parts(v, n, rank, batch_size, mode)?;
    // SAFETY: the caller retains and orders all device and host participants through completion.
    unsafe {
        try_ffi!(sys::cusolverDnCgesvdaStridedBatched(
            context.as_raw(),
            mode.into(),
            to_i32(rank, "rank")?,
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            a.data.as_ptr().cast(),
            to_i32(a.leading_dimension, "lda")?,
            to_i64(a.stride, "stride_a")?,
            s.data.as_mut_ptr(),
            to_i64(s.stride, "stride_s")?,
            u.cast(),
            ldu,
            su,
            v.cast(),
            ldv,
            sv,
            workspace.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr(),
            residual.map_or(ptr::null_mut(), |values| values.as_mut_ptr().cast()),
            to_i32(batch_size, "batch_size")?
        ))?;
    }
    Ok(())
}

/// Queries native scratch length in Complex64 elements without allocating buffers.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a required vector layout is missing or a batched matrix exceeds the native size limit, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn zgesvdj_buffer_size(
    context: &Context,
    mode: EigenMode,
    economy: bool,
    m: usize,
    n: usize,
    layout: SvdLayout,
    params: &GesvdjInfo,
) -> Result<usize> {
    let (lda, ldu, ldv) = jacobi_layout(m, n, mode, economy, layout, 1, false)?;
    context.bind()?;
    let mut lwork = 0;
    // SAFETY: buffer-size entry points inspect layout metadata, not matrix contents.
    unsafe {
        try_ffi!(sys::cusolverDnZgesvdj_bufferSize(
            context.as_raw(),
            mode.into(),
            i32::from(economy),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            ptr::null(),
            lda,
            ptr::null(),
            ptr::null(),
            ldu,
            ptr::null(),
            ldv,
            &raw mut lwork,
            params.as_raw(),
        ))?;
    }
    to_usize(lwork, "lwork")
}

/// Use the matching buffer-size helper to calculate the sizes needed for pre-allocated workspace.
///
/// The S and D data types are real valued single and double precision, respectively.
///
/// The C and Z data types are complex valued single and double precision, respectively.
///
/// Computes the singular value decomposition (SVD) of an $m \times n$ matrix `A` and the corresponding left and/or right singular vectors.
/// The SVD is written as $A = U \Sigma V^{H}$, where `Σ` is an $m \times n$ matrix which is zero except for its `min(m,n)` diagonal elements, `U` is an $m \times m$ unitary matrix, and `V` is an $n \times n$ unitary matrix.
/// The diagonal elements of `Σ` are the singular values of `A`; they are real and non-negative, and are returned in descending order.
/// The first `min(m,n)` columns of `U` and `V` are the left and right singular vectors of `A`.
///
/// `gesvdj` computes the same decomposition as `gesvd`.
/// The difference is that `gesvd` uses a QR algorithm and `gesvdj` uses the Jacobi method.
/// The Jacobi method gives GPUs better parallelism on small and medium-size matrices.
/// Callers can configure `gesvdj` to target a chosen accuracy.
///
/// `gesvdj` iteratively generates a sequence of unitary matrices that transform `A` toward $A = U(S + E)V^{H}$, where `S` is diagonal and the diagonal of `E` is zero.
///
/// During the iterations, the Frobenius norm of `E` decreases monotonically.
/// As `E` goes down to zero, `S` is the set of singular values.
/// In practice, the Jacobi method stops when the off-diagonal residual is below the configured tolerance `eps`.
/// If the real residual norm is computed, it differs from ${\\|{E}\\|}\_{F}$ by roundoff errors of order $N = max(m, n)$ while still meeting the standard SVD accuracy expectation.
///
/// $O(N)$ is typically $N$, but the constant depends on the number of sweeps, which gives an upper roundoff error bound of $sweeps \cdot N$.
///
/// `gesvdj` has two parameters to control the accuracy.
/// The first parameter is the tolerance (`eps`).
/// The default value is machine accuracy, but [`GesvdjInfo::set_tolerance`] can set an a priori tolerance.
/// The maximum-sweep parameter is the maximum number of sweeps, which controls the number of Jacobi iterations.
/// The default value is 100, but [`GesvdjInfo::set_max_sweeps`] can set a different bound.
/// Experiments show that 15 sweeps are enough to converge to machine accuracy.
/// `gesvdj` stops when either the tolerance or the maximum number of sweeps is reached.
///
/// The Jacobi method has quadratic convergence, so the accuracy is not proportional to the number of sweeps.
/// To guarantee a target accuracy, configure only the tolerance.
///
/// Provide workspace through `workspace`.
/// Use the corresponding `*_buffer_size` helper to query the required workspace length.
/// The workspace size in bytes is `size_of::<T>() * lwork`.
///
/// If the reported `info` value is `-i`, the `i`th parameter is invalid.
/// If `info == min(m, n) + 1`, `gesvdj` did not converge within the given tolerance and maximum sweep count.
///
/// If the tolerance is too small, `gesvdj` may not converge.
/// Use a tolerance no smaller than machine accuracy.
///
/// - `gesvdj` supports any combination of `m` and `n`.
///
/// - Returns `V`, not $V^{H}$.
///   This is different from `gesvd`.
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions, leading dimensions, or vector-computation mode are
/// invalid, or if cuSOLVER reports an internal failure.
/// # Safety
///
/// Follow the [raw SVD contracts](crate::svd::raw).
pub unsafe fn zgesvdj(
    context: &Context,
    mode: EigenMode,
    economy: bool,
    m: usize,
    n: usize,
    mut a: MatrixMut<'_, Complex64>,
    mut s: DeviceOutputView<'_, f64>,
    u: Option<MatrixOutput<'_, Complex64>>,
    v: Option<MatrixOutput<'_, Complex64>>,
    mut workspace: DeviceOutputView<'_, Complex64>,
    mut dev_info: DeviceOutputView<'_, i32>,
    params: &mut GesvdjInfo,
) -> Result<()> {
    let layout = SvdLayout {
        input_leading_dimension: a.leading_dimension,
        left_leading_dimension: u.as_ref().map(|matrix| matrix.leading_dimension),
        right_leading_dimension: v.as_ref().map(|matrix| matrix.leading_dimension),
    };
    jacobi_layout(m, n, mode, economy, layout, 1, false)?;
    require_matrix(
        m,
        n.checked_mul(1).ok_or(Error::InvalidMatrixShape)?,
        a.leading_dimension,
        a.data.len(),
    )?;
    validate_output(
        &s,
        m.min(n).checked_mul(1).ok_or(Error::InvalidVectorShape)?,
    )?;
    validate_output(&dev_info, 1)?;
    context.bind()?;
    context.validate_region(&a.data)?;
    context.validate_region(&s)?;
    context.validate_region(&workspace)?;
    context.validate_region(&dev_info)?;
    for matrix in [u.as_ref(), v.as_ref()].into_iter().flatten() {
        context.validate_region(&matrix.data)?;
    }
    let lwork = zgesvdj_buffer_size(context, mode, economy, m, n, layout, params)?;
    if workspace.len() < lwork {
        return Err(Error::InsufficientWorkspaceSize {
            required: lwork,
            actual: workspace.len(),
        });
    }
    let (left, right) = jacobi_columns(m, n, economy);
    let (u, ldu) = output_parts(u, m, left, 1, mode)?;
    let (v, ldv) = output_parts(v, n, right, 1, mode)?;
    // SAFETY: native ABI and extents are checked; retention and ordering belong to the caller.
    unsafe {
        try_ffi!(sys::cusolverDnZgesvdj(
            context.as_raw(),
            mode.into(),
            i32::from(economy),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            a.data.as_mut_ptr().cast(),
            to_i32(a.leading_dimension, "lda")?,
            s.as_mut_ptr(),
            u.cast(),
            ldu,
            v.cast(),
            ldv,
            workspace.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr(),
            params.as_raw(),
        ))?;
    }
    Ok(())
}

/// Queries native scratch length in Complex64 elements without allocating buffers.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a required vector layout is missing or a batched matrix exceeds the native size limit, the batch count or packed storage extent is invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn zgesvdj_batched_buffer_size(
    context: &Context,
    mode: EigenMode,
    m: usize,
    n: usize,
    layout: SvdLayout,
    params: &GesvdjInfo,
    batch_size: usize,
) -> Result<usize> {
    let (lda, ldu, ldv) = jacobi_layout(m, n, mode, false, layout, batch_size, true)?;
    context.bind()?;
    let mut lwork = 0;
    // SAFETY: buffer-size entry points inspect layout metadata, not matrix contents.
    unsafe {
        try_ffi!(sys::cusolverDnZgesvdjBatched_bufferSize(
            context.as_raw(),
            mode.into(),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            ptr::null(),
            lda,
            ptr::null(),
            ptr::null(),
            ldu,
            ptr::null(),
            ldv,
            &raw mut lwork,
            params.as_raw(),
            to_i32(batch_size, "batch_size")?,
        ))?;
    }
    to_usize(lwork, "lwork")
}

/// Use the matching buffer-size helper to calculate the sizes needed for pre-allocated workspace.
///
/// The S and D data types are real valued single and double precision, respectively.
///
/// The C and Z data types are complex valued single and double precision, respectively.
///
/// Computes singular values and singular vectors of a sequence of general $m \times n$ matrices
///
/// where $\Sigma\_{j}$ is a real $m \times n$ diagonal matrix which is zero except for its `min(m,n)` diagonal elements. $U\_{j}$ (left singular vectors) is an $m \times m$ unitary matrix and $V\_{j}$ (right singular vectors) is a $n \times n$ unitary matrix.
/// The diagonal elements of $\Sigma\_{j}$ are the singular values of $A\_{j}$ in either descending order or non-sorting order.
///
/// `gesvdjBatched` performs `gesvdj` on each matrix.
/// It requires that all matrices are of the same size `m,n` no greater than 32 and are packed contiguously,
///
/// Each matrix is column-major with leading dimension `lda`, so the formula for random access is $A\_{k}\operatorname{(i,j)} = {A\lbrack\ i\ +\ lda\cdot j\ +\ lda\cdot n\cdot k\rbrack}$.
///
/// The `S` parameter also contains the singular values of each matrix contiguously,
///
/// The formula for random access of `S` is $S\_{k}\operatorname{(j)} = {S\lbrack\ j\ +\ min(m,n)\cdot k\rbrack}$.
///
/// Except for tolerance and maximum sweeps, `gesvdjBatched` can either sort the singular values in descending order (default) or choose as-is (without sorting) with [`GesvdjInfo::set_sort_eigenvalues`].
/// If several tiny matrices are packed into diagonal blocks of one matrix, the non-sorting option can separate the singular values of those tiny matrices.
///
/// `gesvdjBatched` cannot report residual and executed sweeps through [`GesvdjInfo::residual`] and [`GesvdjInfo::executed_sweeps`].
/// Calling either accessor returns [`crate::error::Status::NotSupported`].
/// Compute the residual explicitly when needed.
///
/// Provide workspace through `workspace`.
/// Use the corresponding `*_buffer_size` helper to query the required workspace length.
/// The workspace size in bytes is `size_of::<T>() * lwork`.
///
/// `dev_info` has one entry per batch item.
/// If the call returns [`crate::error::Status::InvalidValue`], `dev_info[0] == -i` indicates that the `i`th parameter is invalid.
/// Otherwise, `dev_info[i] == min(m, n) + 1` indicates that `gesvdjBatched` did not converge on the `i`th matrix within the given tolerance and maximum sweep count.
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions, leading dimensions, vector-computation mode, or batch
/// size are invalid, or if cuSOLVER reports an internal failure.
/// # Safety
///
/// Follow the [raw SVD contracts](crate::svd::raw).
pub unsafe fn zgesvdj_batched(
    context: &Context,
    mode: EigenMode,
    m: usize,
    n: usize,
    mut a: MatrixMut<'_, Complex64>,
    mut s: DeviceOutputView<'_, f64>,
    u: Option<MatrixOutput<'_, Complex64>>,
    v: Option<MatrixOutput<'_, Complex64>>,
    mut workspace: DeviceOutputView<'_, Complex64>,
    mut dev_info: DeviceOutputView<'_, i32>,
    params: &mut GesvdjInfo,
    batch_size: usize,
) -> Result<()> {
    if mode == EigenMode::NoVector && if m >= n { u.is_none() } else { v.is_none() } {
        return Err(Error::InvalidMatrixShape);
    }

    let layout = SvdLayout {
        input_leading_dimension: a.leading_dimension,
        left_leading_dimension: u.as_ref().map(|matrix| matrix.leading_dimension),
        right_leading_dimension: v.as_ref().map(|matrix| matrix.leading_dimension),
    };
    jacobi_layout(m, n, mode, false, layout, batch_size, true)?;
    require_matrix(
        m,
        n.checked_mul(batch_size).ok_or(Error::InvalidMatrixShape)?,
        a.leading_dimension,
        a.data.len(),
    )?;
    validate_output(
        &s,
        m.min(n)
            .checked_mul(batch_size)
            .ok_or(Error::InvalidVectorShape)?,
    )?;
    validate_output(&dev_info, batch_size)?;
    context.bind()?;
    context.validate_region(&a.data)?;
    context.validate_region(&s)?;
    context.validate_region(&workspace)?;
    context.validate_region(&dev_info)?;
    for matrix in [u.as_ref(), v.as_ref()].into_iter().flatten() {
        context.validate_region(&matrix.data)?;
    }
    let lwork = zgesvdj_batched_buffer_size(context, mode, m, n, layout, params, batch_size)?;
    if workspace.len() < lwork {
        return Err(Error::InsufficientWorkspaceSize {
            required: lwork,
            actual: workspace.len(),
        });
    }
    let (left, right) = jacobi_columns(m, n, false);
    let (u, ldu) = output_parts(u, m, left, batch_size, mode)?;
    let (v, ldv) = output_parts(v, n, right, batch_size, mode)?;
    // SAFETY: native ABI and extents are checked; retention and ordering belong to the caller.
    unsafe {
        try_ffi!(sys::cusolverDnZgesvdjBatched(
            context.as_raw(),
            mode.into(),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            a.data.as_mut_ptr().cast(),
            to_i32(a.leading_dimension, "lda")?,
            s.as_mut_ptr(),
            u.cast(),
            ldu,
            v.cast(),
            ldv,
            workspace.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr(),
            params.as_raw(),
            to_i32(batch_size, "batch_size")?,
        ))?;
    }
    Ok(())
}

/// Queries approximate SVD scratch length in Complex64 elements from batch layouts.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, rank or a matrix or singular-value stride is invalid, the batch count or packed storage extent is invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn zgesvda_strided_batched_buffer_size(
    context: &Context,
    mode: EigenMode,
    rank: usize,
    m: usize,
    n: usize,
    layout: StridedSvdLayout,
    batch_size: usize,
) -> Result<usize> {
    let (lda, ldu, su, ldv, sv) = approximate_layout(rank, m, n, mode, layout, batch_size)?;
    context.bind()?;
    let mut lwork = 0;
    // SAFETY: this query uses dimensions, datatype, and strides, not matrix contents.
    unsafe {
        try_ffi!(sys::cusolverDnZgesvdaStridedBatched_bufferSize(
            context.as_raw(),
            mode.into(),
            to_i32(rank, "rank")?,
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            ptr::null(),
            lda,
            to_i64(layout.input.stride, "stride_a")?,
            ptr::null(),
            to_i64(layout.singular_value_stride, "stride_s")?,
            ptr::null(),
            ldu,
            su,
            ptr::null(),
            ldv,
            sv,
            &raw mut lwork,
            to_i32(batch_size, "batch_size")?
        ))?;
    }
    to_usize(lwork, "lwork")
}

/// Use the matching buffer-size helper to calculate the sizes needed for pre-allocated workspace.
///
/// The S and D data types are real valued single and double precision, respectively.
///
/// The C and Z data types are complex valued single and double precision, respectively.
///
/// `gesvda` (`a` stands for approximate) approximates the singular value decomposition of a tall skinny $m \times n$ matrix `A` and the corresponding left and right singular vectors.
/// The economy form of SVD is written as $A = U \Sigma V^{H}$, where `Σ` is
/// an $n \times n$ matrix.
/// `U` is an $m \times n$ unitary matrix, and `V` is an $n \times n$ unitary matrix.
/// The diagonal elements of `Σ` are the singular values of `A`; they are real and non-negative, and are returned in descending order.
/// `U` and `V` are the left and right singular vectors of `A`.
///
/// `gesvda` computes eigenvalues of $A^{T}A$, or $A^{H}A$ if `A` is
/// complex, to approximate singular values and singular vectors.
/// It generates matrices `U` and `V` and transforms matrix `A` to
/// $A = U(S + E)V^{H}$, where `S` is diagonal and `E` depends on rounding
/// errors.
/// To certain conditions, `U`, `V` and `S` approximate singular values and singular vectors up to machine zero of single precision.
/// In general, `V` is unitary, `S` is more accurate than `U`.
/// If singular value is far from zero, then left singular vector `U` is accurate.
/// In other words, the accuracy of singular values and left singular vectors depend on the distance between singular value and zero.
/// Since computing $A^{T}A$ or $A^{H}A$ can greatly amplify errors, use
/// `gesvda` only with well-conditioned data.
///
/// `rank` controls how many singular values and singular vectors are computed in `S`, `U`, and `V`.
///
/// `residual`, when requested, receives one Frobenius residual norm per matrix.
/// When `rank == n`, it measures how well `A` is approximated by the computed SVD.
/// Otherwise, it reports in the Frobenius norm sense how far `U` is from unitary.
///
/// `gesvdaStridedBatched` performs `gesvda` on each matrix.
/// It requires that all matrices are of the same size `m,n` and are packed contiguously,
///
/// Each matrix is column-major with leading dimension `lda`, so the formula for random access is $A\_{k}\operatorname{(i,j)} = {A\lbrack\ i\ +\ lda\cdot j\ +\ `stride_a\cdot` k\rbrack}$.
/// Similarly, the formula for random access of `S` is $S\_{k}\operatorname{(j)} = {S\lbrack\ j\ +\ `stride_s\cdot` k\rbrack}$, the formula for random access of `U` is $U\_{k}\operatorname{(i,j)} = {U\lbrack\ i\ +\ ldu\cdot j\ +\ `stride_u\cdot` k\rbrack}$ and the formula for random access of `V` is $V\_{k}\operatorname{(i,j)} = {V\lbrack\ i\ +\ ldv\cdot j\ +\ `stride_v\cdot` k\rbrack}$.
///
/// Provide workspace through `workspace`.
/// Use the corresponding `*_buffer_size` helper to query the required workspace length.
/// The workspace size in bytes is `size_of::<T>() * lwork`.
///
/// `dev_info` has one entry per batch item.
/// If the call returns [`crate::error::Status::InvalidValue`], `dev_info[0] == -i` indicates that the `i`th parameter is invalid.
/// Otherwise, `dev_info[i] == min(m, n) + 1` indicates that `gesvdaStridedBatched` did not converge on the `i`th matrix.
/// If `0 < dev_info[i] < min(m, n) + 1`, `gesvdaStridedBatched` could not compute an SVD of the `i`th matrix fully; the leading singular values `S_i[k]`, `0 <= k <= dev_info[i] - 1`, and corresponding singular vectors may still be useful.
/// In this case, if `residual` is requested, it is reported as if `rank` was set to `dev_info[i] - 1`.
///
/// The problem size is limited by `batch_size * stride{A/S/U/V} <= INT32_MAX` primarily due to the current implementation constraints.
///
/// - Returns `V`, not $V^{H}$.
///   This is different from `gesvd`.
///
/// - Only supports `m >= n`.
///
/// - Prefer an FP64 data type, such as `DgesvdaStridedBatched` or `ZgesvdaStridedBatched`.
///
/// - If singular values and singular vectors are known to be accurate, for example when the required singular value is far from zero, performance can be improved by passing `None` for `residual`, with no residual norm computation.
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions, leading dimensions, vector-computation mode, strides, or
/// batch size are invalid, or if cuSOLVER reports an internal failure.
/// # Safety
///
/// Follow the [raw SVD contracts](crate::svd::raw), retaining the host residual array too.
pub unsafe fn zgesvda_strided_batched(
    context: &Context,
    mode: EigenMode,
    rank: usize,
    m: usize,
    n: usize,
    a: StridedBatchedMatrixRef<'_, Complex64>,
    mut s: StridedBatchedVectorOutput<'_, f64>,
    u: Option<StridedBatchedMatrixOutput<'_, Complex64>>,
    v: Option<StridedBatchedMatrixOutput<'_, Complex64>>,
    mut workspace: DeviceOutputView<'_, Complex64>,
    mut dev_info: DeviceOutputView<'_, i32>,
    residual: Option<&mut [MaybeUninit<f64>]>,
    batch_size: usize,
) -> Result<()> {
    let layout = StridedSvdLayout {
        input: a.layout(),
        singular_value_stride: s.stride,
        left_vectors: u.as_ref().map(StridedBatchedMatrixOutput::layout),
        right_vectors: v.as_ref().map(StridedBatchedMatrixOutput::layout),
    };
    approximate_layout(rank, m, n, mode, layout, batch_size)?;
    require_strided(a.data.len(), a.stride, batch_size)?;
    require_strided(s.data.len(), s.stride, batch_size)?;
    validate_output(&dev_info, batch_size)?;
    if residual
        .as_ref()
        .is_some_and(|values| values.len() < batch_size)
    {
        return Err(Error::InvalidVectorShape);
    }
    context.bind()?;
    context.validate_region(&a.data)?;
    context.validate_region(&s.data)?;
    context.validate_region(&workspace)?;
    context.validate_region(&dev_info)?;
    for matrix in [u.as_ref(), v.as_ref()].into_iter().flatten() {
        context.validate_region(&matrix.data)?;
    }
    let lwork = zgesvda_strided_batched_buffer_size(context, mode, rank, m, n, layout, batch_size)?;
    if workspace.len() < lwork {
        return Err(Error::InsufficientWorkspaceSize {
            required: lwork,
            actual: workspace.len(),
        });
    }
    let (u, ldu, su) = strided_output_parts(u, m, rank, batch_size, mode)?;
    let (v, ldv, sv) = strided_output_parts(v, n, rank, batch_size, mode)?;
    // SAFETY: the caller retains and orders all device and host participants through completion.
    unsafe {
        try_ffi!(sys::cusolverDnZgesvdaStridedBatched(
            context.as_raw(),
            mode.into(),
            to_i32(rank, "rank")?,
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            a.data.as_ptr().cast(),
            to_i32(a.leading_dimension, "lda")?,
            to_i64(a.stride, "stride_a")?,
            s.data.as_mut_ptr(),
            to_i64(s.stride, "stride_s")?,
            u.cast(),
            ldu,
            su,
            v.cast(),
            ldv,
            sv,
            workspace.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr(),
            residual.map_or(ptr::null_mut(), |values| values.as_mut_ptr().cast()),
            to_i32(batch_size, "batch_size")?
        ))?;
    }
    Ok(())
}
