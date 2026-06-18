use singe_cuda::{
    memory::DeviceMemory,
    types::{Complex32, Complex64},
};

use crate::{
    context::Context,
    dense::validation::{
        require_info_buffer, require_info_entries, require_workspace,
        validate_batched_square_matrix_pointers, validate_batched_vector_pointers,
        validate_square_matrix,
    },
    error::{Error, Result},
    layout::{BatchedMatrixRef, BatchedVectorRef},
    sys, try_ffi,
    types::FillMode,
    utility::{to_i32, to_usize},
};

/// The S and D data types are real valued single and double precision, respectively.
///
/// The C and Z data types are complex valued single and double precision, respectively.
///
/// Computes the Cholesky factorization of a sequence of Hermitian positive-definite matrices.
///
/// Each `a[i]` for `i = 0, 1, ..., batch_size - 1` is a $n \times n$ Hermitian matrix, only lower or upper part is meaningful.
/// `fill_mode` indicates which part of the matrix is used.
///
/// If `fill_mode` is [`FillMode::Lower`], only the lower triangular part of `A` is processed and replaced by the lower triangular Cholesky factor `L`.
///
/// If `fill_mode` is [`FillMode::Upper`], only the upper triangular part of `A` is processed and replaced by the upper triangular Cholesky factor `U`.
///
/// If Cholesky factorization failed, that is, some leading minor of `A` is not positive definite, or equivalently some diagonal elements of `L` or `U` are not real.
/// `info` contains one entry per matrix and reports the smallest leading minor of `A` that is not positive definite.
///
/// `info` must have one integer entry for each matrix in the batch.
/// If cuSOLVER reports [`crate::error::Status::InvalidValue`], `info[0] == -i` indicates that the `i`th parameter is invalid.
/// If `potrf_batched` returns [`Ok`] and `info[i] == k` is positive, the `i`th matrix is not positive definite and the Cholesky factorization failed at row `k`.
///
/// The other part of `A` is used as workspace.
/// For example, if `fill_mode` is [`FillMode::Upper`], upper triangle of `A` contains Cholesky factor `U` and lower triangle of `A` is destroyed after `potrf_batched`.
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions, leading dimension, or batch size are invalid, or if
/// cuSOLVER reports an internal failure.
pub fn spotrf_batched(
    ctx: &Context,
    fill_mode: FillMode,
    n: usize,
    a: BatchedMatrixRef<'_, f32>,
    info: &mut DeviceMemory<i32>,
) -> Result<()> {
    ctx.bind()?;
    validate_batched_square_matrix_pointers(n, a)?;
    require_info_entries(info, a.len())?;
    unsafe {
        try_ffi!(sys::cusolverDnSpotrfBatched(
            ctx.as_raw(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_mut_ptr(),
            to_i32(a.leading_dimension, "lda")?,
            info.as_mut_ptr().cast(),
            to_i32(a.len(), "batch_size")?,
        ))?;
    }
    Ok(())
}

/// The S and D data types are real valued single and double precision, respectively.
///
/// The C and Z data types are complex valued single and double precision, respectively.
///
/// Computes the Cholesky factorization of a sequence of Hermitian positive-definite matrices.
///
/// Each `a[i]` for `i = 0, 1, ..., batch_size - 1` is a $n \times n$ Hermitian matrix, only lower or upper part is meaningful.
/// `fill_mode` indicates which part of the matrix is used.
///
/// If `fill_mode` is [`FillMode::Lower`], only the lower triangular part of `A` is processed and replaced by the lower triangular Cholesky factor `L`.
///
/// If `fill_mode` is [`FillMode::Upper`], only the upper triangular part of `A` is processed and replaced by the upper triangular Cholesky factor `U`.
///
/// If Cholesky factorization failed, that is, some leading minor of `A` is not positive definite, or equivalently some diagonal elements of `L` or `U` are not real.
/// `info` contains one entry per matrix and reports the smallest leading minor of `A` that is not positive definite.
///
/// `info` must have one integer entry for each matrix in the batch.
/// If cuSOLVER reports [`crate::error::Status::InvalidValue`], `info[0] == -i` indicates that the `i`th parameter is invalid.
/// If `potrf_batched` returns [`Ok`] and `info[i] == k` is positive, the `i`th matrix is not positive definite and the Cholesky factorization failed at row `k`.
///
/// The other part of `A` is used as workspace.
/// For example, if `fill_mode` is [`FillMode::Upper`], upper triangle of `A` contains Cholesky factor `U` and lower triangle of `A` is destroyed after `potrf_batched`.
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions, leading dimension, or batch size are invalid, or if
/// cuSOLVER reports an internal failure.
pub fn dpotrf_batched(
    ctx: &Context,
    fill_mode: FillMode,
    n: usize,
    a: BatchedMatrixRef<'_, f64>,
    info: &mut DeviceMemory<i32>,
) -> Result<()> {
    ctx.bind()?;
    validate_batched_square_matrix_pointers(n, a)?;
    require_info_entries(info, a.len())?;
    unsafe {
        try_ffi!(sys::cusolverDnDpotrfBatched(
            ctx.as_raw(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_mut_ptr(),
            to_i32(a.leading_dimension, "lda")?,
            info.as_mut_ptr().cast(),
            to_i32(a.len(), "batch_size")?,
        ))?;
    }
    Ok(())
}

/// The S and D data types are real valued single and double precision, respectively.
///
/// The C and Z data types are complex valued single and double precision, respectively.
///
/// Computes the Cholesky factorization of a sequence of Hermitian positive-definite matrices.
///
/// Each `a[i]` for `i = 0, 1, ..., batch_size - 1` is a $n \times n$ Hermitian matrix, only lower or upper part is meaningful.
/// `fill_mode` indicates which part of the matrix is used.
///
/// If `fill_mode` is [`FillMode::Lower`], only the lower triangular part of `A` is processed and replaced by the lower triangular Cholesky factor `L`.
///
/// If `fill_mode` is [`FillMode::Upper`], only the upper triangular part of `A` is processed and replaced by the upper triangular Cholesky factor `U`.
///
/// If Cholesky factorization failed, that is, some leading minor of `A` is not positive definite, or equivalently some diagonal elements of `L` or `U` are not real.
/// `info` contains one entry per matrix and reports the smallest leading minor of `A` that is not positive definite.
///
/// `info` must have one integer entry for each matrix in the batch.
/// If cuSOLVER reports [`crate::error::Status::InvalidValue`], `info[0] == -i` indicates that the `i`th parameter is invalid.
/// If `potrf_batched` returns [`Ok`] and `info[i] == k` is positive, the `i`th matrix is not positive definite and the Cholesky factorization failed at row `k`.
///
/// The other part of `A` is used as workspace.
/// For example, if `fill_mode` is [`FillMode::Upper`], upper triangle of `A` contains Cholesky factor `U` and lower triangle of `A` is destroyed after `potrf_batched`.
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions, leading dimension, or batch size are invalid, or if
/// cuSOLVER reports an internal failure.
pub fn cpotrf_batched(
    ctx: &Context,
    fill_mode: FillMode,
    n: usize,
    a: BatchedMatrixRef<'_, Complex32>,
    info: &mut DeviceMemory<i32>,
) -> Result<()> {
    ctx.bind()?;
    validate_batched_square_matrix_pointers(n, a)?;
    require_info_entries(info, a.len())?;
    unsafe {
        try_ffi!(sys::cusolverDnCpotrfBatched(
            ctx.as_raw(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_mut_ptr().cast(),
            to_i32(a.leading_dimension, "lda")?,
            info.as_mut_ptr().cast(),
            to_i32(a.len(), "batch_size")?,
        ))?;
    }
    Ok(())
}

/// The S and D data types are real valued single and double precision, respectively.
///
/// The C and Z data types are complex valued single and double precision, respectively.
///
/// Computes the Cholesky factorization of a sequence of Hermitian positive-definite matrices.
///
/// Each `a[i]` for `i = 0, 1, ..., batch_size - 1` is a $n \times n$ Hermitian matrix, only lower or upper part is meaningful.
/// `fill_mode` indicates which part of the matrix is used.
///
/// If `fill_mode` is [`FillMode::Lower`], only the lower triangular part of `A` is processed and replaced by the lower triangular Cholesky factor `L`.
///
/// If `fill_mode` is [`FillMode::Upper`], only the upper triangular part of `A` is processed and replaced by the upper triangular Cholesky factor `U`.
///
/// If Cholesky factorization failed, that is, some leading minor of `A` is not positive definite, or equivalently some diagonal elements of `L` or `U` are not real.
/// `info` contains one entry per matrix and reports the smallest leading minor of `A` that is not positive definite.
///
/// `info` must have one integer entry for each matrix in the batch.
/// If cuSOLVER reports [`crate::error::Status::InvalidValue`], `info[0] == -i` indicates that the `i`th parameter is invalid.
/// If `potrf_batched` returns [`Ok`] and `info[i] == k` is positive, the `i`th matrix is not positive definite and the Cholesky factorization failed at row `k`.
///
/// The other part of `A` is used as workspace.
/// For example, if `fill_mode` is [`FillMode::Upper`], upper triangle of `A` contains Cholesky factor `U` and lower triangle of `A` is destroyed after `potrf_batched`.
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions, leading dimension, or batch size are invalid, or if
/// cuSOLVER reports an internal failure.
pub fn zpotrf_batched(
    ctx: &Context,
    fill_mode: FillMode,
    n: usize,
    a: BatchedMatrixRef<'_, Complex64>,
    info: &mut DeviceMemory<i32>,
) -> Result<()> {
    ctx.bind()?;
    validate_batched_square_matrix_pointers(n, a)?;
    require_info_entries(info, a.len())?;
    unsafe {
        try_ffi!(sys::cusolverDnZpotrfBatched(
            ctx.as_raw(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_mut_ptr().cast(),
            to_i32(a.leading_dimension, "lda")?,
            info.as_mut_ptr().cast(),
            to_i32(a.len(), "batch_size")?,
        ))?;
    }
    Ok(())
}

/// Solves a sequence of linear systems
///
/// where each `a[i]` for `i = 0, 1, ..., batch_size - 1` is a $n \times n$ Hermitian matrix, only lower or upper part is meaningful.
/// `fill_mode` indicates which part of the matrix is used.
///
/// Call `potrf_batched` first to factorize matrix `a[i]`.
/// If `fill_mode` is [`FillMode::Lower`], `A` is lower triangular Cholesky factor `L` corresponding to $A = L\cdot L^{H}$.
/// If `fill_mode` is [`FillMode::Upper`], `A` is upper triangular Cholesky factor `U` corresponding to $A = U^{H}\cdot U$.
///
/// The operation is in-place, that is, matrix `X` overwrites matrix `B` with the same leading dimension `ldb`.
///
/// `info` is a single status value for the whole batched call.
/// If the reported `info` value is `-i`, the `i`th parameter is invalid.
///
/// - only `nrhs=1` is supported.
///
/// - `info` from `potrf_batched` indicates whether each matrix is positive definite.
///   `info` from `potrsBatched` only reports invalid arguments for the batched call.
///
/// - the other part of `A` is used as a workspace.
///   For example, if `fill_mode` is [`FillMode::Upper`], upper triangle of `A` contains Cholesky factor `U` and lower triangle of `A` is destroyed after `potrsBatched`.
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions, right-hand-side count, leading dimensions, or batch
/// size are invalid, or if cuSOLVER reports an internal failure.
pub fn spotrs_batched(
    ctx: &Context,
    fill_mode: FillMode,
    n: usize,
    a: BatchedMatrixRef<'_, f32>,
    b: BatchedVectorRef<'_, f32>,
    info: &mut DeviceMemory<i32>,
) -> Result<()> {
    ctx.bind()?;
    validate_batched_square_matrix_pointers(n, a)?;
    validate_batched_vector_pointers(n, b)?;
    require_info_buffer(info)?;
    if a.len() != b.len() {
        return Err(Error::InvalidMatrixShape);
    }
    unsafe {
        try_ffi!(sys::cusolverDnSpotrsBatched(
            ctx.as_raw(),
            fill_mode.into(),
            to_i32(n, "n")?,
            1,
            a.as_mut_ptr(),
            to_i32(a.leading_dimension, "lda")?,
            b.as_mut_ptr(),
            to_i32(b.leading_dimension, "ldb")?,
            info.as_mut_ptr().cast(),
            to_i32(a.len(), "batch_size")?,
        ))?;
    }
    Ok(())
}

/// Solves a sequence of linear systems
///
/// where each `a[i]` for `i = 0, 1, ..., batch_size - 1` is a $n \times n$ Hermitian matrix, only lower or upper part is meaningful.
/// `fill_mode` indicates which part of the matrix is used.
///
/// Call `potrf_batched` first to factorize matrix `a[i]`.
/// If `fill_mode` is [`FillMode::Lower`], `A` is lower triangular Cholesky factor `L` corresponding to $A = L\cdot L^{H}$.
/// If `fill_mode` is [`FillMode::Upper`], `A` is upper triangular Cholesky factor `U` corresponding to $A = U^{H}\cdot U$.
///
/// The operation is in-place, that is, matrix `X` overwrites matrix `B` with the same leading dimension `ldb`.
///
/// `info` is a single status value for the whole batched call.
/// If the reported `info` value is `-i`, the `i`th parameter is invalid.
///
/// - only `nrhs=1` is supported.
///
/// - `info` from `potrf_batched` indicates whether each matrix is positive definite.
///   `info` from `potrsBatched` only reports invalid arguments for the batched call.
///
/// - the other part of `A` is used as a workspace.
///   For example, if `fill_mode` is [`FillMode::Upper`], upper triangle of `A` contains Cholesky factor `U` and lower triangle of `A` is destroyed after `potrsBatched`.
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions, right-hand-side count, leading dimensions, or batch
/// size are invalid, or if cuSOLVER reports an internal failure.
pub fn dpotrs_batched(
    ctx: &Context,
    fill_mode: FillMode,
    n: usize,
    a: BatchedMatrixRef<'_, f64>,
    b: BatchedVectorRef<'_, f64>,
    info: &mut DeviceMemory<i32>,
) -> Result<()> {
    ctx.bind()?;
    validate_batched_square_matrix_pointers(n, a)?;
    validate_batched_vector_pointers(n, b)?;
    require_info_buffer(info)?;
    if a.len() != b.len() {
        return Err(Error::InvalidMatrixShape);
    }
    unsafe {
        try_ffi!(sys::cusolverDnDpotrsBatched(
            ctx.as_raw(),
            fill_mode.into(),
            to_i32(n, "n")?,
            1,
            a.as_mut_ptr(),
            to_i32(a.leading_dimension, "lda")?,
            b.as_mut_ptr(),
            to_i32(b.leading_dimension, "ldb")?,
            info.as_mut_ptr().cast(),
            to_i32(a.len(), "batch_size")?,
        ))?;
    }
    Ok(())
}

/// Solves a sequence of linear systems
///
/// where each `a[i]` for `i = 0, 1, ..., batch_size - 1` is a $n \times n$ Hermitian matrix, only lower or upper part is meaningful.
/// `fill_mode` indicates which part of the matrix is used.
///
/// Call `potrf_batched` first to factorize matrix `a[i]`.
/// If `fill_mode` is [`FillMode::Lower`], `A` is lower triangular Cholesky factor `L` corresponding to $A = L\cdot L^{H}$.
/// If `fill_mode` is [`FillMode::Upper`], `A` is upper triangular Cholesky factor `U` corresponding to $A = U^{H}\cdot U$.
///
/// The operation is in-place, that is, matrix `X` overwrites matrix `B` with the same leading dimension `ldb`.
///
/// `info` is a single status value for the whole batched call.
/// If the reported `info` value is `-i`, the `i`th parameter is invalid.
///
/// - only `nrhs=1` is supported.
///
/// - `info` from `potrf_batched` indicates whether each matrix is positive definite.
///   `info` from `potrsBatched` only reports invalid arguments for the batched call.
///
/// - the other part of `A` is used as a workspace.
///   For example, if `fill_mode` is [`FillMode::Upper`], upper triangle of `A` contains Cholesky factor `U` and lower triangle of `A` is destroyed after `potrsBatched`.
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions, right-hand-side count, leading dimensions, or batch
/// size are invalid, or if cuSOLVER reports an internal failure.
pub fn zpotrs_batched(
    ctx: &Context,
    fill_mode: FillMode,
    n: usize,
    a: BatchedMatrixRef<'_, Complex64>,
    b: BatchedVectorRef<'_, Complex64>,
    info: &mut DeviceMemory<i32>,
) -> Result<()> {
    ctx.bind()?;
    validate_batched_square_matrix_pointers(n, a)?;
    validate_batched_vector_pointers(n, b)?;
    require_info_buffer(info)?;
    if a.len() != b.len() {
        return Err(Error::InvalidMatrixShape);
    }
    unsafe {
        try_ffi!(sys::cusolverDnZpotrsBatched(
            ctx.as_raw(),
            fill_mode.into(),
            to_i32(n, "n")?,
            1,
            a.as_mut_ptr().cast(),
            to_i32(a.leading_dimension, "lda")?,
            b.as_mut_ptr().cast(),
            to_i32(b.leading_dimension, "ldb")?,
            info.as_mut_ptr().cast(),
            to_i32(a.len(), "batch_size")?,
        ))?;
    }
    Ok(())
}

pub fn spotri_buffer_size(
    ctx: &Context,
    fill_mode: FillMode,
    n: usize,
    a: &mut DeviceMemory<f32>,
    lda: usize,
) -> Result<usize> {
    ctx.bind()?;
    validate_square_matrix(n, a.len(), lda)?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnSpotri_bufferSize(
            ctx.as_raw(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_mut_ptr().cast(),
            to_i32(lda, "lda")?,
            &raw mut lwork,
        ))?;
    }
    to_usize(lwork, "lwork")
}

pub fn dpotri_buffer_size(
    ctx: &Context,
    fill_mode: FillMode,
    n: usize,
    a: &mut DeviceMemory<f64>,
    lda: usize,
) -> Result<usize> {
    ctx.bind()?;
    validate_square_matrix(n, a.len(), lda)?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnDpotri_bufferSize(
            ctx.as_raw(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_mut_ptr().cast(),
            to_i32(lda, "lda")?,
            &raw mut lwork,
        ))?;
    }
    to_usize(lwork, "lwork")
}

pub fn cpotri_buffer_size(
    ctx: &Context,
    fill_mode: FillMode,
    n: usize,
    a: &mut DeviceMemory<Complex32>,
    lda: usize,
) -> Result<usize> {
    ctx.bind()?;
    validate_square_matrix(n, a.len(), lda)?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnCpotri_bufferSize(
            ctx.as_raw(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_mut_ptr().cast(),
            to_i32(lda, "lda")?,
            &raw mut lwork,
        ))?;
    }
    to_usize(lwork, "lwork")
}

pub fn zpotri_buffer_size(
    ctx: &Context,
    fill_mode: FillMode,
    n: usize,
    a: &mut DeviceMemory<Complex64>,
    lda: usize,
) -> Result<usize> {
    ctx.bind()?;
    validate_square_matrix(n, a.len(), lda)?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnZpotri_bufferSize(
            ctx.as_raw(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_mut_ptr().cast(),
            to_i32(lda, "lda")?,
            &raw mut lwork,
        ))?;
    }
    to_usize(lwork, "lwork")
}

/// Use the matching buffer-size helper to calculate the required workspace size.
///
/// The S and D data types are real valued single and double precision, respectively.
///
/// The C and Z data types are complex valued single and double precision, respectively.
///
/// Computes the inverse of a positive-definite matrix `A` using the Cholesky factorization
///
/// computed by `potrf()`.
///
/// `A` is an $n \times n$ matrix containing the triangular factor `L` or `U` computed by the Cholesky factorization.
/// Only the lower or upper part is meaningful, as selected by `fill_mode`.
/// The other triangular part is left unchanged.
///
/// If `fill_mode` is [`FillMode::Lower`], only the lower triangular part of `A` is processed and replaced by the lower triangular part of the inverse.
///
/// If `fill_mode` is [`FillMode::Upper`], only the upper triangular part of `A` is processed and replaced by the upper triangular part of the inverse.
///
/// Provide workspace through `workspace`.
/// Use the corresponding `*_buffer_size` helper to query the required workspace length.
///
/// If the inverse computation fails because a leading minor of `L` or `U` is singular, `dev_info` indicates the smallest leading minor that is not positive definite.
///
/// If the reported `dev_info` value is `-i`, the `i`th parameter is invalid.
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions or leading dimension are invalid, if the current GPU
/// architecture is unsupported, or if cuSOLVER reports an internal failure.
pub fn spotri(
    ctx: &Context,
    fill_mode: FillMode,
    n: usize,
    a: &mut DeviceMemory<f32>,
    lda: usize,
    workspace: &mut DeviceMemory<f32>,
    dev_info: &mut DeviceMemory<i32>,
) -> Result<()> {
    ctx.bind()?;
    validate_square_matrix(n, a.len(), lda)?;
    require_info_buffer(dev_info)?;
    let lwork = spotri_buffer_size(ctx, fill_mode, n, a, lda)?;
    require_workspace(workspace.len(), lwork)?;
    unsafe {
        try_ffi!(sys::cusolverDnSpotri(
            ctx.as_raw(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_mut_ptr().cast(),
            to_i32(lda, "lda")?,
            workspace.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast(),
        ))?;
    }
    Ok(())
}

/// Use the matching buffer-size helper to calculate the required workspace size.
///
/// The S and D data types are real valued single and double precision, respectively.
///
/// The C and Z data types are complex valued single and double precision, respectively.
///
/// Computes the inverse of a positive-definite matrix `A` using the Cholesky factorization
///
/// computed by `potrf()`.
///
/// `A` is an $n \times n$ matrix containing the triangular factor `L` or `U` computed by the Cholesky factorization.
/// Only the lower or upper part is meaningful, as selected by `fill_mode`.
/// The other triangular part is left unchanged.
///
/// If `fill_mode` is [`FillMode::Lower`], only the lower triangular part of `A` is processed and replaced by the lower triangular part of the inverse.
///
/// If `fill_mode` is [`FillMode::Upper`], only the upper triangular part of `A` is processed and replaced by the upper triangular part of the inverse.
///
/// Provide workspace through `workspace`.
/// Use the corresponding `*_buffer_size` helper to query the required workspace length.
///
/// If the inverse computation fails because a leading minor of `L` or `U` is singular, `dev_info` indicates the smallest leading minor that is not positive definite.
///
/// If the reported `dev_info` value is `-i`, the `i`th parameter is invalid.
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions or leading dimension are invalid, if the current GPU
/// architecture is unsupported, or if cuSOLVER reports an internal failure.
pub fn dpotri(
    ctx: &Context,
    fill_mode: FillMode,
    n: usize,
    a: &mut DeviceMemory<f64>,
    lda: usize,
    workspace: &mut DeviceMemory<f64>,
    dev_info: &mut DeviceMemory<i32>,
) -> Result<()> {
    ctx.bind()?;
    validate_square_matrix(n, a.len(), lda)?;
    require_info_buffer(dev_info)?;
    let lwork = dpotri_buffer_size(ctx, fill_mode, n, a, lda)?;
    require_workspace(workspace.len(), lwork)?;
    unsafe {
        try_ffi!(sys::cusolverDnDpotri(
            ctx.as_raw(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_mut_ptr().cast(),
            to_i32(lda, "lda")?,
            workspace.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast(),
        ))?;
    }
    Ok(())
}

/// Use the matching buffer-size helper to calculate the required workspace size.
///
/// The S and D data types are real valued single and double precision, respectively.
///
/// The C and Z data types are complex valued single and double precision, respectively.
///
/// Computes the inverse of a positive-definite matrix `A` using the Cholesky factorization
///
/// computed by `potrf()`.
///
/// `A` is an $n \times n$ matrix containing the triangular factor `L` or `U` computed by the Cholesky factorization.
/// Only the lower or upper part is meaningful, as selected by `fill_mode`.
/// The other triangular part is left unchanged.
///
/// If `fill_mode` is [`FillMode::Lower`], only the lower triangular part of `A` is processed and replaced by the lower triangular part of the inverse.
///
/// If `fill_mode` is [`FillMode::Upper`], only the upper triangular part of `A` is processed and replaced by the upper triangular part of the inverse.
///
/// Provide workspace through `workspace`.
/// Use the corresponding `*_buffer_size` helper to query the required workspace length.
///
/// If the inverse computation fails because a leading minor of `L` or `U` is singular, `dev_info` indicates the smallest leading minor that is not positive definite.
///
/// If the reported `dev_info` value is `-i`, the `i`th parameter is invalid.
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions or leading dimension are invalid, if the current GPU
/// architecture is unsupported, or if cuSOLVER reports an internal failure.
pub fn cpotri(
    ctx: &Context,
    fill_mode: FillMode,
    n: usize,
    a: &mut DeviceMemory<Complex32>,
    lda: usize,
    workspace: &mut DeviceMemory<Complex32>,
    dev_info: &mut DeviceMemory<i32>,
) -> Result<()> {
    ctx.bind()?;
    validate_square_matrix(n, a.len(), lda)?;
    require_info_buffer(dev_info)?;
    let lwork = cpotri_buffer_size(ctx, fill_mode, n, a, lda)?;
    require_workspace(workspace.len(), lwork)?;
    unsafe {
        try_ffi!(sys::cusolverDnCpotri(
            ctx.as_raw(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_mut_ptr().cast(),
            to_i32(lda, "lda")?,
            workspace.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast(),
        ))?;
    }
    Ok(())
}

/// Use the matching buffer-size helper to calculate the required workspace size.
///
/// The S and D data types are real valued single and double precision, respectively.
///
/// The C and Z data types are complex valued single and double precision, respectively.
///
/// Computes the inverse of a positive-definite matrix `A` using the Cholesky factorization
///
/// computed by `potrf()`.
///
/// `A` is an $n \times n$ matrix containing the triangular factor `L` or `U` computed by the Cholesky factorization.
/// Only the lower or upper part is meaningful, as selected by `fill_mode`.
/// The other triangular part is left unchanged.
///
/// If `fill_mode` is [`FillMode::Lower`], only the lower triangular part of `A` is processed and replaced by the lower triangular part of the inverse.
///
/// If `fill_mode` is [`FillMode::Upper`], only the upper triangular part of `A` is processed and replaced by the upper triangular part of the inverse.
///
/// Provide workspace through `workspace`.
/// Use the corresponding `*_buffer_size` helper to query the required workspace length.
///
/// If the inverse computation fails because a leading minor of `L` or `U` is singular, `dev_info` indicates the smallest leading minor that is not positive definite.
///
/// If the reported `dev_info` value is `-i`, the `i`th parameter is invalid.
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions or leading dimension are invalid, if the current GPU
/// architecture is unsupported, or if cuSOLVER reports an internal failure.
pub fn zpotri(
    ctx: &Context,
    fill_mode: FillMode,
    n: usize,
    a: &mut DeviceMemory<Complex64>,
    lda: usize,
    workspace: &mut DeviceMemory<Complex64>,
    dev_info: &mut DeviceMemory<i32>,
) -> Result<()> {
    ctx.bind()?;
    validate_square_matrix(n, a.len(), lda)?;
    require_info_buffer(dev_info)?;
    let lwork = zpotri_buffer_size(ctx, fill_mode, n, a, lda)?;
    require_workspace(workspace.len(), lwork)?;
    unsafe {
        try_ffi!(sys::cusolverDnZpotri(
            ctx.as_raw(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_mut_ptr().cast(),
            to_i32(lda, "lda")?,
            workspace.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast(),
        ))?;
    }
    Ok(())
}
