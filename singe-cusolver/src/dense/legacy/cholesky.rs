use std::ptr;

use singe_cuda::{
    types::{Complex32, Complex64},
    view::{DeviceOutputView, DeviceViewMut},
};

use crate::{
    context::Context,
    dense::validation::{
        require_info_buffer, require_info_entries, require_workspace,
        validate_batched_square_matrix_pointers, validate_batched_vector_pointers,
        validate_square_matrix,
    },
    error::{Error, Result},
    layout::{BatchedMatrixRef, BatchedVectorRef, MatrixLayout},
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
/// # Safety
///
/// Follow the [raw solver contracts](crate::dense#raw-solver-contracts).
pub unsafe fn spotrf_batched(
    context: &Context,
    fill_mode: FillMode,
    n: usize,
    a: BatchedMatrixRef<'_, f32>,
    mut info: DeviceOutputView<'_, i32>,
) -> Result<()> {
    context.bind()?;
    context.validate_region(&info)?;
    context.validate_region(&a.pointers)?;
    validate_batched_square_matrix_pointers(n, a)?;
    require_info_entries(info.len(), a.len())?;
    // SAFETY: `cusolverDnSpotrfBatched` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnSpotrfBatched(
            context.as_raw(),
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
/// # Safety
///
/// Follow the [raw solver contracts](crate::dense#raw-solver-contracts).
pub unsafe fn dpotrf_batched(
    context: &Context,
    fill_mode: FillMode,
    n: usize,
    a: BatchedMatrixRef<'_, f64>,
    mut info: DeviceOutputView<'_, i32>,
) -> Result<()> {
    context.bind()?;
    context.validate_region(&info)?;
    context.validate_region(&a.pointers)?;
    validate_batched_square_matrix_pointers(n, a)?;
    require_info_entries(info.len(), a.len())?;
    // SAFETY: `cusolverDnDpotrfBatched` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnDpotrfBatched(
            context.as_raw(),
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
/// # Safety
///
/// Follow the [raw solver contracts](crate::dense#raw-solver-contracts).
pub unsafe fn cpotrf_batched(
    context: &Context,
    fill_mode: FillMode,
    n: usize,
    a: BatchedMatrixRef<'_, Complex32>,
    mut info: DeviceOutputView<'_, i32>,
) -> Result<()> {
    context.bind()?;
    context.validate_region(&info)?;
    context.validate_region(&a.pointers)?;
    validate_batched_square_matrix_pointers(n, a)?;
    require_info_entries(info.len(), a.len())?;
    // SAFETY: `cusolverDnCpotrfBatched` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnCpotrfBatched(
            context.as_raw(),
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
/// # Safety
///
/// Follow the [raw solver contracts](crate::dense#raw-solver-contracts).
pub unsafe fn zpotrf_batched(
    context: &Context,
    fill_mode: FillMode,
    n: usize,
    a: BatchedMatrixRef<'_, Complex64>,
    mut info: DeviceOutputView<'_, i32>,
) -> Result<()> {
    context.bind()?;
    context.validate_region(&info)?;
    context.validate_region(&a.pointers)?;
    validate_batched_square_matrix_pointers(n, a)?;
    require_info_entries(info.len(), a.len())?;
    // SAFETY: `cusolverDnZpotrfBatched` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnZpotrfBatched(
            context.as_raw(),
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
/// # Safety
///
/// Follow the [raw solver contracts](crate::dense#raw-solver-contracts).
pub unsafe fn spotrs_batched(
    context: &Context,
    fill_mode: FillMode,
    n: usize,
    a: BatchedMatrixRef<'_, f32>,
    b: BatchedVectorRef<'_, f32>,
    mut info: DeviceOutputView<'_, i32>,
) -> Result<()> {
    context.bind()?;
    context.validate_region(&info)?;
    context.validate_region(&a.pointers)?;
    context.validate_region(&b.pointers)?;
    validate_batched_square_matrix_pointers(n, a)?;
    validate_batched_vector_pointers(n, b)?;
    require_info_buffer(info.len())?;
    if a.len() != b.len() {
        return Err(Error::InvalidMatrixShape);
    }
    // SAFETY: `cusolverDnSpotrsBatched` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnSpotrsBatched(
            context.as_raw(),
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
/// # Safety
///
/// Follow the [raw solver contracts](crate::dense#raw-solver-contracts).
pub unsafe fn dpotrs_batched(
    context: &Context,
    fill_mode: FillMode,
    n: usize,
    a: BatchedMatrixRef<'_, f64>,
    b: BatchedVectorRef<'_, f64>,
    mut info: DeviceOutputView<'_, i32>,
) -> Result<()> {
    context.bind()?;
    context.validate_region(&info)?;
    context.validate_region(&a.pointers)?;
    context.validate_region(&b.pointers)?;
    validate_batched_square_matrix_pointers(n, a)?;
    validate_batched_vector_pointers(n, b)?;
    require_info_buffer(info.len())?;
    if a.len() != b.len() {
        return Err(Error::InvalidMatrixShape);
    }
    // SAFETY: `cusolverDnDpotrsBatched` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnDpotrsBatched(
            context.as_raw(),
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
/// # Safety
///
/// Follow the [raw solver contracts](crate::dense#raw-solver-contracts).
pub unsafe fn zpotrs_batched(
    context: &Context,
    fill_mode: FillMode,
    n: usize,
    a: BatchedMatrixRef<'_, Complex64>,
    b: BatchedVectorRef<'_, Complex64>,
    mut info: DeviceOutputView<'_, i32>,
) -> Result<()> {
    context.bind()?;
    context.validate_region(&info)?;
    context.validate_region(&a.pointers)?;
    context.validate_region(&b.pointers)?;
    validate_batched_square_matrix_pointers(n, a)?;
    validate_batched_vector_pointers(n, b)?;
    require_info_buffer(info.len())?;
    if a.len() != b.len() {
        return Err(Error::InvalidMatrixShape);
    }
    // SAFETY: `cusolverDnZpotrsBatched` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnZpotrsBatched(
            context.as_raw(),
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

/// Solves one right-hand side for each complex single-precision Cholesky factor.
///
/// `info` produces one status for the call. Each right-hand side is overwritten
/// by its solution; the unused triangle of each factor may be used as scratch.
///
/// # Safety
///
/// Follow the [raw solver contracts](crate::dense#raw-solver-contracts).
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a device buffer belongs to another CUDA context, matrix dimensions, leading dimensions, or storage capacities are invalid, checked size arithmetic or native integer conversion fails, the status output capacity is insufficient, batch metadata or related pointer-table lengths are invalid, or cuSOLVER rejects the configuration or returns an error while submitting execution.
pub unsafe fn cpotrs_batched(
    context: &Context,
    fill_mode: FillMode,
    n: usize,
    a: BatchedMatrixRef<'_, Complex32>,
    b: BatchedVectorRef<'_, Complex32>,
    mut info: DeviceOutputView<'_, i32>,
) -> Result<()> {
    context.bind()?;
    context.validate_region(&info)?;
    context.validate_region(&a.pointers)?;
    context.validate_region(&b.pointers)?;
    validate_batched_square_matrix_pointers(n, a)?;
    validate_batched_vector_pointers(n, b)?;
    require_info_buffer(info.len())?;
    if a.len() != b.len() {
        return Err(Error::InvalidMatrixShape);
    }
    // SAFETY: `cusolverDnCpotrsBatched` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnCpotrsBatched(
            context.as_raw(),
            fill_mode.into(),
            to_i32(n, "n")?,
            1,
            a.as_mut_ptr().cast(),
            to_i32(a.leading_dimension, "lda")?,
            b.as_mut_ptr().cast(),
            to_i32(b.leading_dimension, "ldb")?,
            info.as_mut_ptr(),
            to_i32(a.len(), "batch_size")?,
        ))?;
    }
    Ok(())
}

/// Queries device workspace capacity in elements for `f32` Cholesky inversion.
///
/// Uses shape and layout metadata without borrowing matrix storage.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn spotri_buffer_size(
    context: &Context,
    fill_mode: FillMode,
    n: usize,
    lda: usize,
) -> Result<usize> {
    context.bind()?;
    MatrixLayout::new(lda).validate(n, n)?;
    let mut lwork = 0;
    // SAFETY: `cusolverDnSpotri_bufferSize` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnSpotri_bufferSize(
            context.as_raw(),
            fill_mode.into(),
            to_i32(n, "n")?,
            ptr::null_mut::<u8>().cast(),
            to_i32(lda, "lda")?,
            &raw mut lwork,
        ))?;
    }
    to_usize(lwork, "lwork")
}

/// Queries device workspace capacity in elements for `f64` Cholesky inversion.
///
/// Uses shape and layout metadata without borrowing matrix storage.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn dpotri_buffer_size(
    context: &Context,
    fill_mode: FillMode,
    n: usize,
    lda: usize,
) -> Result<usize> {
    context.bind()?;
    MatrixLayout::new(lda).validate(n, n)?;
    let mut lwork = 0;
    // SAFETY: `cusolverDnDpotri_bufferSize` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnDpotri_bufferSize(
            context.as_raw(),
            fill_mode.into(),
            to_i32(n, "n")?,
            ptr::null_mut::<u8>().cast(),
            to_i32(lda, "lda")?,
            &raw mut lwork,
        ))?;
    }
    to_usize(lwork, "lwork")
}

/// Queries device workspace capacity in elements for complex single-precision Cholesky inversion.
///
/// Uses shape and layout metadata without borrowing matrix storage.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn cpotri_buffer_size(
    context: &Context,
    fill_mode: FillMode,
    n: usize,
    lda: usize,
) -> Result<usize> {
    context.bind()?;
    MatrixLayout::new(lda).validate(n, n)?;
    let mut lwork = 0;
    // SAFETY: `cusolverDnCpotri_bufferSize` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnCpotri_bufferSize(
            context.as_raw(),
            fill_mode.into(),
            to_i32(n, "n")?,
            ptr::null_mut::<u8>().cast(),
            to_i32(lda, "lda")?,
            &raw mut lwork,
        ))?;
    }
    to_usize(lwork, "lwork")
}

/// Queries device workspace capacity in elements for complex double-precision Cholesky inversion.
///
/// Uses shape and layout metadata without borrowing matrix storage.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn zpotri_buffer_size(
    context: &Context,
    fill_mode: FillMode,
    n: usize,
    lda: usize,
) -> Result<usize> {
    context.bind()?;
    MatrixLayout::new(lda).validate(n, n)?;
    let mut lwork = 0;
    // SAFETY: `cusolverDnZpotri_bufferSize` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnZpotri_bufferSize(
            context.as_raw(),
            fill_mode.into(),
            to_i32(n, "n")?,
            ptr::null_mut::<u8>().cast(),
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
/// # Safety
///
/// Follow the [raw solver contracts](crate::dense#raw-solver-contracts).
pub unsafe fn spotri(
    context: &Context,
    fill_mode: FillMode,
    n: usize,
    mut a: DeviceViewMut<'_, f32>,
    lda: usize,
    mut workspace: DeviceOutputView<'_, f32>,
    mut dev_info: DeviceOutputView<'_, i32>,
) -> Result<()> {
    context.bind()?;
    context.validate_region(&a)?;
    context.validate_region(&workspace)?;
    context.validate_region(&dev_info)?;
    validate_square_matrix(n, a.len(), lda)?;
    require_info_buffer(dev_info.len())?;
    let lwork = spotri_buffer_size(context, fill_mode, n, lda)?;
    require_workspace(workspace.len(), lwork)?;
    // SAFETY: `cusolverDnSpotri` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnSpotri(
            context.as_raw(),
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
/// # Safety
///
/// Follow the [raw solver contracts](crate::dense#raw-solver-contracts).
pub unsafe fn dpotri(
    context: &Context,
    fill_mode: FillMode,
    n: usize,
    mut a: DeviceViewMut<'_, f64>,
    lda: usize,
    mut workspace: DeviceOutputView<'_, f64>,
    mut dev_info: DeviceOutputView<'_, i32>,
) -> Result<()> {
    context.bind()?;
    context.validate_region(&a)?;
    context.validate_region(&workspace)?;
    context.validate_region(&dev_info)?;
    validate_square_matrix(n, a.len(), lda)?;
    require_info_buffer(dev_info.len())?;
    let lwork = dpotri_buffer_size(context, fill_mode, n, lda)?;
    require_workspace(workspace.len(), lwork)?;
    // SAFETY: `cusolverDnDpotri` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnDpotri(
            context.as_raw(),
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
/// # Safety
///
/// Follow the [raw solver contracts](crate::dense#raw-solver-contracts).
pub unsafe fn cpotri(
    context: &Context,
    fill_mode: FillMode,
    n: usize,
    mut a: DeviceViewMut<'_, Complex32>,
    lda: usize,
    mut workspace: DeviceOutputView<'_, Complex32>,
    mut dev_info: DeviceOutputView<'_, i32>,
) -> Result<()> {
    context.bind()?;
    context.validate_region(&a)?;
    context.validate_region(&workspace)?;
    context.validate_region(&dev_info)?;
    validate_square_matrix(n, a.len(), lda)?;
    require_info_buffer(dev_info.len())?;
    let lwork = cpotri_buffer_size(context, fill_mode, n, lda)?;
    require_workspace(workspace.len(), lwork)?;
    // SAFETY: `cusolverDnCpotri` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnCpotri(
            context.as_raw(),
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
/// # Safety
///
/// Follow the [raw solver contracts](crate::dense#raw-solver-contracts).
pub unsafe fn zpotri(
    context: &Context,
    fill_mode: FillMode,
    n: usize,
    mut a: DeviceViewMut<'_, Complex64>,
    lda: usize,
    mut workspace: DeviceOutputView<'_, Complex64>,
    mut dev_info: DeviceOutputView<'_, i32>,
) -> Result<()> {
    context.bind()?;
    context.validate_region(&a)?;
    context.validate_region(&workspace)?;
    context.validate_region(&dev_info)?;
    validate_square_matrix(n, a.len(), lda)?;
    require_info_buffer(dev_info.len())?;
    let lwork = zpotri_buffer_size(context, fill_mode, n, lda)?;
    require_workspace(workspace.len(), lwork)?;
    // SAFETY: `cusolverDnZpotri` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnZpotri(
            context.as_raw(),
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
