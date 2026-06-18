use singe_cuda::{
    memory::DeviceMemory,
    types::{Complex32, Complex64},
};

use crate::{
    context::Context,
    dense::validation::{
        require_info_buffer, require_pivot_buffer, require_workspace, validate_square_matrix,
    },
    error::Result,
    sys, try_ffi,
    types::FillMode,
    utility::{to_i32, to_usize},
};

pub fn ssytrf_buffer_size(
    ctx: &Context,
    n: usize,
    a: &mut DeviceMemory<f32>,
    lda: usize,
) -> Result<usize> {
    ctx.bind()?;
    validate_square_matrix(n, a.len(), lda)?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnSsytrf_bufferSize(
            ctx.as_raw(),
            to_i32(n, "n")?,
            a.as_mut_ptr().cast(),
            to_i32(lda, "lda")?,
            &raw mut lwork,
        ))?;
    }
    to_usize(lwork, "lwork")
}

pub fn dsytrf_buffer_size(
    ctx: &Context,
    n: usize,
    a: &mut DeviceMemory<f64>,
    lda: usize,
) -> Result<usize> {
    ctx.bind()?;
    validate_square_matrix(n, a.len(), lda)?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnDsytrf_bufferSize(
            ctx.as_raw(),
            to_i32(n, "n")?,
            a.as_mut_ptr().cast(),
            to_i32(lda, "lda")?,
            &raw mut lwork,
        ))?;
    }
    to_usize(lwork, "lwork")
}

pub fn csytrf_buffer_size(
    ctx: &Context,
    n: usize,
    a: &mut DeviceMemory<Complex32>,
    lda: usize,
) -> Result<usize> {
    ctx.bind()?;
    validate_square_matrix(n, a.len(), lda)?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnCsytrf_bufferSize(
            ctx.as_raw(),
            to_i32(n, "n")?,
            a.as_mut_ptr().cast(),
            to_i32(lda, "lda")?,
            &raw mut lwork,
        ))?;
    }
    to_usize(lwork, "lwork")
}

pub fn zsytrf_buffer_size(
    ctx: &Context,
    n: usize,
    a: &mut DeviceMemory<Complex64>,
    lda: usize,
) -> Result<usize> {
    ctx.bind()?;
    validate_square_matrix(n, a.len(), lda)?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnZsytrf_bufferSize(
            ctx.as_raw(),
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
/// Computes the factorization of a symmetric indefinite matrix using the Bunch-Kaufman diagonal pivoting.
///
/// `A` is a $n \times n$ symmetric matrix, only lower or upper part is meaningful.
/// `fill_mode` indicates which part of the matrix is used.
/// If `pivots` is `None`, no pivoting is performed, which is not numerically stable.
///
/// If `fill_mode` is [`FillMode::Lower`], only the lower triangular part of `A` is processed and replaced by the lower triangular factor `L` and block diagonal matrix `D`.
/// Each block of `D` is either 1x1 or 2x2 block, depending on pivoting.
///
/// If `fill_mode` is [`FillMode::Upper`], only the upper triangular part of `A` is processed and replaced by the upper triangular factor `U` and block diagonal matrix `D`.
///
/// Provide workspace through `workspace`.
/// Use the corresponding `*_buffer_size` helper to query the required workspace length.
/// The workspace size in bytes is `size_of::<T>() * lwork`.
/// When no pivoting is performed, the other triangular part of the input matrix `A` is used as workspace.
///
/// If Bunch-Kaufman factorization failed, that is, `A` is singular,
/// `dev_info = i` indicates `D(i, i) = 0`.
///
/// If the reported `dev_info` value is `-i`, the `i`th parameter is invalid.
///
/// `pivots` contains the pivoting sequence.
/// If `pivots[i] = k` with `k > 0`, `D(i, i)` is a 1x1 block, and row/column `i` of `A`
/// is interchanged with row/column `k`.
/// If `fill_mode` is [`FillMode::Upper`] and `pivots[i - 1] = pivots[i] = -m` with `m > 0`,
/// `D(i-1:i,i-1:i)` is a 2x2 block, and row/column `i - 1` is interchanged
/// with row/column `m`.
/// If `fill_mode` is [`FillMode::Lower`] and `pivots[i + 1] = pivots[i] = -m` with `m > 0`,
/// `D(i:i+1,i:i+1)` is a 2x2 block, and row/column `i + 1` is interchanged
/// with row/column `m`.
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions or leading dimension are invalid, if the current GPU
/// architecture is unsupported, or if cuSOLVER reports an internal failure.
pub fn ssytrf(
    ctx: &Context,
    fill_mode: FillMode,
    n: usize,
    a: &mut DeviceMemory<f32>,
    lda: usize,
    pivots: Option<&mut DeviceMemory<i32>>,
    workspace: &mut DeviceMemory<f32>,
    dev_info: &mut DeviceMemory<i32>,
) -> Result<()> {
    ctx.bind()?;
    validate_square_matrix(n, a.len(), lda)?;
    if let Some(pivots) = pivots.as_ref() {
        require_pivot_buffer(pivots, n)?;
    }
    require_info_buffer(dev_info)?;
    let lwork = ssytrf_buffer_size(ctx, n, a, lda)?;
    require_workspace(workspace.len(), lwork)?;
    unsafe {
        try_ffi!(sys::cusolverDnSsytrf(
            ctx.as_raw(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_mut_ptr().cast(),
            to_i32(lda, "lda")?,
            pivots.map_or(std::ptr::null_mut(), |p| p.as_mut_ptr()),
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
/// Computes the factorization of a symmetric indefinite matrix using the Bunch-Kaufman diagonal pivoting.
///
/// `A` is a $n \times n$ symmetric matrix, only lower or upper part is meaningful.
/// `fill_mode` indicates which part of the matrix is used.
/// If `pivots` is `None`, no pivoting is performed, which is not numerically stable.
///
/// If `fill_mode` is [`FillMode::Lower`], only the lower triangular part of `A` is processed and replaced by the lower triangular factor `L` and block diagonal matrix `D`.
/// Each block of `D` is either 1x1 or 2x2 block, depending on pivoting.
///
/// If `fill_mode` is [`FillMode::Upper`], only the upper triangular part of `A` is processed and replaced by the upper triangular factor `U` and block diagonal matrix `D`.
///
/// Provide workspace through `workspace`.
/// Use the corresponding `*_buffer_size` helper to query the required workspace length.
/// The workspace size in bytes is `size_of::<T>() * lwork`.
/// When no pivoting is performed, the other triangular part of the input matrix `A` is used as workspace.
///
/// If Bunch-Kaufman factorization failed, that is, `A` is singular,
/// `dev_info = i` indicates `D(i, i) = 0`.
///
/// If the reported `dev_info` value is `-i`, the `i`th parameter is invalid.
///
/// `pivots` contains the pivoting sequence.
/// If `pivots[i] = k` with `k > 0`, `D(i, i)` is a 1x1 block, and row/column `i` of `A`
/// is interchanged with row/column `k`.
/// If `fill_mode` is [`FillMode::Upper`] and `pivots[i - 1] = pivots[i] = -m` with `m > 0`,
/// `D(i-1:i,i-1:i)` is a 2x2 block, and row/column `i - 1` is interchanged
/// with row/column `m`.
/// If `fill_mode` is [`FillMode::Lower`] and `pivots[i + 1] = pivots[i] = -m` with `m > 0`,
/// `D(i:i+1,i:i+1)` is a 2x2 block, and row/column `i + 1` is interchanged
/// with row/column `m`.
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions or leading dimension are invalid, if the current GPU
/// architecture is unsupported, or if cuSOLVER reports an internal failure.
pub fn dsytrf(
    ctx: &Context,
    fill_mode: FillMode,
    n: usize,
    a: &mut DeviceMemory<f64>,
    lda: usize,
    pivots: Option<&mut DeviceMemory<i32>>,
    workspace: &mut DeviceMemory<f64>,
    dev_info: &mut DeviceMemory<i32>,
) -> Result<()> {
    ctx.bind()?;
    validate_square_matrix(n, a.len(), lda)?;
    if let Some(pivots) = pivots.as_ref() {
        require_pivot_buffer(pivots, n)?;
    }
    require_info_buffer(dev_info)?;
    let lwork = dsytrf_buffer_size(ctx, n, a, lda)?;
    require_workspace(workspace.len(), lwork)?;
    unsafe {
        try_ffi!(sys::cusolverDnDsytrf(
            ctx.as_raw(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_mut_ptr().cast(),
            to_i32(lda, "lda")?,
            pivots.map_or(std::ptr::null_mut(), |p| p.as_mut_ptr()),
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
/// Computes the factorization of a symmetric indefinite matrix using the Bunch-Kaufman diagonal pivoting.
///
/// `A` is a $n \times n$ symmetric matrix, only lower or upper part is meaningful.
/// `fill_mode` indicates which part of the matrix is used.
/// If `pivots` is `None`, no pivoting is performed, which is not numerically stable.
///
/// If `fill_mode` is [`FillMode::Lower`], only the lower triangular part of `A` is processed and replaced by the lower triangular factor `L` and block diagonal matrix `D`.
/// Each block of `D` is either 1x1 or 2x2 block, depending on pivoting.
///
/// If `fill_mode` is [`FillMode::Upper`], only the upper triangular part of `A` is processed and replaced by the upper triangular factor `U` and block diagonal matrix `D`.
///
/// Provide workspace through `workspace`.
/// Use the corresponding `*_buffer_size` helper to query the required workspace length.
/// The workspace size in bytes is `size_of::<T>() * lwork`.
/// When no pivoting is performed, the other triangular part of the input matrix `A` is used as workspace.
///
/// If Bunch-Kaufman factorization failed, that is, `A` is singular,
/// `dev_info = i` indicates `D(i, i) = 0`.
///
/// If the reported `dev_info` value is `-i`, the `i`th parameter is invalid.
///
/// `pivots` contains the pivoting sequence.
/// If `pivots[i] = k` with `k > 0`, `D(i, i)` is a 1x1 block, and row/column `i` of `A`
/// is interchanged with row/column `k`.
/// If `fill_mode` is [`FillMode::Upper`] and `pivots[i - 1] = pivots[i] = -m` with `m > 0`,
/// `D(i-1:i,i-1:i)` is a 2x2 block, and row/column `i - 1` is interchanged
/// with row/column `m`.
/// If `fill_mode` is [`FillMode::Lower`] and `pivots[i + 1] = pivots[i] = -m` with `m > 0`,
/// `D(i:i+1,i:i+1)` is a 2x2 block, and row/column `i + 1` is interchanged
/// with row/column `m`.
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions or leading dimension are invalid, if the current GPU
/// architecture is unsupported, or if cuSOLVER reports an internal failure.
pub fn csytrf(
    ctx: &Context,
    fill_mode: FillMode,
    n: usize,
    a: &mut DeviceMemory<Complex32>,
    lda: usize,
    pivots: Option<&mut DeviceMemory<i32>>,
    workspace: &mut DeviceMemory<Complex32>,
    dev_info: &mut DeviceMemory<i32>,
) -> Result<()> {
    ctx.bind()?;
    validate_square_matrix(n, a.len(), lda)?;
    if let Some(pivots) = pivots.as_ref() {
        require_pivot_buffer(pivots, n)?;
    }
    require_info_buffer(dev_info)?;
    let lwork = csytrf_buffer_size(ctx, n, a, lda)?;
    require_workspace(workspace.len(), lwork)?;
    unsafe {
        try_ffi!(sys::cusolverDnCsytrf(
            ctx.as_raw(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_mut_ptr().cast(),
            to_i32(lda, "lda")?,
            pivots.map_or(std::ptr::null_mut(), |p| p.as_mut_ptr()),
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
/// Computes the factorization of a symmetric indefinite matrix using the Bunch-Kaufman diagonal pivoting.
///
/// `A` is a $n \times n$ symmetric matrix, only lower or upper part is meaningful.
/// `fill_mode` indicates which part of the matrix is used.
/// If `pivots` is `None`, no pivoting is performed, which is not numerically stable.
///
/// If `fill_mode` is [`FillMode::Lower`], only the lower triangular part of `A` is processed and replaced by the lower triangular factor `L` and block diagonal matrix `D`.
/// Each block of `D` is either 1x1 or 2x2 block, depending on pivoting.
///
/// If `fill_mode` is [`FillMode::Upper`], only the upper triangular part of `A` is processed and replaced by the upper triangular factor `U` and block diagonal matrix `D`.
///
/// Provide workspace through `workspace`.
/// Use the corresponding `*_buffer_size` helper to query the required workspace length.
/// The workspace size in bytes is `size_of::<T>() * lwork`.
/// When no pivoting is performed, the other triangular part of the input matrix `A` is used as workspace.
///
/// If Bunch-Kaufman factorization failed, that is, `A` is singular,
/// `dev_info = i` indicates `D(i, i) = 0`.
///
/// If the reported `dev_info` value is `-i`, the `i`th parameter is invalid.
///
/// `pivots` contains the pivoting sequence.
/// If `pivots[i] = k` with `k > 0`, `D(i, i)` is a 1x1 block, and row/column `i` of `A`
/// is interchanged with row/column `k`.
/// If `fill_mode` is [`FillMode::Upper`] and `pivots[i - 1] = pivots[i] = -m` with `m > 0`,
/// `D(i-1:i,i-1:i)` is a 2x2 block, and row/column `i - 1` is interchanged
/// with row/column `m`.
/// If `fill_mode` is [`FillMode::Lower`] and `pivots[i + 1] = pivots[i] = -m` with `m > 0`,
/// `D(i:i+1,i:i+1)` is a 2x2 block, and row/column `i + 1` is interchanged
/// with row/column `m`.
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions or leading dimension are invalid, if the current GPU
/// architecture is unsupported, or if cuSOLVER reports an internal failure.
pub fn zsytrf(
    ctx: &Context,
    fill_mode: FillMode,
    n: usize,
    a: &mut DeviceMemory<Complex64>,
    lda: usize,
    pivots: Option<&mut DeviceMemory<i32>>,
    workspace: &mut DeviceMemory<Complex64>,
    dev_info: &mut DeviceMemory<i32>,
) -> Result<()> {
    ctx.bind()?;
    validate_square_matrix(n, a.len(), lda)?;
    if let Some(pivots) = pivots.as_ref() {
        require_pivot_buffer(pivots, n)?;
    }
    require_info_buffer(dev_info)?;
    let lwork = zsytrf_buffer_size(ctx, n, a, lda)?;
    require_workspace(workspace.len(), lwork)?;
    unsafe {
        try_ffi!(sys::cusolverDnZsytrf(
            ctx.as_raw(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_mut_ptr().cast(),
            to_i32(lda, "lda")?,
            pivots.map_or(std::ptr::null_mut(), |p| p.as_mut_ptr()),
            workspace.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast(),
        ))?;
    }
    Ok(())
}
