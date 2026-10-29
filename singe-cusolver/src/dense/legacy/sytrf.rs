use std::ptr;

use singe_cuda::{
    types::{Complex32, Complex64},
    view::{DeviceOutputView, DeviceViewMut},
};

use crate::{
    context::Context,
    dense::validation::{
        require_info_buffer, require_pivot_buffer, require_workspace, validate_square_matrix,
    },
    error::Result,
    layout::MatrixLayout,
    sys, try_ffi,
    types::FillMode,
    utility::{to_i32, to_usize},
};

/// Queries device workspace capacity in elements for `f32` symmetric indefinite factorization.
///
/// Uses shape and layout metadata without borrowing matrix storage.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn ssytrf_buffer_size(context: &Context, n: usize, lda: usize) -> Result<usize> {
    context.bind()?;
    MatrixLayout::new(lda).validate(n, n)?;
    let mut lwork = 0;
    // SAFETY: `cusolverDnSsytrf_bufferSize` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnSsytrf_bufferSize(
            context.as_raw(),
            to_i32(n, "n")?,
            ptr::null_mut::<u8>().cast(),
            to_i32(lda, "lda")?,
            &raw mut lwork,
        ))?;
    }
    to_usize(lwork, "lwork")
}

/// Queries device workspace capacity in elements for `f64` symmetric indefinite factorization.
///
/// Uses shape and layout metadata without borrowing matrix storage.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn dsytrf_buffer_size(context: &Context, n: usize, lda: usize) -> Result<usize> {
    context.bind()?;
    MatrixLayout::new(lda).validate(n, n)?;
    let mut lwork = 0;
    // SAFETY: `cusolverDnDsytrf_bufferSize` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnDsytrf_bufferSize(
            context.as_raw(),
            to_i32(n, "n")?,
            ptr::null_mut::<u8>().cast(),
            to_i32(lda, "lda")?,
            &raw mut lwork,
        ))?;
    }
    to_usize(lwork, "lwork")
}

/// Queries device workspace capacity in elements for complex single-precision symmetric indefinite factorization.
///
/// Uses shape and layout metadata without borrowing matrix storage.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn csytrf_buffer_size(context: &Context, n: usize, lda: usize) -> Result<usize> {
    context.bind()?;
    MatrixLayout::new(lda).validate(n, n)?;
    let mut lwork = 0;
    // SAFETY: `cusolverDnCsytrf_bufferSize` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnCsytrf_bufferSize(
            context.as_raw(),
            to_i32(n, "n")?,
            ptr::null_mut::<u8>().cast(),
            to_i32(lda, "lda")?,
            &raw mut lwork,
        ))?;
    }
    to_usize(lwork, "lwork")
}

/// Queries device workspace capacity in elements for complex double-precision symmetric indefinite factorization.
///
/// Uses shape and layout metadata without borrowing matrix storage.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn zsytrf_buffer_size(context: &Context, n: usize, lda: usize) -> Result<usize> {
    context.bind()?;
    MatrixLayout::new(lda).validate(n, n)?;
    let mut lwork = 0;
    // SAFETY: `cusolverDnZsytrf_bufferSize` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnZsytrf_bufferSize(
            context.as_raw(),
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
/// # Safety
///
/// Follow the [raw solver contracts](crate::dense#raw-solver-contracts).
pub unsafe fn ssytrf(
    context: &Context,
    fill_mode: FillMode,
    n: usize,
    mut a: DeviceViewMut<'_, f32>,
    lda: usize,
    pivots: Option<DeviceOutputView<'_, i32>>,
    mut workspace: DeviceOutputView<'_, f32>,
    mut dev_info: DeviceOutputView<'_, i32>,
) -> Result<()> {
    context.bind()?;
    context.validate_region(&a)?;
    context.validate_region(&workspace)?;
    context.validate_region(&dev_info)?;
    if let Some(value) = pivots.as_ref() {
        context.validate_region(value)?;
    }
    validate_square_matrix(n, a.len(), lda)?;
    if let Some(pivots) = pivots.as_ref() {
        require_pivot_buffer(pivots.len(), n)?;
    }
    require_info_buffer(dev_info.len())?;
    let lwork = ssytrf_buffer_size(context, n, lda)?;
    require_workspace(workspace.len(), lwork)?;
    // SAFETY: `cusolverDnSsytrf` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnSsytrf(
            context.as_raw(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_mut_ptr().cast(),
            to_i32(lda, "lda")?,
            pivots.map_or(ptr::null_mut(), |mut p| p.as_mut_ptr()),
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
/// # Safety
///
/// Follow the [raw solver contracts](crate::dense#raw-solver-contracts).
pub unsafe fn dsytrf(
    context: &Context,
    fill_mode: FillMode,
    n: usize,
    mut a: DeviceViewMut<'_, f64>,
    lda: usize,
    pivots: Option<DeviceOutputView<'_, i32>>,
    mut workspace: DeviceOutputView<'_, f64>,
    mut dev_info: DeviceOutputView<'_, i32>,
) -> Result<()> {
    context.bind()?;
    context.validate_region(&a)?;
    context.validate_region(&workspace)?;
    context.validate_region(&dev_info)?;
    if let Some(value) = pivots.as_ref() {
        context.validate_region(value)?;
    }
    validate_square_matrix(n, a.len(), lda)?;
    if let Some(pivots) = pivots.as_ref() {
        require_pivot_buffer(pivots.len(), n)?;
    }
    require_info_buffer(dev_info.len())?;
    let lwork = dsytrf_buffer_size(context, n, lda)?;
    require_workspace(workspace.len(), lwork)?;
    // SAFETY: `cusolverDnDsytrf` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnDsytrf(
            context.as_raw(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_mut_ptr().cast(),
            to_i32(lda, "lda")?,
            pivots.map_or(ptr::null_mut(), |mut p| p.as_mut_ptr()),
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
/// # Safety
///
/// Follow the [raw solver contracts](crate::dense#raw-solver-contracts).
pub unsafe fn csytrf(
    context: &Context,
    fill_mode: FillMode,
    n: usize,
    mut a: DeviceViewMut<'_, Complex32>,
    lda: usize,
    pivots: Option<DeviceOutputView<'_, i32>>,
    mut workspace: DeviceOutputView<'_, Complex32>,
    mut dev_info: DeviceOutputView<'_, i32>,
) -> Result<()> {
    context.bind()?;
    context.validate_region(&a)?;
    context.validate_region(&workspace)?;
    context.validate_region(&dev_info)?;
    if let Some(value) = pivots.as_ref() {
        context.validate_region(value)?;
    }
    validate_square_matrix(n, a.len(), lda)?;
    if let Some(pivots) = pivots.as_ref() {
        require_pivot_buffer(pivots.len(), n)?;
    }
    require_info_buffer(dev_info.len())?;
    let lwork = csytrf_buffer_size(context, n, lda)?;
    require_workspace(workspace.len(), lwork)?;
    // SAFETY: `cusolverDnCsytrf` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnCsytrf(
            context.as_raw(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_mut_ptr().cast(),
            to_i32(lda, "lda")?,
            pivots.map_or(ptr::null_mut(), |mut p| p.as_mut_ptr()),
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
/// # Safety
///
/// Follow the [raw solver contracts](crate::dense#raw-solver-contracts).
pub unsafe fn zsytrf(
    context: &Context,
    fill_mode: FillMode,
    n: usize,
    mut a: DeviceViewMut<'_, Complex64>,
    lda: usize,
    pivots: Option<DeviceOutputView<'_, i32>>,
    mut workspace: DeviceOutputView<'_, Complex64>,
    mut dev_info: DeviceOutputView<'_, i32>,
) -> Result<()> {
    context.bind()?;
    context.validate_region(&a)?;
    context.validate_region(&workspace)?;
    context.validate_region(&dev_info)?;
    if let Some(value) = pivots.as_ref() {
        context.validate_region(value)?;
    }
    validate_square_matrix(n, a.len(), lda)?;
    if let Some(pivots) = pivots.as_ref() {
        require_pivot_buffer(pivots.len(), n)?;
    }
    require_info_buffer(dev_info.len())?;
    let lwork = zsytrf_buffer_size(context, n, lda)?;
    require_workspace(workspace.len(), lwork)?;
    // SAFETY: `cusolverDnZsytrf` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnZsytrf(
            context.as_raw(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_mut_ptr().cast(),
            to_i32(lda, "lda")?,
            pivots.map_or(ptr::null_mut(), |mut p| p.as_mut_ptr()),
            workspace.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast(),
        ))?;
    }
    Ok(())
}
