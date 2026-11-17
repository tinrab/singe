use std::ptr;

use singe_cuda::{
    data_type::{DataType, DataTypeLike},
    view::{DeviceOutputView, DeviceView},
};

use crate::{
    context::Context,
    dense::validation::{
        require_info_buffer, require_pivot64_buffer, validate_x_matrix, validate_xlarft_inputs,
    },
    error::{Error, Result},
    layout::{ByteWorkspaceMut, MatrixLayout, MatrixMut, MatrixRef, VectorRef, WorkspaceSizes},
    params::Params,
    sys, try_ffi,
    types::{DiagonalType, DirectMode, FillMode, Operation, StorevMode},
    utility::{require_workspace_bytes, to_i64, to_size_t, to_usize},
};

/// Returns the device and host workspace bytes that [`xpotrf_async`] needs for an `n` by `n` matrix with layout `a`.
///
/// # Errors
///
/// Returns an error if the CUDA context cannot be bound, a dimension or leading dimension is invalid or does not fit the native integer types, or cuSOLVER rejects the configuration.
pub fn xpotrf_workspace_size<TA: DataTypeLike>(
    context: &Context,
    params: &Params,
    fill_mode: FillMode,
    n: usize,
    a: MatrixLayout,
    compute_type: DataType,
) -> Result<WorkspaceSizes> {
    context.bind()?;
    let a_type = TA::data_type();
    a.validate(n, n)?;
    let mut device_bytes = 0;
    let mut host_bytes = 0;
    // SAFETY: the handles are live and the context is bound, the layouts were validated above, size queries accept null data pointers, and the call writes only the local byte counts.
    unsafe {
        try_ffi!(sys::cusolverDnXpotrf_bufferSize(
            context.as_raw(),
            params.as_raw(),
            fill_mode.into(),
            to_i64(n, "n")?,
            a_type.into(),
            ptr::null_mut::<u8>().cast(),
            to_i64(a.leading_dimension, "lda")?,
            compute_type.into(),
            &raw mut device_bytes,
            &raw mut host_bytes,
        ))?;
    }
    Ok(WorkspaceSizes::new(
        to_usize(device_bytes, "device workspace size")?,
        to_usize(host_bytes, "host workspace size")?,
    ))
}

/// Enqueues the Cholesky factorization of the Hermitian positive-definite `n` by `n` matrix `a`.
///
/// Only the triangle selected by `fill_mode` is read and replaced by its Cholesky factor; the other triangle is left untouched.
/// `dev_info` receives `i > 0` when the leading minor of order `i` is not positive definite, and `-i` when the `i`th argument is invalid.
/// Size `workspace` with [`xpotrf_workspace_size`].
/// Supported element types are `f32`, `f64`, `Complex32`, and `Complex64`, with every element type and `compute_type` equal.
///
/// # Safety
///
/// Follow the [raw execution contract](crate#raw-execution).
///
/// # Errors
///
/// Returns an error if the CUDA context cannot be bound, a buffer belongs to another CUDA context, a dimension, leading dimension, or buffer capacity is invalid or does not fit the native integer types, the workspace is smaller than the queried size, or cuSOLVER rejects the call.
pub unsafe fn xpotrf_async<TA: DataTypeLike>(
    context: &Context,
    params: &Params,
    fill_mode: FillMode,
    n: usize,
    mut a: MatrixMut<'_, TA>,
    compute_type: DataType,
    mut workspace: ByteWorkspaceMut<'_>,
    mut dev_info: DeviceOutputView<'_, i32>,
) -> Result<()> {
    context.bind()?;
    context.prepare_region(&a.data)?;
    context.prepare_region(&dev_info)?;
    context.prepare_region(&workspace.device)?;
    let a_type = TA::data_type();
    validate_x_matrix(
        n,
        n,
        a.data.region().byte_len(),
        a.leading_dimension,
        a_type,
    )?;
    require_info_buffer(dev_info.len())?;
    let workspace_sizes =
        xpotrf_workspace_size::<TA>(context, params, fill_mode, n, a.layout(), compute_type)?;
    require_workspace_bytes(
        "device workspace",
        workspace.device.len(),
        workspace_sizes.device_bytes,
    )?;
    require_workspace_bytes(
        "host workspace",
        workspace.host.len(),
        workspace_sizes.host_bytes,
    )?;
    // SAFETY: the handles are live and the context is bound, and every buffer belongs to the context and was checked above against the dimensions, element types, and queried workspace bytes.
    // The caller retains the buffers, including the host workspace, through completion under the raw execution contract.
    unsafe {
        try_ffi!(sys::cusolverDnXpotrf(
            context.as_raw(),
            params.as_raw(),
            fill_mode.into(),
            to_i64(n, "n")?,
            a_type.into(),
            a.data.as_mut_ptr().cast(),
            to_i64(a.leading_dimension, "lda")?,
            compute_type.into(),
            workspace.device.as_mut_ptr().cast(),
            to_size_t(workspace_sizes.device_bytes, "device workspace size")?,
            workspace.host.as_mut_ptr().cast(),
            to_size_t(workspace_sizes.host_bytes, "host workspace size")?,
            dev_info.as_mut_ptr().cast(),
        ))?;
    }
    Ok(())
}

/// Enqueues the solve of `A * X = B` for `nrhs` right-hand sides, using the Cholesky factor of `a` from [`xpotrf_async`].
///
/// `b` is overwritten by the solution `X`.
/// `dev_info` receives `-i` when the `i`th argument is invalid.
/// Supported element types are `f32`, `f64`, `Complex32`, and `Complex64`, with `TA` equal to `TB`.
///
/// # Safety
///
/// Follow the [raw execution contract](crate#raw-execution).
///
/// # Errors
///
/// Returns an error if the CUDA context cannot be bound, a buffer belongs to another CUDA context, a dimension, leading dimension, or buffer capacity is invalid or does not fit the native integer types, or cuSOLVER rejects the call.
pub unsafe fn xpotrs_async<TA: DataTypeLike, TB: DataTypeLike>(
    context: &Context,
    params: &Params,
    fill_mode: FillMode,
    n: usize,
    nrhs: usize,
    a: MatrixRef<'_, TA>,
    mut b: MatrixMut<'_, TB>,
    mut dev_info: DeviceOutputView<'_, i32>,
) -> Result<()> {
    context.bind()?;
    context.prepare_region(&a.data)?;
    context.prepare_region(&b.data)?;
    context.prepare_region(&dev_info)?;
    let a_type = TA::data_type();
    let b_type = TB::data_type();
    validate_x_matrix(
        n,
        n,
        a.data.region().byte_len(),
        a.leading_dimension,
        a_type,
    )?;
    validate_x_matrix(
        n,
        nrhs,
        b.data.region().byte_len(),
        b.leading_dimension,
        b_type,
    )?;
    require_info_buffer(dev_info.len())?;
    // SAFETY: the handles are live and the context is bound, and every buffer belongs to the context and was checked above against the dimensions and element types.
    // The caller retains the buffers through completion under the raw execution contract.
    unsafe {
        try_ffi!(sys::cusolverDnXpotrs(
            context.as_raw(),
            params.as_raw(),
            fill_mode.into(),
            to_i64(n, "n")?,
            to_i64(nrhs, "nrhs")?,
            a_type.into(),
            a.data.as_ptr().cast(),
            to_i64(a.leading_dimension, "lda")?,
            b_type.into(),
            b.data.as_mut_ptr().cast(),
            to_i64(b.leading_dimension, "ldb")?,
            dev_info.as_mut_ptr().cast(),
        ))?;
    }
    Ok(())
}

/// Returns the device and host workspace bytes that [`xtrtri_async`] needs for an `n` by `n` triangular matrix with layout `a`.
///
/// # Errors
///
/// Returns an error if the CUDA context cannot be bound, a dimension or leading dimension is invalid or does not fit the native integer types, or cuSOLVER rejects the configuration.
pub fn xtrtri_workspace_size<TA: DataTypeLike>(
    context: &Context,
    fill_mode: FillMode,
    diagonal_type: DiagonalType,
    n: usize,
    a: MatrixLayout,
) -> Result<WorkspaceSizes> {
    context.bind()?;
    a.validate(n, n)?;
    let mut device_bytes = 0;
    let mut host_bytes = 0;
    // SAFETY: the handles are live and the context is bound, the layouts were validated above, size queries accept null data pointers, and the call writes only the local byte counts.
    unsafe {
        try_ffi!(sys::cusolverDnXtrtri_bufferSize(
            context.as_raw(),
            fill_mode.into(),
            diagonal_type.into(),
            to_i64(n, "n")?,
            TA::data_type().into(),
            ptr::null_mut(),
            to_i64(a.leading_dimension, "lda")?,
            &raw mut device_bytes,
            &raw mut host_bytes,
        ))?;
    }
    Ok(WorkspaceSizes::new(
        to_usize(device_bytes, "device workspace size")?,
        to_usize(host_bytes, "host workspace size")?,
    ))
}

/// Enqueues the in-place inversion of the triangle of `a` selected by `fill_mode`.
///
/// The other triangle is left unchanged.
/// `dev_info` receives `i > 0` when `A(i, i)` is zero, and `-i` when the `i`th argument is invalid.
/// Size `workspace` with [`xtrtri_workspace_size`].
/// Supported element types are `f32`, `f64`, `Complex32`, and `Complex64`.
///
/// # Safety
///
/// Follow the [raw execution contract](crate#raw-execution).
///
/// # Errors
///
/// Returns an error if the CUDA context cannot be bound, a buffer belongs to another CUDA context, a dimension, leading dimension, or buffer capacity is invalid or does not fit the native integer types, the workspace is smaller than the queried size, or cuSOLVER rejects the call.
pub unsafe fn xtrtri_async<TA: DataTypeLike>(
    context: &Context,
    fill_mode: FillMode,
    diagonal_type: DiagonalType,
    n: usize,
    mut a: MatrixMut<'_, TA>,
    mut workspace: ByteWorkspaceMut<'_>,
    mut dev_info: DeviceOutputView<'_, i32>,
) -> Result<()> {
    context.bind()?;
    context.prepare_region(&a.data)?;
    context.prepare_region(&dev_info)?;
    context.prepare_region(&workspace.device)?;
    validate_x_matrix(
        n,
        n,
        a.data.region().byte_len(),
        a.leading_dimension,
        TA::data_type(),
    )?;
    require_info_buffer(dev_info.len())?;
    let workspace_sizes =
        xtrtri_workspace_size::<TA>(context, fill_mode, diagonal_type, n, a.layout())?;
    require_workspace_bytes(
        "device workspace",
        workspace.device.len(),
        workspace_sizes.device_bytes,
    )?;
    require_workspace_bytes(
        "host workspace",
        workspace.host.len(),
        workspace_sizes.host_bytes,
    )?;
    // SAFETY: the handles are live and the context is bound, and every buffer belongs to the context and was checked above against the dimensions, element types, and queried workspace bytes.
    // The caller retains the buffers, including the host workspace, through completion under the raw execution contract.
    unsafe {
        try_ffi!(sys::cusolverDnXtrtri(
            context.as_raw(),
            fill_mode.into(),
            diagonal_type.into(),
            to_i64(n, "n")?,
            TA::data_type().into(),
            a.data.as_mut_ptr().cast(),
            to_i64(a.leading_dimension, "lda")?,
            workspace.device.as_mut_ptr().cast(),
            to_size_t(workspace_sizes.device_bytes, "device workspace size")?,
            workspace.host.as_mut_ptr().cast(),
            to_size_t(workspace_sizes.host_bytes, "host workspace size")?,
            dev_info.as_mut_ptr().cast(),
        ))?;
    }
    Ok(())
}

/// Returns the device and host workspace bytes that [`xgetrf_async`] needs for an `m` by `n` matrix with layout `a`.
///
/// # Errors
///
/// Returns an error if the CUDA context cannot be bound, a dimension or leading dimension is invalid or does not fit the native integer types, or cuSOLVER rejects the configuration.
pub fn xgetrf_workspace_size<TA: DataTypeLike>(
    context: &Context,
    params: &Params,
    m: usize,
    n: usize,
    a: MatrixLayout,
    compute_type: DataType,
) -> Result<WorkspaceSizes> {
    context.bind()?;
    let a_type = TA::data_type();
    a.validate(m, n)?;
    let mut device_bytes = 0;
    let mut host_bytes = 0;
    // SAFETY: the handles are live and the context is bound, the layouts were validated above, size queries accept null data pointers, and the call writes only the local byte counts.
    unsafe {
        try_ffi!(sys::cusolverDnXgetrf_bufferSize(
            context.as_raw(),
            params.as_raw(),
            to_i64(m, "m")?,
            to_i64(n, "n")?,
            a_type.into(),
            ptr::null_mut::<u8>().cast(),
            to_i64(a.leading_dimension, "lda")?,
            compute_type.into(),
            &raw mut device_bytes,
            &raw mut host_bytes,
        ))?;
    }
    Ok(WorkspaceSizes::new(
        to_usize(device_bytes, "device workspace size")?,
        to_usize(host_bytes, "host workspace size")?,
    ))
}

/// Enqueues the LU factorization `P * A = L * U` of the `m` by `n` matrix `a`, in place.
///
/// With `pivots`, row `i` is interchanged with row `pivots[i]`; with `None`, no pivoting is performed, which is not numerically stable.
/// `dev_info` receives `i > 0` when `U(i, i)` is zero, and `-i` when the `i`th argument is invalid.
/// Size `workspace` with [`xgetrf_workspace_size`].
/// [`Params::set_algorithm`] selects between [`AlgorithmMode::Default`](crate::types::AlgorithmMode::Default), which needs `m * n` elements of workspace, and the legacy [`AlgorithmMode::Algorithm1`](crate::types::AlgorithmMode::Algorithm1).
/// Supported element types are `f32`, `f64`, `Complex32`, and `Complex64`, with every element type and `compute_type` equal.
///
/// # Safety
///
/// Follow the [raw execution contract](crate#raw-execution).
///
/// # Errors
///
/// Returns an error if the CUDA context cannot be bound, a buffer belongs to another CUDA context, a dimension, leading dimension, or buffer capacity is invalid or does not fit the native integer types, the workspace is smaller than the queried size, or cuSOLVER rejects the call.
pub unsafe fn xgetrf_async<TA: DataTypeLike>(
    context: &Context,
    params: &Params,
    m: usize,
    n: usize,
    mut a: MatrixMut<'_, TA>,
    pivots: Option<DeviceOutputView<'_, i64>>,
    compute_type: DataType,
    mut workspace: ByteWorkspaceMut<'_>,
    mut dev_info: DeviceOutputView<'_, i32>,
) -> Result<()> {
    context.bind()?;
    context.prepare_region(&a.data)?;
    context.prepare_region(&dev_info)?;
    if let Some(value) = pivots.as_ref() {
        context.prepare_region(value)?;
    }

    context.prepare_region(&workspace.device)?;
    let a_type = TA::data_type();
    validate_x_matrix(
        m,
        n,
        a.data.region().byte_len(),
        a.leading_dimension,
        a_type,
    )?;
    if let Some(pivots) = pivots.as_ref() {
        require_pivot64_buffer(pivots.len(), m.min(n))?;
    }

    require_info_buffer(dev_info.len())?;
    let workspace_sizes =
        xgetrf_workspace_size::<TA>(context, params, m, n, a.layout(), compute_type)?;
    require_workspace_bytes(
        "device workspace",
        workspace.device.len(),
        workspace_sizes.device_bytes,
    )?;
    require_workspace_bytes(
        "host workspace",
        workspace.host.len(),
        workspace_sizes.host_bytes,
    )?;
    // SAFETY: the handles are live and the context is bound, and every buffer belongs to the context and was checked above against the dimensions, element types, and queried workspace bytes.
    // The caller retains the buffers, including the host workspace, through completion under the raw execution contract.
    unsafe {
        try_ffi!(sys::cusolverDnXgetrf(
            context.as_raw(),
            params.as_raw(),
            to_i64(m, "m")?,
            to_i64(n, "n")?,
            a_type.into(),
            a.data.as_mut_ptr().cast(),
            to_i64(a.leading_dimension, "lda")?,
            pivots.map_or(ptr::null_mut(), |mut p| p.as_mut_ptr()),
            compute_type.into(),
            workspace.device.as_mut_ptr().cast(),
            to_size_t(workspace_sizes.device_bytes, "device workspace size")?,
            workspace.host.as_mut_ptr().cast(),
            to_size_t(workspace_sizes.host_bytes, "host workspace size")?,
            dev_info.as_mut_ptr().cast(),
        ))?;
    }
    Ok(())
}

/// Enqueues the solve of `op(A) * X = B` for `nrhs` right-hand sides, using the LU factors and `pivots` from [`xgetrf_async`].
///
/// `b` is overwritten by the solution `X`.
/// `dev_info` receives `-i` when the `i`th argument is invalid.
/// Supported element types are `f32`, `f64`, `Complex32`, and `Complex64`, with `TA` equal to `TB`.
///
/// # Safety
///
/// Follow the [raw execution contract](crate#raw-execution).
///
/// # Errors
///
/// Returns an error if the CUDA context cannot be bound, a buffer belongs to another CUDA context, a dimension, leading dimension, or buffer capacity is invalid or does not fit the native integer types, or cuSOLVER rejects the call.
pub unsafe fn xgetrs_async<TA: DataTypeLike, TB: DataTypeLike>(
    context: &Context,
    params: &Params,
    operation: Operation,
    n: usize,
    nrhs: usize,
    a: MatrixRef<'_, TA>,
    pivots: DeviceView<'_, i64>,
    mut b: MatrixMut<'_, TB>,
    mut dev_info: DeviceOutputView<'_, i32>,
) -> Result<()> {
    context.bind()?;
    context.prepare_region(&a.data)?;
    context.prepare_region(&b.data)?;
    context.prepare_region(&pivots)?;
    context.prepare_region(&dev_info)?;
    let a_type = TA::data_type();
    let b_type = TB::data_type();
    validate_x_matrix(
        n,
        n,
        a.data.region().byte_len(),
        a.leading_dimension,
        a_type,
    )?;
    require_pivot64_buffer(pivots.len(), n)?;
    validate_x_matrix(
        n,
        nrhs,
        b.data.region().byte_len(),
        b.leading_dimension,
        b_type,
    )?;
    require_info_buffer(dev_info.len())?;
    // SAFETY: the handles are live and the context is bound, and every buffer belongs to the context and was checked above against the dimensions and element types.
    // The caller retains the buffers through completion under the raw execution contract.
    unsafe {
        try_ffi!(sys::cusolverDnXgetrs(
            context.as_raw(),
            params.as_raw(),
            operation.into(),
            to_i64(n, "n")?,
            to_i64(nrhs, "nrhs")?,
            a_type.into(),
            a.data.as_ptr().cast(),
            to_i64(a.leading_dimension, "lda")?,
            pivots.as_ptr().cast(),
            b_type.into(),
            b.data.as_mut_ptr().cast(),
            to_i64(b.leading_dimension, "ldb")?,
            dev_info.as_mut_ptr().cast(),
        ))?;
    }
    Ok(())
}

/// Returns the device and host workspace bytes that [`xsytrs_async`] needs for `n` by `n` and `n` by `nrhs` matrices with layouts `a` and `b`.
///
/// # Errors
///
/// Returns an error if the CUDA context cannot be bound, a dimension or leading dimension is invalid or does not fit the native integer types, or cuSOLVER rejects the configuration.
pub fn xsytrs_workspace_size<TA: DataTypeLike, TB: DataTypeLike>(
    context: &Context,
    fill_mode: FillMode,
    n: usize,
    nrhs: usize,
    a: MatrixLayout,
    b: MatrixLayout,
) -> Result<WorkspaceSizes> {
    context.bind()?;
    a.validate(n, n)?;
    b.validate(n, nrhs)?;

    let mut device_bytes = 0;
    let mut host_bytes = 0;
    // SAFETY: the handles are live and the context is bound, the layouts were validated above, size queries accept null data pointers, and the call writes only the local byte counts.
    unsafe {
        try_ffi!(sys::cusolverDnXsytrs_bufferSize(
            context.as_raw(),
            fill_mode.into(),
            to_i64(n, "n")?,
            to_i64(nrhs, "nrhs")?,
            TA::data_type().into(),
            ptr::null_mut::<u8>().cast(),
            to_i64(a.leading_dimension, "lda")?,
            ptr::null(),
            TB::data_type().into(),
            ptr::null_mut(),
            to_i64(b.leading_dimension, "ldb")?,
            &raw mut device_bytes,
            &raw mut host_bytes,
        ))?;
    }
    Ok(WorkspaceSizes::new(
        to_usize(device_bytes, "device workspace size")?,
        to_usize(host_bytes, "host workspace size")?,
    ))
}

/// Enqueues the solve of `A * X = B` for `nrhs` right-hand sides, using the symmetric indefinite factorization of `a` from [`xsytrf_async`].
///
/// Pass the pivots from that factorization, or `None` when it ran without pivoting.
/// `b` is overwritten by the solution `X`.
/// `dev_info` receives `-i` when the `i`th argument is invalid.
/// Size `workspace` with [`xsytrs_workspace_size`].
/// Supported element types are `f32`, `f64`, `Complex32`, and `Complex64`, with `TA` equal to `TB`.
///
/// # Safety
///
/// Follow the [raw execution contract](crate#raw-execution).
///
/// # Errors
///
/// Returns an error if the CUDA context cannot be bound, a buffer belongs to another CUDA context, a dimension, leading dimension, or buffer capacity is invalid or does not fit the native integer types, the workspace is smaller than the queried size, or cuSOLVER rejects the call.
pub unsafe fn xsytrs_async<TA: DataTypeLike, TB: DataTypeLike>(
    context: &Context,
    fill_mode: FillMode,
    n: usize,
    nrhs: usize,
    a: MatrixRef<'_, TA>,
    pivots: Option<DeviceView<'_, i64>>,
    mut b: MatrixMut<'_, TB>,
    mut workspace: ByteWorkspaceMut<'_>,
    mut dev_info: DeviceOutputView<'_, i32>,
) -> Result<()> {
    context.bind()?;
    context.prepare_region(&a.data)?;
    context.prepare_region(&b.data)?;
    context.prepare_region(&dev_info)?;
    if let Some(value) = pivots.as_ref() {
        context.prepare_region(value)?;
    }

    context.prepare_region(&workspace.device)?;
    validate_x_matrix(
        n,
        n,
        a.data.region().byte_len(),
        a.leading_dimension,
        TA::data_type(),
    )?;
    validate_x_matrix(
        n,
        nrhs,
        b.data.region().byte_len(),
        b.leading_dimension,
        TB::data_type(),
    )?;
    if let Some(pivots) = pivots {
        require_pivot64_buffer(pivots.len(), n)?;
    }

    require_info_buffer(dev_info.len())?;
    let workspace_sizes =
        xsytrs_workspace_size::<TA, TB>(context, fill_mode, n, nrhs, a.layout(), b.layout())?;
    require_workspace_bytes(
        "device workspace",
        workspace.device.len(),
        workspace_sizes.device_bytes,
    )?;
    require_workspace_bytes(
        "host workspace",
        workspace.host.len(),
        workspace_sizes.host_bytes,
    )?;
    // SAFETY: the handles are live and the context is bound, and every buffer belongs to the context and was checked above against the dimensions, element types, and queried workspace bytes.
    // The caller retains the buffers, including the host workspace, through completion under the raw execution contract.
    unsafe {
        try_ffi!(sys::cusolverDnXsytrs(
            context.as_raw(),
            fill_mode.into(),
            to_i64(n, "n")?,
            to_i64(nrhs, "nrhs")?,
            TA::data_type().into(),
            a.data.as_ptr().cast(),
            to_i64(a.leading_dimension, "lda")?,
            pivots.map_or(ptr::null(), |buffer| buffer.as_ptr()),
            TB::data_type().into(),
            b.data.as_mut_ptr().cast(),
            to_i64(b.leading_dimension, "ldb")?,
            workspace.device.as_mut_ptr().cast(),
            to_size_t(workspace_sizes.device_bytes, "device workspace size")?,
            workspace.host.as_mut_ptr().cast(),
            to_size_t(workspace_sizes.host_bytes, "host workspace size")?,
            dev_info.as_mut_ptr().cast(),
        ))?;
    }
    Ok(())
}

/// Returns the device and host workspace bytes that [`xlarft_async`] needs for `k` reflectors of order `n` with layouts `v` and `t`.
///
/// # Errors
///
/// Returns an error if the CUDA context cannot be bound, `k` exceeds `n`, `storev` is not [`StorevMode::Columnwise`], a dimension or leading dimension is invalid or does not fit the native integer types, or cuSOLVER rejects the configuration.
pub fn xlarft_workspace_size<TV: DataTypeLike, TTau: DataTypeLike, TT: DataTypeLike>(
    context: &Context,
    params: &Params,
    direct: DirectMode,
    storev: StorevMode,
    n: usize,
    k: usize,
    v: MatrixLayout,
    t: MatrixLayout,
    compute_type: DataType,
) -> Result<WorkspaceSizes> {
    context.bind()?;
    let v_type = TV::data_type();
    let tau_type = TTau::data_type();
    let t_type = TT::data_type();
    v.validate(n, k)
        .and_then(|()| t.validate(k, k))
        .and_then(|()| {
            if k > n || storev != StorevMode::Columnwise {
                Err(Error::InvalidMatrixShape)
            } else {
                Ok(())
            }
        })?;
    let mut device_bytes = 0;
    let mut host_bytes = 0;
    // SAFETY: the handles are live and the context is bound, the layouts were validated above, size queries accept null data pointers, and the call writes only the local byte counts.
    unsafe {
        try_ffi!(sys::cusolverDnXlarft_bufferSize(
            context.as_raw(),
            params.as_raw(),
            direct.into(),
            storev.into(),
            to_i64(n, "n")?,
            to_i64(k, "k")?,
            v_type.into(),
            ptr::null_mut::<u8>().cast(),
            to_i64(v.leading_dimension, "ldv")?,
            tau_type.into(),
            ptr::null(),
            t_type.into(),
            ptr::null_mut(),
            to_i64(t.leading_dimension, "ldt")?,
            compute_type.into(),
            &raw mut device_bytes,
            &raw mut host_bytes,
        ))?;
    }
    Ok(WorkspaceSizes::new(
        to_usize(device_bytes, "device workspace size")?,
        to_usize(host_bytes, "host workspace size")?,
    ))
}

/// Enqueues the formation of the `k` by `k` triangular factor `t` of the block reflector `H = I - V * T * V^H` built from `k` elementary reflectors of order `n`.
///
/// The `i`th column of `v` holds the `i`th reflector, so only [`StorevMode::Columnwise`] storage and `k <= n` are supported.
/// `t` must be initialized, because only its selected triangle is written.
/// Size `workspace` with [`xlarft_workspace_size`].
/// Supported element types are `f32`, `f64`, `Complex32`, and `Complex64`, with `TV`, `TTau`, `TT`, and `compute_type` equal.
///
/// # Safety
///
/// Follow the [raw execution contract](crate#raw-execution).
///
/// # Errors
///
/// Returns an error if the CUDA context cannot be bound, a buffer belongs to another CUDA context, `k` exceeds `n`, `storev` is not [`StorevMode::Columnwise`], a dimension, leading dimension, or buffer capacity is invalid or does not fit the native integer types, the workspace is smaller than the queried size, or cuSOLVER rejects the call.
pub unsafe fn xlarft_async<TV: DataTypeLike, TTau: DataTypeLike, TT: DataTypeLike>(
    context: &Context,
    params: &Params,
    direct: DirectMode,
    storev: StorevMode,
    n: usize,
    k: usize,
    v: MatrixRef<'_, TV>,
    tau: VectorRef<'_, TTau>,
    mut t: MatrixMut<'_, TT>,
    compute_type: DataType,
    mut workspace: ByteWorkspaceMut<'_>,
) -> Result<()> {
    context.bind()?;
    context.prepare_region(&v.data)?;
    context.prepare_region(&tau.data)?;
    context.prepare_region(&t.data)?;
    context.prepare_region(&workspace.device)?;
    let v_type = TV::data_type();
    let tau_type = TTau::data_type();
    let t_type = TT::data_type();
    validate_xlarft_inputs(
        n,
        k,
        storev,
        v.data.region().byte_len(),
        v.leading_dimension,
        v_type,
        tau.data.region().byte_len(),
        tau_type,
        t.data.region().byte_len(),
        t.leading_dimension,
        t_type,
    )?;
    let workspace_sizes = xlarft_workspace_size::<TV, TTau, TT>(
        context,
        params,
        direct,
        storev,
        n,
        k,
        v.layout(),
        t.layout(),
        compute_type,
    )?;
    require_workspace_bytes(
        "device workspace",
        workspace.device.len(),
        workspace_sizes.device_bytes,
    )?;
    require_workspace_bytes(
        "host workspace",
        workspace.host.len(),
        workspace_sizes.host_bytes,
    )?;
    // SAFETY: the handles are live and the context is bound, and every buffer belongs to the context and was checked above against the dimensions, element types, and queried workspace bytes.
    // The caller retains the buffers, including the host workspace, through completion under the raw execution contract.
    unsafe {
        try_ffi!(sys::cusolverDnXlarft(
            context.as_raw(),
            params.as_raw(),
            direct.into(),
            storev.into(),
            to_i64(n, "n")?,
            to_i64(k, "k")?,
            v_type.into(),
            v.data.as_ptr().cast(),
            to_i64(v.leading_dimension, "ldv")?,
            tau_type.into(),
            tau.data.as_ptr().cast(),
            t_type.into(),
            t.data.as_mut_ptr().cast(),
            to_i64(t.leading_dimension, "ldt")?,
            compute_type.into(),
            workspace.device.as_mut_ptr().cast(),
            to_size_t(workspace_sizes.device_bytes, "device workspace size")?,
            workspace.host.as_mut_ptr().cast(),
            to_size_t(workspace_sizes.host_bytes, "host workspace size")?,
        ))?;
    }
    Ok(())
}

/// Returns the device and host workspace bytes that [`xsytrf_async`] needs for an `n` by `n` matrix with layout `a`.
///
/// # Errors
///
/// Returns an error if the CUDA context cannot be bound, a dimension or leading dimension is invalid or does not fit the native integer types, or cuSOLVER rejects the configuration.
pub fn xsytrf_workspace_size<TA: DataTypeLike>(
    context: &Context,
    params: &Params,
    fill_mode: FillMode,
    n: usize,
    a: MatrixLayout,
    compute_type: DataType,
) -> Result<WorkspaceSizes> {
    context.bind()?;
    let a_type = TA::data_type();
    a.validate(n, n)?;

    let mut device_bytes = 0;
    let mut host_bytes = 0;
    // SAFETY: the handles are live and the context is bound, the layouts were validated above, size queries accept null data pointers, and the call writes only the local byte counts.
    unsafe {
        try_ffi!(sys::cusolverDnXsytrf_bufferSize(
            context.as_raw(),
            params.as_raw(),
            fill_mode.into(),
            to_i64(n, "n")?,
            a_type.into(),
            ptr::null_mut::<u8>().cast(),
            to_i64(a.leading_dimension, "lda")?,
            ptr::null(),
            compute_type.into(),
            &raw mut device_bytes,
            &raw mut host_bytes,
        ))?;
    }
    Ok(WorkspaceSizes::new(
        to_usize(device_bytes, "device workspace size")?,
        to_usize(host_bytes, "host workspace size")?,
    ))
}

/// Enqueues the Bunch-Kaufman factorization of the symmetric indefinite `n` by `n` matrix `a`, in place.
///
/// Only the triangle selected by `fill_mode` is meaningful; the other triangle may be used as scratch.
/// With `None` for `pivots`, no pivoting is performed, which is not numerically stable.
/// Each diagonal block of the factor is 1 by 1 or 2 by 2.
/// Size `workspace` with [`xsytrf_workspace_size`].
/// Supported element types are `f32`, `f64`, `Complex32`, and `Complex64`, with every element type and `compute_type` equal.
///
/// # Safety
///
/// Follow the [raw execution contract](crate#raw-execution).
///
/// # Errors
///
/// Returns an error if the CUDA context cannot be bound, a buffer belongs to another CUDA context, a dimension, leading dimension, or buffer capacity is invalid or does not fit the native integer types, the workspace is smaller than the queried size, or cuSOLVER rejects the call.
pub unsafe fn xsytrf_async<TA: DataTypeLike>(
    context: &Context,
    params: &Params,
    fill_mode: FillMode,
    n: usize,
    mut a: MatrixMut<'_, TA>,
    pivots: Option<DeviceOutputView<'_, i64>>,
    compute_type: DataType,
    mut workspace: ByteWorkspaceMut<'_>,
    mut dev_info: DeviceOutputView<'_, i32>,
) -> Result<()> {
    context.bind()?;
    context.prepare_region(&a.data)?;
    context.prepare_region(&dev_info)?;
    if let Some(value) = pivots.as_ref() {
        context.prepare_region(value)?;
    }

    context.prepare_region(&workspace.device)?;
    let a_type = TA::data_type();
    validate_x_matrix(
        n,
        n,
        a.data.region().byte_len(),
        a.leading_dimension,
        a_type,
    )?;
    if let Some(pivots) = pivots.as_ref() {
        require_pivot64_buffer(pivots.len(), n)?;
    }

    require_info_buffer(dev_info.len())?;
    let workspace_sizes =
        xsytrf_workspace_size::<TA>(context, params, fill_mode, n, a.layout(), compute_type)?;
    require_workspace_bytes(
        "device workspace",
        workspace.device.len(),
        workspace_sizes.device_bytes,
    )?;
    require_workspace_bytes(
        "host workspace",
        workspace.host.len(),
        workspace_sizes.host_bytes,
    )?;
    // SAFETY: the handles are live and the context is bound, and every buffer belongs to the context and was checked above against the dimensions, element types, and queried workspace bytes.
    // The caller retains the buffers, including the host workspace, through completion under the raw execution contract.
    unsafe {
        try_ffi!(sys::cusolverDnXsytrf(
            context.as_raw(),
            params.as_raw(),
            fill_mode.into(),
            to_i64(n, "n")?,
            a_type.into(),
            a.data.as_mut_ptr().cast(),
            to_i64(a.leading_dimension, "lda")?,
            pivots.map_or(ptr::null_mut(), |mut buffer| buffer.as_mut_ptr()),
            compute_type.into(),
            workspace.device.as_mut_ptr().cast(),
            to_size_t(workspace_sizes.device_bytes, "device workspace size")?,
            workspace.host.as_mut_ptr().cast(),
            to_size_t(workspace_sizes.host_bytes, "host workspace size")?,
            dev_info.as_mut_ptr().cast(),
        ))?;
    }
    Ok(())
}

/// Returns the device and host workspace bytes that [`xhetrf_async`] needs for an `n` by `n` matrix with layout `a`.
///
/// # Errors
///
/// Returns an error if the CUDA context cannot be bound, a dimension or leading dimension is invalid or does not fit the native integer types, or cuSOLVER rejects the configuration.
pub fn xhetrf_workspace_size<TA: DataTypeLike>(
    context: &Context,
    params: &Params,
    fill_mode: FillMode,
    n: usize,
    a: MatrixLayout,
    compute_type: DataType,
) -> Result<WorkspaceSizes> {
    context.bind()?;
    let a_type = TA::data_type();
    a.validate(n, n)?;

    let mut device_bytes = 0;
    let mut host_bytes = 0;
    // SAFETY: the handles are live and the context is bound, the layouts were validated above, size queries accept null data pointers, and the call writes only the local byte counts.
    unsafe {
        try_ffi!(sys::cusolverDnXhetrf_bufferSize(
            context.as_raw(),
            params.as_raw(),
            fill_mode.into(),
            to_i64(n, "n")?,
            a_type.into(),
            ptr::null_mut::<u8>().cast(),
            to_i64(a.leading_dimension, "lda")?,
            ptr::null(),
            compute_type.into(),
            &raw mut device_bytes,
            &raw mut host_bytes,
        ))?;
    }
    Ok(WorkspaceSizes::new(
        to_usize(device_bytes, "device workspace size")?,
        to_usize(host_bytes, "host workspace size")?,
    ))
}

/// Enqueues the Bunch-Kaufman factorization of the Hermitian indefinite `n` by `n` matrix `a`, in place.
///
/// Only the triangle selected by `fill_mode` is meaningful; the other triangle may be used as scratch.
/// With `None` for `pivots`, no pivoting is performed, which is not numerically stable.
/// Each diagonal block of the factor is 1 by 1 or 2 by 2.
/// Size `workspace` with [`xhetrf_workspace_size`].
/// Supported element types are `Complex32` and `Complex64`, with `compute_type` equal to the element type.
///
/// # Safety
///
/// Follow the [raw execution contract](crate#raw-execution).
///
/// # Errors
///
/// Returns an error if the CUDA context cannot be bound, a buffer belongs to another CUDA context, a dimension, leading dimension, or buffer capacity is invalid or does not fit the native integer types, the workspace is smaller than the queried size, or cuSOLVER rejects the call.
pub unsafe fn xhetrf_async<TA: DataTypeLike>(
    context: &Context,
    params: &Params,
    fill_mode: FillMode,
    n: usize,
    mut a: MatrixMut<'_, TA>,
    pivots: Option<DeviceOutputView<'_, i64>>,
    compute_type: DataType,
    mut workspace: ByteWorkspaceMut<'_>,
    mut dev_info: DeviceOutputView<'_, i32>,
) -> Result<()> {
    context.bind()?;
    context.prepare_region(&a.data)?;
    context.prepare_region(&dev_info)?;
    if let Some(value) = pivots.as_ref() {
        context.prepare_region(value)?;
    }

    context.prepare_region(&workspace.device)?;
    let a_type = TA::data_type();
    validate_x_matrix(
        n,
        n,
        a.data.region().byte_len(),
        a.leading_dimension,
        a_type,
    )?;
    if let Some(pivots) = pivots.as_ref() {
        require_pivot64_buffer(pivots.len(), n)?;
    }

    require_info_buffer(dev_info.len())?;
    let workspace_sizes =
        xhetrf_workspace_size::<TA>(context, params, fill_mode, n, a.layout(), compute_type)?;
    require_workspace_bytes(
        "device workspace",
        workspace.device.len(),
        workspace_sizes.device_bytes,
    )?;
    require_workspace_bytes(
        "host workspace",
        workspace.host.len(),
        workspace_sizes.host_bytes,
    )?;
    // SAFETY: the handles are live and the context is bound, and every buffer belongs to the context and was checked above against the dimensions, element types, and queried workspace bytes.
    // The caller retains the buffers, including the host workspace, through completion under the raw execution contract.
    unsafe {
        try_ffi!(sys::cusolverDnXhetrf(
            context.as_raw(),
            params.as_raw(),
            fill_mode.into(),
            to_i64(n, "n")?,
            a_type.into(),
            a.data.as_mut_ptr().cast(),
            to_i64(a.leading_dimension, "lda")?,
            pivots.map_or(ptr::null_mut(), |mut buffer| buffer.as_mut_ptr()),
            compute_type.into(),
            workspace.device.as_mut_ptr().cast(),
            to_size_t(workspace_sizes.device_bytes, "device workspace size")?,
            workspace.host.as_mut_ptr().cast(),
            to_size_t(workspace_sizes.host_bytes, "host workspace size")?,
            dev_info.as_mut_ptr().cast(),
        ))?;
    }
    Ok(())
}

/// Returns the device and host workspace bytes that [`xhetrs_async`] needs for `n` by `n` and `n` by `nrhs` matrices with layouts `a` and `b`.
///
/// # Errors
///
/// Returns an error if the CUDA context cannot be bound, a dimension or leading dimension is invalid or does not fit the native integer types, or cuSOLVER rejects the configuration.
pub fn xhetrs_workspace_size<TA: DataTypeLike, TB: DataTypeLike>(
    context: &Context,
    params: &Params,
    fill_mode: FillMode,
    n: usize,
    nrhs: usize,
    a: MatrixLayout,
    b: MatrixLayout,
) -> Result<WorkspaceSizes> {
    context.bind()?;
    a.validate(n, n)?;
    b.validate(n, nrhs)?;

    let mut device_bytes = 0;
    let mut host_bytes = 0;
    // SAFETY: the handles are live and the context is bound, the layouts were validated above, size queries accept null data pointers, and the call writes only the local byte counts.
    unsafe {
        try_ffi!(sys::cusolverDnXhetrs_bufferSize(
            context.as_raw(),
            params.as_raw(),
            fill_mode.into(),
            to_i64(n, "n")?,
            to_i64(nrhs, "nrhs")?,
            TA::data_type().into(),
            ptr::null_mut::<u8>().cast(),
            to_i64(a.leading_dimension, "lda")?,
            ptr::null(),
            TB::data_type().into(),
            ptr::null_mut(),
            to_i64(b.leading_dimension, "ldb")?,
            &raw mut device_bytes,
            &raw mut host_bytes,
        ))?;
    }
    Ok(WorkspaceSizes::new(
        to_usize(device_bytes, "device workspace size")?,
        to_usize(host_bytes, "host workspace size")?,
    ))
}

/// Enqueues the solve of `A * X = B` for `nrhs` right-hand sides, using the Hermitian indefinite factorization of `a` from [`xhetrf_async`].
///
/// Pass the pivots from that factorization, or `None` when it ran without pivoting.
/// `b` is overwritten by the solution `X`.
/// `dev_info` receives `-i` when the `i`th argument is invalid.
/// Size `workspace` with [`xhetrs_workspace_size`].
/// Supported element types are `Complex32` and `Complex64`, with `TA` equal to `TB`.
///
/// # Safety
///
/// Follow the [raw execution contract](crate#raw-execution).
///
/// # Errors
///
/// Returns an error if the CUDA context cannot be bound, a buffer belongs to another CUDA context, a dimension, leading dimension, or buffer capacity is invalid or does not fit the native integer types, the workspace is smaller than the queried size, or cuSOLVER rejects the call.
pub unsafe fn xhetrs_async<TA: DataTypeLike, TB: DataTypeLike>(
    context: &Context,
    params: &Params,
    fill_mode: FillMode,
    n: usize,
    nrhs: usize,
    a: MatrixRef<'_, TA>,
    pivots: Option<DeviceView<'_, i64>>,
    mut b: MatrixMut<'_, TB>,
    mut workspace: ByteWorkspaceMut<'_>,
    mut dev_info: DeviceOutputView<'_, i32>,
) -> Result<()> {
    context.bind()?;
    context.prepare_region(&a.data)?;
    context.prepare_region(&b.data)?;
    context.prepare_region(&dev_info)?;
    if let Some(value) = pivots.as_ref() {
        context.prepare_region(value)?;
    }

    context.prepare_region(&workspace.device)?;
    validate_x_matrix(
        n,
        n,
        a.data.region().byte_len(),
        a.leading_dimension,
        TA::data_type(),
    )?;
    validate_x_matrix(
        n,
        nrhs,
        b.data.region().byte_len(),
        b.leading_dimension,
        TB::data_type(),
    )?;
    if let Some(pivots) = pivots {
        require_pivot64_buffer(pivots.len(), n)?;
    }

    require_info_buffer(dev_info.len())?;
    let workspace_sizes = xhetrs_workspace_size::<TA, TB>(
        context,
        params,
        fill_mode,
        n,
        nrhs,
        a.layout(),
        b.layout(),
    )?;
    require_workspace_bytes(
        "device workspace",
        workspace.device.len(),
        workspace_sizes.device_bytes,
    )?;
    require_workspace_bytes(
        "host workspace",
        workspace.host.len(),
        workspace_sizes.host_bytes,
    )?;
    // SAFETY: the handles are live and the context is bound, and every buffer belongs to the context and was checked above against the dimensions, element types, and queried workspace bytes.
    // The caller retains the buffers, including the host workspace, through completion under the raw execution contract.
    unsafe {
        try_ffi!(sys::cusolverDnXhetrs(
            context.as_raw(),
            params.as_raw(),
            fill_mode.into(),
            to_i64(n, "n")?,
            to_i64(nrhs, "nrhs")?,
            TA::data_type().into(),
            a.data.as_ptr().cast(),
            to_i64(a.leading_dimension, "lda")?,
            pivots.map_or(ptr::null(), |buffer| buffer.as_ptr()),
            TB::data_type().into(),
            b.data.as_mut_ptr().cast(),
            to_i64(b.leading_dimension, "ldb")?,
            workspace.device.as_mut_ptr().cast(),
            to_size_t(workspace_sizes.device_bytes, "device workspace size")?,
            workspace.host.as_mut_ptr().cast(),
            to_size_t(workspace_sizes.host_bytes, "host workspace size")?,
            dev_info.as_mut_ptr().cast(),
        ))?;
    }
    Ok(())
}
