//! Native symmetric, generalized, and general eigenvalue operations.
//!
//! Execution enqueues work without completing its borrows.
//! Keep the context, parameter objects, matrices, outputs, workspaces, selection
//! bounds, and host count destinations alive through completion and captured replay.
//! Order prior writes before execution and exclude overlapping accesses to writes.
//! A successful enqueue does not establish a successful numerical result.
//! Inspect completed device status before promoting outputs.
//!
//! Symmetric operations produce real eigenvalues and overwrite A with vectors
//! when requested. Generalized operations also overwrite B with its selected
//! triangular factor. Padding, scratch, and allocation tails are not results.
//! Selected operations produce only the returned count of values and columns.
//! Batched operations pack A at lda*n elements per matrix and values at n;
//! they produce one status per matrix.
//!
//! Capture support follows the native routine.
//! Batched Jacobi supports captured replay in cuSOLVER 12.3.4.
//! A successful workspace query during capture does not establish execution support.

use std::{mem::MaybeUninit, ptr};

use singe_cuda::{
    data_type::DataTypeLike,
    types::{Complex32, Complex64},
    view::DeviceOutputView,
};

use crate::{
    context::Context,
    data_type::{EigenvalueElement, SolverElement},
    eigen::{
        EigenSelection, SyevjInfo,
        validation::{batch_layout, native_selection, output, workspace},
    },
    error::{Error, Result},
    layout::{ByteWorkspaceMut, MatrixLayout, MatrixMut, MatrixOutput, WorkspaceSizes},
    params::Params,
    sys, try_ffi,
    types::{EigenMode, EigenType, FillMode},
    utility::{to_i32, to_i64, to_usize},
};

/// Queries scratch capacity from layout metadata without allocating buffers.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn ssyevj_buffer_size(
    context: &Context,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    a: MatrixLayout,
    params: &SyevjInfo,
) -> Result<usize> {
    context.bind()?;
    a.validate(n, n)?;
    let mut lwork = 0;
    // SAFETY: native workspace queries inspect metadata without accessing matrices.
    unsafe {
        try_ffi!(sys::cusolverDnSsyevj_bufferSize(
            context.as_raw(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            ptr::null(),
            to_i32(a.leading_dimension, "lda")?,
            ptr::null(),
            &raw mut lwork,
            params.as_raw()
        ))?;
    }
    to_usize(lwork, "lwork")
}

/// Enqueues the native eigenvalue operation.
///
/// # Safety
/// Keep all borrowed resources alive and order their accesses through completion
/// and every replay. Inspect completed status before reading or promoting results.
/// Jacobi parameter storage is exclusive through completion when present.
/// Selected host counts and bounds must also survive completion and replay.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a device buffer belongs to another CUDA context, matrix dimensions, leading dimensions, or storage capacities are invalid, checked size arithmetic or native integer conversion fails, the status output capacity is insufficient, a pivot, reflector, or numerical output buffer is too small, device scratch capacity is smaller than the queried requirement, or cuSOLVER rejects the configuration or returns an error while querying workspace or submitting execution.
pub unsafe fn ssyevj(
    context: &Context,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    mut a: MatrixMut<'_, f32>,
    mut w: DeviceOutputView<'_, f32>,
    mut scratch: DeviceOutputView<'_, f32>,
    mut dev_info: DeviceOutputView<'_, i32>,
    params: &mut SyevjInfo,
) -> Result<()> {
    context.bind()?;
    let values = n;
    a.layout().require_capacity(n, n, a.data.len())?;
    output(&w, values)?;
    output(&dev_info, 1)?;
    context.validate_region(&a.data)?;
    context.validate_region(&w)?;
    context.validate_region(&scratch)?;
    context.validate_region(&dev_info)?;
    let lwork = ssyevj_buffer_size(context, mode, fill_mode, n, a.layout(), params)?;
    workspace(scratch.len(), lwork)?;
    // SAFETY: validated metadata and the caller's queued-access contract cover native use.
    unsafe {
        try_ffi!(sys::cusolverDnSsyevj(
            context.as_raw(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.data.as_mut_ptr().cast(),
            to_i32(a.leading_dimension, "lda")?,
            w.as_mut_ptr().cast(),
            scratch.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast(),
            params.as_raw()
        ))?;
    }
    Ok(())
}

/// Queries scratch capacity from layout metadata without allocating buffers.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, the batch count or packed storage extent is invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn ssyevj_batched_buffer_size(
    context: &Context,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    a: MatrixLayout,
    params: &SyevjInfo,
    batch_count: usize,
) -> Result<usize> {
    context.bind()?;
    batch_layout(n, a, batch_count, false)?;
    let mut lwork = 0;
    // SAFETY: native workspace queries inspect metadata without accessing matrices.
    unsafe {
        try_ffi!(sys::cusolverDnSsyevjBatched_bufferSize(
            context.as_raw(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            ptr::null(),
            to_i32(a.leading_dimension, "lda")?,
            ptr::null(),
            &raw mut lwork,
            params.as_raw(),
            to_i32(batch_count, "batch_count")?
        ))?;
    }
    to_usize(lwork, "lwork")
}

/// Enqueues the native eigenvalue operation.
///
/// # Safety
/// Keep all borrowed resources alive and order their accesses through completion
/// and every replay. Inspect completed status before reading or promoting results.
/// Jacobi parameter storage is exclusive through completion when present.
/// Selected host counts and bounds must also survive completion and replay.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a device buffer belongs to another CUDA context, matrix dimensions, leading dimensions, or storage capacities are invalid, checked size arithmetic or native integer conversion fails, the status output capacity is insufficient, a pivot, reflector, or numerical output buffer is too small, device scratch capacity is smaller than the queried requirement, batch metadata or related pointer-table lengths are invalid, or cuSOLVER rejects the configuration or returns an error while querying workspace or submitting execution.
pub unsafe fn ssyevj_batched(
    context: &Context,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    mut a: MatrixMut<'_, f32>,
    mut w: DeviceOutputView<'_, f32>,
    mut scratch: DeviceOutputView<'_, f32>,
    mut dev_info: DeviceOutputView<'_, i32>,
    params: &mut SyevjInfo,
    batch_count: usize,
) -> Result<()> {
    context.bind()?;
    let values = batch_layout(n, a.layout(), batch_count, false)?;
    a.layout().require_capacity(n, values, a.data.len())?;
    output(&w, values)?;
    output(&dev_info, batch_count)?;
    context.validate_region(&a.data)?;
    context.validate_region(&w)?;
    context.validate_region(&scratch)?;
    context.validate_region(&dev_info)?;
    let lwork =
        ssyevj_batched_buffer_size(context, mode, fill_mode, n, a.layout(), params, batch_count)?;
    workspace(scratch.len(), lwork)?;
    // SAFETY: validated metadata and the caller's queued-access contract cover native use.
    unsafe {
        try_ffi!(sys::cusolverDnSsyevjBatched(
            context.as_raw(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.data.as_mut_ptr().cast(),
            to_i32(a.leading_dimension, "lda")?,
            w.as_mut_ptr().cast(),
            scratch.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast(),
            params.as_raw(),
            to_i32(batch_count, "batch_count")?
        ))?;
    }
    Ok(())
}

/// Queries scratch capacity from layout metadata without allocating buffers.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn ssygvj_buffer_size(
    context: &Context,
    eig_type: EigenType,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    a: MatrixLayout,
    b: MatrixLayout,
    params: &SyevjInfo,
) -> Result<usize> {
    context.bind()?;
    a.validate(n, n)?;
    b.validate(n, n)?;
    let mut lwork = 0;
    // SAFETY: native workspace queries inspect metadata without accessing matrices.
    unsafe {
        try_ffi!(sys::cusolverDnSsygvj_bufferSize(
            context.as_raw(),
            eig_type.into(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            ptr::null(),
            to_i32(a.leading_dimension, "lda")?,
            ptr::null(),
            to_i32(b.leading_dimension, "ldb")?,
            ptr::null(),
            &raw mut lwork,
            params.as_raw()
        ))?;
    }
    to_usize(lwork, "lwork")
}

/// Enqueues the native eigenvalue operation.
///
/// # Safety
/// Keep all borrowed resources alive and order their accesses through completion
/// and every replay. Inspect completed status before reading or promoting results.
/// Jacobi parameter storage is exclusive through completion when present.
/// Selected host counts and bounds must also survive completion and replay.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a device buffer belongs to another CUDA context, matrix dimensions, leading dimensions, or storage capacities are invalid, checked size arithmetic or native integer conversion fails, the status output capacity is insufficient, a pivot, reflector, or numerical output buffer is too small, device scratch capacity is smaller than the queried requirement, or cuSOLVER rejects the configuration or returns an error while querying workspace or submitting execution.
pub unsafe fn ssygvj(
    context: &Context,
    eig_type: EigenType,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    mut a: MatrixMut<'_, f32>,
    mut b: MatrixMut<'_, f32>,
    mut w: DeviceOutputView<'_, f32>,
    mut scratch: DeviceOutputView<'_, f32>,
    mut dev_info: DeviceOutputView<'_, i32>,
    params: &mut SyevjInfo,
) -> Result<()> {
    context.bind()?;
    let values = n;
    a.layout().require_capacity(n, n, a.data.len())?;
    b.layout().require_capacity(n, n, b.data.len())?;
    output(&w, values)?;
    output(&dev_info, 1)?;
    context.validate_region(&a.data)?;
    context.validate_region(&b.data)?;
    context.validate_region(&w)?;
    context.validate_region(&scratch)?;
    context.validate_region(&dev_info)?;
    let lwork = ssygvj_buffer_size(
        context,
        eig_type,
        mode,
        fill_mode,
        n,
        a.layout(),
        b.layout(),
        params,
    )?;
    workspace(scratch.len(), lwork)?;
    // SAFETY: validated metadata and the caller's queued-access contract cover native use.
    unsafe {
        try_ffi!(sys::cusolverDnSsygvj(
            context.as_raw(),
            eig_type.into(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.data.as_mut_ptr().cast(),
            to_i32(a.leading_dimension, "lda")?,
            b.data.as_mut_ptr().cast(),
            to_i32(b.leading_dimension, "ldb")?,
            w.as_mut_ptr().cast(),
            scratch.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast(),
            params.as_raw()
        ))?;
    }
    Ok(())
}

/// Queries scratch capacity from layout metadata without allocating buffers.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn ssygvd_buffer_size(
    context: &Context,
    eig_type: EigenType,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    a: MatrixLayout,
    b: MatrixLayout,
) -> Result<usize> {
    context.bind()?;
    a.validate(n, n)?;
    b.validate(n, n)?;
    let mut lwork = 0;
    // SAFETY: native workspace queries inspect metadata without accessing matrices.
    unsafe {
        try_ffi!(sys::cusolverDnSsygvd_bufferSize(
            context.as_raw(),
            eig_type.into(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            ptr::null(),
            to_i32(a.leading_dimension, "lda")?,
            ptr::null(),
            to_i32(b.leading_dimension, "ldb")?,
            ptr::null(),
            &raw mut lwork
        ))?;
    }
    to_usize(lwork, "lwork")
}

/// Enqueues the native eigenvalue operation.
///
/// # Safety
/// Keep all borrowed resources alive and order their accesses through completion
/// and every replay. Inspect completed status before reading or promoting results.
/// Jacobi parameter storage is exclusive through completion when present.
/// Selected host counts and bounds must also survive completion and replay.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a device buffer belongs to another CUDA context, matrix dimensions, leading dimensions, or storage capacities are invalid, checked size arithmetic or native integer conversion fails, the status output capacity is insufficient, a pivot, reflector, or numerical output buffer is too small, device scratch capacity is smaller than the queried requirement, or cuSOLVER rejects the configuration or returns an error while querying workspace or submitting execution.
pub unsafe fn ssygvd(
    context: &Context,
    eig_type: EigenType,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    mut a: MatrixMut<'_, f32>,
    mut b: MatrixMut<'_, f32>,
    mut w: DeviceOutputView<'_, f32>,
    mut scratch: DeviceOutputView<'_, f32>,
    mut dev_info: DeviceOutputView<'_, i32>,
) -> Result<()> {
    context.bind()?;
    let values = n;
    a.layout().require_capacity(n, n, a.data.len())?;
    b.layout().require_capacity(n, n, b.data.len())?;
    output(&w, values)?;
    output(&dev_info, 1)?;
    context.validate_region(&a.data)?;
    context.validate_region(&b.data)?;
    context.validate_region(&w)?;
    context.validate_region(&scratch)?;
    context.validate_region(&dev_info)?;
    let lwork = ssygvd_buffer_size(
        context,
        eig_type,
        mode,
        fill_mode,
        n,
        a.layout(),
        b.layout(),
    )?;
    workspace(scratch.len(), lwork)?;
    // SAFETY: validated metadata and the caller's queued-access contract cover native use.
    unsafe {
        try_ffi!(sys::cusolverDnSsygvd(
            context.as_raw(),
            eig_type.into(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.data.as_mut_ptr().cast(),
            to_i32(a.leading_dimension, "lda")?,
            b.data.as_mut_ptr().cast(),
            to_i32(b.leading_dimension, "ldb")?,
            w.as_mut_ptr().cast(),
            scratch.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast()
        ))?;
    }
    Ok(())
}

/// Queries scratch capacity from layout metadata without allocating buffers.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, the selected eigenvalue interval is invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn ssygvdx_selected_buffer_size(
    context: &Context,
    eig_type: EigenType,
    mode: EigenMode,
    fill_mode: FillMode,
    selection: &EigenSelection<f32>,
    n: usize,
    a: MatrixLayout,
    b: MatrixLayout,
) -> Result<usize> {
    context.bind()?;
    a.validate(n, n)?;
    b.validate(n, n)?;
    let (range, vl, vu, il, iu) = native_selection(selection, n)?;
    let mut lwork = 0;
    let mut ignored_count = MaybeUninit::<i32>::uninit();
    // SAFETY: native workspace queries inspect metadata without accessing matrices.
    unsafe {
        try_ffi!(sys::cusolverDnSsygvdx_bufferSize(
            context.as_raw(),
            eig_type.into(),
            mode.into(),
            range.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            ptr::null(),
            to_i32(a.leading_dimension, "lda")?,
            ptr::null(),
            to_i32(b.leading_dimension, "ldb")?,
            *vl,
            *vu,
            to_i32(il, "il")?,
            to_i32(iu, "iu")?,
            ignored_count.as_mut_ptr(),
            ptr::null(),
            &raw mut lwork
        ))?;
    }
    to_usize(lwork, "lwork")
}

/// Enqueues the native eigenvalue operation.
///
/// # Safety
/// Keep all borrowed resources alive and order their accesses through completion
/// and every replay. Inspect completed status before reading or promoting results.
/// Jacobi parameter storage is exclusive through completion when present.
/// Selected host counts and bounds must also survive completion and replay.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a device buffer belongs to another CUDA context, matrix dimensions, leading dimensions, or storage capacities are invalid, checked size arithmetic or native integer conversion fails, the status output capacity is insufficient, a pivot, reflector, or numerical output buffer is too small, device scratch capacity is smaller than the queried requirement, the eigenvalue selection is invalid, or cuSOLVER rejects the configuration or returns an error while querying workspace or submitting execution.
pub unsafe fn ssygvdx_selected(
    context: &Context,
    eig_type: EigenType,
    mode: EigenMode,
    fill_mode: FillMode,
    selection: &EigenSelection<f32>,
    n: usize,
    mut a: MatrixMut<'_, f32>,
    mut b: MatrixMut<'_, f32>,
    mut w: DeviceOutputView<'_, f32>,
    mut scratch: DeviceOutputView<'_, f32>,
    mut dev_info: DeviceOutputView<'_, i32>,
    selected_count: &mut MaybeUninit<i32>,
) -> Result<()> {
    context.bind()?;
    let values = n;
    a.layout().require_capacity(n, n, a.data.len())?;
    b.layout().require_capacity(n, n, b.data.len())?;
    output(&w, values)?;
    output(&dev_info, 1)?;
    context.validate_region(&a.data)?;
    context.validate_region(&b.data)?;
    context.validate_region(&w)?;
    context.validate_region(&scratch)?;
    context.validate_region(&dev_info)?;
    let lwork = ssygvdx_selected_buffer_size(
        context,
        eig_type,
        mode,
        fill_mode,
        selection,
        n,
        a.layout(),
        b.layout(),
    )?;
    workspace(scratch.len(), lwork)?;
    let (range, vl, vu, il, iu) = native_selection(selection, n)?;
    // SAFETY: validated metadata and the caller's queued-access contract cover native use.
    unsafe {
        try_ffi!(sys::cusolverDnSsygvdx(
            context.as_raw(),
            eig_type.into(),
            mode.into(),
            range.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.data.as_mut_ptr().cast(),
            to_i32(a.leading_dimension, "lda")?,
            b.data.as_mut_ptr().cast(),
            to_i32(b.leading_dimension, "ldb")?,
            *vl,
            *vu,
            to_i32(il, "il")?,
            to_i32(iu, "iu")?,
            selected_count.as_mut_ptr(),
            w.as_mut_ptr().cast(),
            scratch.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast()
        ))?;
    }
    Ok(())
}

/// Queries scratch capacity from layout metadata without allocating buffers.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn dsyevj_buffer_size(
    context: &Context,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    a: MatrixLayout,
    params: &SyevjInfo,
) -> Result<usize> {
    context.bind()?;
    a.validate(n, n)?;
    let mut lwork = 0;
    // SAFETY: native workspace queries inspect metadata without accessing matrices.
    unsafe {
        try_ffi!(sys::cusolverDnDsyevj_bufferSize(
            context.as_raw(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            ptr::null(),
            to_i32(a.leading_dimension, "lda")?,
            ptr::null(),
            &raw mut lwork,
            params.as_raw()
        ))?;
    }
    to_usize(lwork, "lwork")
}

/// Enqueues the native eigenvalue operation.
///
/// # Safety
/// Keep all borrowed resources alive and order their accesses through completion
/// and every replay. Inspect completed status before reading or promoting results.
/// Jacobi parameter storage is exclusive through completion when present.
/// Selected host counts and bounds must also survive completion and replay.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a device buffer belongs to another CUDA context, matrix dimensions, leading dimensions, or storage capacities are invalid, checked size arithmetic or native integer conversion fails, the status output capacity is insufficient, a pivot, reflector, or numerical output buffer is too small, device scratch capacity is smaller than the queried requirement, or cuSOLVER rejects the configuration or returns an error while querying workspace or submitting execution.
pub unsafe fn dsyevj(
    context: &Context,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    mut a: MatrixMut<'_, f64>,
    mut w: DeviceOutputView<'_, f64>,
    mut scratch: DeviceOutputView<'_, f64>,
    mut dev_info: DeviceOutputView<'_, i32>,
    params: &mut SyevjInfo,
) -> Result<()> {
    context.bind()?;
    let values = n;
    a.layout().require_capacity(n, n, a.data.len())?;
    output(&w, values)?;
    output(&dev_info, 1)?;
    context.validate_region(&a.data)?;
    context.validate_region(&w)?;
    context.validate_region(&scratch)?;
    context.validate_region(&dev_info)?;
    let lwork = dsyevj_buffer_size(context, mode, fill_mode, n, a.layout(), params)?;
    workspace(scratch.len(), lwork)?;
    // SAFETY: validated metadata and the caller's queued-access contract cover native use.
    unsafe {
        try_ffi!(sys::cusolverDnDsyevj(
            context.as_raw(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.data.as_mut_ptr().cast(),
            to_i32(a.leading_dimension, "lda")?,
            w.as_mut_ptr().cast(),
            scratch.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast(),
            params.as_raw()
        ))?;
    }
    Ok(())
}

/// Queries scratch capacity from layout metadata without allocating buffers.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, the batch count or packed storage extent is invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn dsyevj_batched_buffer_size(
    context: &Context,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    a: MatrixLayout,
    params: &SyevjInfo,
    batch_count: usize,
) -> Result<usize> {
    context.bind()?;
    batch_layout(n, a, batch_count, false)?;
    let mut lwork = 0;
    // SAFETY: native workspace queries inspect metadata without accessing matrices.
    unsafe {
        try_ffi!(sys::cusolverDnDsyevjBatched_bufferSize(
            context.as_raw(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            ptr::null(),
            to_i32(a.leading_dimension, "lda")?,
            ptr::null(),
            &raw mut lwork,
            params.as_raw(),
            to_i32(batch_count, "batch_count")?
        ))?;
    }
    to_usize(lwork, "lwork")
}

/// Enqueues the native eigenvalue operation.
///
/// # Safety
/// Keep all borrowed resources alive and order their accesses through completion
/// and every replay. Inspect completed status before reading or promoting results.
/// Jacobi parameter storage is exclusive through completion when present.
/// Selected host counts and bounds must also survive completion and replay.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a device buffer belongs to another CUDA context, matrix dimensions, leading dimensions, or storage capacities are invalid, checked size arithmetic or native integer conversion fails, the status output capacity is insufficient, a pivot, reflector, or numerical output buffer is too small, device scratch capacity is smaller than the queried requirement, batch metadata or related pointer-table lengths are invalid, or cuSOLVER rejects the configuration or returns an error while querying workspace or submitting execution.
pub unsafe fn dsyevj_batched(
    context: &Context,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    mut a: MatrixMut<'_, f64>,
    mut w: DeviceOutputView<'_, f64>,
    mut scratch: DeviceOutputView<'_, f64>,
    mut dev_info: DeviceOutputView<'_, i32>,
    params: &mut SyevjInfo,
    batch_count: usize,
) -> Result<()> {
    context.bind()?;
    let values = batch_layout(n, a.layout(), batch_count, false)?;
    a.layout().require_capacity(n, values, a.data.len())?;
    output(&w, values)?;
    output(&dev_info, batch_count)?;
    context.validate_region(&a.data)?;
    context.validate_region(&w)?;
    context.validate_region(&scratch)?;
    context.validate_region(&dev_info)?;
    let lwork =
        dsyevj_batched_buffer_size(context, mode, fill_mode, n, a.layout(), params, batch_count)?;
    workspace(scratch.len(), lwork)?;
    // SAFETY: validated metadata and the caller's queued-access contract cover native use.
    unsafe {
        try_ffi!(sys::cusolverDnDsyevjBatched(
            context.as_raw(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.data.as_mut_ptr().cast(),
            to_i32(a.leading_dimension, "lda")?,
            w.as_mut_ptr().cast(),
            scratch.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast(),
            params.as_raw(),
            to_i32(batch_count, "batch_count")?
        ))?;
    }
    Ok(())
}

/// Queries scratch capacity from layout metadata without allocating buffers.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn dsygvj_buffer_size(
    context: &Context,
    eig_type: EigenType,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    a: MatrixLayout,
    b: MatrixLayout,
    params: &SyevjInfo,
) -> Result<usize> {
    context.bind()?;
    a.validate(n, n)?;
    b.validate(n, n)?;
    let mut lwork = 0;
    // SAFETY: native workspace queries inspect metadata without accessing matrices.
    unsafe {
        try_ffi!(sys::cusolverDnDsygvj_bufferSize(
            context.as_raw(),
            eig_type.into(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            ptr::null(),
            to_i32(a.leading_dimension, "lda")?,
            ptr::null(),
            to_i32(b.leading_dimension, "ldb")?,
            ptr::null(),
            &raw mut lwork,
            params.as_raw()
        ))?;
    }
    to_usize(lwork, "lwork")
}

/// Enqueues the native eigenvalue operation.
///
/// # Safety
/// Keep all borrowed resources alive and order their accesses through completion
/// and every replay. Inspect completed status before reading or promoting results.
/// Jacobi parameter storage is exclusive through completion when present.
/// Selected host counts and bounds must also survive completion and replay.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a device buffer belongs to another CUDA context, matrix dimensions, leading dimensions, or storage capacities are invalid, checked size arithmetic or native integer conversion fails, the status output capacity is insufficient, a pivot, reflector, or numerical output buffer is too small, device scratch capacity is smaller than the queried requirement, or cuSOLVER rejects the configuration or returns an error while querying workspace or submitting execution.
pub unsafe fn dsygvj(
    context: &Context,
    eig_type: EigenType,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    mut a: MatrixMut<'_, f64>,
    mut b: MatrixMut<'_, f64>,
    mut w: DeviceOutputView<'_, f64>,
    mut scratch: DeviceOutputView<'_, f64>,
    mut dev_info: DeviceOutputView<'_, i32>,
    params: &mut SyevjInfo,
) -> Result<()> {
    context.bind()?;
    let values = n;
    a.layout().require_capacity(n, n, a.data.len())?;
    b.layout().require_capacity(n, n, b.data.len())?;
    output(&w, values)?;
    output(&dev_info, 1)?;
    context.validate_region(&a.data)?;
    context.validate_region(&b.data)?;
    context.validate_region(&w)?;
    context.validate_region(&scratch)?;
    context.validate_region(&dev_info)?;
    let lwork = dsygvj_buffer_size(
        context,
        eig_type,
        mode,
        fill_mode,
        n,
        a.layout(),
        b.layout(),
        params,
    )?;
    workspace(scratch.len(), lwork)?;
    // SAFETY: validated metadata and the caller's queued-access contract cover native use.
    unsafe {
        try_ffi!(sys::cusolverDnDsygvj(
            context.as_raw(),
            eig_type.into(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.data.as_mut_ptr().cast(),
            to_i32(a.leading_dimension, "lda")?,
            b.data.as_mut_ptr().cast(),
            to_i32(b.leading_dimension, "ldb")?,
            w.as_mut_ptr().cast(),
            scratch.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast(),
            params.as_raw()
        ))?;
    }
    Ok(())
}

/// Queries scratch capacity from layout metadata without allocating buffers.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn dsygvd_buffer_size(
    context: &Context,
    eig_type: EigenType,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    a: MatrixLayout,
    b: MatrixLayout,
) -> Result<usize> {
    context.bind()?;
    a.validate(n, n)?;
    b.validate(n, n)?;
    let mut lwork = 0;
    // SAFETY: native workspace queries inspect metadata without accessing matrices.
    unsafe {
        try_ffi!(sys::cusolverDnDsygvd_bufferSize(
            context.as_raw(),
            eig_type.into(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            ptr::null(),
            to_i32(a.leading_dimension, "lda")?,
            ptr::null(),
            to_i32(b.leading_dimension, "ldb")?,
            ptr::null(),
            &raw mut lwork
        ))?;
    }
    to_usize(lwork, "lwork")
}

/// Enqueues the native eigenvalue operation.
///
/// # Safety
/// Keep all borrowed resources alive and order their accesses through completion
/// and every replay. Inspect completed status before reading or promoting results.
/// Jacobi parameter storage is exclusive through completion when present.
/// Selected host counts and bounds must also survive completion and replay.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a device buffer belongs to another CUDA context, matrix dimensions, leading dimensions, or storage capacities are invalid, checked size arithmetic or native integer conversion fails, the status output capacity is insufficient, a pivot, reflector, or numerical output buffer is too small, device scratch capacity is smaller than the queried requirement, or cuSOLVER rejects the configuration or returns an error while querying workspace or submitting execution.
pub unsafe fn dsygvd(
    context: &Context,
    eig_type: EigenType,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    mut a: MatrixMut<'_, f64>,
    mut b: MatrixMut<'_, f64>,
    mut w: DeviceOutputView<'_, f64>,
    mut scratch: DeviceOutputView<'_, f64>,
    mut dev_info: DeviceOutputView<'_, i32>,
) -> Result<()> {
    context.bind()?;
    let values = n;
    a.layout().require_capacity(n, n, a.data.len())?;
    b.layout().require_capacity(n, n, b.data.len())?;
    output(&w, values)?;
    output(&dev_info, 1)?;
    context.validate_region(&a.data)?;
    context.validate_region(&b.data)?;
    context.validate_region(&w)?;
    context.validate_region(&scratch)?;
    context.validate_region(&dev_info)?;
    let lwork = dsygvd_buffer_size(
        context,
        eig_type,
        mode,
        fill_mode,
        n,
        a.layout(),
        b.layout(),
    )?;
    workspace(scratch.len(), lwork)?;
    // SAFETY: validated metadata and the caller's queued-access contract cover native use.
    unsafe {
        try_ffi!(sys::cusolverDnDsygvd(
            context.as_raw(),
            eig_type.into(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.data.as_mut_ptr().cast(),
            to_i32(a.leading_dimension, "lda")?,
            b.data.as_mut_ptr().cast(),
            to_i32(b.leading_dimension, "ldb")?,
            w.as_mut_ptr().cast(),
            scratch.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast()
        ))?;
    }
    Ok(())
}

/// Queries scratch capacity from layout metadata without allocating buffers.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, the selected eigenvalue interval is invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn dsygvdx_selected_buffer_size(
    context: &Context,
    eig_type: EigenType,
    mode: EigenMode,
    fill_mode: FillMode,
    selection: &EigenSelection<f64>,
    n: usize,
    a: MatrixLayout,
    b: MatrixLayout,
) -> Result<usize> {
    context.bind()?;
    a.validate(n, n)?;
    b.validate(n, n)?;
    let (range, vl, vu, il, iu) = native_selection(selection, n)?;
    let mut lwork = 0;
    let mut ignored_count = MaybeUninit::<i32>::uninit();
    // SAFETY: native workspace queries inspect metadata without accessing matrices.
    unsafe {
        try_ffi!(sys::cusolverDnDsygvdx_bufferSize(
            context.as_raw(),
            eig_type.into(),
            mode.into(),
            range.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            ptr::null(),
            to_i32(a.leading_dimension, "lda")?,
            ptr::null(),
            to_i32(b.leading_dimension, "ldb")?,
            *vl,
            *vu,
            to_i32(il, "il")?,
            to_i32(iu, "iu")?,
            ignored_count.as_mut_ptr(),
            ptr::null(),
            &raw mut lwork
        ))?;
    }
    to_usize(lwork, "lwork")
}

/// Enqueues the native eigenvalue operation.
///
/// # Safety
/// Keep all borrowed resources alive and order their accesses through completion
/// and every replay. Inspect completed status before reading or promoting results.
/// Jacobi parameter storage is exclusive through completion when present.
/// Selected host counts and bounds must also survive completion and replay.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a device buffer belongs to another CUDA context, matrix dimensions, leading dimensions, or storage capacities are invalid, checked size arithmetic or native integer conversion fails, the status output capacity is insufficient, a pivot, reflector, or numerical output buffer is too small, device scratch capacity is smaller than the queried requirement, the eigenvalue selection is invalid, or cuSOLVER rejects the configuration or returns an error while querying workspace or submitting execution.
pub unsafe fn dsygvdx_selected(
    context: &Context,
    eig_type: EigenType,
    mode: EigenMode,
    fill_mode: FillMode,
    selection: &EigenSelection<f64>,
    n: usize,
    mut a: MatrixMut<'_, f64>,
    mut b: MatrixMut<'_, f64>,
    mut w: DeviceOutputView<'_, f64>,
    mut scratch: DeviceOutputView<'_, f64>,
    mut dev_info: DeviceOutputView<'_, i32>,
    selected_count: &mut MaybeUninit<i32>,
) -> Result<()> {
    context.bind()?;
    let values = n;
    a.layout().require_capacity(n, n, a.data.len())?;
    b.layout().require_capacity(n, n, b.data.len())?;
    output(&w, values)?;
    output(&dev_info, 1)?;
    context.validate_region(&a.data)?;
    context.validate_region(&b.data)?;
    context.validate_region(&w)?;
    context.validate_region(&scratch)?;
    context.validate_region(&dev_info)?;
    let lwork = dsygvdx_selected_buffer_size(
        context,
        eig_type,
        mode,
        fill_mode,
        selection,
        n,
        a.layout(),
        b.layout(),
    )?;
    workspace(scratch.len(), lwork)?;
    let (range, vl, vu, il, iu) = native_selection(selection, n)?;
    // SAFETY: validated metadata and the caller's queued-access contract cover native use.
    unsafe {
        try_ffi!(sys::cusolverDnDsygvdx(
            context.as_raw(),
            eig_type.into(),
            mode.into(),
            range.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.data.as_mut_ptr().cast(),
            to_i32(a.leading_dimension, "lda")?,
            b.data.as_mut_ptr().cast(),
            to_i32(b.leading_dimension, "ldb")?,
            *vl,
            *vu,
            to_i32(il, "il")?,
            to_i32(iu, "iu")?,
            selected_count.as_mut_ptr(),
            w.as_mut_ptr().cast(),
            scratch.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast()
        ))?;
    }
    Ok(())
}

/// Queries scratch capacity from layout metadata without allocating buffers.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn cheevj_buffer_size(
    context: &Context,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    a: MatrixLayout,
    params: &SyevjInfo,
) -> Result<usize> {
    context.bind()?;
    a.validate(n, n)?;
    let mut lwork = 0;
    // SAFETY: native workspace queries inspect metadata without accessing matrices.
    unsafe {
        try_ffi!(sys::cusolverDnCheevj_bufferSize(
            context.as_raw(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            ptr::null(),
            to_i32(a.leading_dimension, "lda")?,
            ptr::null(),
            &raw mut lwork,
            params.as_raw()
        ))?;
    }
    to_usize(lwork, "lwork")
}

/// Enqueues the native eigenvalue operation.
///
/// # Safety
/// Keep all borrowed resources alive and order their accesses through completion
/// and every replay. Inspect completed status before reading or promoting results.
/// Jacobi parameter storage is exclusive through completion when present.
/// Selected host counts and bounds must also survive completion and replay.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a device buffer belongs to another CUDA context, matrix dimensions, leading dimensions, or storage capacities are invalid, checked size arithmetic or native integer conversion fails, the status output capacity is insufficient, a pivot, reflector, or numerical output buffer is too small, device scratch capacity is smaller than the queried requirement, or cuSOLVER rejects the configuration or returns an error while querying workspace or submitting execution.
pub unsafe fn cheevj(
    context: &Context,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    mut a: MatrixMut<'_, Complex32>,
    mut w: DeviceOutputView<'_, f32>,
    mut scratch: DeviceOutputView<'_, Complex32>,
    mut dev_info: DeviceOutputView<'_, i32>,
    params: &mut SyevjInfo,
) -> Result<()> {
    context.bind()?;
    let values = n;
    a.layout().require_capacity(n, n, a.data.len())?;
    output(&w, values)?;
    output(&dev_info, 1)?;
    context.validate_region(&a.data)?;
    context.validate_region(&w)?;
    context.validate_region(&scratch)?;
    context.validate_region(&dev_info)?;
    let lwork = cheevj_buffer_size(context, mode, fill_mode, n, a.layout(), params)?;
    workspace(scratch.len(), lwork)?;
    // SAFETY: validated metadata and the caller's queued-access contract cover native use.
    unsafe {
        try_ffi!(sys::cusolverDnCheevj(
            context.as_raw(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.data.as_mut_ptr().cast(),
            to_i32(a.leading_dimension, "lda")?,
            w.as_mut_ptr().cast(),
            scratch.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast(),
            params.as_raw()
        ))?;
    }
    Ok(())
}

/// Queries scratch capacity from layout metadata without allocating buffers.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, the batch count or packed storage extent is invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn cheevj_batched_buffer_size(
    context: &Context,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    a: MatrixLayout,
    params: &SyevjInfo,
    batch_count: usize,
) -> Result<usize> {
    context.bind()?;
    batch_layout(n, a, batch_count, false)?;
    let mut lwork = 0;
    // SAFETY: native workspace queries inspect metadata without accessing matrices.
    unsafe {
        try_ffi!(sys::cusolverDnCheevjBatched_bufferSize(
            context.as_raw(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            ptr::null(),
            to_i32(a.leading_dimension, "lda")?,
            ptr::null(),
            &raw mut lwork,
            params.as_raw(),
            to_i32(batch_count, "batch_count")?
        ))?;
    }
    to_usize(lwork, "lwork")
}

/// Enqueues the native eigenvalue operation.
///
/// # Safety
/// Keep all borrowed resources alive and order their accesses through completion
/// and every replay. Inspect completed status before reading or promoting results.
/// Jacobi parameter storage is exclusive through completion when present.
/// Selected host counts and bounds must also survive completion and replay.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a device buffer belongs to another CUDA context, matrix dimensions, leading dimensions, or storage capacities are invalid, checked size arithmetic or native integer conversion fails, the status output capacity is insufficient, a pivot, reflector, or numerical output buffer is too small, device scratch capacity is smaller than the queried requirement, batch metadata or related pointer-table lengths are invalid, or cuSOLVER rejects the configuration or returns an error while querying workspace or submitting execution.
pub unsafe fn cheevj_batched(
    context: &Context,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    mut a: MatrixMut<'_, Complex32>,
    mut w: DeviceOutputView<'_, f32>,
    mut scratch: DeviceOutputView<'_, Complex32>,
    mut dev_info: DeviceOutputView<'_, i32>,
    params: &mut SyevjInfo,
    batch_count: usize,
) -> Result<()> {
    context.bind()?;
    let values = batch_layout(n, a.layout(), batch_count, false)?;
    a.layout().require_capacity(n, values, a.data.len())?;
    output(&w, values)?;
    output(&dev_info, batch_count)?;
    context.validate_region(&a.data)?;
    context.validate_region(&w)?;
    context.validate_region(&scratch)?;
    context.validate_region(&dev_info)?;
    let lwork =
        cheevj_batched_buffer_size(context, mode, fill_mode, n, a.layout(), params, batch_count)?;
    workspace(scratch.len(), lwork)?;
    // SAFETY: validated metadata and the caller's queued-access contract cover native use.
    unsafe {
        try_ffi!(sys::cusolverDnCheevjBatched(
            context.as_raw(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.data.as_mut_ptr().cast(),
            to_i32(a.leading_dimension, "lda")?,
            w.as_mut_ptr().cast(),
            scratch.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast(),
            params.as_raw(),
            to_i32(batch_count, "batch_count")?
        ))?;
    }
    Ok(())
}

/// Queries scratch capacity from layout metadata without allocating buffers.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn chegvj_buffer_size(
    context: &Context,
    eig_type: EigenType,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    a: MatrixLayout,
    b: MatrixLayout,
    params: &SyevjInfo,
) -> Result<usize> {
    context.bind()?;
    a.validate(n, n)?;
    b.validate(n, n)?;
    let mut lwork = 0;
    // SAFETY: native workspace queries inspect metadata without accessing matrices.
    unsafe {
        try_ffi!(sys::cusolverDnChegvj_bufferSize(
            context.as_raw(),
            eig_type.into(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            ptr::null(),
            to_i32(a.leading_dimension, "lda")?,
            ptr::null(),
            to_i32(b.leading_dimension, "ldb")?,
            ptr::null(),
            &raw mut lwork,
            params.as_raw()
        ))?;
    }
    to_usize(lwork, "lwork")
}

/// Enqueues the native eigenvalue operation.
///
/// # Safety
/// Keep all borrowed resources alive and order their accesses through completion
/// and every replay. Inspect completed status before reading or promoting results.
/// Jacobi parameter storage is exclusive through completion when present.
/// Selected host counts and bounds must also survive completion and replay.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a device buffer belongs to another CUDA context, matrix dimensions, leading dimensions, or storage capacities are invalid, checked size arithmetic or native integer conversion fails, the status output capacity is insufficient, a pivot, reflector, or numerical output buffer is too small, device scratch capacity is smaller than the queried requirement, or cuSOLVER rejects the configuration or returns an error while querying workspace or submitting execution.
pub unsafe fn chegvj(
    context: &Context,
    eig_type: EigenType,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    mut a: MatrixMut<'_, Complex32>,
    mut b: MatrixMut<'_, Complex32>,
    mut w: DeviceOutputView<'_, f32>,
    mut scratch: DeviceOutputView<'_, Complex32>,
    mut dev_info: DeviceOutputView<'_, i32>,
    params: &mut SyevjInfo,
) -> Result<()> {
    context.bind()?;
    let values = n;
    a.layout().require_capacity(n, n, a.data.len())?;
    b.layout().require_capacity(n, n, b.data.len())?;
    output(&w, values)?;
    output(&dev_info, 1)?;
    context.validate_region(&a.data)?;
    context.validate_region(&b.data)?;
    context.validate_region(&w)?;
    context.validate_region(&scratch)?;
    context.validate_region(&dev_info)?;
    let lwork = chegvj_buffer_size(
        context,
        eig_type,
        mode,
        fill_mode,
        n,
        a.layout(),
        b.layout(),
        params,
    )?;
    workspace(scratch.len(), lwork)?;
    // SAFETY: validated metadata and the caller's queued-access contract cover native use.
    unsafe {
        try_ffi!(sys::cusolverDnChegvj(
            context.as_raw(),
            eig_type.into(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.data.as_mut_ptr().cast(),
            to_i32(a.leading_dimension, "lda")?,
            b.data.as_mut_ptr().cast(),
            to_i32(b.leading_dimension, "ldb")?,
            w.as_mut_ptr().cast(),
            scratch.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast(),
            params.as_raw()
        ))?;
    }
    Ok(())
}

/// Queries scratch capacity from layout metadata without allocating buffers.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn chegvd_buffer_size(
    context: &Context,
    eig_type: EigenType,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    a: MatrixLayout,
    b: MatrixLayout,
) -> Result<usize> {
    context.bind()?;
    a.validate(n, n)?;
    b.validate(n, n)?;
    let mut lwork = 0;
    // SAFETY: native workspace queries inspect metadata without accessing matrices.
    unsafe {
        try_ffi!(sys::cusolverDnChegvd_bufferSize(
            context.as_raw(),
            eig_type.into(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            ptr::null(),
            to_i32(a.leading_dimension, "lda")?,
            ptr::null(),
            to_i32(b.leading_dimension, "ldb")?,
            ptr::null(),
            &raw mut lwork
        ))?;
    }
    to_usize(lwork, "lwork")
}

/// Enqueues the native eigenvalue operation.
///
/// # Safety
/// Keep all borrowed resources alive and order their accesses through completion
/// and every replay. Inspect completed status before reading or promoting results.
/// Jacobi parameter storage is exclusive through completion when present.
/// Selected host counts and bounds must also survive completion and replay.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a device buffer belongs to another CUDA context, matrix dimensions, leading dimensions, or storage capacities are invalid, checked size arithmetic or native integer conversion fails, the status output capacity is insufficient, a pivot, reflector, or numerical output buffer is too small, device scratch capacity is smaller than the queried requirement, or cuSOLVER rejects the configuration or returns an error while querying workspace or submitting execution.
pub unsafe fn chegvd(
    context: &Context,
    eig_type: EigenType,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    mut a: MatrixMut<'_, Complex32>,
    mut b: MatrixMut<'_, Complex32>,
    mut w: DeviceOutputView<'_, f32>,
    mut scratch: DeviceOutputView<'_, Complex32>,
    mut dev_info: DeviceOutputView<'_, i32>,
) -> Result<()> {
    context.bind()?;
    let values = n;
    a.layout().require_capacity(n, n, a.data.len())?;
    b.layout().require_capacity(n, n, b.data.len())?;
    output(&w, values)?;
    output(&dev_info, 1)?;
    context.validate_region(&a.data)?;
    context.validate_region(&b.data)?;
    context.validate_region(&w)?;
    context.validate_region(&scratch)?;
    context.validate_region(&dev_info)?;
    let lwork = chegvd_buffer_size(
        context,
        eig_type,
        mode,
        fill_mode,
        n,
        a.layout(),
        b.layout(),
    )?;
    workspace(scratch.len(), lwork)?;
    // SAFETY: validated metadata and the caller's queued-access contract cover native use.
    unsafe {
        try_ffi!(sys::cusolverDnChegvd(
            context.as_raw(),
            eig_type.into(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.data.as_mut_ptr().cast(),
            to_i32(a.leading_dimension, "lda")?,
            b.data.as_mut_ptr().cast(),
            to_i32(b.leading_dimension, "ldb")?,
            w.as_mut_ptr().cast(),
            scratch.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast()
        ))?;
    }
    Ok(())
}

/// Queries scratch capacity from layout metadata without allocating buffers.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, the selected eigenvalue interval is invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn chegvdx_selected_buffer_size(
    context: &Context,
    eig_type: EigenType,
    mode: EigenMode,
    fill_mode: FillMode,
    selection: &EigenSelection<f32>,
    n: usize,
    a: MatrixLayout,
    b: MatrixLayout,
) -> Result<usize> {
    context.bind()?;
    a.validate(n, n)?;
    b.validate(n, n)?;
    let (range, vl, vu, il, iu) = native_selection(selection, n)?;
    let mut lwork = 0;
    let mut ignored_count = MaybeUninit::<i32>::uninit();
    // SAFETY: native workspace queries inspect metadata without accessing matrices.
    unsafe {
        try_ffi!(sys::cusolverDnChegvdx_bufferSize(
            context.as_raw(),
            eig_type.into(),
            mode.into(),
            range.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            ptr::null(),
            to_i32(a.leading_dimension, "lda")?,
            ptr::null(),
            to_i32(b.leading_dimension, "ldb")?,
            *vl,
            *vu,
            to_i32(il, "il")?,
            to_i32(iu, "iu")?,
            ignored_count.as_mut_ptr(),
            ptr::null(),
            &raw mut lwork
        ))?;
    }
    to_usize(lwork, "lwork")
}

/// Enqueues the native eigenvalue operation.
///
/// # Safety
/// Keep all borrowed resources alive and order their accesses through completion
/// and every replay. Inspect completed status before reading or promoting results.
/// Jacobi parameter storage is exclusive through completion when present.
/// Selected host counts and bounds must also survive completion and replay.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a device buffer belongs to another CUDA context, matrix dimensions, leading dimensions, or storage capacities are invalid, checked size arithmetic or native integer conversion fails, the status output capacity is insufficient, a pivot, reflector, or numerical output buffer is too small, device scratch capacity is smaller than the queried requirement, the eigenvalue selection is invalid, or cuSOLVER rejects the configuration or returns an error while querying workspace or submitting execution.
pub unsafe fn chegvdx_selected(
    context: &Context,
    eig_type: EigenType,
    mode: EigenMode,
    fill_mode: FillMode,
    selection: &EigenSelection<f32>,
    n: usize,
    mut a: MatrixMut<'_, Complex32>,
    mut b: MatrixMut<'_, Complex32>,
    mut w: DeviceOutputView<'_, f32>,
    mut scratch: DeviceOutputView<'_, Complex32>,
    mut dev_info: DeviceOutputView<'_, i32>,
    selected_count: &mut MaybeUninit<i32>,
) -> Result<()> {
    context.bind()?;
    let values = n;
    a.layout().require_capacity(n, n, a.data.len())?;
    b.layout().require_capacity(n, n, b.data.len())?;
    output(&w, values)?;
    output(&dev_info, 1)?;
    context.validate_region(&a.data)?;
    context.validate_region(&b.data)?;
    context.validate_region(&w)?;
    context.validate_region(&scratch)?;
    context.validate_region(&dev_info)?;
    let lwork = chegvdx_selected_buffer_size(
        context,
        eig_type,
        mode,
        fill_mode,
        selection,
        n,
        a.layout(),
        b.layout(),
    )?;
    workspace(scratch.len(), lwork)?;
    let (range, vl, vu, il, iu) = native_selection(selection, n)?;
    // SAFETY: validated metadata and the caller's queued-access contract cover native use.
    unsafe {
        try_ffi!(sys::cusolverDnChegvdx(
            context.as_raw(),
            eig_type.into(),
            mode.into(),
            range.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.data.as_mut_ptr().cast(),
            to_i32(a.leading_dimension, "lda")?,
            b.data.as_mut_ptr().cast(),
            to_i32(b.leading_dimension, "ldb")?,
            *vl,
            *vu,
            to_i32(il, "il")?,
            to_i32(iu, "iu")?,
            selected_count.as_mut_ptr(),
            w.as_mut_ptr().cast(),
            scratch.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast()
        ))?;
    }
    Ok(())
}

/// Queries scratch capacity from layout metadata without allocating buffers.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn zheevj_buffer_size(
    context: &Context,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    a: MatrixLayout,
    params: &SyevjInfo,
) -> Result<usize> {
    context.bind()?;
    a.validate(n, n)?;
    let mut lwork = 0;
    // SAFETY: native workspace queries inspect metadata without accessing matrices.
    unsafe {
        try_ffi!(sys::cusolverDnZheevj_bufferSize(
            context.as_raw(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            ptr::null(),
            to_i32(a.leading_dimension, "lda")?,
            ptr::null(),
            &raw mut lwork,
            params.as_raw()
        ))?;
    }
    to_usize(lwork, "lwork")
}

/// Enqueues the native eigenvalue operation.
///
/// # Safety
/// Keep all borrowed resources alive and order their accesses through completion
/// and every replay. Inspect completed status before reading or promoting results.
/// Jacobi parameter storage is exclusive through completion when present.
/// Selected host counts and bounds must also survive completion and replay.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a device buffer belongs to another CUDA context, matrix dimensions, leading dimensions, or storage capacities are invalid, checked size arithmetic or native integer conversion fails, the status output capacity is insufficient, a pivot, reflector, or numerical output buffer is too small, device scratch capacity is smaller than the queried requirement, or cuSOLVER rejects the configuration or returns an error while querying workspace or submitting execution.
pub unsafe fn zheevj(
    context: &Context,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    mut a: MatrixMut<'_, Complex64>,
    mut w: DeviceOutputView<'_, f64>,
    mut scratch: DeviceOutputView<'_, Complex64>,
    mut dev_info: DeviceOutputView<'_, i32>,
    params: &mut SyevjInfo,
) -> Result<()> {
    context.bind()?;
    let values = n;
    a.layout().require_capacity(n, n, a.data.len())?;
    output(&w, values)?;
    output(&dev_info, 1)?;
    context.validate_region(&a.data)?;
    context.validate_region(&w)?;
    context.validate_region(&scratch)?;
    context.validate_region(&dev_info)?;
    let lwork = zheevj_buffer_size(context, mode, fill_mode, n, a.layout(), params)?;
    workspace(scratch.len(), lwork)?;
    // SAFETY: validated metadata and the caller's queued-access contract cover native use.
    unsafe {
        try_ffi!(sys::cusolverDnZheevj(
            context.as_raw(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.data.as_mut_ptr().cast(),
            to_i32(a.leading_dimension, "lda")?,
            w.as_mut_ptr().cast(),
            scratch.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast(),
            params.as_raw()
        ))?;
    }
    Ok(())
}

/// Queries scratch capacity from layout metadata without allocating buffers.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, the batch count or packed storage extent is invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn zheevj_batched_buffer_size(
    context: &Context,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    a: MatrixLayout,
    params: &SyevjInfo,
    batch_count: usize,
) -> Result<usize> {
    context.bind()?;
    batch_layout(n, a, batch_count, false)?;
    let mut lwork = 0;
    // SAFETY: native workspace queries inspect metadata without accessing matrices.
    unsafe {
        try_ffi!(sys::cusolverDnZheevjBatched_bufferSize(
            context.as_raw(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            ptr::null(),
            to_i32(a.leading_dimension, "lda")?,
            ptr::null(),
            &raw mut lwork,
            params.as_raw(),
            to_i32(batch_count, "batch_count")?
        ))?;
    }
    to_usize(lwork, "lwork")
}

/// Enqueues the native eigenvalue operation.
///
/// # Safety
/// Keep all borrowed resources alive and order their accesses through completion
/// and every replay. Inspect completed status before reading or promoting results.
/// Jacobi parameter storage is exclusive through completion when present.
/// Selected host counts and bounds must also survive completion and replay.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a device buffer belongs to another CUDA context, matrix dimensions, leading dimensions, or storage capacities are invalid, checked size arithmetic or native integer conversion fails, the status output capacity is insufficient, a pivot, reflector, or numerical output buffer is too small, device scratch capacity is smaller than the queried requirement, batch metadata or related pointer-table lengths are invalid, or cuSOLVER rejects the configuration or returns an error while querying workspace or submitting execution.
pub unsafe fn zheevj_batched(
    context: &Context,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    mut a: MatrixMut<'_, Complex64>,
    mut w: DeviceOutputView<'_, f64>,
    mut scratch: DeviceOutputView<'_, Complex64>,
    mut dev_info: DeviceOutputView<'_, i32>,
    params: &mut SyevjInfo,
    batch_count: usize,
) -> Result<()> {
    context.bind()?;
    let values = batch_layout(n, a.layout(), batch_count, false)?;
    a.layout().require_capacity(n, values, a.data.len())?;
    output(&w, values)?;
    output(&dev_info, batch_count)?;
    context.validate_region(&a.data)?;
    context.validate_region(&w)?;
    context.validate_region(&scratch)?;
    context.validate_region(&dev_info)?;
    let lwork =
        zheevj_batched_buffer_size(context, mode, fill_mode, n, a.layout(), params, batch_count)?;
    workspace(scratch.len(), lwork)?;
    // SAFETY: validated metadata and the caller's queued-access contract cover native use.
    unsafe {
        try_ffi!(sys::cusolverDnZheevjBatched(
            context.as_raw(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.data.as_mut_ptr().cast(),
            to_i32(a.leading_dimension, "lda")?,
            w.as_mut_ptr().cast(),
            scratch.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast(),
            params.as_raw(),
            to_i32(batch_count, "batch_count")?
        ))?;
    }
    Ok(())
}

/// Queries scratch capacity from layout metadata without allocating buffers.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn zhegvj_buffer_size(
    context: &Context,
    eig_type: EigenType,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    a: MatrixLayout,
    b: MatrixLayout,
    params: &SyevjInfo,
) -> Result<usize> {
    context.bind()?;
    a.validate(n, n)?;
    b.validate(n, n)?;
    let mut lwork = 0;
    // SAFETY: native workspace queries inspect metadata without accessing matrices.
    unsafe {
        try_ffi!(sys::cusolverDnZhegvj_bufferSize(
            context.as_raw(),
            eig_type.into(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            ptr::null(),
            to_i32(a.leading_dimension, "lda")?,
            ptr::null(),
            to_i32(b.leading_dimension, "ldb")?,
            ptr::null(),
            &raw mut lwork,
            params.as_raw()
        ))?;
    }
    to_usize(lwork, "lwork")
}

/// Enqueues the native eigenvalue operation.
///
/// # Safety
/// Keep all borrowed resources alive and order their accesses through completion
/// and every replay. Inspect completed status before reading or promoting results.
/// Jacobi parameter storage is exclusive through completion when present.
/// Selected host counts and bounds must also survive completion and replay.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a device buffer belongs to another CUDA context, matrix dimensions, leading dimensions, or storage capacities are invalid, checked size arithmetic or native integer conversion fails, the status output capacity is insufficient, a pivot, reflector, or numerical output buffer is too small, device scratch capacity is smaller than the queried requirement, or cuSOLVER rejects the configuration or returns an error while querying workspace or submitting execution.
pub unsafe fn zhegvj(
    context: &Context,
    eig_type: EigenType,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    mut a: MatrixMut<'_, Complex64>,
    mut b: MatrixMut<'_, Complex64>,
    mut w: DeviceOutputView<'_, f64>,
    mut scratch: DeviceOutputView<'_, Complex64>,
    mut dev_info: DeviceOutputView<'_, i32>,
    params: &mut SyevjInfo,
) -> Result<()> {
    context.bind()?;
    let values = n;
    a.layout().require_capacity(n, n, a.data.len())?;
    b.layout().require_capacity(n, n, b.data.len())?;
    output(&w, values)?;
    output(&dev_info, 1)?;
    context.validate_region(&a.data)?;
    context.validate_region(&b.data)?;
    context.validate_region(&w)?;
    context.validate_region(&scratch)?;
    context.validate_region(&dev_info)?;
    let lwork = zhegvj_buffer_size(
        context,
        eig_type,
        mode,
        fill_mode,
        n,
        a.layout(),
        b.layout(),
        params,
    )?;
    workspace(scratch.len(), lwork)?;
    // SAFETY: validated metadata and the caller's queued-access contract cover native use.
    unsafe {
        try_ffi!(sys::cusolverDnZhegvj(
            context.as_raw(),
            eig_type.into(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.data.as_mut_ptr().cast(),
            to_i32(a.leading_dimension, "lda")?,
            b.data.as_mut_ptr().cast(),
            to_i32(b.leading_dimension, "ldb")?,
            w.as_mut_ptr().cast(),
            scratch.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast(),
            params.as_raw()
        ))?;
    }
    Ok(())
}

/// Queries scratch capacity from layout metadata without allocating buffers.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn zhegvd_buffer_size(
    context: &Context,
    eig_type: EigenType,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    a: MatrixLayout,
    b: MatrixLayout,
) -> Result<usize> {
    context.bind()?;
    a.validate(n, n)?;
    b.validate(n, n)?;
    let mut lwork = 0;
    // SAFETY: native workspace queries inspect metadata without accessing matrices.
    unsafe {
        try_ffi!(sys::cusolverDnZhegvd_bufferSize(
            context.as_raw(),
            eig_type.into(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            ptr::null(),
            to_i32(a.leading_dimension, "lda")?,
            ptr::null(),
            to_i32(b.leading_dimension, "ldb")?,
            ptr::null(),
            &raw mut lwork
        ))?;
    }
    to_usize(lwork, "lwork")
}

/// Enqueues the native eigenvalue operation.
///
/// # Safety
/// Keep all borrowed resources alive and order their accesses through completion
/// and every replay. Inspect completed status before reading or promoting results.
/// Jacobi parameter storage is exclusive through completion when present.
/// Selected host counts and bounds must also survive completion and replay.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a device buffer belongs to another CUDA context, matrix dimensions, leading dimensions, or storage capacities are invalid, checked size arithmetic or native integer conversion fails, the status output capacity is insufficient, a pivot, reflector, or numerical output buffer is too small, device scratch capacity is smaller than the queried requirement, or cuSOLVER rejects the configuration or returns an error while querying workspace or submitting execution.
pub unsafe fn zhegvd(
    context: &Context,
    eig_type: EigenType,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    mut a: MatrixMut<'_, Complex64>,
    mut b: MatrixMut<'_, Complex64>,
    mut w: DeviceOutputView<'_, f64>,
    mut scratch: DeviceOutputView<'_, Complex64>,
    mut dev_info: DeviceOutputView<'_, i32>,
) -> Result<()> {
    context.bind()?;
    let values = n;
    a.layout().require_capacity(n, n, a.data.len())?;
    b.layout().require_capacity(n, n, b.data.len())?;
    output(&w, values)?;
    output(&dev_info, 1)?;
    context.validate_region(&a.data)?;
    context.validate_region(&b.data)?;
    context.validate_region(&w)?;
    context.validate_region(&scratch)?;
    context.validate_region(&dev_info)?;
    let lwork = zhegvd_buffer_size(
        context,
        eig_type,
        mode,
        fill_mode,
        n,
        a.layout(),
        b.layout(),
    )?;
    workspace(scratch.len(), lwork)?;
    // SAFETY: validated metadata and the caller's queued-access contract cover native use.
    unsafe {
        try_ffi!(sys::cusolverDnZhegvd(
            context.as_raw(),
            eig_type.into(),
            mode.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.data.as_mut_ptr().cast(),
            to_i32(a.leading_dimension, "lda")?,
            b.data.as_mut_ptr().cast(),
            to_i32(b.leading_dimension, "ldb")?,
            w.as_mut_ptr().cast(),
            scratch.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast()
        ))?;
    }
    Ok(())
}

/// Queries scratch capacity from layout metadata without allocating buffers.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, the selected eigenvalue interval is invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn zhegvdx_selected_buffer_size(
    context: &Context,
    eig_type: EigenType,
    mode: EigenMode,
    fill_mode: FillMode,
    selection: &EigenSelection<f64>,
    n: usize,
    a: MatrixLayout,
    b: MatrixLayout,
) -> Result<usize> {
    context.bind()?;
    a.validate(n, n)?;
    b.validate(n, n)?;
    let (range, vl, vu, il, iu) = native_selection(selection, n)?;
    let mut lwork = 0;
    let mut ignored_count = MaybeUninit::<i32>::uninit();
    // SAFETY: native workspace queries inspect metadata without accessing matrices.
    unsafe {
        try_ffi!(sys::cusolverDnZhegvdx_bufferSize(
            context.as_raw(),
            eig_type.into(),
            mode.into(),
            range.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            ptr::null(),
            to_i32(a.leading_dimension, "lda")?,
            ptr::null(),
            to_i32(b.leading_dimension, "ldb")?,
            *vl,
            *vu,
            to_i32(il, "il")?,
            to_i32(iu, "iu")?,
            ignored_count.as_mut_ptr(),
            ptr::null(),
            &raw mut lwork
        ))?;
    }
    to_usize(lwork, "lwork")
}

/// Enqueues the native eigenvalue operation.
///
/// # Safety
/// Keep all borrowed resources alive and order their accesses through completion
/// and every replay. Inspect completed status before reading or promoting results.
/// Jacobi parameter storage is exclusive through completion when present.
/// Selected host counts and bounds must also survive completion and replay.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a device buffer belongs to another CUDA context, matrix dimensions, leading dimensions, or storage capacities are invalid, checked size arithmetic or native integer conversion fails, the status output capacity is insufficient, a pivot, reflector, or numerical output buffer is too small, device scratch capacity is smaller than the queried requirement, the eigenvalue selection is invalid, or cuSOLVER rejects the configuration or returns an error while querying workspace or submitting execution.
pub unsafe fn zhegvdx_selected(
    context: &Context,
    eig_type: EigenType,
    mode: EigenMode,
    fill_mode: FillMode,
    selection: &EigenSelection<f64>,
    n: usize,
    mut a: MatrixMut<'_, Complex64>,
    mut b: MatrixMut<'_, Complex64>,
    mut w: DeviceOutputView<'_, f64>,
    mut scratch: DeviceOutputView<'_, Complex64>,
    mut dev_info: DeviceOutputView<'_, i32>,
    selected_count: &mut MaybeUninit<i32>,
) -> Result<()> {
    context.bind()?;
    let values = n;
    a.layout().require_capacity(n, n, a.data.len())?;
    b.layout().require_capacity(n, n, b.data.len())?;
    output(&w, values)?;
    output(&dev_info, 1)?;
    context.validate_region(&a.data)?;
    context.validate_region(&b.data)?;
    context.validate_region(&w)?;
    context.validate_region(&scratch)?;
    context.validate_region(&dev_info)?;
    let lwork = zhegvdx_selected_buffer_size(
        context,
        eig_type,
        mode,
        fill_mode,
        selection,
        n,
        a.layout(),
        b.layout(),
    )?;
    workspace(scratch.len(), lwork)?;
    let (range, vl, vu, il, iu) = native_selection(selection, n)?;
    // SAFETY: validated metadata and the caller's queued-access contract cover native use.
    unsafe {
        try_ffi!(sys::cusolverDnZhegvdx(
            context.as_raw(),
            eig_type.into(),
            mode.into(),
            range.into(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.data.as_mut_ptr().cast(),
            to_i32(a.leading_dimension, "lda")?,
            b.data.as_mut_ptr().cast(),
            to_i32(b.leading_dimension, "ldb")?,
            *vl,
            *vu,
            to_i32(il, "il")?,
            to_i32(iu, "iu")?,
            selected_count.as_mut_ptr(),
            w.as_mut_ptr().cast(),
            scratch.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast()
        ))?;
    }
    Ok(())
}

/// Queries scratch capacity from layout metadata without allocating buffers.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn xsyevd_buffer_size<T: SolverElement>(
    context: &Context,
    params: &Params,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    a: MatrixLayout,
) -> Result<WorkspaceSizes> {
    context.bind()?;
    a.validate(n, n)?;
    let mut device_bytes = 0;
    let mut host_bytes = 0;
    // SAFETY: the query reads metadata only.
    unsafe {
        try_ffi!(sys::cusolverDnXsyevd_bufferSize(
            context.as_raw(),
            params.as_raw(),
            mode.into(),
            fill_mode.into(),
            to_i64(n, "n")?,
            T::data_type().into(),
            ptr::null(),
            to_i64(a.leading_dimension, "lda")?,
            T::Real::data_type().into(),
            ptr::null(),
            T::data_type().into(),
            &raw mut device_bytes,
            &raw mut host_bytes
        ))?;
    }
    Ok(WorkspaceSizes::new(
        to_usize(device_bytes, "device workspace size")?,
        to_usize(host_bytes, "host workspace size")?,
    ))
}

/// Enqueues the native eigenvalue operation.
///
/// # Safety
/// Keep all borrowed resources alive and order their accesses through completion
/// and every replay. Inspect completed status before reading or promoting results.
/// Jacobi parameter storage is exclusive through completion when present.
/// Selected host counts and bounds must also survive completion and replay.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a device buffer belongs to another CUDA context, matrix dimensions, leading dimensions, or storage capacities are invalid, checked size arithmetic or native integer conversion fails, the status output capacity is insufficient, a pivot, reflector, or numerical output buffer is too small, device scratch capacity is smaller than the queried requirement, or cuSOLVER rejects the configuration or returns an error while querying workspace or submitting execution.
pub unsafe fn xsyevd<T: SolverElement>(
    context: &Context,
    params: &Params,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    mut a: MatrixMut<'_, T>,
    mut w: DeviceOutputView<'_, T::Real>,
    mut scratch: ByteWorkspaceMut<'_>,
    mut dev_info: DeviceOutputView<'_, i32>,
) -> Result<()> {
    context.bind()?;
    let values = n;
    a.layout().require_capacity(n, n, a.data.len())?;
    output(&w, values)?;
    output(&dev_info, 1)?;
    context.validate_region(&a.data)?;
    context.validate_region(&w)?;
    context.validate_region(&scratch.device)?;
    context.validate_region(&dev_info)?;
    let sizes = xsyevd_buffer_size::<T>(context, params, mode, fill_mode, n, a.layout())?;
    workspace(scratch.device.len(), sizes.device_bytes)?;
    workspace(scratch.host.len(), sizes.host_bytes)?;
    // SAFETY: validated storage and the caller's queued-access contract cover native use.
    unsafe {
        try_ffi!(sys::cusolverDnXsyevd(
            context.as_raw(),
            params.as_raw(),
            mode.into(),
            fill_mode.into(),
            to_i64(n, "n")?,
            T::data_type().into(),
            a.data.as_mut_ptr().cast(),
            to_i64(a.leading_dimension, "lda")?,
            T::Real::data_type().into(),
            w.as_mut_ptr().cast(),
            T::data_type().into(),
            scratch.device.as_mut_ptr().cast(),
            sizes.device_bytes as _,
            scratch.host.as_mut_ptr().cast(),
            sizes.host_bytes as _,
            dev_info.as_mut_ptr().cast()
        ))?;
    }
    Ok(())
}

/// Queries scratch capacity from layout metadata without allocating buffers.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, the selected eigenvalue interval is invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn xsyevdx_buffer_size<T: SolverElement>(
    context: &Context,
    params: &Params,
    mode: EigenMode,
    fill_mode: FillMode,
    selection: &EigenSelection<T::Real>,
    n: usize,
    a: MatrixLayout,
) -> Result<WorkspaceSizes> {
    context.bind()?;
    a.validate(n, n)?;
    let (range, vl, vu, il, iu) = native_selection(selection, n)?;
    let mut device_bytes = 0;
    let mut host_bytes = 0;
    let mut ignored_count = MaybeUninit::<i64>::uninit();
    // SAFETY: the query reads metadata only.
    unsafe {
        try_ffi!(sys::cusolverDnXsyevdx_bufferSize(
            context.as_raw(),
            params.as_raw(),
            mode.into(),
            range.into(),
            fill_mode.into(),
            to_i64(n, "n")?,
            T::data_type().into(),
            ptr::null(),
            to_i64(a.leading_dimension, "lda")?,
            ptr::from_ref(vl).cast_mut().cast(),
            ptr::from_ref(vu).cast_mut().cast(),
            to_i64(il, "il")?,
            to_i64(iu, "iu")?,
            ignored_count.as_mut_ptr(),
            T::Real::data_type().into(),
            ptr::null(),
            T::data_type().into(),
            &raw mut device_bytes,
            &raw mut host_bytes
        ))?;
    }
    Ok(WorkspaceSizes::new(
        to_usize(device_bytes, "device workspace size")?,
        to_usize(host_bytes, "host workspace size")?,
    ))
}

/// Enqueues the native eigenvalue operation.
///
/// # Safety
/// Keep all borrowed resources alive and order their accesses through completion
/// and every replay. Inspect completed status before reading or promoting results.
/// Jacobi parameter storage is exclusive through completion when present.
/// Selected host counts and bounds must also survive completion and replay.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a device buffer belongs to another CUDA context, matrix dimensions, leading dimensions, or storage capacities are invalid, checked size arithmetic or native integer conversion fails, the status output capacity is insufficient, a pivot, reflector, or numerical output buffer is too small, device scratch capacity is smaller than the queried requirement, the eigenvalue selection is invalid, or cuSOLVER rejects the configuration or returns an error while querying workspace or submitting execution.
pub unsafe fn xsyevdx<T: SolverElement>(
    context: &Context,
    params: &Params,
    mode: EigenMode,
    fill_mode: FillMode,
    selection: &EigenSelection<T::Real>,
    n: usize,
    mut a: MatrixMut<'_, T>,
    mut w: DeviceOutputView<'_, T::Real>,
    mut scratch: ByteWorkspaceMut<'_>,
    mut dev_info: DeviceOutputView<'_, i32>,
    selected_count: &mut MaybeUninit<i64>,
) -> Result<()> {
    context.bind()?;
    let values = n;
    a.layout().require_capacity(n, n, a.data.len())?;
    output(&w, values)?;
    output(&dev_info, 1)?;
    context.validate_region(&a.data)?;
    context.validate_region(&w)?;
    context.validate_region(&scratch.device)?;
    context.validate_region(&dev_info)?;
    let sizes =
        xsyevdx_buffer_size::<T>(context, params, mode, fill_mode, selection, n, a.layout())?;
    workspace(scratch.device.len(), sizes.device_bytes)?;
    workspace(scratch.host.len(), sizes.host_bytes)?;
    let (range, vl, vu, il, iu) = native_selection(selection, n)?;
    // SAFETY: validated storage and the caller's queued-access contract cover native use.
    unsafe {
        try_ffi!(sys::cusolverDnXsyevdx(
            context.as_raw(),
            params.as_raw(),
            mode.into(),
            range.into(),
            fill_mode.into(),
            to_i64(n, "n")?,
            T::data_type().into(),
            a.data.as_mut_ptr().cast(),
            to_i64(a.leading_dimension, "lda")?,
            ptr::from_ref(vl).cast_mut().cast(),
            ptr::from_ref(vu).cast_mut().cast(),
            to_i64(il, "il")?,
            to_i64(iu, "iu")?,
            selected_count.as_mut_ptr(),
            T::Real::data_type().into(),
            w.as_mut_ptr().cast(),
            T::data_type().into(),
            scratch.device.as_mut_ptr().cast(),
            sizes.device_bytes as _,
            scratch.host.as_mut_ptr().cast(),
            sizes.host_bytes as _,
            dev_info.as_mut_ptr().cast()
        ))?;
    }
    Ok(())
}

/// Queries scratch capacity from layout metadata without allocating buffers.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, the batch count or packed storage extent is invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn xsyev_batched_buffer_size<T: SolverElement>(
    context: &Context,
    params: &Params,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    a: MatrixLayout,
    batch_count: usize,
) -> Result<WorkspaceSizes> {
    context.bind()?;
    batch_layout(n, a, batch_count, true)?;
    let mut device_bytes = 0;
    let mut host_bytes = 0;
    // SAFETY: the query reads metadata only.
    unsafe {
        try_ffi!(sys::cusolverDnXsyevBatched_bufferSize(
            context.as_raw(),
            params.as_raw(),
            mode.into(),
            fill_mode.into(),
            to_i64(n, "n")?,
            T::data_type().into(),
            ptr::null(),
            to_i64(a.leading_dimension, "lda")?,
            T::Real::data_type().into(),
            ptr::null(),
            T::data_type().into(),
            &raw mut device_bytes,
            &raw mut host_bytes,
            to_i64(batch_count, "batch_count")?
        ))?;
    }
    Ok(WorkspaceSizes::new(
        to_usize(device_bytes, "device workspace size")?,
        to_usize(host_bytes, "host workspace size")?,
    ))
}

/// Enqueues the native eigenvalue operation.
///
/// # Safety
/// Keep all borrowed resources alive and order their accesses through completion
/// and every replay. Inspect completed status before reading or promoting results.
/// Jacobi parameter storage is exclusive through completion when present.
/// Selected host counts and bounds must also survive completion and replay.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a device buffer belongs to another CUDA context, matrix dimensions, leading dimensions, or storage capacities are invalid, checked size arithmetic or native integer conversion fails, the status output capacity is insufficient, a pivot, reflector, or numerical output buffer is too small, device scratch capacity is smaller than the queried requirement, batch metadata or related pointer-table lengths are invalid, or cuSOLVER rejects the configuration or returns an error while querying workspace or submitting execution.
pub unsafe fn xsyev_batched<T: SolverElement>(
    context: &Context,
    params: &Params,
    mode: EigenMode,
    fill_mode: FillMode,
    n: usize,
    mut a: MatrixMut<'_, T>,
    mut w: DeviceOutputView<'_, T::Real>,
    mut scratch: ByteWorkspaceMut<'_>,
    mut dev_info: DeviceOutputView<'_, i32>,
    batch_count: usize,
) -> Result<()> {
    context.bind()?;
    let values = batch_layout(n, a.layout(), batch_count, true)?;
    a.layout().require_capacity(n, values, a.data.len())?;
    output(&w, values)?;
    output(&dev_info, batch_count)?;
    context.validate_region(&a.data)?;
    context.validate_region(&w)?;
    context.validate_region(&scratch.device)?;
    context.validate_region(&dev_info)?;
    let sizes = xsyev_batched_buffer_size::<T>(
        context,
        params,
        mode,
        fill_mode,
        n,
        a.layout(),
        batch_count,
    )?;
    workspace(scratch.device.len(), sizes.device_bytes)?;
    workspace(scratch.host.len(), sizes.host_bytes)?;
    // SAFETY: validated storage and the caller's queued-access contract cover native use.
    unsafe {
        try_ffi!(sys::cusolverDnXsyevBatched(
            context.as_raw(),
            params.as_raw(),
            mode.into(),
            fill_mode.into(),
            to_i64(n, "n")?,
            T::data_type().into(),
            a.data.as_mut_ptr().cast(),
            to_i64(a.leading_dimension, "lda")?,
            T::Real::data_type().into(),
            w.as_mut_ptr().cast(),
            T::data_type().into(),
            scratch.device.as_mut_ptr().cast(),
            sizes.device_bytes as _,
            scratch.host.as_mut_ptr().cast(),
            sizes.host_bytes as _,
            dev_info.as_mut_ptr().cast(),
            to_i64(batch_count, "batch_count")?
        ))?;
    }
    Ok(())
}

/// Queries scratch capacity from layout metadata without allocating buffers.
///
/// Real inputs support n complex eigenvalues or two n-element real blocks
/// holding real and imaginary parts. Eigenvectors always use the input type.
/// For real A, a conjugate pair uses adjacent real and imaginary vector columns:
/// `v[j] + i*v[j+1]` and `v[j] - i*v[j+1]`. Complex A produces complex columns.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn xgeev_buffer_size<T: SolverElement, W: EigenvalueElement<T>>(
    context: &Context,
    params: &Params,
    n: usize,
    a: MatrixLayout,
    right_vectors: Option<MatrixLayout>,
) -> Result<WorkspaceSizes> {
    context.bind()?;
    a.validate(n, n)?;
    if let Some(vectors) = right_vectors {
        vectors.validate(n, n)?;
    }
    let mode = if right_vectors.is_some() {
        EigenMode::Vector
    } else {
        EigenMode::NoVector
    };
    let ldvr = to_i64(
        right_vectors.as_ref().map_or(1, |v| v.leading_dimension),
        "ldvr",
    )?;
    let mut device_bytes = 0;
    let mut host_bytes = 0;
    // SAFETY: the query reads metadata only.
    unsafe {
        try_ffi!(sys::cusolverDnXgeev_bufferSize(
            context.as_raw(),
            params.as_raw(),
            EigenMode::NoVector.into(),
            mode.into(),
            to_i64(n, "n")?,
            T::data_type().into(),
            ptr::null(),
            to_i64(a.leading_dimension, "lda")?,
            W::data_type().into(),
            ptr::null(),
            T::data_type().into(),
            ptr::null(),
            1,
            T::data_type().into(),
            ptr::null(),
            ldvr,
            T::data_type().into(),
            &raw mut device_bytes,
            &raw mut host_bytes
        ))?;
    }
    Ok(WorkspaceSizes::new(
        to_usize(device_bytes, "device workspace size")?,
        to_usize(host_bytes, "host workspace size")?,
    ))
}

/// Enqueues the native eigenvalue operation.
///
/// # Safety
/// Keep all borrowed resources alive and order their accesses through completion
/// and every replay. Inspect completed status before reading or promoting results.
/// Jacobi parameter storage is exclusive through completion when present.
/// Selected host counts and bounds must also survive completion and replay.
///
/// Real inputs support n complex eigenvalues or two n-element real blocks
/// holding real and imaginary parts. Eigenvectors always use the input type.
/// For real A, a conjugate pair uses adjacent real and imaginary vector columns:
/// `v[j] + i*v[j+1]` and `v[j] - i*v[j+1]`. Complex A produces complex columns.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a device buffer belongs to another CUDA context, matrix dimensions, leading dimensions, or storage capacities are invalid, checked size arithmetic or native integer conversion fails, the status output capacity is insufficient, a pivot, reflector, or numerical output buffer is too small, device scratch capacity is smaller than the queried requirement, or cuSOLVER rejects the configuration or returns an error while querying workspace or submitting execution.
pub unsafe fn xgeev<T: SolverElement, W: EigenvalueElement<T>>(
    context: &Context,
    params: &Params,
    n: usize,
    mut a: MatrixMut<'_, T>,
    mut right_vectors: Option<MatrixOutput<'_, T>>,
    mut w: DeviceOutputView<'_, W>,
    mut scratch: ByteWorkspaceMut<'_>,
    mut dev_info: DeviceOutputView<'_, i32>,
) -> Result<()> {
    context.bind()?;
    let values = n;
    a.layout().require_capacity(n, n, a.data.len())?;
    let values = values
        .checked_mul(W::ELEMENTS_PER_VALUE)
        .ok_or(Error::InvalidVectorShape)?;
    if let Some(vectors) = &right_vectors {
        MatrixLayout::new(vectors.leading_dimension).require_capacity(n, n, vectors.data.len())?;
        context.validate_region(&vectors.data)?;
    }
    output(&w, values)?;
    output(&dev_info, 1)?;
    context.validate_region(&a.data)?;
    context.validate_region(&w)?;
    context.validate_region(&scratch.device)?;
    context.validate_region(&dev_info)?;
    let sizes = xgeev_buffer_size::<T, W>(
        context,
        params,
        n,
        a.layout(),
        right_vectors
            .as_ref()
            .map(|v| MatrixLayout::new(v.leading_dimension)),
    )?;
    workspace(scratch.device.len(), sizes.device_bytes)?;
    workspace(scratch.host.len(), sizes.host_bytes)?;
    let mode = if right_vectors.is_some() {
        EigenMode::Vector
    } else {
        EigenMode::NoVector
    };
    let ldvr = to_i64(
        right_vectors.as_ref().map_or(1, |v| v.leading_dimension),
        "ldvr",
    )?;
    let vr_ptr = right_vectors
        .as_mut()
        .map_or(ptr::null_mut(), |v| v.data.as_mut_ptr().cast());
    // SAFETY: validated storage and the caller's queued-access contract cover native use.
    unsafe {
        try_ffi!(sys::cusolverDnXgeev(
            context.as_raw(),
            params.as_raw(),
            EigenMode::NoVector.into(),
            mode.into(),
            to_i64(n, "n")?,
            T::data_type().into(),
            a.data.as_mut_ptr().cast(),
            to_i64(a.leading_dimension, "lda")?,
            W::data_type().into(),
            w.as_mut_ptr().cast(),
            T::data_type().into(),
            ptr::null_mut(),
            1,
            T::data_type().into(),
            vr_ptr,
            ldvr,
            T::data_type().into(),
            scratch.device.as_mut_ptr().cast(),
            sizes.device_bytes as _,
            scratch.host.as_mut_ptr().cast(),
            sizes.host_bytes as _,
            dev_info.as_mut_ptr().cast()
        ))?;
    }
    Ok(())
}
