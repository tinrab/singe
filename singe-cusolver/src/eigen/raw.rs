//! Native symmetric, generalized, and general eigenvalue operations.
//!
//! Execution enqueues work without completing its borrows.
//! Keep the context, parameter objects, matrices, outputs, workspaces, selection bounds, and host count destinations alive through completion and captured replay.
//! Order prior writes before execution and exclude overlapping accesses to writes.
//! A successful enqueue does not establish a successful numerical result.
//! Inspect completed device status before promoting outputs.
//!
//! Symmetric operations produce real eigenvalues and overwrite A with vectors when requested.
//! Generalized operations also overwrite B with its selected triangular factor.
//! Padding, scratch, and allocation tails are not results.
//! Selected operations produce only the returned count of values and columns.
//! Batched operations pack A at lda*n elements per matrix and values at n; they produce one status per matrix.
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
    utility::{require_workspace_bytes, to_i32, to_i64, to_size_t, to_usize},
};

macro_rules! syevj_buffer_size_by_precision {
    ($($name:ident, $ty:ty, $sys:ident;)+) => {
        $(
            pub(crate) fn $name(
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
                // SAFETY: `context` and `params` are live handles, the size query reads no device data so its matrix pointers may be null, and `lwork` is a writable local.
                unsafe {
                    try_ffi!(sys::$sys(
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
        )+
    };
}

syevj_buffer_size_by_precision! {
    ssyevj_buffer_size, f32, cusolverDnSsyevj_bufferSize;
    dsyevj_buffer_size, f64, cusolverDnDsyevj_bufferSize;
    cheevj_buffer_size, Complex32, cusolverDnCheevj_bufferSize;
    zheevj_buffer_size, Complex64, cusolverDnZheevj_bufferSize;
}

macro_rules! syevj_by_precision {
    ($($name:ident, $ty:ty, $real:ty, $sys:ident, $buffer_size:ident;)+) => {
        $(
            pub(crate) unsafe fn $name(
                context: &Context,
                mode: EigenMode,
                fill_mode: FillMode,
                n: usize,
                mut a: MatrixMut<'_, $ty>,
                mut w: DeviceOutputView<'_, $real>,
                mut scratch: DeviceOutputView<'_, $ty>,
                mut dev_info: DeviceOutputView<'_, i32>,
                params: &mut SyevjInfo,
            ) -> Result<()> {
                context.bind()?;
                let values = n;
                a.layout().require_capacity("a", n, n, a.data.len())?;
                output(&w, values)?;
                output(&dev_info, 1)?;
                context.prepare_region(&a.data)?;
                context.prepare_region(&w)?;
                context.prepare_region(&scratch)?;
                context.prepare_region(&dev_info)?;
                let lwork = $buffer_size(context, mode, fill_mode, n, a.layout(), params)?;
                workspace(scratch.len(), lwork)?;
                // SAFETY: the capacities of `a`, `w`, and `dev_info` were checked above, `scratch` covers the queried `lwork`, and `params` is borrowed exclusively.
                // The caller retains and orders every operand through completion per the crate's raw execution contract.
                unsafe {
                    try_ffi!(sys::$sys(
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
        )+
    };
}

syevj_by_precision! {
    ssyevj, f32, f32, cusolverDnSsyevj, ssyevj_buffer_size;
    dsyevj, f64, f64, cusolverDnDsyevj, dsyevj_buffer_size;
    cheevj, Complex32, f32, cusolverDnCheevj, cheevj_buffer_size;
    zheevj, Complex64, f64, cusolverDnZheevj, zheevj_buffer_size;
}

macro_rules! syevj_batched_buffer_size_by_precision {
    ($($name:ident, $ty:ty, $sys:ident;)+) => {
        $(
            pub(crate) fn $name(
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
                // SAFETY: `context` and `params` are live handles, the size query reads no device data so its matrix pointers may be null, and `lwork` is a writable local.
                unsafe {
                    try_ffi!(sys::$sys(
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
        )+
    };
}

syevj_batched_buffer_size_by_precision! {
    ssyevj_batched_buffer_size, f32, cusolverDnSsyevjBatched_bufferSize;
    dsyevj_batched_buffer_size, f64, cusolverDnDsyevjBatched_bufferSize;
    cheevj_batched_buffer_size, Complex32, cusolverDnCheevjBatched_bufferSize;
    zheevj_batched_buffer_size, Complex64, cusolverDnZheevjBatched_bufferSize;
}

macro_rules! syevj_batched_by_precision {
    ($($name:ident, $ty:ty, $real:ty, $sys:ident, $buffer_size:ident;)+) => {
        $(
            pub(crate) unsafe fn $name(
                context: &Context,
                mode: EigenMode,
                fill_mode: FillMode,
                n: usize,
                mut a: MatrixMut<'_, $ty>,
                mut w: DeviceOutputView<'_, $real>,
                mut scratch: DeviceOutputView<'_, $ty>,
                mut dev_info: DeviceOutputView<'_, i32>,
                params: &mut SyevjInfo,
                batch_count: usize,
            ) -> Result<()> {
                context.bind()?;
                let values = batch_layout(n, a.layout(), batch_count, false)?;
                a.layout().require_capacity("a", n, values, a.data.len())?;
                output(&w, values)?;
                output(&dev_info, batch_count)?;
                context.prepare_region(&a.data)?;
                context.prepare_region(&w)?;
                context.prepare_region(&scratch)?;
                context.prepare_region(&dev_info)?;
                let lwork =
                    $buffer_size(context, mode, fill_mode, n, a.layout(), params, batch_count)?;
                workspace(scratch.len(), lwork)?;
                // SAFETY: the capacities of `a`, `w`, and `dev_info` were checked above, `scratch` covers the queried `lwork`, and `params` is borrowed exclusively.
                // The caller retains and orders every operand through completion per the crate's raw execution contract.
                unsafe {
                    try_ffi!(sys::$sys(
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
        )+
    };
}

syevj_batched_by_precision! {
    ssyevj_batched, f32, f32, cusolverDnSsyevjBatched, ssyevj_batched_buffer_size;
    dsyevj_batched, f64, f64, cusolverDnDsyevjBatched, dsyevj_batched_buffer_size;
    cheevj_batched, Complex32, f32, cusolverDnCheevjBatched, cheevj_batched_buffer_size;
    zheevj_batched, Complex64, f64, cusolverDnZheevjBatched, zheevj_batched_buffer_size;
}

macro_rules! sygvj_buffer_size_by_precision {
    ($($name:ident, $ty:ty, $sys:ident;)+) => {
        $(
            pub(crate) fn $name(
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
                // SAFETY: `context` and `params` are live handles, the size query reads no device data so its matrix pointers may be null, and `lwork` is a writable local.
                unsafe {
                    try_ffi!(sys::$sys(
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
        )+
    };
}

sygvj_buffer_size_by_precision! {
    ssygvj_buffer_size, f32, cusolverDnSsygvj_bufferSize;
    dsygvj_buffer_size, f64, cusolverDnDsygvj_bufferSize;
    chegvj_buffer_size, Complex32, cusolverDnChegvj_bufferSize;
    zhegvj_buffer_size, Complex64, cusolverDnZhegvj_bufferSize;
}

macro_rules! sygvj_by_precision {
    ($($name:ident, $ty:ty, $real:ty, $sys:ident, $buffer_size:ident;)+) => {
        $(
            pub(crate) unsafe fn $name(
                context: &Context,
                eig_type: EigenType,
                mode: EigenMode,
                fill_mode: FillMode,
                n: usize,
                mut a: MatrixMut<'_, $ty>,
                mut b: MatrixMut<'_, $ty>,
                mut w: DeviceOutputView<'_, $real>,
                mut scratch: DeviceOutputView<'_, $ty>,
                mut dev_info: DeviceOutputView<'_, i32>,
                params: &mut SyevjInfo,
            ) -> Result<()> {
                context.bind()?;
                let values = n;
                a.layout().require_capacity("a", n, n, a.data.len())?;
                b.layout().require_capacity("b", n, n, b.data.len())?;
                output(&w, values)?;
                output(&dev_info, 1)?;
                context.prepare_region(&a.data)?;
                context.prepare_region(&b.data)?;
                context.prepare_region(&w)?;
                context.prepare_region(&scratch)?;
                context.prepare_region(&dev_info)?;
                let lwork = $buffer_size(
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
                // SAFETY: the capacities of `a`, `b`, `w`, and `dev_info` were checked above, `scratch` covers the queried `lwork`, and `params` is borrowed exclusively.
                // The caller retains and orders every operand through completion per the crate's raw execution contract.
                unsafe {
                    try_ffi!(sys::$sys(
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
        )+
    };
}

sygvj_by_precision! {
    ssygvj, f32, f32, cusolverDnSsygvj, ssygvj_buffer_size;
    dsygvj, f64, f64, cusolverDnDsygvj, dsygvj_buffer_size;
    chegvj, Complex32, f32, cusolverDnChegvj, chegvj_buffer_size;
    zhegvj, Complex64, f64, cusolverDnZhegvj, zhegvj_buffer_size;
}

macro_rules! sygvd_buffer_size_by_precision {
    ($($name:ident, $ty:ty, $sys:ident;)+) => {
        $(
            pub(crate) fn $name(
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
                // SAFETY: `context` is a live handle, the size query reads no device data so its matrix pointers may be null, and `lwork` is a writable local.
                unsafe {
                    try_ffi!(sys::$sys(
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
        )+
    };
}

sygvd_buffer_size_by_precision! {
    ssygvd_buffer_size, f32, cusolverDnSsygvd_bufferSize;
    dsygvd_buffer_size, f64, cusolverDnDsygvd_bufferSize;
    chegvd_buffer_size, Complex32, cusolverDnChegvd_bufferSize;
    zhegvd_buffer_size, Complex64, cusolverDnZhegvd_bufferSize;
}

macro_rules! sygvd_by_precision {
    ($($name:ident, $ty:ty, $real:ty, $sys:ident, $buffer_size:ident;)+) => {
        $(
            pub(crate) unsafe fn $name(
                context: &Context,
                eig_type: EigenType,
                mode: EigenMode,
                fill_mode: FillMode,
                n: usize,
                mut a: MatrixMut<'_, $ty>,
                mut b: MatrixMut<'_, $ty>,
                mut w: DeviceOutputView<'_, $real>,
                mut scratch: DeviceOutputView<'_, $ty>,
                mut dev_info: DeviceOutputView<'_, i32>,
            ) -> Result<()> {
                context.bind()?;
                let values = n;
                a.layout().require_capacity("a", n, n, a.data.len())?;
                b.layout().require_capacity("b", n, n, b.data.len())?;
                output(&w, values)?;
                output(&dev_info, 1)?;
                context.prepare_region(&a.data)?;
                context.prepare_region(&b.data)?;
                context.prepare_region(&w)?;
                context.prepare_region(&scratch)?;
                context.prepare_region(&dev_info)?;
                let lwork = $buffer_size(
                    context,
                    eig_type,
                    mode,
                    fill_mode,
                    n,
                    a.layout(),
                    b.layout(),
                )?;
                workspace(scratch.len(), lwork)?;
                // SAFETY: the capacities of `a`, `b`, `w`, and `dev_info` were checked above and `scratch` covers the queried `lwork`.
                // The caller retains and orders every operand through completion per the crate's raw execution contract.
                unsafe {
                    try_ffi!(sys::$sys(
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
        )+
    };
}

sygvd_by_precision! {
    ssygvd, f32, f32, cusolverDnSsygvd, ssygvd_buffer_size;
    dsygvd, f64, f64, cusolverDnDsygvd, dsygvd_buffer_size;
    chegvd, Complex32, f32, cusolverDnChegvd, chegvd_buffer_size;
    zhegvd, Complex64, f64, cusolverDnZhegvd, zhegvd_buffer_size;
}

macro_rules! sygvdx_selected_buffer_size_by_precision {
    ($($name:ident, $ty:ty, $real:ty, $sys:ident;)+) => {
        $(
            pub(crate) fn $name(
                context: &Context,
                eig_type: EigenType,
                mode: EigenMode,
                fill_mode: FillMode,
                selection: &EigenSelection<$real>,
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
                // SAFETY: `context` is a live handle, the size query reads no device data so its matrix pointers may be null, and `lwork` and `ignored_count` are writable locals.
                unsafe {
                    try_ffi!(sys::$sys(
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
        )+
    };
}

sygvdx_selected_buffer_size_by_precision! {
    ssygvdx_selected_buffer_size, f32, f32, cusolverDnSsygvdx_bufferSize;
    dsygvdx_selected_buffer_size, f64, f64, cusolverDnDsygvdx_bufferSize;
    chegvdx_selected_buffer_size, Complex32, f32, cusolverDnChegvdx_bufferSize;
    zhegvdx_selected_buffer_size, Complex64, f64, cusolverDnZhegvdx_bufferSize;
}

macro_rules! sygvdx_selected_by_precision {
    ($($name:ident, $ty:ty, $real:ty, $sys:ident, $buffer_size:ident;)+) => {
        $(
            pub(crate) unsafe fn $name(
                context: &Context,
                eig_type: EigenType,
                mode: EigenMode,
                fill_mode: FillMode,
                selection: &EigenSelection<$real>,
                n: usize,
                mut a: MatrixMut<'_, $ty>,
                mut b: MatrixMut<'_, $ty>,
                mut w: DeviceOutputView<'_, $real>,
                mut scratch: DeviceOutputView<'_, $ty>,
                mut dev_info: DeviceOutputView<'_, i32>,
                selected_count: &mut MaybeUninit<i32>,
            ) -> Result<()> {
                context.bind()?;
                let values = n;
                a.layout().require_capacity("a", n, n, a.data.len())?;
                b.layout().require_capacity("b", n, n, b.data.len())?;
                output(&w, values)?;
                output(&dev_info, 1)?;
                context.prepare_region(&a.data)?;
                context.prepare_region(&b.data)?;
                context.prepare_region(&w)?;
                context.prepare_region(&scratch)?;
                context.prepare_region(&dev_info)?;
                let lwork = $buffer_size(
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
                // SAFETY: the capacities of `a`, `b`, `w`, and `dev_info` were checked above and `scratch` covers the queried `lwork`.
                // The caller retains and orders every operand, including `selected_count`, through completion per the crate's raw execution contract.
                unsafe {
                    try_ffi!(sys::$sys(
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
        )+
    };
}

sygvdx_selected_by_precision! {
    ssygvdx_selected, f32, f32, cusolverDnSsygvdx, ssygvdx_selected_buffer_size;
    dsygvdx_selected, f64, f64, cusolverDnDsygvdx, dsygvdx_selected_buffer_size;
    chegvdx_selected, Complex32, f32, cusolverDnChegvdx, chegvdx_selected_buffer_size;
    zhegvdx_selected, Complex64, f64, cusolverDnZhegvdx, zhegvdx_selected_buffer_size;
}

/// Returns the device and host workspace bytes the operation needs, reading only layout metadata.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn xsyevd_workspace_size<T: SolverElement>(
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
    // SAFETY: `context` and `params` are live handles, the size query reads no device data so its matrix pointers may be null, and `device_bytes` and `host_bytes` are writable locals.
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

/// Enqueues a symmetric or Hermitian eigendecomposition through the native 64-bit interface.
///
/// # Safety
///
/// Follow the [raw execution contract](crate#raw-execution).
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a device buffer belongs to another CUDA context, matrix dimensions, leading dimensions, or storage capacities are invalid, checked size arithmetic or native integer conversion fails, an eigenvalue or status output is too small, device or host workspace is smaller than the queried requirement, or cuSOLVER rejects the configuration or returns an error while querying workspace or submitting execution.
pub unsafe fn xsyevd_async<T: SolverElement>(
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
    a.layout().require_capacity("a", n, n, a.data.len())?;
    output(&w, values)?;
    output(&dev_info, 1)?;
    context.prepare_region(&a.data)?;
    context.prepare_region(&w)?;
    context.prepare_region(&scratch.device)?;
    context.prepare_region(&dev_info)?;
    let sizes = xsyevd_workspace_size::<T>(context, params, mode, fill_mode, n, a.layout())?;
    require_workspace_bytes("device workspace", scratch.device.len(), sizes.device_bytes)?;
    require_workspace_bytes("host workspace", scratch.host.len(), sizes.host_bytes)?;
    // SAFETY: the capacities of `a`, `w`, and `dev_info` were checked above, `scratch` covers both queried byte sizes, and `params` is a live handle.
    // The caller retains and orders every operand through completion per the crate's raw execution contract.
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
            to_size_t(sizes.device_bytes, "device workspace size")?,
            scratch.host.as_mut_ptr().cast(),
            to_size_t(sizes.host_bytes, "host workspace size")?,
            dev_info.as_mut_ptr().cast()
        ))?;
    }
    Ok(())
}

/// Returns the device and host workspace bytes the operation needs, reading only layout metadata.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, the selected eigenvalue interval is invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn xsyevdx_workspace_size<T: SolverElement>(
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
    // SAFETY: `context` and `params` are live handles, the size query reads no device data so its matrix pointers may be null, and `device_bytes`, `host_bytes`, and `ignored_count` are writable locals.
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

/// Enqueues a selected symmetric or Hermitian eigendecomposition through the native 64-bit interface.
///
/// `selected_count` receives the number of produced eigenvalues when the operation completes.
///
/// # Safety
///
/// Follow the [raw execution contract](crate#raw-execution).
/// `selected_count` is written at completion, so it must also stay alive through completion and replay.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a device buffer belongs to another CUDA context, matrix dimensions, leading dimensions, or storage capacities are invalid, checked size arithmetic or native integer conversion fails, an eigenvalue or status output is too small, device or host workspace is smaller than the queried requirement, the eigenvalue selection is invalid, or cuSOLVER rejects the configuration or returns an error while querying workspace or submitting execution.
pub unsafe fn xsyevdx_async<T: SolverElement>(
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
    a.layout().require_capacity("a", n, n, a.data.len())?;
    output(&w, values)?;
    output(&dev_info, 1)?;
    context.prepare_region(&a.data)?;
    context.prepare_region(&w)?;
    context.prepare_region(&scratch.device)?;
    context.prepare_region(&dev_info)?;
    let sizes =
        xsyevdx_workspace_size::<T>(context, params, mode, fill_mode, selection, n, a.layout())?;
    require_workspace_bytes("device workspace", scratch.device.len(), sizes.device_bytes)?;
    require_workspace_bytes("host workspace", scratch.host.len(), sizes.host_bytes)?;
    let (range, vl, vu, il, iu) = native_selection(selection, n)?;
    // SAFETY: the capacities of `a`, `w`, and `dev_info` were checked above, `scratch` covers both queried byte sizes, and `params` is a live handle.
    // The caller retains and orders every operand, including `selected_count`, through completion per the crate's raw execution contract.
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
            to_size_t(sizes.device_bytes, "device workspace size")?,
            scratch.host.as_mut_ptr().cast(),
            to_size_t(sizes.host_bytes, "host workspace size")?,
            dev_info.as_mut_ptr().cast()
        ))?;
    }
    Ok(())
}

/// Returns the device and host workspace bytes the operation needs, reading only layout metadata.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, the batch count or packed storage extent is invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn xsyev_batched_workspace_size<T: SolverElement>(
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
    // SAFETY: `context` and `params` are live handles, the size query reads no device data so its matrix pointers may be null, and `device_bytes` and `host_bytes` are writable locals.
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

/// Enqueues a packed batched symmetric or Hermitian eigendecomposition through the native 64-bit interface.
///
/// # Safety
///
/// Follow the [raw execution contract](crate#raw-execution).
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a device buffer belongs to another CUDA context, matrix dimensions, leading dimensions, or storage capacities are invalid, checked size arithmetic or native integer conversion fails, an eigenvalue or status output is too small, device or host workspace is smaller than the queried requirement, the batch count is zero or the packed batch exceeds native limits, or cuSOLVER rejects the configuration or returns an error while querying workspace or submitting execution.
pub unsafe fn xsyev_batched_async<T: SolverElement>(
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
    a.layout().require_capacity("a", n, values, a.data.len())?;
    output(&w, values)?;
    output(&dev_info, batch_count)?;
    context.prepare_region(&a.data)?;
    context.prepare_region(&w)?;
    context.prepare_region(&scratch.device)?;
    context.prepare_region(&dev_info)?;
    let sizes = xsyev_batched_workspace_size::<T>(
        context,
        params,
        mode,
        fill_mode,
        n,
        a.layout(),
        batch_count,
    )?;
    require_workspace_bytes("device workspace", scratch.device.len(), sizes.device_bytes)?;
    require_workspace_bytes("host workspace", scratch.host.len(), sizes.host_bytes)?;
    // SAFETY: the capacities of `a`, `w`, and `dev_info` were checked above, `scratch` covers both queried byte sizes, and `params` is a live handle.
    // The caller retains and orders every operand through completion per the crate's raw execution contract.
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
            to_size_t(sizes.device_bytes, "device workspace size")?,
            scratch.host.as_mut_ptr().cast(),
            to_size_t(sizes.host_bytes, "host workspace size")?,
            dev_info.as_mut_ptr().cast(),
            to_i64(batch_count, "batch_count")?
        ))?;
    }
    Ok(())
}

/// Returns the device and host workspace bytes the operation needs, reading only layout metadata.
///
/// Real inputs support n complex eigenvalues or two n-element real blocks holding real and imaginary parts.
/// Eigenvectors always use the input type.
/// For real A, a conjugate pair uses adjacent real and imaginary vector columns: `v[j] + i*v[j+1]` and `v[j] - i*v[j+1]`.
/// Complex A produces complex columns.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn xgeev_workspace_size<T: SolverElement, W: EigenvalueElement<T>>(
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
    // SAFETY: `context` and `params` are live handles, the size query reads no device data so its matrix pointers may be null, and `device_bytes` and `host_bytes` are writable locals.
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

/// Enqueues a general eigendecomposition through the native 64-bit interface.
///
/// Real inputs support n complex eigenvalues or two n-element real blocks holding real and imaginary parts.
/// Eigenvectors always use the input type.
/// For real A, a conjugate pair uses adjacent real and imaginary vector columns: `v[j] + i*v[j+1]` and `v[j] - i*v[j+1]`.
/// Complex A produces complex columns.
///
/// # Safety
///
/// Follow the [raw execution contract](crate#raw-execution).
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a device buffer belongs to another CUDA context, matrix dimensions, leading dimensions, or storage capacities are invalid, checked size arithmetic or native integer conversion fails, an eigenvalue or status output is too small, device or host workspace is smaller than the queried requirement, or cuSOLVER rejects the configuration or returns an error while querying workspace or submitting execution.
pub unsafe fn xgeev_async<T: SolverElement, W: EigenvalueElement<T>>(
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
    a.layout().require_capacity("a", n, n, a.data.len())?;
    let values = values
        .checked_mul(W::ELEMENTS_PER_VALUE)
        .ok_or(Error::InvalidVectorShape)?;
    if let Some(vectors) = &right_vectors {
        MatrixLayout::new(vectors.leading_dimension).require_capacity(
            "vectors",
            n,
            n,
            vectors.data.len(),
        )?;
        context.prepare_region(&vectors.data)?;
    }

    output(&w, values)?;
    output(&dev_info, 1)?;
    context.prepare_region(&a.data)?;
    context.prepare_region(&w)?;
    context.prepare_region(&scratch.device)?;
    context.prepare_region(&dev_info)?;
    let sizes = xgeev_workspace_size::<T, W>(
        context,
        params,
        n,
        a.layout(),
        right_vectors
            .as_ref()
            .map(|v| MatrixLayout::new(v.leading_dimension)),
    )?;
    require_workspace_bytes("device workspace", scratch.device.len(), sizes.device_bytes)?;
    require_workspace_bytes("host workspace", scratch.host.len(), sizes.host_bytes)?;
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
    // SAFETY: the capacities of `a`, `w`, `right_vectors`, and `dev_info` were checked above, `scratch` covers both queried byte sizes, and `params` is a live handle.
    // The caller retains and orders every operand through completion per the crate's raw execution contract.
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
            to_size_t(sizes.device_bytes, "device workspace size")?,
            scratch.host.as_mut_ptr().cast(),
            to_size_t(sizes.host_bytes, "host workspace size")?,
            dev_info.as_mut_ptr().cast()
        ))?;
    }
    Ok(())
}
