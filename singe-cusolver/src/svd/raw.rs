//! Direct native Jacobi and approximate SVD entry points.
//!
//! Raw execution requires inputs, outputs, scratch, parameter objects, and the context to survive completion and every captured replay.
//! Order inputs before use and exclude concurrent access to writes and mutable Jacobi parameters.
//! Successful enqueue does not establish initialization or a valid numerical result.
//! Inspect completed status before promoting the documented logical results.
//! Leading-dimension padding, stride gaps, allocation tails, and scratch are not results.
//!
//! Jacobi SVD produces min(m,n) singular values.
//! Full vectors have m columns in U and n columns in V; economy vectors both have min(m,n) columns.
//! Batched Jacobi always produces full vectors and one status per matrix.
//! Approximate SVD produces rank values and rank columns of U and V per matrix, one status per matrix, and one host residual per matrix when requested with Vector mode.
//! `NoVector` mode leaves host residuals unwritten.
//! Both families return V, rather than its conjugate transpose.
//!
//! The linked cuSOLVER 12.3.4 batched Jacobi values-only implementation requires full U storage when m >= n, or full V storage when m < n, as temporary storage.
//! A null pointer for that buffer causes a device fault even in `NoVector` mode.
//! Those contents are scratch.
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

macro_rules! gesvdj_buffer_size_by_precision {
    ($($name:ident, $ty:ty, $sys:ident;)+) => {
        $(
            pub(crate) fn $name(
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
                // SAFETY: `context` and `params` are live handles, the size query reads no device data so its matrix pointers may be null, and `lwork` is a writable local.
                unsafe {
                    try_ffi!(sys::$sys(
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
        )+
    };
}

gesvdj_buffer_size_by_precision! {
    sgesvdj_buffer_size, f32, cusolverDnSgesvdj_bufferSize;
    dgesvdj_buffer_size, f64, cusolverDnDgesvdj_bufferSize;
    cgesvdj_buffer_size, Complex32, cusolverDnCgesvdj_bufferSize;
    zgesvdj_buffer_size, Complex64, cusolverDnZgesvdj_bufferSize;
}

macro_rules! gesvdj_by_precision {
    ($($name:ident, $ty:ty, $real:ty, $sys:ident, $buffer_size:ident;)+) => {
        $(
            pub(crate) unsafe fn $name(
                context: &Context,
                mode: EigenMode,
                economy: bool,
                m: usize,
                n: usize,
                mut a: MatrixMut<'_, $ty>,
                mut s: DeviceOutputView<'_, $real>,
                u: Option<MatrixOutput<'_, $ty>>,
                v: Option<MatrixOutput<'_, $ty>>,
                mut workspace: DeviceOutputView<'_, $ty>,
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
                context.prepare_region(&a.data)?;
                context.prepare_region(&s)?;
                context.prepare_region(&workspace)?;
                context.prepare_region(&dev_info)?;
                for matrix in [u.as_ref(), v.as_ref()].into_iter().flatten() {
                    context.prepare_region(&matrix.data)?;
                }

                let lwork = $buffer_size(context, mode, economy, m, n, layout, params)?;
                if workspace.len() < lwork {
                    return Err(Error::BufferTooSmall {
                        name: "workspace".into(),
                        required: lwork,
                        actual: workspace.len(),
                    });
                }

                let (left, right) = jacobi_columns(m, n, economy);
                let (u, ldu) = output_parts(u, m, left, 1, mode)?;
                let (v, ldv) = output_parts(v, n, right, 1, mode)?;
                // SAFETY: the capacities of `a`, `s`, `u`, `v`, and `dev_info` were checked above, `workspace` covers the queried `lwork`, and `params` is borrowed exclusively.
                // The caller retains and orders every operand through completion per the crate's raw execution contract.
                unsafe {
                    try_ffi!(sys::$sys(
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
        )+
    };
}

gesvdj_by_precision! {
    sgesvdj, f32, f32, cusolverDnSgesvdj, sgesvdj_buffer_size;
    dgesvdj, f64, f64, cusolverDnDgesvdj, dgesvdj_buffer_size;
    cgesvdj, Complex32, f32, cusolverDnCgesvdj, cgesvdj_buffer_size;
    zgesvdj, Complex64, f64, cusolverDnZgesvdj, zgesvdj_buffer_size;
}

macro_rules! gesvdj_batched_buffer_size_by_precision {
    ($($name:ident, $ty:ty, $sys:ident;)+) => {
        $(
            pub(crate) fn $name(
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
                // SAFETY: `context` and `params` are live handles, the size query reads no device data so its matrix pointers may be null, and `lwork` is a writable local.
                unsafe {
                    try_ffi!(sys::$sys(
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
        )+
    };
}

gesvdj_batched_buffer_size_by_precision! {
    sgesvdj_batched_buffer_size, f32, cusolverDnSgesvdjBatched_bufferSize;
    dgesvdj_batched_buffer_size, f64, cusolverDnDgesvdjBatched_bufferSize;
    cgesvdj_batched_buffer_size, Complex32, cusolverDnCgesvdjBatched_bufferSize;
    zgesvdj_batched_buffer_size, Complex64, cusolverDnZgesvdjBatched_bufferSize;
}

macro_rules! gesvdj_batched_by_precision {
    ($($name:ident, $ty:ty, $real:ty, $sys:ident, $buffer_size:ident;)+) => {
        $(
            pub(crate) unsafe fn $name(
                context: &Context,
                mode: EigenMode,
                m: usize,
                n: usize,
                mut a: MatrixMut<'_, $ty>,
                mut s: DeviceOutputView<'_, $real>,
                u: Option<MatrixOutput<'_, $ty>>,
                v: Option<MatrixOutput<'_, $ty>>,
                mut workspace: DeviceOutputView<'_, $ty>,
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
                context.prepare_region(&a.data)?;
                context.prepare_region(&s)?;
                context.prepare_region(&workspace)?;
                context.prepare_region(&dev_info)?;
                for matrix in [u.as_ref(), v.as_ref()].into_iter().flatten() {
                    context.prepare_region(&matrix.data)?;
                }

                let lwork = $buffer_size(context, mode, m, n, layout, params, batch_size)?;
                if workspace.len() < lwork {
                    return Err(Error::BufferTooSmall {
                        name: "workspace".into(),
                        required: lwork,
                        actual: workspace.len(),
                    });
                }

                let (left, right) = jacobi_columns(m, n, false);
                let (u, ldu) = output_parts(u, m, left, batch_size, mode)?;
                let (v, ldv) = output_parts(v, n, right, batch_size, mode)?;
                // SAFETY: the capacities of `a`, `s`, `u`, `v`, and `dev_info` were checked above, `workspace` covers the queried `lwork`, and `params` is borrowed exclusively.
                // The caller retains and orders every operand through completion per the crate's raw execution contract.
                unsafe {
                    try_ffi!(sys::$sys(
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
        )+
    };
}

gesvdj_batched_by_precision! {
    sgesvdj_batched, f32, f32, cusolverDnSgesvdjBatched, sgesvdj_batched_buffer_size;
    dgesvdj_batched, f64, f64, cusolverDnDgesvdjBatched, dgesvdj_batched_buffer_size;
    cgesvdj_batched, Complex32, f32, cusolverDnCgesvdjBatched, cgesvdj_batched_buffer_size;
    zgesvdj_batched, Complex64, f64, cusolverDnZgesvdjBatched, zgesvdj_batched_buffer_size;
}

macro_rules! gesvda_strided_batched_buffer_size_by_precision {
    ($($name:ident, $ty:ty, $sys:ident;)+) => {
        $(
            pub(crate) fn $name(
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
                // SAFETY: `context` is a live handle, the size query reads no device data so its matrix pointers may be null, and `lwork` is a writable local.
                unsafe {
                    try_ffi!(sys::$sys(
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
        )+
    };
}

gesvda_strided_batched_buffer_size_by_precision! {
    sgesvda_strided_batched_buffer_size, f32, cusolverDnSgesvdaStridedBatched_bufferSize;
    dgesvda_strided_batched_buffer_size, f64, cusolverDnDgesvdaStridedBatched_bufferSize;
    cgesvda_strided_batched_buffer_size, Complex32, cusolverDnCgesvdaStridedBatched_bufferSize;
    zgesvda_strided_batched_buffer_size, Complex64, cusolverDnZgesvdaStridedBatched_bufferSize;
}

macro_rules! gesvda_strided_batched_by_precision {
    ($($name:ident, $ty:ty, $real:ty, $sys:ident, $buffer_size:ident;)+) => {
        $(
            pub(crate) unsafe fn $name(
                context: &Context,
                mode: EigenMode,
                rank: usize,
                m: usize,
                n: usize,
                a: StridedBatchedMatrixRef<'_, $ty>,
                mut s: StridedBatchedVectorOutput<'_, $real>,
                u: Option<StridedBatchedMatrixOutput<'_, $ty>>,
                v: Option<StridedBatchedMatrixOutput<'_, $ty>>,
                mut workspace: DeviceOutputView<'_, $ty>,
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
                context.prepare_region(&a.data)?;
                context.prepare_region(&s.data)?;
                context.prepare_region(&workspace)?;
                context.prepare_region(&dev_info)?;
                for matrix in [u.as_ref(), v.as_ref()].into_iter().flatten() {
                    context.prepare_region(&matrix.data)?;
                }

                let lwork = $buffer_size(context, mode, rank, m, n, layout, batch_size)?;
                if workspace.len() < lwork {
                    return Err(Error::BufferTooSmall {
                        name: "workspace".into(),
                        required: lwork,
                        actual: workspace.len(),
                    });
                }

                let (u, ldu, su) = strided_output_parts(u, m, rank, batch_size, mode)?;
                let (v, ldv, sv) = strided_output_parts(v, n, rank, batch_size, mode)?;
                // SAFETY: the capacities of `a`, `s`, `u`, `v`, and `dev_info` were checked above and `workspace` covers the queried `lwork`.
                // The caller retains and orders every operand, including the host residuals, through completion per the crate's raw execution contract.
                unsafe {
                    try_ffi!(sys::$sys(
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
        )+
    };
}

gesvda_strided_batched_by_precision! {
    sgesvda_strided_batched, f32, f32, cusolverDnSgesvdaStridedBatched, sgesvda_strided_batched_buffer_size;
    dgesvda_strided_batched, f64, f64, cusolverDnDgesvdaStridedBatched, dgesvda_strided_batched_buffer_size;
    cgesvda_strided_batched, Complex32, f32, cusolverDnCgesvdaStridedBatched, cgesvda_strided_batched_buffer_size;
    zgesvda_strided_batched, Complex64, f64, cusolverDnZgesvdaStridedBatched, zgesvda_strided_batched_buffer_size;
}
