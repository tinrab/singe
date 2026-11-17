use std::ptr;

use singe_cuda::{
    data_type::{DataType, DataTypeLike},
    types::{Complex32, Complex64},
    view::{DeviceOutputView, DeviceRegionProvider, DeviceView, DeviceViewMut},
};

use crate::{
    context::Context,
    dense::validation::{
        qr_rows, require_info_buffer, require_tau_buffer, require_workspace, validate_x_matrix,
        validate_x_vector,
    },
    error::Result,
    layout::{ByteWorkspaceMut, MatrixLayout, MatrixMut, VectorOutput, WorkspaceSizes},
    params::Params,
    sys, try_ffi,
    types::{Operation, SideMode},
    utility::{require_workspace_bytes, to_i32, to_i64, to_size_t, to_usize, validate_matrix},
};

macro_rules! orgqr_buffer_size_by_precision {
    ($($name:ident, $ty:ty, $sys:ident;)+) => {
        $(
            pub(crate) fn $name(
                context: &Context,
                m: usize,
                n: usize,
                k: usize,
                lda: usize,
            ) -> Result<usize> {
                context.bind()?;
                MatrixLayout::new(lda).validate(m, n)?;
                let mut lwork = 0;
                // SAFETY: the context handle is live and bound, the dimensions were validated above, and the call writes only the local `lwork`.
                unsafe {
                    try_ffi!(sys::$sys(
                        context.as_raw(),
                        to_i32(m, "m")?,
                        to_i32(n, "n")?,
                        to_i32(k, "k")?,
                        ptr::null_mut::<u8>().cast(),
                        to_i32(lda, "lda")?,
                        ptr::null_mut::<u8>().cast(),
                        &raw mut lwork,
                    ))?;
                }
                to_usize(lwork, "lwork")
            }
        )+
    };
}

orgqr_buffer_size_by_precision! {
    sorgqr_buffer_size, f32, cusolverDnSorgqr_bufferSize;
    dorgqr_buffer_size, f64, cusolverDnDorgqr_bufferSize;
    cungqr_buffer_size, Complex32, cusolverDnCungqr_bufferSize;
    zungqr_buffer_size, Complex64, cusolverDnZungqr_bufferSize;
}

macro_rules! orgqr_by_precision {
    ($($name:ident, $ty:ty, $sys:ident, $buffer_size:ident;)+) => {
        $(
            pub(crate) unsafe fn $name(
                context: &Context,
                m: usize,
                n: usize,
                k: usize,
                mut a: DeviceViewMut<'_, $ty>,
                lda: usize,
                tau: DeviceView<'_, $ty>,
                mut workspace: DeviceOutputView<'_, $ty>,
                mut dev_info: DeviceOutputView<'_, i32>,
            ) -> Result<()> {
                context.bind()?;
                context.prepare_region(&a)?;
                context.prepare_region(&tau)?;
                context.prepare_region(&workspace)?;
                context.prepare_region(&dev_info)?;
                validate_matrix(m, n, a.len(), lda)?;
                require_tau_buffer(tau.len(), k)?;
                require_info_buffer(dev_info.len())?;
                let lwork = $buffer_size(context, m, n, k, lda)?;
                require_workspace(workspace.len(), lwork)?;
                // SAFETY: the context handle is live and bound, every buffer was checked above against the dimensions and the required workspace and belongs to the context.
                // The caller retains the buffers through completion under the raw execution contract.
                unsafe {
                    try_ffi!(sys::$sys(
                        context.as_raw(),
                        to_i32(m, "m")?,
                        to_i32(n, "n")?,
                        to_i32(k, "k")?,
                        a.as_mut_ptr().cast(),
                        to_i32(lda, "lda")?,
                        tau.as_ptr().cast(),
                        workspace.as_mut_ptr().cast(),
                        to_i32(lwork, "lwork")?,
                        dev_info.as_mut_ptr().cast(),
                    ))?;
                }
                Ok(())
            }
        )+
    };
}

orgqr_by_precision! {
    sorgqr, f32, cusolverDnSorgqr, sorgqr_buffer_size;
    dorgqr, f64, cusolverDnDorgqr, dorgqr_buffer_size;
    cungqr, Complex32, cusolverDnCungqr, cungqr_buffer_size;
    zungqr, Complex64, cusolverDnZungqr, zungqr_buffer_size;
}

macro_rules! ormqr_buffer_size_by_precision {
    ($($name:ident, $ty:ty, $sys:ident;)+) => {
        $(
            pub(crate) fn $name(
                context: &Context,
                side: SideMode,
                operation: Operation,
                m: usize,
                n: usize,
                k: usize,
                lda: usize,
                ldc: usize,
            ) -> Result<usize> {
                context.bind()?;
                MatrixLayout::new(lda).validate(qr_rows(side, m, n), k)?;
                MatrixLayout::new(ldc).validate(m, n)?;
                let mut lwork = 0;
                // SAFETY: the context handle is live and bound, the dimensions were validated above, and the call writes only the local `lwork`.
                unsafe {
                    try_ffi!(sys::$sys(
                        context.as_raw(),
                        side.into(),
                        operation.into(),
                        to_i32(m, "m")?,
                        to_i32(n, "n")?,
                        to_i32(k, "k")?,
                        ptr::null_mut::<u8>().cast(),
                        to_i32(lda, "lda")?,
                        ptr::null_mut::<u8>().cast(),
                        ptr::null_mut::<u8>().cast(),
                        to_i32(ldc, "ldc")?,
                        &raw mut lwork,
                    ))?;
                }
                to_usize(lwork, "lwork")
            }
        )+
    };
}

ormqr_buffer_size_by_precision! {
    sormqr_buffer_size, f32, cusolverDnSormqr_bufferSize;
    dormqr_buffer_size, f64, cusolverDnDormqr_bufferSize;
    cunmqr_buffer_size, Complex32, cusolverDnCunmqr_bufferSize;
    zunmqr_buffer_size, Complex64, cusolverDnZunmqr_bufferSize;
}

macro_rules! ormqr_by_precision {
    ($($name:ident, $ty:ty, $sys:ident, $buffer_size:ident;)+) => {
        $(
            pub(crate) unsafe fn $name(
                context: &Context,
                side: SideMode,
                operation: Operation,
                m: usize,
                n: usize,
                k: usize,
                a: DeviceView<'_, $ty>,
                lda: usize,
                tau: DeviceView<'_, $ty>,
                mut c: DeviceViewMut<'_, $ty>,
                ldc: usize,
                mut workspace: DeviceOutputView<'_, $ty>,
                mut dev_info: DeviceOutputView<'_, i32>,
            ) -> Result<()> {
                context.bind()?;
                context.prepare_region(&a)?;
                context.prepare_region(&tau)?;
                context.prepare_region(&c)?;
                context.prepare_region(&workspace)?;
                context.prepare_region(&dev_info)?;
                validate_matrix(qr_rows(side, m, n), k, a.len(), lda)?;
                require_tau_buffer(tau.len(), k)?;
                validate_matrix(m, n, c.len(), ldc)?;
                require_info_buffer(dev_info.len())?;
                let lwork = $buffer_size(context, side, operation, m, n, k, lda, ldc)?;
                require_workspace(workspace.len(), lwork)?;
                // SAFETY: the context handle is live and bound, every buffer was checked above against the dimensions and the required workspace and belongs to the context.
                // The caller retains the buffers through completion under the raw execution contract.
                unsafe {
                    try_ffi!(sys::$sys(
                        context.as_raw(),
                        side.into(),
                        operation.into(),
                        to_i32(m, "m")?,
                        to_i32(n, "n")?,
                        to_i32(k, "k")?,
                        a.as_ptr().cast(),
                        to_i32(lda, "lda")?,
                        tau.as_ptr().cast(),
                        c.as_mut_ptr().cast(),
                        to_i32(ldc, "ldc")?,
                        workspace.as_mut_ptr().cast(),
                        to_i32(lwork, "lwork")?,
                        dev_info.as_mut_ptr().cast(),
                    ))?;
                }
                Ok(())
            }
        )+
    };
}

ormqr_by_precision! {
    sormqr, f32, cusolverDnSormqr, sormqr_buffer_size;
    dormqr, f64, cusolverDnDormqr, dormqr_buffer_size;
    cunmqr, Complex32, cusolverDnCunmqr, cunmqr_buffer_size;
    zunmqr, Complex64, cusolverDnZunmqr, zunmqr_buffer_size;
}

/// Returns the device and host workspace bytes that [`xgeqrf_async`] needs for an `m` by `n` matrix with layout `a`.
///
/// # Errors
///
/// Returns an error if the CUDA context cannot be bound, a dimension or leading dimension is invalid or does not fit the native integer types, or cuSOLVER rejects the configuration.
pub fn xgeqrf_workspace_size<TA: DataTypeLike, TTau: DataTypeLike>(
    context: &Context,
    params: &Params,
    m: usize,
    n: usize,
    a: MatrixLayout,
    compute_type: DataType,
) -> Result<WorkspaceSizes> {
    context.bind()?;
    let a_type = TA::data_type();
    let tau_type = TTau::data_type();
    a.validate(m, n)?;
    let mut device_bytes = 0;
    let mut host_bytes = 0;
    // SAFETY: the handles are live and the context is bound, the layouts were validated above, size queries accept null data pointers, and the call writes only the local byte counts.
    unsafe {
        try_ffi!(sys::cusolverDnXgeqrf_bufferSize(
            context.as_raw(),
            params.as_raw(),
            to_i64(m, "m")?,
            to_i64(n, "n")?,
            a_type.into(),
            ptr::null_mut::<u8>().cast(),
            to_i64(a.leading_dimension, "lda")?,
            tau_type.into(),
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

/// Enqueues the QR factorization `A = Q * R` of the `m` by `n` matrix `a`, in place.
///
/// `R` overwrites the upper triangle of `a`, including the diagonal.
/// `Q` is not formed: the lower triangle holds the Householder vectors, each with an implicit leading 1, and `tau` receives their `min(m, n)` scaling factors.
/// `dev_info` receives `-i` when the `i`th argument is invalid.
/// Size `workspace` with [`xgeqrf_workspace_size`].
/// Supported element types are `f32`, `f64`, `Complex32`, and `Complex64`, with `TA`, `TTau`, and `compute_type` equal.
///
/// # Safety
///
/// Follow the [raw execution contract](crate#raw-execution).
///
/// # Errors
///
/// Returns an error if the CUDA context cannot be bound, a buffer belongs to another CUDA context, a dimension, leading dimension, or buffer capacity is invalid or does not fit the native integer types, the workspace is smaller than the queried size, or cuSOLVER rejects the call.
pub unsafe fn xgeqrf_async<TA: DataTypeLike, TTau: DataTypeLike>(
    context: &Context,
    params: &Params,
    m: usize,
    n: usize,
    mut a: MatrixMut<'_, TA>,
    mut tau: VectorOutput<'_, TTau>,
    compute_type: DataType,
    mut workspace: ByteWorkspaceMut<'_>,
    mut dev_info: DeviceOutputView<'_, i32>,
) -> Result<()> {
    context.bind()?;
    context.prepare_region(&a.data)?;
    context.prepare_region(&tau.data)?;
    context.prepare_region(&dev_info)?;
    context.prepare_region(&workspace.device)?;
    let a_type = TA::data_type();
    let tau_type = TTau::data_type();
    validate_x_matrix(
        m,
        n,
        a.data.region().byte_len(),
        a.leading_dimension,
        a_type,
    )?;
    validate_x_vector(m.min(n), tau.data.region().byte_len(), tau_type)?;
    require_info_buffer(dev_info.len())?;
    let workspace_sizes =
        xgeqrf_workspace_size::<TA, TTau>(context, params, m, n, a.layout(), compute_type)?;
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
        try_ffi!(sys::cusolverDnXgeqrf(
            context.as_raw(),
            params.as_raw(),
            to_i64(m, "m")?,
            to_i64(n, "n")?,
            a_type.into(),
            a.data.as_mut_ptr().cast(),
            to_i64(a.leading_dimension, "lda")?,
            tau_type.into(),
            tau.data.as_mut_ptr().cast(),
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
