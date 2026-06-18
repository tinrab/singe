use singe_cuda::{
    data_type::{DataType, DataTypeLike},
    memory::DeviceMemory,
    types::{Complex32, Complex64},
};

use crate::{
    context::Context,
    dense::validation::{
        qr_rows, require_host_workspace, require_info_buffer, require_tau_buffer,
        require_workspace, require_workspace_bytes, validate_matrix, validate_x_matrix,
        validate_x_vector,
    },
    error::Result,
    layout::{ByteWorkspaceMut, MatrixMut, MatrixRef, VectorMut, VectorRef, WorkspaceSizes},
    params::Params,
    sys, try_ffi,
    types::{Operation, SideMode},
    utility::{to_i32, to_i64, to_usize},
};

pub fn sorgqr_buffer_size(
    ctx: &Context,
    m: usize,
    n: usize,
    k: usize,
    a: &DeviceMemory<f32>,
    lda: usize,
    tau: &DeviceMemory<f32>,
) -> Result<usize> {
    ctx.bind()?;
    validate_matrix(m, n, a.len(), lda)?;
    require_tau_buffer(tau, k)?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnSorgqr_bufferSize(
            ctx.as_raw(),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            to_i32(k, "k")?,
            a.as_ptr().cast(),
            to_i32(lda, "lda")?,
            tau.as_ptr().cast(),
            &raw mut lwork,
        ))?;
    }
    to_usize(lwork, "lwork")
}

pub fn dorgqr_buffer_size(
    ctx: &Context,
    m: usize,
    n: usize,
    k: usize,
    a: &DeviceMemory<f64>,
    lda: usize,
    tau: &DeviceMemory<f64>,
) -> Result<usize> {
    ctx.bind()?;
    validate_matrix(m, n, a.len(), lda)?;
    require_tau_buffer(tau, k)?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnDorgqr_bufferSize(
            ctx.as_raw(),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            to_i32(k, "k")?,
            a.as_ptr().cast(),
            to_i32(lda, "lda")?,
            tau.as_ptr().cast(),
            &raw mut lwork,
        ))?;
    }
    to_usize(lwork, "lwork")
}

pub fn cungqr_buffer_size(
    ctx: &Context,
    m: usize,
    n: usize,
    k: usize,
    a: &DeviceMemory<Complex32>,
    lda: usize,
    tau: &DeviceMemory<Complex32>,
) -> Result<usize> {
    ctx.bind()?;
    validate_matrix(m, n, a.len(), lda)?;
    require_tau_buffer(tau, k)?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnCungqr_bufferSize(
            ctx.as_raw(),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            to_i32(k, "k")?,
            a.as_ptr().cast(),
            to_i32(lda, "lda")?,
            tau.as_ptr().cast(),
            &raw mut lwork,
        ))?;
    }
    to_usize(lwork, "lwork")
}

pub fn zungqr_buffer_size(
    ctx: &Context,
    m: usize,
    n: usize,
    k: usize,
    a: &DeviceMemory<Complex64>,
    lda: usize,
    tau: &DeviceMemory<Complex64>,
) -> Result<usize> {
    ctx.bind()?;
    validate_matrix(m, n, a.len(), lda)?;
    require_tau_buffer(tau, k)?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnZungqr_bufferSize(
            ctx.as_raw(),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            to_i32(k, "k")?,
            a.as_ptr().cast(),
            to_i32(lda, "lda")?,
            tau.as_ptr().cast(),
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
/// Generates the first `n` columns of the orthogonal matrix `Q` from the
/// elementary reflectors returned by `geqrf` and stores them in `A`.
///
/// Provide workspace through `workspace`.
/// Use the corresponding `*_buffer_size` helper to query the required workspace length.
/// The workspace size in bytes is `size_of::<T>() * lwork`.
///
/// If the reported `dev_info` value is `-i`, the `i`th parameter is invalid.
///
/// Callers can combine `geqrf` and `orgqr` to complete orthogonalization.
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions, reflector count, or leading dimension are invalid, if
/// the current GPU architecture is unsupported, or if cuSOLVER reports an
/// internal failure.
pub fn sorgqr(
    ctx: &Context,
    m: usize,
    n: usize,
    k: usize,
    a: &mut DeviceMemory<f32>,
    lda: usize,
    tau: &DeviceMemory<f32>,
    workspace: &mut DeviceMemory<f32>,
    dev_info: &mut DeviceMemory<i32>,
) -> Result<()> {
    ctx.bind()?;
    validate_matrix(m, n, a.len(), lda)?;
    require_tau_buffer(tau, k)?;
    require_info_buffer(dev_info)?;
    let lwork = sorgqr_buffer_size(ctx, m, n, k, a, lda, tau)?;
    require_workspace(workspace.len(), lwork)?;
    unsafe {
        try_ffi!(sys::cusolverDnSorgqr(
            ctx.as_raw(),
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

/// Use the matching buffer-size helper to calculate the required workspace size.
///
/// The S and D data types are real valued single and double precision, respectively.
///
/// The C and Z data types are complex valued single and double precision, respectively.
///
/// Generates the first `n` columns of the orthogonal matrix `Q` from the
/// elementary reflectors returned by `geqrf` and stores them in `A`.
///
/// Provide workspace through `workspace`.
/// Use the corresponding `*_buffer_size` helper to query the required workspace length.
/// The workspace size in bytes is `size_of::<T>() * lwork`.
///
/// If the reported `dev_info` value is `-i`, the `i`th parameter is invalid.
///
/// Callers can combine `geqrf` and `orgqr` to complete orthogonalization.
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions, reflector count, or leading dimension are invalid, if
/// the current GPU architecture is unsupported, or if cuSOLVER reports an
/// internal failure.
pub fn dorgqr(
    ctx: &Context,
    m: usize,
    n: usize,
    k: usize,
    a: &mut DeviceMemory<f64>,
    lda: usize,
    tau: &DeviceMemory<f64>,
    workspace: &mut DeviceMemory<f64>,
    dev_info: &mut DeviceMemory<i32>,
) -> Result<()> {
    ctx.bind()?;
    validate_matrix(m, n, a.len(), lda)?;
    require_tau_buffer(tau, k)?;
    require_info_buffer(dev_info)?;
    let lwork = dorgqr_buffer_size(ctx, m, n, k, a, lda, tau)?;
    require_workspace(workspace.len(), lwork)?;
    unsafe {
        try_ffi!(sys::cusolverDnDorgqr(
            ctx.as_raw(),
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

pub fn cungqr(
    ctx: &Context,
    m: usize,
    n: usize,
    k: usize,
    a: &mut DeviceMemory<Complex32>,
    lda: usize,
    tau: &DeviceMemory<Complex32>,
    workspace: &mut DeviceMemory<Complex32>,
    dev_info: &mut DeviceMemory<i32>,
) -> Result<()> {
    ctx.bind()?;
    validate_matrix(m, n, a.len(), lda)?;
    require_tau_buffer(tau, k)?;
    require_info_buffer(dev_info)?;
    let lwork = cungqr_buffer_size(ctx, m, n, k, a, lda, tau)?;
    require_workspace(workspace.len(), lwork)?;
    unsafe {
        try_ffi!(sys::cusolverDnCungqr(
            ctx.as_raw(),
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

pub fn zungqr(
    ctx: &Context,
    m: usize,
    n: usize,
    k: usize,
    a: &mut DeviceMemory<Complex64>,
    lda: usize,
    tau: &DeviceMemory<Complex64>,
    workspace: &mut DeviceMemory<Complex64>,
    dev_info: &mut DeviceMemory<i32>,
) -> Result<()> {
    ctx.bind()?;
    validate_matrix(m, n, a.len(), lda)?;
    require_tau_buffer(tau, k)?;
    require_info_buffer(dev_info)?;
    let lwork = zungqr_buffer_size(ctx, m, n, k, a, lda, tau)?;
    require_workspace(workspace.len(), lwork)?;
    unsafe {
        try_ffi!(sys::cusolverDnZungqr(
            ctx.as_raw(),
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

pub fn sormqr_buffer_size(
    ctx: &Context,
    side: SideMode,
    operation: Operation,
    m: usize,
    n: usize,
    k: usize,
    a: &DeviceMemory<f32>,
    lda: usize,
    tau: &DeviceMemory<f32>,
    c: &DeviceMemory<f32>,
    ldc: usize,
) -> Result<usize> {
    ctx.bind()?;
    validate_matrix(qr_rows(side, m, n), k, a.len(), lda)?;
    require_tau_buffer(tau, k)?;
    validate_matrix(m, n, c.len(), ldc)?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnSormqr_bufferSize(
            ctx.as_raw(),
            side.into(),
            operation.into(),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            to_i32(k, "k")?,
            a.as_ptr().cast(),
            to_i32(lda, "lda")?,
            tau.as_ptr().cast(),
            c.as_ptr().cast(),
            to_i32(ldc, "ldc")?,
            &raw mut lwork,
        ))?;
    }
    to_usize(lwork, "lwork")
}

pub fn dormqr_buffer_size(
    ctx: &Context,
    side: SideMode,
    operation: Operation,
    m: usize,
    n: usize,
    k: usize,
    a: &DeviceMemory<f64>,
    lda: usize,
    tau: &DeviceMemory<f64>,
    c: &DeviceMemory<f64>,
    ldc: usize,
) -> Result<usize> {
    ctx.bind()?;
    validate_matrix(qr_rows(side, m, n), k, a.len(), lda)?;
    require_tau_buffer(tau, k)?;
    validate_matrix(m, n, c.len(), ldc)?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnDormqr_bufferSize(
            ctx.as_raw(),
            side.into(),
            operation.into(),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            to_i32(k, "k")?,
            a.as_ptr().cast(),
            to_i32(lda, "lda")?,
            tau.as_ptr().cast(),
            c.as_ptr().cast(),
            to_i32(ldc, "ldc")?,
            &raw mut lwork,
        ))?;
    }
    to_usize(lwork, "lwork")
}

pub fn cunmqr_buffer_size(
    ctx: &Context,
    side: SideMode,
    operation: Operation,
    m: usize,
    n: usize,
    k: usize,
    a: &DeviceMemory<Complex32>,
    lda: usize,
    tau: &DeviceMemory<Complex32>,
    c: &DeviceMemory<Complex32>,
    ldc: usize,
) -> Result<usize> {
    ctx.bind()?;
    validate_matrix(qr_rows(side, m, n), k, a.len(), lda)?;
    require_tau_buffer(tau, k)?;
    validate_matrix(m, n, c.len(), ldc)?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnCunmqr_bufferSize(
            ctx.as_raw(),
            side.into(),
            operation.into(),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            to_i32(k, "k")?,
            a.as_ptr().cast(),
            to_i32(lda, "lda")?,
            tau.as_ptr().cast(),
            c.as_ptr().cast(),
            to_i32(ldc, "ldc")?,
            &raw mut lwork,
        ))?;
    }
    to_usize(lwork, "lwork")
}

pub fn zunmqr_buffer_size(
    ctx: &Context,
    side: SideMode,
    operation: Operation,
    m: usize,
    n: usize,
    k: usize,
    a: &DeviceMemory<Complex64>,
    lda: usize,
    tau: &DeviceMemory<Complex64>,
    c: &DeviceMemory<Complex64>,
    ldc: usize,
) -> Result<usize> {
    ctx.bind()?;
    validate_matrix(qr_rows(side, m, n), k, a.len(), lda)?;
    require_tau_buffer(tau, k)?;
    validate_matrix(m, n, c.len(), ldc)?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnZunmqr_bufferSize(
            ctx.as_raw(),
            side.into(),
            operation.into(),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            to_i32(k, "k")?,
            a.as_ptr().cast(),
            to_i32(lda, "lda")?,
            tau.as_ptr().cast(),
            c.as_ptr().cast(),
            to_i32(ldc, "ldc")?,
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
/// Applies the orthogonal matrix `Q`, represented by the elementary reflectors
/// returned by `geqrf`, to `C` and stores the result in `C`.
///
/// `operation` selects whether `Q` is transposed.
///
/// `Q` is of order `m` if `side` = [`SideMode::Left`] and of order `n` if `side` = [`SideMode::Right`].
///
/// Provide workspace through `workspace`.
/// Use the corresponding `*_buffer_size` helper to query the required workspace length.
/// The workspace size in bytes is `size_of::<T>() * lwork`.
///
/// If the reported `dev_info` value is `-i`, the `i`th parameter is invalid.
///
/// Callers can combine `geqrf`, `ormqr`, and `trsm` to complete a linear solver or a least-square solver.
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions, reflector count, side/operation mode, or leading
/// dimensions are invalid, if the current GPU architecture is unsupported, or
/// if cuSOLVER reports an internal failure.
pub fn sormqr(
    ctx: &Context,
    side: SideMode,
    operation: Operation,
    m: usize,
    n: usize,
    k: usize,
    a: &DeviceMemory<f32>,
    lda: usize,
    tau: &DeviceMemory<f32>,
    c: &mut DeviceMemory<f32>,
    ldc: usize,
    workspace: &mut DeviceMemory<f32>,
    dev_info: &mut DeviceMemory<i32>,
) -> Result<()> {
    ctx.bind()?;
    validate_matrix(qr_rows(side, m, n), k, a.len(), lda)?;
    require_tau_buffer(tau, k)?;
    validate_matrix(m, n, c.len(), ldc)?;
    require_info_buffer(dev_info)?;
    let lwork = sormqr_buffer_size(ctx, side, operation, m, n, k, a, lda, tau, c, ldc)?;
    require_workspace(workspace.len(), lwork)?;
    unsafe {
        try_ffi!(sys::cusolverDnSormqr(
            ctx.as_raw(),
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

/// Use the matching buffer-size helper to calculate the required workspace size.
///
/// The S and D data types are real valued single and double precision, respectively.
///
/// The C and Z data types are complex valued single and double precision, respectively.
///
/// Applies the orthogonal matrix `Q`, represented by the elementary reflectors
/// returned by `geqrf`, to `C` and stores the result in `C`.
///
/// `operation` selects whether `Q` is transposed.
///
/// `Q` is of order `m` if `side` = [`SideMode::Left`] and of order `n` if `side` = [`SideMode::Right`].
///
/// Provide workspace through `workspace`.
/// Use the corresponding `*_buffer_size` helper to query the required workspace length.
/// The workspace size in bytes is `size_of::<T>() * lwork`.
///
/// If the reported `dev_info` value is `-i`, the `i`th parameter is invalid.
///
/// Callers can combine `geqrf`, `ormqr`, and `trsm` to complete a linear solver or a least-square solver.
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions, reflector count, side/operation mode, or leading
/// dimensions are invalid, if the current GPU architecture is unsupported, or
/// if cuSOLVER reports an internal failure.
pub fn dormqr(
    ctx: &Context,
    side: SideMode,
    operation: Operation,
    m: usize,
    n: usize,
    k: usize,
    a: &DeviceMemory<f64>,
    lda: usize,
    tau: &DeviceMemory<f64>,
    c: &mut DeviceMemory<f64>,
    ldc: usize,
    workspace: &mut DeviceMemory<f64>,
    dev_info: &mut DeviceMemory<i32>,
) -> Result<()> {
    ctx.bind()?;
    validate_matrix(qr_rows(side, m, n), k, a.len(), lda)?;
    require_tau_buffer(tau, k)?;
    validate_matrix(m, n, c.len(), ldc)?;
    require_info_buffer(dev_info)?;
    let lwork = dormqr_buffer_size(ctx, side, operation, m, n, k, a, lda, tau, c, ldc)?;
    require_workspace(workspace.len(), lwork)?;
    unsafe {
        try_ffi!(sys::cusolverDnDormqr(
            ctx.as_raw(),
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

pub fn cunmqr(
    ctx: &Context,
    side: SideMode,
    operation: Operation,
    m: usize,
    n: usize,
    k: usize,
    a: &DeviceMemory<Complex32>,
    lda: usize,
    tau: &DeviceMemory<Complex32>,
    c: &mut DeviceMemory<Complex32>,
    ldc: usize,
    workspace: &mut DeviceMemory<Complex32>,
    dev_info: &mut DeviceMemory<i32>,
) -> Result<()> {
    ctx.bind()?;
    validate_matrix(qr_rows(side, m, n), k, a.len(), lda)?;
    require_tau_buffer(tau, k)?;
    validate_matrix(m, n, c.len(), ldc)?;
    require_info_buffer(dev_info)?;
    let lwork = cunmqr_buffer_size(ctx, side, operation, m, n, k, a, lda, tau, c, ldc)?;
    require_workspace(workspace.len(), lwork)?;
    unsafe {
        try_ffi!(sys::cusolverDnCunmqr(
            ctx.as_raw(),
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

pub fn zunmqr(
    ctx: &Context,
    side: SideMode,
    operation: Operation,
    m: usize,
    n: usize,
    k: usize,
    a: &DeviceMemory<Complex64>,
    lda: usize,
    tau: &DeviceMemory<Complex64>,
    c: &mut DeviceMemory<Complex64>,
    ldc: usize,
    workspace: &mut DeviceMemory<Complex64>,
    dev_info: &mut DeviceMemory<i32>,
) -> Result<()> {
    ctx.bind()?;
    validate_matrix(qr_rows(side, m, n), k, a.len(), lda)?;
    require_tau_buffer(tau, k)?;
    validate_matrix(m, n, c.len(), ldc)?;
    require_info_buffer(dev_info)?;
    let lwork = zunmqr_buffer_size(ctx, side, operation, m, n, k, a, lda, tau, c, ldc)?;
    require_workspace(workspace.len(), lwork)?;
    unsafe {
        try_ffi!(sys::cusolverDnZunmqr(
            ctx.as_raw(),
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

pub fn xgeqrf_buffer_size<TA: DataTypeLike, TTau: DataTypeLike>(
    ctx: &Context,
    params: &Params,
    m: usize,
    n: usize,
    a: MatrixRef<'_, TA>,
    tau: VectorRef<'_, TTau>,
    compute_type: DataType,
) -> Result<WorkspaceSizes> {
    ctx.bind()?;
    let a_type = TA::data_type();
    let tau_type = TTau::data_type();
    validate_x_matrix(m, n, a.data.byte_len(), a.leading_dimension, a_type)?;
    validate_x_vector(m.min(n), tau.data.byte_len(), tau_type)?;
    let mut device_bytes = 0;
    let mut host_bytes = 0;
    unsafe {
        try_ffi!(sys::cusolverDnXgeqrf_bufferSize(
            ctx.as_raw(),
            params.as_raw(),
            to_i64(m, "m")?,
            to_i64(n, "n")?,
            a_type.into(),
            a.data.as_ptr().cast(),
            to_i64(a.leading_dimension, "lda")?,
            tau_type.into(),
            tau.data.as_ptr().cast(),
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

/// Use [`xgeqrf_buffer_size`] to calculate the sizes needed for pre-allocated
/// workspace.
///
/// Computes the QR factorization of an $m \times n$ matrix.
///
/// Here `A` is an $m \times n$ matrix, `Q` is an $m \times n$ matrix, and
/// `R` is an $n \times n$ upper triangular matrix.
///
/// Provide device and host workspace through `workspace`.
/// Use [`xgeqrf_buffer_size`] to determine the required sizes for
/// `workspace.device` and `workspace.host`.
///
/// The matrix `R` overwrites the upper triangular part of `A`, including the
/// diagonal elements.
///
/// The matrix `Q` is not formed explicitly. Instead, a sequence of Householder
/// vectors is stored in the lower triangular part of `A`.
/// The leading nonzero element of the Householder vector is assumed to be 1, so `tau` contains the scaling factor `τ`.
/// If `v` is the original Householder vector, `q` is the new Householder vector
/// corresponding to `τ`.
///
/// If the reported `info` value is `-i`, the `i`th parameter is invalid.
///
/// Currently, [`xgeqrf`] supports only the default algorithm.
///
/// **Algorithms supported by [`xgeqrf`]**
///
/// | Algorithm | Notes |
/// | --- | --- |
/// | [`AlgorithmMode::Default`](crate::types::AlgorithmMode::Default) | Default algorithm. |
///
/// List of input arguments for [`xgeqrf_buffer_size`] and [`xgeqrf`]:
///
/// The generic cuSOLVER routine separates matrix, tau-vector, and compute data types:
/// `data_type_a` is the data type of matrix `A`, `data_type_tau` is the data
/// type of `tau`, and `compute_type` is the operation's compute type.
/// [`xgeqrf`] only supports the following four combinations.
///
/// **Valid combination of data type and compute type**
///
/// | **data_type_a** | **compute_type** | **Meaning** |
/// | --- | --- | --- |
/// | [`DataType::F32`] | [`DataType::F32`] | `SGEQRF` |
/// | [`DataType::F64`] | [`DataType::F64`] | `DGEQRF` |
/// | [`DataType::ComplexF32`] | [`DataType::ComplexF32`] | `CGEQRF` |
/// | [`DataType::ComplexF64`] | [`DataType::ComplexF64`] | `ZGEQRF` |
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions or leading dimension are invalid, or if cuSOLVER reports
/// an internal failure.
pub fn xgeqrf<TA: DataTypeLike, TTau: DataTypeLike>(
    ctx: &Context,
    params: &Params,
    m: usize,
    n: usize,
    a: MatrixMut<'_, TA>,
    tau: VectorMut<'_, TTau>,
    compute_type: DataType,
    workspace: ByteWorkspaceMut<'_>,
    dev_info: &mut DeviceMemory<i32>,
) -> Result<()> {
    ctx.bind()?;
    let a_type = TA::data_type();
    let tau_type = TTau::data_type();
    validate_x_matrix(m, n, a.data.byte_len(), a.leading_dimension, a_type)?;
    validate_x_vector(m.min(n), tau.data.byte_len(), tau_type)?;
    require_info_buffer(dev_info)?;
    let workspace_sizes =
        xgeqrf_buffer_size(ctx, params, m, n, a.as_ref(), tau.as_ref(), compute_type)?;
    require_workspace_bytes(workspace.device.byte_len(), workspace_sizes.device_bytes)?;
    require_host_workspace(workspace.host.len(), workspace_sizes.host_bytes)?;
    unsafe {
        try_ffi!(sys::cusolverDnXgeqrf(
            ctx.as_raw(),
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
            workspace_sizes.device_bytes as _,
            workspace.host.as_mut_ptr().cast(),
            workspace_sizes.host_bytes as _,
            dev_info.as_mut_ptr().cast(),
        ))?;
    }
    Ok(())
}
