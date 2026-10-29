#![expect(
    clippy::many_single_char_names,
    reason = "BLAS and factorization operand names follow the standard mathematical API"
)]
use std::ptr;

use singe_cuda::{
    data_type::{DataType, DataTypeLike},
    types::{Complex32, Complex64},
    view::{DeviceOutputView, DeviceRegionProvider, DeviceView, DeviceViewMut},
};

use crate::{
    context::Context,
    dense::validation::{
        qr_rows, require_host_workspace, require_info_buffer, require_tau_buffer,
        require_workspace, require_workspace_bytes, validate_matrix, validate_x_matrix,
        validate_x_vector,
    },
    error::Result,
    layout::{ByteWorkspaceMut, MatrixLayout, MatrixMut, VectorOutput, WorkspaceSizes},
    params::Params,
    sys, try_ffi,
    types::{Operation, SideMode},
    utility::{to_i32, to_i64, to_usize},
};

/// Queries device workspace capacity in elements for `f32` generation of orthogonal QR factors.
///
/// Uses shape and layout metadata without borrowing matrix storage.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn sorgqr_buffer_size(
    context: &Context,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
) -> Result<usize> {
    context.bind()?;
    MatrixLayout::new(lda).validate(m, n)?;
    let mut lwork = 0;
    // SAFETY: `cusolverDnSorgqr_bufferSize` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnSorgqr_bufferSize(
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

/// Queries device workspace capacity in elements for `f64` generation of orthogonal QR factors.
///
/// Uses shape and layout metadata without borrowing matrix storage.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn dorgqr_buffer_size(
    context: &Context,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
) -> Result<usize> {
    context.bind()?;
    MatrixLayout::new(lda).validate(m, n)?;
    let mut lwork = 0;
    // SAFETY: `cusolverDnDorgqr_bufferSize` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnDorgqr_bufferSize(
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

/// Queries device workspace capacity in elements for complex single-precision generation of unitary QR factors.
///
/// Uses shape and layout metadata without borrowing matrix storage.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn cungqr_buffer_size(
    context: &Context,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
) -> Result<usize> {
    context.bind()?;
    MatrixLayout::new(lda).validate(m, n)?;
    let mut lwork = 0;
    // SAFETY: `cusolverDnCungqr_bufferSize` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnCungqr_bufferSize(
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

/// Queries device workspace capacity in elements for complex double-precision generation of unitary QR factors.
///
/// Uses shape and layout metadata without borrowing matrix storage.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn zungqr_buffer_size(
    context: &Context,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
) -> Result<usize> {
    context.bind()?;
    MatrixLayout::new(lda).validate(m, n)?;
    let mut lwork = 0;
    // SAFETY: `cusolverDnZungqr_bufferSize` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnZungqr_bufferSize(
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
/// # Safety
///
/// Follow the [raw solver contracts](crate::dense#raw-solver-contracts).
pub unsafe fn sorgqr(
    context: &Context,
    m: usize,
    n: usize,
    k: usize,
    mut a: DeviceViewMut<'_, f32>,
    lda: usize,
    tau: DeviceView<'_, f32>,
    mut workspace: DeviceOutputView<'_, f32>,
    mut dev_info: DeviceOutputView<'_, i32>,
) -> Result<()> {
    context.bind()?;
    context.validate_region(&a)?;
    context.validate_region(&tau)?;
    context.validate_region(&workspace)?;
    context.validate_region(&dev_info)?;
    validate_matrix(m, n, a.len(), lda)?;
    require_tau_buffer(tau.len(), k)?;
    require_info_buffer(dev_info.len())?;
    let lwork = sorgqr_buffer_size(context, m, n, k, lda)?;
    require_workspace(workspace.len(), lwork)?;
    // SAFETY: `cusolverDnSorgqr` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cusolverDnSorgqr(
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
/// # Safety
///
/// Follow the [raw solver contracts](crate::dense#raw-solver-contracts).
pub unsafe fn dorgqr(
    context: &Context,
    m: usize,
    n: usize,
    k: usize,
    mut a: DeviceViewMut<'_, f64>,
    lda: usize,
    tau: DeviceView<'_, f64>,
    mut workspace: DeviceOutputView<'_, f64>,
    mut dev_info: DeviceOutputView<'_, i32>,
) -> Result<()> {
    context.bind()?;
    context.validate_region(&a)?;
    context.validate_region(&tau)?;
    context.validate_region(&workspace)?;
    context.validate_region(&dev_info)?;
    validate_matrix(m, n, a.len(), lda)?;
    require_tau_buffer(tau.len(), k)?;
    require_info_buffer(dev_info.len())?;
    let lwork = dorgqr_buffer_size(context, m, n, k, lda)?;
    require_workspace(workspace.len(), lwork)?;
    // SAFETY: `cusolverDnDorgqr` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cusolverDnDorgqr(
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

/// # Safety
///
/// Follow the [raw solver contracts](crate::dense#raw-solver-contracts).
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a device buffer belongs to another CUDA context, matrix dimensions, leading dimensions, or storage capacities are invalid, checked size arithmetic or native integer conversion fails, the status output capacity is insufficient, a pivot, reflector, or numerical output buffer is too small, device scratch capacity is smaller than the queried requirement, or cuSOLVER rejects the configuration or returns an error while querying workspace or submitting execution.
pub unsafe fn cungqr(
    context: &Context,
    m: usize,
    n: usize,
    k: usize,
    mut a: DeviceViewMut<'_, Complex32>,
    lda: usize,
    tau: DeviceView<'_, Complex32>,
    mut workspace: DeviceOutputView<'_, Complex32>,
    mut dev_info: DeviceOutputView<'_, i32>,
) -> Result<()> {
    context.bind()?;
    context.validate_region(&a)?;
    context.validate_region(&tau)?;
    context.validate_region(&workspace)?;
    context.validate_region(&dev_info)?;
    validate_matrix(m, n, a.len(), lda)?;
    require_tau_buffer(tau.len(), k)?;
    require_info_buffer(dev_info.len())?;
    let lwork = cungqr_buffer_size(context, m, n, k, lda)?;
    require_workspace(workspace.len(), lwork)?;
    // SAFETY: `cusolverDnCungqr` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cusolverDnCungqr(
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

/// # Safety
///
/// Follow the [raw solver contracts](crate::dense#raw-solver-contracts).
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a device buffer belongs to another CUDA context, matrix dimensions, leading dimensions, or storage capacities are invalid, checked size arithmetic or native integer conversion fails, the status output capacity is insufficient, a pivot, reflector, or numerical output buffer is too small, device scratch capacity is smaller than the queried requirement, or cuSOLVER rejects the configuration or returns an error while querying workspace or submitting execution.
pub unsafe fn zungqr(
    context: &Context,
    m: usize,
    n: usize,
    k: usize,
    mut a: DeviceViewMut<'_, Complex64>,
    lda: usize,
    tau: DeviceView<'_, Complex64>,
    mut workspace: DeviceOutputView<'_, Complex64>,
    mut dev_info: DeviceOutputView<'_, i32>,
) -> Result<()> {
    context.bind()?;
    context.validate_region(&a)?;
    context.validate_region(&tau)?;
    context.validate_region(&workspace)?;
    context.validate_region(&dev_info)?;
    validate_matrix(m, n, a.len(), lda)?;
    require_tau_buffer(tau.len(), k)?;
    require_info_buffer(dev_info.len())?;
    let lwork = zungqr_buffer_size(context, m, n, k, lda)?;
    require_workspace(workspace.len(), lwork)?;
    // SAFETY: `cusolverDnZungqr` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cusolverDnZungqr(
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

/// Queries device workspace capacity in elements for `f32` application of orthogonal QR factors.
///
/// Uses shape and layout metadata without borrowing matrix storage.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn sormqr_buffer_size(
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
    // SAFETY: `cusolverDnSormqr_bufferSize` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnSormqr_bufferSize(
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

/// Queries device workspace capacity in elements for `f64` application of orthogonal QR factors.
///
/// Uses shape and layout metadata without borrowing matrix storage.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn dormqr_buffer_size(
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
    // SAFETY: `cusolverDnDormqr_bufferSize` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnDormqr_bufferSize(
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

/// Queries device workspace capacity in elements for complex single-precision application of unitary QR factors.
///
/// Uses shape and layout metadata without borrowing matrix storage.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn cunmqr_buffer_size(
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
    // SAFETY: `cusolverDnCunmqr_bufferSize` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnCunmqr_bufferSize(
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

/// Queries device workspace capacity in elements for complex double-precision application of unitary QR factors.
///
/// Uses shape and layout metadata without borrowing matrix storage.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn zunmqr_buffer_size(
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
    // SAFETY: `cusolverDnZunmqr_bufferSize` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnZunmqr_bufferSize(
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
/// # Safety
///
/// Follow the [raw solver contracts](crate::dense#raw-solver-contracts).
pub unsafe fn sormqr(
    context: &Context,
    side: SideMode,
    operation: Operation,
    m: usize,
    n: usize,
    k: usize,
    a: DeviceView<'_, f32>,
    lda: usize,
    tau: DeviceView<'_, f32>,
    mut c: DeviceViewMut<'_, f32>,
    ldc: usize,
    mut workspace: DeviceOutputView<'_, f32>,
    mut dev_info: DeviceOutputView<'_, i32>,
) -> Result<()> {
    context.bind()?;
    context.validate_region(&a)?;
    context.validate_region(&tau)?;
    context.validate_region(&c)?;
    context.validate_region(&workspace)?;
    context.validate_region(&dev_info)?;
    validate_matrix(qr_rows(side, m, n), k, a.len(), lda)?;
    require_tau_buffer(tau.len(), k)?;
    validate_matrix(m, n, c.len(), ldc)?;
    require_info_buffer(dev_info.len())?;
    let lwork = sormqr_buffer_size(context, side, operation, m, n, k, lda, ldc)?;
    require_workspace(workspace.len(), lwork)?;
    // SAFETY: `cusolverDnSormqr` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cusolverDnSormqr(
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
/// # Safety
///
/// Follow the [raw solver contracts](crate::dense#raw-solver-contracts).
pub unsafe fn dormqr(
    context: &Context,
    side: SideMode,
    operation: Operation,
    m: usize,
    n: usize,
    k: usize,
    a: DeviceView<'_, f64>,
    lda: usize,
    tau: DeviceView<'_, f64>,
    mut c: DeviceViewMut<'_, f64>,
    ldc: usize,
    mut workspace: DeviceOutputView<'_, f64>,
    mut dev_info: DeviceOutputView<'_, i32>,
) -> Result<()> {
    context.bind()?;
    context.validate_region(&a)?;
    context.validate_region(&tau)?;
    context.validate_region(&c)?;
    context.validate_region(&workspace)?;
    context.validate_region(&dev_info)?;
    validate_matrix(qr_rows(side, m, n), k, a.len(), lda)?;
    require_tau_buffer(tau.len(), k)?;
    validate_matrix(m, n, c.len(), ldc)?;
    require_info_buffer(dev_info.len())?;
    let lwork = dormqr_buffer_size(context, side, operation, m, n, k, lda, ldc)?;
    require_workspace(workspace.len(), lwork)?;
    // SAFETY: `cusolverDnDormqr` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cusolverDnDormqr(
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

/// # Safety
///
/// Follow the [raw solver contracts](crate::dense#raw-solver-contracts).
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a device buffer belongs to another CUDA context, matrix dimensions, leading dimensions, or storage capacities are invalid, checked size arithmetic or native integer conversion fails, the status output capacity is insufficient, a pivot, reflector, or numerical output buffer is too small, device scratch capacity is smaller than the queried requirement, or cuSOLVER rejects the configuration or returns an error while querying workspace or submitting execution.
pub unsafe fn cunmqr(
    context: &Context,
    side: SideMode,
    operation: Operation,
    m: usize,
    n: usize,
    k: usize,
    a: DeviceView<'_, Complex32>,
    lda: usize,
    tau: DeviceView<'_, Complex32>,
    mut c: DeviceViewMut<'_, Complex32>,
    ldc: usize,
    mut workspace: DeviceOutputView<'_, Complex32>,
    mut dev_info: DeviceOutputView<'_, i32>,
) -> Result<()> {
    context.bind()?;
    context.validate_region(&a)?;
    context.validate_region(&tau)?;
    context.validate_region(&c)?;
    context.validate_region(&workspace)?;
    context.validate_region(&dev_info)?;
    validate_matrix(qr_rows(side, m, n), k, a.len(), lda)?;
    require_tau_buffer(tau.len(), k)?;
    validate_matrix(m, n, c.len(), ldc)?;
    require_info_buffer(dev_info.len())?;
    let lwork = cunmqr_buffer_size(context, side, operation, m, n, k, lda, ldc)?;
    require_workspace(workspace.len(), lwork)?;
    // SAFETY: `cusolverDnCunmqr` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cusolverDnCunmqr(
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

/// # Safety
///
/// Follow the [raw solver contracts](crate::dense#raw-solver-contracts).
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, a device buffer belongs to another CUDA context, matrix dimensions, leading dimensions, or storage capacities are invalid, checked size arithmetic or native integer conversion fails, the status output capacity is insufficient, a pivot, reflector, or numerical output buffer is too small, device scratch capacity is smaller than the queried requirement, or cuSOLVER rejects the configuration or returns an error while querying workspace or submitting execution.
pub unsafe fn zunmqr(
    context: &Context,
    side: SideMode,
    operation: Operation,
    m: usize,
    n: usize,
    k: usize,
    a: DeviceView<'_, Complex64>,
    lda: usize,
    tau: DeviceView<'_, Complex64>,
    mut c: DeviceViewMut<'_, Complex64>,
    ldc: usize,
    mut workspace: DeviceOutputView<'_, Complex64>,
    mut dev_info: DeviceOutputView<'_, i32>,
) -> Result<()> {
    context.bind()?;
    context.validate_region(&a)?;
    context.validate_region(&tau)?;
    context.validate_region(&c)?;
    context.validate_region(&workspace)?;
    context.validate_region(&dev_info)?;
    validate_matrix(qr_rows(side, m, n), k, a.len(), lda)?;
    require_tau_buffer(tau.len(), k)?;
    validate_matrix(m, n, c.len(), ldc)?;
    require_info_buffer(dev_info.len())?;
    let lwork = zunmqr_buffer_size(context, side, operation, m, n, k, lda, ldc)?;
    require_workspace(workspace.len(), lwork)?;
    // SAFETY: `cusolverDnZunmqr` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cusolverDnZunmqr(
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

/// Queries device and host workspace capacities in bytes for QR factorization.
///
/// Uses shape and layout metadata without borrowing matrix storage.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn xgeqrf_buffer_size<TA: DataTypeLike, TTau: DataTypeLike>(
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
    // SAFETY: `cusolverDnXgeqrf_bufferSize` receives live handles owned by the wrappers and valid output locations.
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
/// | **`data_type_a`** | **`compute_type`** | **Meaning** |
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
/// # Safety
///
/// Follow the [raw solver contracts](crate::dense#raw-solver-contracts).
pub unsafe fn xgeqrf<TA: DataTypeLike, TTau: DataTypeLike>(
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
    context.validate_region(&a.data)?;
    context.validate_region(&tau.data)?;
    context.validate_region(&dev_info)?;
    context.validate_region(&workspace.device)?;
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
        xgeqrf_buffer_size::<TA, TTau>(context, params, m, n, a.layout(), compute_type)?;
    require_workspace_bytes(workspace.device.len(), workspace_sizes.device_bytes)?;
    require_host_workspace(workspace.host.len(), workspace_sizes.host_bytes)?;
    // SAFETY: `cusolverDnXgeqrf` receives live handles owned by the wrappers and valid output locations.
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
            workspace_sizes.device_bytes as _,
            workspace.host.as_mut_ptr().cast(),
            workspace_sizes.host_bytes as _,
            dev_info.as_mut_ptr().cast(),
        ))?;
    }
    Ok(())
}
