#![expect(
    clippy::many_single_char_names,
    clippy::similar_names,
    reason = "BLAS and factorization operand names follow the standard mathematical API"
)]
use std::ptr;

use singe_cuda::{
    types::{Complex32, Complex64},
    view::{DeviceOutputView, DeviceView, DeviceViewMut},
};

use crate::{
    context::Context,
    dense::validation::{
        require_info_buffer, require_workspace, validate_bidiagonal_buffers,
        validate_bidiagonal_dims, validate_orgbr_inputs, validate_orgbr_shape,
    },
    error::Result,
    layout::MatrixLayout,
    sys, try_ffi,
    types::SideMode,
    utility::{to_i32, to_usize},
};

/// Queries device workspace capacity in elements for `f32` bidiagonal reduction.
///
/// Uses shape and layout metadata without borrowing matrix storage.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, `m < n`, a dimension or returned workspace size cannot be represented by the required integer type, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn sgebrd_buffer_size(context: &Context, m: usize, n: usize) -> Result<usize> {
    context.bind()?;
    validate_bidiagonal_dims(m, n)?;
    let mut lwork = 0;
    // SAFETY: `cusolverDnSgebrd_bufferSize` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnSgebrd_bufferSize(
            context.as_raw(),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            &raw mut lwork,
        ))?;
    }
    to_usize(lwork, "lwork")
}

/// Queries device workspace capacity in elements for `f64` bidiagonal reduction.
///
/// Uses shape and layout metadata without borrowing matrix storage.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, `m < n`, a dimension or returned workspace size cannot be represented by the required integer type, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn dgebrd_buffer_size(context: &Context, m: usize, n: usize) -> Result<usize> {
    context.bind()?;
    validate_bidiagonal_dims(m, n)?;
    let mut lwork = 0;
    // SAFETY: `cusolverDnDgebrd_bufferSize` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnDgebrd_bufferSize(
            context.as_raw(),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            &raw mut lwork,
        ))?;
    }
    to_usize(lwork, "lwork")
}

/// Queries device workspace capacity in elements for complex single-precision bidiagonal reduction.
///
/// Uses shape and layout metadata without borrowing matrix storage.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, `m < n`, a dimension or returned workspace size cannot be represented by the required integer type, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn cgebrd_buffer_size(context: &Context, m: usize, n: usize) -> Result<usize> {
    context.bind()?;
    validate_bidiagonal_dims(m, n)?;
    let mut lwork = 0;
    // SAFETY: `cusolverDnCgebrd_bufferSize` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnCgebrd_bufferSize(
            context.as_raw(),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            &raw mut lwork,
        ))?;
    }
    to_usize(lwork, "lwork")
}

/// Queries device workspace capacity in elements for complex double-precision bidiagonal reduction.
///
/// Uses shape and layout metadata without borrowing matrix storage.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, `m < n`, a dimension or returned workspace size cannot be represented by the required integer type, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn zgebrd_buffer_size(context: &Context, m: usize, n: usize) -> Result<usize> {
    context.bind()?;
    validate_bidiagonal_dims(m, n)?;
    let mut lwork = 0;
    // SAFETY: `cusolverDnZgebrd_bufferSize` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnZgebrd_bufferSize(
            context.as_raw(),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
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
/// Reduces a general $m \times n$ matrix `A` to a real upper or lower
/// bidiagonal form `B` by an orthogonal transformation:
/// $Q^{H}\cdot A\cdot P = B$.
///
/// If `m >= n`, `B` is upper bidiagonal; if `m < n`, `B` is lower
/// bidiagonal.
///
/// The matrix `Q` and `P` are overwritten into matrix `A` in the following sense:
///
/// - If `m >= n`, the diagonal and first superdiagonal are overwritten with
///   the upper bidiagonal matrix `B`. Elements below the diagonal, together
///   with `tauq`, represent `Q`; elements above the first superdiagonal,
///   together with `taup`, represent `P`.
/// - If `m < n`, the diagonal and first subdiagonal are overwritten with the
///   lower bidiagonal matrix `B`. Elements below the first subdiagonal,
///   together with `tauq`, represent `Q`; elements above the diagonal,
///   together with `taup`, represent `P`.
///
/// Provide workspace through `workspace`.
/// Use the corresponding `*_buffer_size` helper to query the required workspace length.
///
/// If the reported `dev_info` value is `-i`, the `i`th parameter is invalid.
///
/// `gebrd` only supports `m >= n`.
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions or leading dimension are invalid, if the current GPU
/// architecture is unsupported, or if cuSOLVER reports an internal failure.
/// # Safety
///
/// Follow the [raw solver contracts](crate::dense#raw-solver-contracts).
pub unsafe fn sgebrd(
    context: &Context,
    m: usize,
    n: usize,
    mut a: DeviceViewMut<'_, f32>,
    lda: usize,
    mut d: DeviceOutputView<'_, f32>,
    mut e: DeviceOutputView<'_, f32>,
    mut tauq: DeviceOutputView<'_, f32>,
    mut taup: DeviceOutputView<'_, f32>,
    mut workspace: DeviceOutputView<'_, f32>,
    mut dev_info: DeviceOutputView<'_, i32>,
) -> Result<()> {
    context.bind()?;
    context.validate_region(&a)?;
    context.validate_region(&d)?;
    context.validate_region(&e)?;
    context.validate_region(&tauq)?;
    context.validate_region(&taup)?;
    context.validate_region(&workspace)?;
    context.validate_region(&dev_info)?;
    validate_bidiagonal_buffers(m, n, a.len(), lda, d.len(), e.len(), tauq.len(), taup.len())?;
    require_info_buffer(dev_info.len())?;
    let lwork = sgebrd_buffer_size(context, m, n)?;
    require_workspace(workspace.len(), lwork)?;
    // SAFETY: `cusolverDnSgebrd` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnSgebrd(
            context.as_raw(),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            a.as_mut_ptr().cast(),
            to_i32(lda, "lda")?,
            d.as_mut_ptr().cast(),
            e.as_mut_ptr().cast(),
            tauq.as_mut_ptr().cast(),
            taup.as_mut_ptr().cast(),
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
/// Reduces a general $m \times n$ matrix `A` to a real upper or lower
/// bidiagonal form `B` by an orthogonal transformation:
/// $Q^{H}\cdot A\cdot P = B$.
///
/// If `m >= n`, `B` is upper bidiagonal; if `m < n`, `B` is lower
/// bidiagonal.
///
/// The matrix `Q` and `P` are overwritten into matrix `A` in the following sense:
///
/// - If `m >= n`, the diagonal and first superdiagonal are overwritten with
///   the upper bidiagonal matrix `B`. Elements below the diagonal, together
///   with `tauq`, represent `Q`; elements above the first superdiagonal,
///   together with `taup`, represent `P`.
/// - If `m < n`, the diagonal and first subdiagonal are overwritten with the
///   lower bidiagonal matrix `B`. Elements below the first subdiagonal,
///   together with `tauq`, represent `Q`; elements above the diagonal,
///   together with `taup`, represent `P`.
///
/// Provide workspace through `workspace`.
/// Use the corresponding `*_buffer_size` helper to query the required workspace length.
///
/// If the reported `dev_info` value is `-i`, the `i`th parameter is invalid.
///
/// `gebrd` only supports `m >= n`.
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions or leading dimension are invalid, if the current GPU
/// architecture is unsupported, or if cuSOLVER reports an internal failure.
/// # Safety
///
/// Follow the [raw solver contracts](crate::dense#raw-solver-contracts).
pub unsafe fn dgebrd(
    context: &Context,
    m: usize,
    n: usize,
    mut a: DeviceViewMut<'_, f64>,
    lda: usize,
    mut d: DeviceOutputView<'_, f64>,
    mut e: DeviceOutputView<'_, f64>,
    mut tauq: DeviceOutputView<'_, f64>,
    mut taup: DeviceOutputView<'_, f64>,
    mut workspace: DeviceOutputView<'_, f64>,
    mut dev_info: DeviceOutputView<'_, i32>,
) -> Result<()> {
    context.bind()?;
    context.validate_region(&a)?;
    context.validate_region(&d)?;
    context.validate_region(&e)?;
    context.validate_region(&tauq)?;
    context.validate_region(&taup)?;
    context.validate_region(&workspace)?;
    context.validate_region(&dev_info)?;
    validate_bidiagonal_buffers(m, n, a.len(), lda, d.len(), e.len(), tauq.len(), taup.len())?;
    require_info_buffer(dev_info.len())?;
    let lwork = dgebrd_buffer_size(context, m, n)?;
    require_workspace(workspace.len(), lwork)?;
    // SAFETY: `cusolverDnDgebrd` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnDgebrd(
            context.as_raw(),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            a.as_mut_ptr().cast(),
            to_i32(lda, "lda")?,
            d.as_mut_ptr().cast(),
            e.as_mut_ptr().cast(),
            tauq.as_mut_ptr().cast(),
            taup.as_mut_ptr().cast(),
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
/// Reduces a general $m \times n$ matrix `A` to a real upper or lower
/// bidiagonal form `B` by an orthogonal transformation:
/// $Q^{H}\cdot A\cdot P = B$.
///
/// If `m >= n`, `B` is upper bidiagonal; if `m < n`, `B` is lower
/// bidiagonal.
///
/// The matrix `Q` and `P` are overwritten into matrix `A` in the following sense:
///
/// - If `m >= n`, the diagonal and first superdiagonal are overwritten with
///   the upper bidiagonal matrix `B`. Elements below the diagonal, together
///   with `tauq`, represent `Q`; elements above the first superdiagonal,
///   together with `taup`, represent `P`.
/// - If `m < n`, the diagonal and first subdiagonal are overwritten with the
///   lower bidiagonal matrix `B`. Elements below the first subdiagonal,
///   together with `tauq`, represent `Q`; elements above the diagonal,
///   together with `taup`, represent `P`.
///
/// Provide workspace through `workspace`.
/// Use the corresponding `*_buffer_size` helper to query the required workspace length.
///
/// If the reported `dev_info` value is `-i`, the `i`th parameter is invalid.
///
/// `gebrd` only supports `m >= n`.
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions or leading dimension are invalid, if the current GPU
/// architecture is unsupported, or if cuSOLVER reports an internal failure.
/// # Safety
///
/// Follow the [raw solver contracts](crate::dense#raw-solver-contracts).
pub unsafe fn cgebrd(
    context: &Context,
    m: usize,
    n: usize,
    mut a: DeviceViewMut<'_, Complex32>,
    lda: usize,
    mut d: DeviceOutputView<'_, f32>,
    mut e: DeviceOutputView<'_, f32>,
    mut tauq: DeviceOutputView<'_, Complex32>,
    mut taup: DeviceOutputView<'_, Complex32>,
    mut workspace: DeviceOutputView<'_, Complex32>,
    mut dev_info: DeviceOutputView<'_, i32>,
) -> Result<()> {
    context.bind()?;
    context.validate_region(&a)?;
    context.validate_region(&d)?;
    context.validate_region(&e)?;
    context.validate_region(&tauq)?;
    context.validate_region(&taup)?;
    context.validate_region(&workspace)?;
    context.validate_region(&dev_info)?;
    validate_bidiagonal_buffers(m, n, a.len(), lda, d.len(), e.len(), tauq.len(), taup.len())?;
    require_info_buffer(dev_info.len())?;
    let lwork = cgebrd_buffer_size(context, m, n)?;
    require_workspace(workspace.len(), lwork)?;
    // SAFETY: `cusolverDnCgebrd` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnCgebrd(
            context.as_raw(),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            a.as_mut_ptr().cast(),
            to_i32(lda, "lda")?,
            d.as_mut_ptr().cast(),
            e.as_mut_ptr().cast(),
            tauq.as_mut_ptr().cast(),
            taup.as_mut_ptr().cast(),
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
/// Reduces a general $m \times n$ matrix `A` to a real upper or lower
/// bidiagonal form `B` by an orthogonal transformation:
/// $Q^{H}\cdot A\cdot P = B$.
///
/// If `m >= n`, `B` is upper bidiagonal; if `m < n`, `B` is lower
/// bidiagonal.
///
/// The matrix `Q` and `P` are overwritten into matrix `A` in the following sense:
///
/// - If `m >= n`, the diagonal and first superdiagonal are overwritten with
///   the upper bidiagonal matrix `B`. Elements below the diagonal, together
///   with `tauq`, represent `Q`; elements above the first superdiagonal,
///   together with `taup`, represent `P`.
/// - If `m < n`, the diagonal and first subdiagonal are overwritten with the
///   lower bidiagonal matrix `B`. Elements below the first subdiagonal,
///   together with `tauq`, represent `Q`; elements above the diagonal,
///   together with `taup`, represent `P`.
///
/// Provide workspace through `workspace`.
/// Use the corresponding `*_buffer_size` helper to query the required workspace length.
///
/// If the reported `dev_info` value is `-i`, the `i`th parameter is invalid.
///
/// `gebrd` only supports `m >= n`.
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions or leading dimension are invalid, if the current GPU
/// architecture is unsupported, or if cuSOLVER reports an internal failure.
/// # Safety
///
/// Follow the [raw solver contracts](crate::dense#raw-solver-contracts).
pub unsafe fn zgebrd(
    context: &Context,
    m: usize,
    n: usize,
    mut a: DeviceViewMut<'_, Complex64>,
    lda: usize,
    mut d: DeviceOutputView<'_, f64>,
    mut e: DeviceOutputView<'_, f64>,
    mut tauq: DeviceOutputView<'_, Complex64>,
    mut taup: DeviceOutputView<'_, Complex64>,
    mut workspace: DeviceOutputView<'_, Complex64>,
    mut dev_info: DeviceOutputView<'_, i32>,
) -> Result<()> {
    context.bind()?;
    context.validate_region(&a)?;
    context.validate_region(&d)?;
    context.validate_region(&e)?;
    context.validate_region(&tauq)?;
    context.validate_region(&taup)?;
    context.validate_region(&workspace)?;
    context.validate_region(&dev_info)?;
    validate_bidiagonal_buffers(m, n, a.len(), lda, d.len(), e.len(), tauq.len(), taup.len())?;
    require_info_buffer(dev_info.len())?;
    let lwork = zgebrd_buffer_size(context, m, n)?;
    require_workspace(workspace.len(), lwork)?;
    // SAFETY: `cusolverDnZgebrd` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnZgebrd(
            context.as_raw(),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            a.as_mut_ptr().cast(),
            to_i32(lda, "lda")?,
            d.as_mut_ptr().cast(),
            e.as_mut_ptr().cast(),
            tauq.as_mut_ptr().cast(),
            taup.as_mut_ptr().cast(),
            workspace.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast(),
        ))?;
    }
    Ok(())
}

/// Queries device workspace capacity in elements for `f32` generation of orthogonal bidiagonalization factors.
///
/// Uses shape and layout metadata without borrowing matrix storage.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn sorgbr_buffer_size(
    context: &Context,
    side: SideMode,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
) -> Result<usize> {
    context.bind()?;
    validate_orgbr_shape(side, m, n, k)?;
    MatrixLayout::new(lda).validate(m, n)?;
    let mut lwork = 0;
    // SAFETY: `cusolverDnSorgbr_bufferSize` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnSorgbr_bufferSize(
            context.as_raw(),
            side.into(),
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

/// Queries device workspace capacity in elements for `f64` generation of orthogonal bidiagonalization factors.
///
/// Uses shape and layout metadata without borrowing matrix storage.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn dorgbr_buffer_size(
    context: &Context,
    side: SideMode,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
) -> Result<usize> {
    context.bind()?;
    validate_orgbr_shape(side, m, n, k)?;
    MatrixLayout::new(lda).validate(m, n)?;
    let mut lwork = 0;
    // SAFETY: `cusolverDnDorgbr_bufferSize` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnDorgbr_bufferSize(
            context.as_raw(),
            side.into(),
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

/// Queries device workspace capacity in elements for complex single-precision generation of unitary bidiagonalization factors.
///
/// Uses shape and layout metadata without borrowing matrix storage.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn cungbr_buffer_size(
    context: &Context,
    side: SideMode,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
) -> Result<usize> {
    context.bind()?;
    validate_orgbr_shape(side, m, n, k)?;
    MatrixLayout::new(lda).validate(m, n)?;
    let mut lwork = 0;
    // SAFETY: `cusolverDnCungbr_bufferSize` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnCungbr_bufferSize(
            context.as_raw(),
            side.into(),
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

/// Queries device workspace capacity in elements for complex double-precision generation of unitary bidiagonalization factors.
///
/// Uses shape and layout metadata without borrowing matrix storage.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn zungbr_buffer_size(
    context: &Context,
    side: SideMode,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
) -> Result<usize> {
    context.bind()?;
    validate_orgbr_shape(side, m, n, k)?;
    MatrixLayout::new(lda).validate(m, n)?;
    let mut lwork = 0;
    // SAFETY: `cusolverDnZungbr_bufferSize` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnZungbr_bufferSize(
            context.as_raw(),
            side.into(),
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
/// Generates one of the unitary matrices `Q` or $P^{H}$ determined by `gebrd`
/// when reducing matrix `A` to bidiagonal form:
/// $Q^{H}\cdot A\cdot P = B$.
///
/// `Q` and $P^{H}$ are defined as products of elementary reflectors `H(i)`
/// or `G(i)`, respectively.
///
/// Provide workspace through `workspace`.
/// Use the corresponding `*_buffer_size` helper to query the required workspace length.
/// The workspace size in bytes is `size_of::<T>() * lwork`.
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
pub unsafe fn sorgbr(
    context: &Context,
    side: SideMode,
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
    validate_orgbr_inputs(side, m, n, k, a.len(), lda, tau.len())?;
    require_info_buffer(dev_info.len())?;
    let lwork = sorgbr_buffer_size(context, side, m, n, k, lda)?;
    require_workspace(workspace.len(), lwork)?;
    // SAFETY: `cusolverDnSorgbr` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cusolverDnSorgbr(
            context.as_raw(),
            side.into(),
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
/// Generates one of the unitary matrices `Q` or $P^{H}$ determined by `gebrd`
/// when reducing matrix `A` to bidiagonal form:
/// $Q^{H}\cdot A\cdot P = B$.
///
/// `Q` and $P^{H}$ are defined as products of elementary reflectors `H(i)`
/// or `G(i)`, respectively.
///
/// Provide workspace through `workspace`.
/// Use the corresponding `*_buffer_size` helper to query the required workspace length.
/// The workspace size in bytes is `size_of::<T>() * lwork`.
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
pub unsafe fn dorgbr(
    context: &Context,
    side: SideMode,
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
    validate_orgbr_inputs(side, m, n, k, a.len(), lda, tau.len())?;
    require_info_buffer(dev_info.len())?;
    let lwork = dorgbr_buffer_size(context, side, m, n, k, lda)?;
    require_workspace(workspace.len(), lwork)?;
    // SAFETY: `cusolverDnDorgbr` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cusolverDnDorgbr(
            context.as_raw(),
            side.into(),
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
pub unsafe fn cungbr(
    context: &Context,
    side: SideMode,
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
    validate_orgbr_inputs(side, m, n, k, a.len(), lda, tau.len())?;
    require_info_buffer(dev_info.len())?;
    let lwork = cungbr_buffer_size(context, side, m, n, k, lda)?;
    require_workspace(workspace.len(), lwork)?;
    // SAFETY: `cusolverDnCungbr` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cusolverDnCungbr(
            context.as_raw(),
            side.into(),
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
pub unsafe fn zungbr(
    context: &Context,
    side: SideMode,
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
    validate_orgbr_inputs(side, m, n, k, a.len(), lda, tau.len())?;
    require_info_buffer(dev_info.len())?;
    let lwork = zungbr_buffer_size(context, side, m, n, k, lda)?;
    require_workspace(workspace.len(), lwork)?;
    // SAFETY: `cusolverDnZungbr` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cusolverDnZungbr(
            context.as_raw(),
            side.into(),
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
