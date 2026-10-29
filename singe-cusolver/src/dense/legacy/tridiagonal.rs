use std::ptr;

use singe_cuda::{
    types::{Complex32, Complex64},
    view::{DeviceOutputView, DeviceView, DeviceViewMut},
};

use crate::{
    context::Context,
    dense::validation::{
        require_info_buffer, require_workspace, tridiagonal_order, validate_orgtr_inputs,
        validate_ormtr_inputs, validate_sytrd_inputs,
    },
    error::Result,
    layout::MatrixLayout,
    sys, try_ffi,
    types::{FillMode, Operation, SideMode},
    utility::{to_i32, to_usize},
};

/// Queries device workspace capacity in elements for `f32` symmetric tridiagonal reduction.
///
/// Uses shape and layout metadata without borrowing matrix storage.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn ssytrd_buffer_size(
    context: &Context,
    fill_mode: FillMode,
    n: usize,
    lda: usize,
) -> Result<usize> {
    context.bind()?;
    MatrixLayout::new(lda).validate(n, n)?;
    let mut lwork = 0;
    // SAFETY: `cusolverDnSsytrd_bufferSize` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnSsytrd_bufferSize(
            context.as_raw(),
            fill_mode.into(),
            to_i32(n, "n")?,
            ptr::null_mut::<u8>().cast(),
            to_i32(lda, "lda")?,
            ptr::null_mut::<u8>().cast(),
            ptr::null_mut::<u8>().cast(),
            ptr::null_mut::<u8>().cast(),
            &raw mut lwork,
        ))?;
    }
    to_usize(lwork, "lwork")
}

/// Queries device workspace capacity in elements for `f64` symmetric tridiagonal reduction.
///
/// Uses shape and layout metadata without borrowing matrix storage.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn dsytrd_buffer_size(
    context: &Context,
    fill_mode: FillMode,
    n: usize,
    lda: usize,
) -> Result<usize> {
    context.bind()?;
    MatrixLayout::new(lda).validate(n, n)?;
    let mut lwork = 0;
    // SAFETY: `cusolverDnDsytrd_bufferSize` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnDsytrd_bufferSize(
            context.as_raw(),
            fill_mode.into(),
            to_i32(n, "n")?,
            ptr::null_mut::<u8>().cast(),
            to_i32(lda, "lda")?,
            ptr::null_mut::<u8>().cast(),
            ptr::null_mut::<u8>().cast(),
            ptr::null_mut::<u8>().cast(),
            &raw mut lwork,
        ))?;
    }
    to_usize(lwork, "lwork")
}

/// Queries device workspace capacity in elements for complex single-precision Hermitian tridiagonal reduction.
///
/// Uses shape and layout metadata without borrowing matrix storage.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn chetrd_buffer_size(
    context: &Context,
    fill_mode: FillMode,
    n: usize,
    lda: usize,
) -> Result<usize> {
    context.bind()?;
    MatrixLayout::new(lda).validate(n, n)?;
    let mut lwork = 0;
    // SAFETY: `cusolverDnChetrd_bufferSize` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnChetrd_bufferSize(
            context.as_raw(),
            fill_mode.into(),
            to_i32(n, "n")?,
            ptr::null_mut::<u8>().cast(),
            to_i32(lda, "lda")?,
            ptr::null_mut::<u8>().cast(),
            ptr::null_mut::<u8>().cast(),
            ptr::null_mut::<u8>().cast(),
            &raw mut lwork,
        ))?;
    }
    to_usize(lwork, "lwork")
}

/// Queries device workspace capacity in elements for complex double-precision Hermitian tridiagonal reduction.
///
/// Uses shape and layout metadata without borrowing matrix storage.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn zhetrd_buffer_size(
    context: &Context,
    fill_mode: FillMode,
    n: usize,
    lda: usize,
) -> Result<usize> {
    context.bind()?;
    MatrixLayout::new(lda).validate(n, n)?;
    let mut lwork = 0;
    // SAFETY: `cusolverDnZhetrd_bufferSize` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnZhetrd_bufferSize(
            context.as_raw(),
            fill_mode.into(),
            to_i32(n, "n")?,
            ptr::null_mut::<u8>().cast(),
            to_i32(lda, "lda")?,
            ptr::null_mut::<u8>().cast(),
            ptr::null_mut::<u8>().cast(),
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
/// Reduces a general symmetric (Hermitian) $n \times n$ matrix `A` to the
/// real symmetric tridiagonal form `T` by an orthogonal transformation:
/// $Q^{H}\cdot A\cdot Q = T$.
///
/// On output, `A` contains `T` and Householder reflection vectors.
/// If `fill_mode` is [`FillMode::Upper`], the diagonal and first
/// superdiagonal of `A` are overwritten by `T`; elements above the first
/// superdiagonal, together with `tau`, represent `Q`.
/// If `fill_mode` is [`FillMode::Lower`], the diagonal and first subdiagonal
/// of `A` are overwritten by `T`; elements below the first subdiagonal,
/// together with `tau`, represent `Q`.
///
/// Provide workspace through `workspace`.
/// Use the corresponding `*_buffer_size` helper to query the required workspace length.
/// The workspace size in bytes is `size_of::<T>() * lwork`.
///
/// If the reported `dev_info` value is `-i`, the `i`th parameter is invalid.
/// The problem size `n` is limited by `n * lda <= INT32_MAX` primarily due to the current implementation constraints.
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions, leading dimension, or fill mode are invalid, if the
/// current GPU architecture is unsupported, or if cuSOLVER reports an
/// internal failure.
/// # Safety
///
/// Follow the [raw solver contracts](crate::dense#raw-solver-contracts).
pub unsafe fn ssytrd(
    context: &Context,
    fill_mode: FillMode,
    n: usize,
    mut a: DeviceViewMut<'_, f32>,
    lda: usize,
    mut d: DeviceOutputView<'_, f32>,
    mut e: DeviceOutputView<'_, f32>,
    mut tau: DeviceOutputView<'_, f32>,
    mut workspace: DeviceOutputView<'_, f32>,
    mut dev_info: DeviceOutputView<'_, i32>,
) -> Result<()> {
    context.bind()?;
    context.validate_region(&a)?;
    context.validate_region(&d)?;
    context.validate_region(&e)?;
    context.validate_region(&tau)?;
    context.validate_region(&workspace)?;
    context.validate_region(&dev_info)?;
    validate_sytrd_inputs(n, a.len(), lda, d.len(), e.len(), tau.len())?;
    require_info_buffer(dev_info.len())?;
    let lwork = ssytrd_buffer_size(context, fill_mode, n, lda)?;
    require_workspace(workspace.len(), lwork)?;
    // SAFETY: `cusolverDnSsytrd` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnSsytrd(
            context.as_raw(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_mut_ptr().cast(),
            to_i32(lda, "lda")?,
            d.as_mut_ptr().cast(),
            e.as_mut_ptr().cast(),
            tau.as_mut_ptr().cast(),
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
/// Reduces a general symmetric (Hermitian) $n \times n$ matrix `A` to the
/// real symmetric tridiagonal form `T` by an orthogonal transformation:
/// $Q^{H}\cdot A\cdot Q = T$.
///
/// On output, `A` contains `T` and Householder reflection vectors.
/// If `fill_mode` is [`FillMode::Upper`], the diagonal and first
/// superdiagonal of `A` are overwritten by `T`; elements above the first
/// superdiagonal, together with `tau`, represent `Q`.
/// If `fill_mode` is [`FillMode::Lower`], the diagonal and first subdiagonal
/// of `A` are overwritten by `T`; elements below the first subdiagonal,
/// together with `tau`, represent `Q`.
///
/// Provide workspace through `workspace`.
/// Use the corresponding `*_buffer_size` helper to query the required workspace length.
/// The workspace size in bytes is `size_of::<T>() * lwork`.
///
/// If the reported `dev_info` value is `-i`, the `i`th parameter is invalid.
/// The problem size `n` is limited by `n * lda <= INT32_MAX` primarily due to the current implementation constraints.
///
/// # Errors
///
/// Returns an error if cuSOLVER has not been initialized, if the
/// matrix dimensions, leading dimension, or fill mode are invalid, if the
/// current GPU architecture is unsupported, or if cuSOLVER reports an
/// internal failure.
/// # Safety
///
/// Follow the [raw solver contracts](crate::dense#raw-solver-contracts).
pub unsafe fn dsytrd(
    context: &Context,
    fill_mode: FillMode,
    n: usize,
    mut a: DeviceViewMut<'_, f64>,
    lda: usize,
    mut d: DeviceOutputView<'_, f64>,
    mut e: DeviceOutputView<'_, f64>,
    mut tau: DeviceOutputView<'_, f64>,
    mut workspace: DeviceOutputView<'_, f64>,
    mut dev_info: DeviceOutputView<'_, i32>,
) -> Result<()> {
    context.bind()?;
    context.validate_region(&a)?;
    context.validate_region(&d)?;
    context.validate_region(&e)?;
    context.validate_region(&tau)?;
    context.validate_region(&workspace)?;
    context.validate_region(&dev_info)?;
    validate_sytrd_inputs(n, a.len(), lda, d.len(), e.len(), tau.len())?;
    require_info_buffer(dev_info.len())?;
    let lwork = dsytrd_buffer_size(context, fill_mode, n, lda)?;
    require_workspace(workspace.len(), lwork)?;
    // SAFETY: `cusolverDnDsytrd` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnDsytrd(
            context.as_raw(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_mut_ptr().cast(),
            to_i32(lda, "lda")?,
            d.as_mut_ptr().cast(),
            e.as_mut_ptr().cast(),
            tau.as_mut_ptr().cast(),
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
pub unsafe fn chetrd(
    context: &Context,
    fill_mode: FillMode,
    n: usize,
    mut a: DeviceViewMut<'_, Complex32>,
    lda: usize,
    mut d: DeviceOutputView<'_, f32>,
    mut e: DeviceOutputView<'_, f32>,
    mut tau: DeviceOutputView<'_, Complex32>,
    mut workspace: DeviceOutputView<'_, Complex32>,
    mut dev_info: DeviceOutputView<'_, i32>,
) -> Result<()> {
    context.bind()?;
    context.validate_region(&a)?;
    context.validate_region(&d)?;
    context.validate_region(&e)?;
    context.validate_region(&tau)?;
    context.validate_region(&workspace)?;
    context.validate_region(&dev_info)?;
    validate_sytrd_inputs(n, a.len(), lda, d.len(), e.len(), tau.len())?;
    require_info_buffer(dev_info.len())?;
    let lwork = chetrd_buffer_size(context, fill_mode, n, lda)?;
    require_workspace(workspace.len(), lwork)?;
    // SAFETY: `cusolverDnChetrd` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnChetrd(
            context.as_raw(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_mut_ptr().cast(),
            to_i32(lda, "lda")?,
            d.as_mut_ptr().cast(),
            e.as_mut_ptr().cast(),
            tau.as_mut_ptr().cast(),
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
pub unsafe fn zhetrd(
    context: &Context,
    fill_mode: FillMode,
    n: usize,
    mut a: DeviceViewMut<'_, Complex64>,
    lda: usize,
    mut d: DeviceOutputView<'_, f64>,
    mut e: DeviceOutputView<'_, f64>,
    mut tau: DeviceOutputView<'_, Complex64>,
    mut workspace: DeviceOutputView<'_, Complex64>,
    mut dev_info: DeviceOutputView<'_, i32>,
) -> Result<()> {
    context.bind()?;
    context.validate_region(&a)?;
    context.validate_region(&d)?;
    context.validate_region(&e)?;
    context.validate_region(&tau)?;
    context.validate_region(&workspace)?;
    context.validate_region(&dev_info)?;
    validate_sytrd_inputs(n, a.len(), lda, d.len(), e.len(), tau.len())?;
    require_info_buffer(dev_info.len())?;
    let lwork = zhetrd_buffer_size(context, fill_mode, n, lda)?;
    require_workspace(workspace.len(), lwork)?;
    // SAFETY: `cusolverDnZhetrd` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnZhetrd(
            context.as_raw(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_mut_ptr().cast(),
            to_i32(lda, "lda")?,
            d.as_mut_ptr().cast(),
            e.as_mut_ptr().cast(),
            tau.as_mut_ptr().cast(),
            workspace.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast(),
        ))?;
    }
    Ok(())
}

/// Queries device workspace capacity in elements for `f32` generation of orthogonal tridiagonalization factors.
///
/// Uses shape and layout metadata without borrowing matrix storage.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn sorgtr_buffer_size(
    context: &Context,
    fill_mode: FillMode,
    n: usize,
    lda: usize,
) -> Result<usize> {
    context.bind()?;
    MatrixLayout::new(lda).validate(n, n)?;
    let mut lwork = 0;
    // SAFETY: `cusolverDnSorgtr_bufferSize` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnSorgtr_bufferSize(
            context.as_raw(),
            fill_mode.into(),
            to_i32(n, "n")?,
            ptr::null_mut::<u8>().cast(),
            to_i32(lda, "lda")?,
            ptr::null_mut::<u8>().cast(),
            &raw mut lwork,
        ))?;
    }
    to_usize(lwork, "lwork")
}

/// Queries device workspace capacity in elements for `f64` generation of orthogonal tridiagonalization factors.
///
/// Uses shape and layout metadata without borrowing matrix storage.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn dorgtr_buffer_size(
    context: &Context,
    fill_mode: FillMode,
    n: usize,
    lda: usize,
) -> Result<usize> {
    context.bind()?;
    MatrixLayout::new(lda).validate(n, n)?;
    let mut lwork = 0;
    // SAFETY: `cusolverDnDorgtr_bufferSize` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnDorgtr_bufferSize(
            context.as_raw(),
            fill_mode.into(),
            to_i32(n, "n")?,
            ptr::null_mut::<u8>().cast(),
            to_i32(lda, "lda")?,
            ptr::null_mut::<u8>().cast(),
            &raw mut lwork,
        ))?;
    }
    to_usize(lwork, "lwork")
}

/// Queries device workspace capacity in elements for complex single-precision generation of unitary tridiagonalization factors.
///
/// Uses shape and layout metadata without borrowing matrix storage.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn cungtr_buffer_size(
    context: &Context,
    fill_mode: FillMode,
    n: usize,
    lda: usize,
) -> Result<usize> {
    context.bind()?;
    MatrixLayout::new(lda).validate(n, n)?;
    let mut lwork = 0;
    // SAFETY: `cusolverDnCungtr_bufferSize` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnCungtr_bufferSize(
            context.as_raw(),
            fill_mode.into(),
            to_i32(n, "n")?,
            ptr::null_mut::<u8>().cast(),
            to_i32(lda, "lda")?,
            ptr::null_mut::<u8>().cast(),
            &raw mut lwork,
        ))?;
    }
    to_usize(lwork, "lwork")
}

/// Queries device workspace capacity in elements for complex double-precision generation of unitary tridiagonalization factors.
///
/// Uses shape and layout metadata without borrowing matrix storage.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn zungtr_buffer_size(
    context: &Context,
    fill_mode: FillMode,
    n: usize,
    lda: usize,
) -> Result<usize> {
    context.bind()?;
    MatrixLayout::new(lda).validate(n, n)?;
    let mut lwork = 0;
    // SAFETY: `cusolverDnZungtr_bufferSize` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnZungtr_bufferSize(
            context.as_raw(),
            fill_mode.into(),
            to_i32(n, "n")?,
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
/// Generates the orthogonal matrix `Q` from the elementary reflectors returned
/// by `sytrd`.
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
pub unsafe fn sorgtr(
    context: &Context,
    fill_mode: FillMode,
    n: usize,
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
    validate_orgtr_inputs(n, a.len(), lda, tau.len())?;
    require_info_buffer(dev_info.len())?;
    let lwork = sorgtr_buffer_size(context, fill_mode, n, lda)?;
    require_workspace(workspace.len(), lwork)?;
    // SAFETY: `cusolverDnSorgtr` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cusolverDnSorgtr(
            context.as_raw(),
            fill_mode.into(),
            to_i32(n, "n")?,
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
/// Generates the orthogonal matrix `Q` from the elementary reflectors returned
/// by `sytrd`.
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
pub unsafe fn dorgtr(
    context: &Context,
    fill_mode: FillMode,
    n: usize,
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
    validate_orgtr_inputs(n, a.len(), lda, tau.len())?;
    require_info_buffer(dev_info.len())?;
    let lwork = dorgtr_buffer_size(context, fill_mode, n, lda)?;
    require_workspace(workspace.len(), lwork)?;
    // SAFETY: `cusolverDnDorgtr` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cusolverDnDorgtr(
            context.as_raw(),
            fill_mode.into(),
            to_i32(n, "n")?,
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
pub unsafe fn cungtr(
    context: &Context,
    fill_mode: FillMode,
    n: usize,
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
    validate_orgtr_inputs(n, a.len(), lda, tau.len())?;
    require_info_buffer(dev_info.len())?;
    let lwork = cungtr_buffer_size(context, fill_mode, n, lda)?;
    require_workspace(workspace.len(), lwork)?;
    // SAFETY: `cusolverDnCungtr` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cusolverDnCungtr(
            context.as_raw(),
            fill_mode.into(),
            to_i32(n, "n")?,
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
pub unsafe fn zungtr(
    context: &Context,
    fill_mode: FillMode,
    n: usize,
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
    validate_orgtr_inputs(n, a.len(), lda, tau.len())?;
    require_info_buffer(dev_info.len())?;
    let lwork = zungtr_buffer_size(context, fill_mode, n, lda)?;
    require_workspace(workspace.len(), lwork)?;
    // SAFETY: `cusolverDnZungtr` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cusolverDnZungtr(
            context.as_raw(),
            fill_mode.into(),
            to_i32(n, "n")?,
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

/// Queries device workspace capacity in elements for `f32` application of orthogonal tridiagonalization factors.
///
/// Uses shape and layout metadata without borrowing matrix storage.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn sormtr_buffer_size(
    context: &Context,
    side: SideMode,
    fill_mode: FillMode,
    operation: Operation,
    m: usize,
    n: usize,
    lda: usize,
    ldc: usize,
) -> Result<usize> {
    context.bind()?;
    MatrixLayout::new(lda)
        .validate(tridiagonal_order(side, m, n), tridiagonal_order(side, m, n))
        .and_then(|()| MatrixLayout::new(ldc).validate(m, n))?;
    let mut lwork = 0;
    // SAFETY: `cusolverDnSormtr_bufferSize` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnSormtr_bufferSize(
            context.as_raw(),
            side.into(),
            fill_mode.into(),
            operation.into(),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
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

/// Queries device workspace capacity in elements for `f64` application of orthogonal tridiagonalization factors.
///
/// Uses shape and layout metadata without borrowing matrix storage.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn dormtr_buffer_size(
    context: &Context,
    side: SideMode,
    fill_mode: FillMode,
    operation: Operation,
    m: usize,
    n: usize,
    lda: usize,
    ldc: usize,
) -> Result<usize> {
    context.bind()?;
    MatrixLayout::new(lda)
        .validate(tridiagonal_order(side, m, n), tridiagonal_order(side, m, n))
        .and_then(|()| MatrixLayout::new(ldc).validate(m, n))?;
    let mut lwork = 0;
    // SAFETY: `cusolverDnDormtr_bufferSize` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnDormtr_bufferSize(
            context.as_raw(),
            side.into(),
            fill_mode.into(),
            operation.into(),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
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

/// Queries device workspace capacity in elements for complex single-precision application of unitary tridiagonalization factors.
///
/// Uses shape and layout metadata without borrowing matrix storage.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn cunmtr_buffer_size(
    context: &Context,
    side: SideMode,
    fill_mode: FillMode,
    operation: Operation,
    m: usize,
    n: usize,
    lda: usize,
    ldc: usize,
) -> Result<usize> {
    context.bind()?;
    MatrixLayout::new(lda)
        .validate(tridiagonal_order(side, m, n), tridiagonal_order(side, m, n))
        .and_then(|()| MatrixLayout::new(ldc).validate(m, n))?;
    let mut lwork = 0;
    // SAFETY: `cusolverDnCunmtr_bufferSize` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnCunmtr_bufferSize(
            context.as_raw(),
            side.into(),
            fill_mode.into(),
            operation.into(),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
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

/// Queries device workspace capacity in elements for complex double-precision application of unitary tridiagonalization factors.
///
/// Uses shape and layout metadata without borrowing matrix storage.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, matrix dimensions or leading dimensions are invalid, a dimension or returned workspace size cannot be represented by the required integer type, checked layout-size arithmetic overflows, or cuSOLVER rejects the configuration or fails to report workspace requirements.
pub fn zunmtr_buffer_size(
    context: &Context,
    side: SideMode,
    fill_mode: FillMode,
    operation: Operation,
    m: usize,
    n: usize,
    lda: usize,
    ldc: usize,
) -> Result<usize> {
    context.bind()?;
    MatrixLayout::new(lda)
        .validate(tridiagonal_order(side, m, n), tridiagonal_order(side, m, n))
        .and_then(|()| MatrixLayout::new(ldc).validate(m, n))?;
    let mut lwork = 0;
    // SAFETY: `cusolverDnZunmtr_bufferSize` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cusolverDnZunmtr_bufferSize(
            context.as_raw(),
            side.into(),
            fill_mode.into(),
            operation.into(),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
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
/// returned by `sytrd`, to `C` and stores the result in `C`.
///
/// `side` selects whether `Q` is applied from the left or right, and
/// `operation` selects whether `Q` is transposed.
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
/// matrix dimensions or leading dimensions are invalid, if the current GPU
/// architecture is unsupported, or if cuSOLVER reports an internal failure.
/// # Safety
///
/// Follow the [raw solver contracts](crate::dense#raw-solver-contracts).
pub unsafe fn sormtr(
    context: &Context,
    side: SideMode,
    fill_mode: FillMode,
    operation: Operation,
    m: usize,
    n: usize,
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
    validate_ormtr_inputs(side, m, n, a.len(), lda, tau.len(), c.len(), ldc)?;
    require_info_buffer(dev_info.len())?;
    let lwork = sormtr_buffer_size(context, side, fill_mode, operation, m, n, lda, ldc)?;
    require_workspace(workspace.len(), lwork)?;
    // SAFETY: `cusolverDnSormtr` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cusolverDnSormtr(
            context.as_raw(),
            side.into(),
            fill_mode.into(),
            operation.into(),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            a.as_ptr().cast_mut().cast(),
            to_i32(lda, "lda")?,
            tau.as_ptr().cast_mut().cast(),
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
/// returned by `sytrd`, to `C` and stores the result in `C`.
///
/// `side` selects whether `Q` is applied from the left or right, and
/// `operation` selects whether `Q` is transposed.
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
/// matrix dimensions or leading dimensions are invalid, if the current GPU
/// architecture is unsupported, or if cuSOLVER reports an internal failure.
/// # Safety
///
/// Follow the [raw solver contracts](crate::dense#raw-solver-contracts).
pub unsafe fn dormtr(
    context: &Context,
    side: SideMode,
    fill_mode: FillMode,
    operation: Operation,
    m: usize,
    n: usize,
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
    validate_ormtr_inputs(side, m, n, a.len(), lda, tau.len(), c.len(), ldc)?;
    require_info_buffer(dev_info.len())?;
    let lwork = dormtr_buffer_size(context, side, fill_mode, operation, m, n, lda, ldc)?;
    require_workspace(workspace.len(), lwork)?;
    // SAFETY: `cusolverDnDormtr` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cusolverDnDormtr(
            context.as_raw(),
            side.into(),
            fill_mode.into(),
            operation.into(),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            a.as_ptr().cast_mut().cast(),
            to_i32(lda, "lda")?,
            tau.as_ptr().cast_mut().cast(),
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
pub unsafe fn cunmtr(
    context: &Context,
    side: SideMode,
    fill_mode: FillMode,
    operation: Operation,
    m: usize,
    n: usize,
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
    validate_ormtr_inputs(side, m, n, a.len(), lda, tau.len(), c.len(), ldc)?;
    require_info_buffer(dev_info.len())?;
    let lwork = cunmtr_buffer_size(context, side, fill_mode, operation, m, n, lda, ldc)?;
    require_workspace(workspace.len(), lwork)?;
    // SAFETY: `cusolverDnCunmtr` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cusolverDnCunmtr(
            context.as_raw(),
            side.into(),
            fill_mode.into(),
            operation.into(),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            a.as_ptr().cast_mut().cast(),
            to_i32(lda, "lda")?,
            tau.as_ptr().cast_mut().cast(),
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
pub unsafe fn zunmtr(
    context: &Context,
    side: SideMode,
    fill_mode: FillMode,
    operation: Operation,
    m: usize,
    n: usize,
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
    validate_ormtr_inputs(side, m, n, a.len(), lda, tau.len(), c.len(), ldc)?;
    require_info_buffer(dev_info.len())?;
    let lwork = zunmtr_buffer_size(context, side, fill_mode, operation, m, n, lda, ldc)?;
    require_workspace(workspace.len(), lwork)?;
    // SAFETY: `cusolverDnZunmtr` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::cusolverDnZunmtr(
            context.as_raw(),
            side.into(),
            fill_mode.into(),
            operation.into(),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            a.as_ptr().cast_mut().cast(),
            to_i32(lda, "lda")?,
            tau.as_ptr().cast_mut().cast(),
            c.as_mut_ptr().cast(),
            to_i32(ldc, "ldc")?,
            workspace.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast(),
        ))?;
    }
    Ok(())
}
