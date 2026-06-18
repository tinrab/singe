use singe_cuda::{
    memory::DeviceMemory,
    types::{Complex32, Complex64},
};

use crate::{
    context::Context,
    dense::validation::{
        require_info_buffer, require_workspace, validate_orgtr_inputs, validate_ormtr_inputs,
        validate_sytrd_inputs,
    },
    error::Result,
    sys, try_ffi,
    types::{FillMode, Operation, SideMode},
    utility::{to_i32, to_usize},
};

pub fn ssytrd_buffer_size(
    ctx: &Context,
    fill_mode: FillMode,
    n: usize,
    a: &DeviceMemory<f32>,
    lda: usize,
    d: &DeviceMemory<f32>,
    e: &DeviceMemory<f32>,
    tau: &DeviceMemory<f32>,
) -> Result<usize> {
    ctx.bind()?;
    validate_sytrd_inputs(n, a.len(), lda, d.len(), e.len(), tau.len())?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnSsytrd_bufferSize(
            ctx.as_raw(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_ptr().cast(),
            to_i32(lda, "lda")?,
            d.as_ptr().cast(),
            e.as_ptr().cast(),
            tau.as_ptr().cast(),
            &raw mut lwork,
        ))?;
    }
    to_usize(lwork, "lwork")
}

pub fn dsytrd_buffer_size(
    ctx: &Context,
    fill_mode: FillMode,
    n: usize,
    a: &DeviceMemory<f64>,
    lda: usize,
    d: &DeviceMemory<f64>,
    e: &DeviceMemory<f64>,
    tau: &DeviceMemory<f64>,
) -> Result<usize> {
    ctx.bind()?;
    validate_sytrd_inputs(n, a.len(), lda, d.len(), e.len(), tau.len())?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnDsytrd_bufferSize(
            ctx.as_raw(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_ptr().cast(),
            to_i32(lda, "lda")?,
            d.as_ptr().cast(),
            e.as_ptr().cast(),
            tau.as_ptr().cast(),
            &raw mut lwork,
        ))?;
    }
    to_usize(lwork, "lwork")
}

pub fn chetrd_buffer_size(
    ctx: &Context,
    fill_mode: FillMode,
    n: usize,
    a: &DeviceMemory<Complex32>,
    lda: usize,
    d: &DeviceMemory<f32>,
    e: &DeviceMemory<f32>,
    tau: &DeviceMemory<Complex32>,
) -> Result<usize> {
    ctx.bind()?;
    validate_sytrd_inputs(n, a.len(), lda, d.len(), e.len(), tau.len())?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnChetrd_bufferSize(
            ctx.as_raw(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_ptr().cast(),
            to_i32(lda, "lda")?,
            d.as_ptr().cast(),
            e.as_ptr().cast(),
            tau.as_ptr().cast(),
            &raw mut lwork,
        ))?;
    }
    to_usize(lwork, "lwork")
}

pub fn zhetrd_buffer_size(
    ctx: &Context,
    fill_mode: FillMode,
    n: usize,
    a: &DeviceMemory<Complex64>,
    lda: usize,
    d: &DeviceMemory<f64>,
    e: &DeviceMemory<f64>,
    tau: &DeviceMemory<Complex64>,
) -> Result<usize> {
    ctx.bind()?;
    validate_sytrd_inputs(n, a.len(), lda, d.len(), e.len(), tau.len())?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnZhetrd_bufferSize(
            ctx.as_raw(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_ptr().cast(),
            to_i32(lda, "lda")?,
            d.as_ptr().cast(),
            e.as_ptr().cast(),
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
pub fn ssytrd(
    ctx: &Context,
    fill_mode: FillMode,
    n: usize,
    a: &mut DeviceMemory<f32>,
    lda: usize,
    d: &mut DeviceMemory<f32>,
    e: &mut DeviceMemory<f32>,
    tau: &mut DeviceMemory<f32>,
    workspace: &mut DeviceMemory<f32>,
    dev_info: &mut DeviceMemory<i32>,
) -> Result<()> {
    ctx.bind()?;
    validate_sytrd_inputs(n, a.len(), lda, d.len(), e.len(), tau.len())?;
    require_info_buffer(dev_info)?;
    let lwork = ssytrd_buffer_size(ctx, fill_mode, n, a, lda, d, e, tau)?;
    require_workspace(workspace.len(), lwork)?;
    unsafe {
        try_ffi!(sys::cusolverDnSsytrd(
            ctx.as_raw(),
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
pub fn dsytrd(
    ctx: &Context,
    fill_mode: FillMode,
    n: usize,
    a: &mut DeviceMemory<f64>,
    lda: usize,
    d: &mut DeviceMemory<f64>,
    e: &mut DeviceMemory<f64>,
    tau: &mut DeviceMemory<f64>,
    workspace: &mut DeviceMemory<f64>,
    dev_info: &mut DeviceMemory<i32>,
) -> Result<()> {
    ctx.bind()?;
    validate_sytrd_inputs(n, a.len(), lda, d.len(), e.len(), tau.len())?;
    require_info_buffer(dev_info)?;
    let lwork = dsytrd_buffer_size(ctx, fill_mode, n, a, lda, d, e, tau)?;
    require_workspace(workspace.len(), lwork)?;
    unsafe {
        try_ffi!(sys::cusolverDnDsytrd(
            ctx.as_raw(),
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

pub fn chetrd(
    ctx: &Context,
    fill_mode: FillMode,
    n: usize,
    a: &mut DeviceMemory<Complex32>,
    lda: usize,
    d: &mut DeviceMemory<f32>,
    e: &mut DeviceMemory<f32>,
    tau: &mut DeviceMemory<Complex32>,
    workspace: &mut DeviceMemory<Complex32>,
    dev_info: &mut DeviceMemory<i32>,
) -> Result<()> {
    ctx.bind()?;
    validate_sytrd_inputs(n, a.len(), lda, d.len(), e.len(), tau.len())?;
    require_info_buffer(dev_info)?;
    let lwork = chetrd_buffer_size(ctx, fill_mode, n, a, lda, d, e, tau)?;
    require_workspace(workspace.len(), lwork)?;
    unsafe {
        try_ffi!(sys::cusolverDnChetrd(
            ctx.as_raw(),
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

pub fn zhetrd(
    ctx: &Context,
    fill_mode: FillMode,
    n: usize,
    a: &mut DeviceMemory<Complex64>,
    lda: usize,
    d: &mut DeviceMemory<f64>,
    e: &mut DeviceMemory<f64>,
    tau: &mut DeviceMemory<Complex64>,
    workspace: &mut DeviceMemory<Complex64>,
    dev_info: &mut DeviceMemory<i32>,
) -> Result<()> {
    ctx.bind()?;
    validate_sytrd_inputs(n, a.len(), lda, d.len(), e.len(), tau.len())?;
    require_info_buffer(dev_info)?;
    let lwork = zhetrd_buffer_size(ctx, fill_mode, n, a, lda, d, e, tau)?;
    require_workspace(workspace.len(), lwork)?;
    unsafe {
        try_ffi!(sys::cusolverDnZhetrd(
            ctx.as_raw(),
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

pub fn sorgtr_buffer_size(
    ctx: &Context,
    fill_mode: FillMode,
    n: usize,
    a: &DeviceMemory<f32>,
    lda: usize,
    tau: &DeviceMemory<f32>,
) -> Result<usize> {
    ctx.bind()?;
    validate_orgtr_inputs(n, a.len(), lda, tau.len())?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnSorgtr_bufferSize(
            ctx.as_raw(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_ptr().cast(),
            to_i32(lda, "lda")?,
            tau.as_ptr().cast(),
            &raw mut lwork,
        ))?;
    }
    to_usize(lwork, "lwork")
}

pub fn dorgtr_buffer_size(
    ctx: &Context,
    fill_mode: FillMode,
    n: usize,
    a: &DeviceMemory<f64>,
    lda: usize,
    tau: &DeviceMemory<f64>,
) -> Result<usize> {
    ctx.bind()?;
    validate_orgtr_inputs(n, a.len(), lda, tau.len())?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnDorgtr_bufferSize(
            ctx.as_raw(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_ptr().cast(),
            to_i32(lda, "lda")?,
            tau.as_ptr().cast(),
            &raw mut lwork,
        ))?;
    }
    to_usize(lwork, "lwork")
}

pub fn cungtr_buffer_size(
    ctx: &Context,
    fill_mode: FillMode,
    n: usize,
    a: &DeviceMemory<Complex32>,
    lda: usize,
    tau: &DeviceMemory<Complex32>,
) -> Result<usize> {
    ctx.bind()?;
    validate_orgtr_inputs(n, a.len(), lda, tau.len())?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnCungtr_bufferSize(
            ctx.as_raw(),
            fill_mode.into(),
            to_i32(n, "n")?,
            a.as_ptr().cast(),
            to_i32(lda, "lda")?,
            tau.as_ptr().cast(),
            &raw mut lwork,
        ))?;
    }
    to_usize(lwork, "lwork")
}

pub fn zungtr_buffer_size(
    ctx: &Context,
    fill_mode: FillMode,
    n: usize,
    a: &DeviceMemory<Complex64>,
    lda: usize,
    tau: &DeviceMemory<Complex64>,
) -> Result<usize> {
    ctx.bind()?;
    validate_orgtr_inputs(n, a.len(), lda, tau.len())?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnZungtr_bufferSize(
            ctx.as_raw(),
            fill_mode.into(),
            to_i32(n, "n")?,
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
pub fn sorgtr(
    ctx: &Context,
    fill_mode: FillMode,
    n: usize,
    a: &mut DeviceMemory<f32>,
    lda: usize,
    tau: &DeviceMemory<f32>,
    workspace: &mut DeviceMemory<f32>,
    dev_info: &mut DeviceMemory<i32>,
) -> Result<()> {
    ctx.bind()?;
    validate_orgtr_inputs(n, a.len(), lda, tau.len())?;
    require_info_buffer(dev_info)?;
    let lwork = sorgtr_buffer_size(ctx, fill_mode, n, a, lda, tau)?;
    require_workspace(workspace.len(), lwork)?;
    unsafe {
        try_ffi!(sys::cusolverDnSorgtr(
            ctx.as_raw(),
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
pub fn dorgtr(
    ctx: &Context,
    fill_mode: FillMode,
    n: usize,
    a: &mut DeviceMemory<f64>,
    lda: usize,
    tau: &DeviceMemory<f64>,
    workspace: &mut DeviceMemory<f64>,
    dev_info: &mut DeviceMemory<i32>,
) -> Result<()> {
    ctx.bind()?;
    validate_orgtr_inputs(n, a.len(), lda, tau.len())?;
    require_info_buffer(dev_info)?;
    let lwork = dorgtr_buffer_size(ctx, fill_mode, n, a, lda, tau)?;
    require_workspace(workspace.len(), lwork)?;
    unsafe {
        try_ffi!(sys::cusolverDnDorgtr(
            ctx.as_raw(),
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

pub fn cungtr(
    ctx: &Context,
    fill_mode: FillMode,
    n: usize,
    a: &mut DeviceMemory<Complex32>,
    lda: usize,
    tau: &DeviceMemory<Complex32>,
    workspace: &mut DeviceMemory<Complex32>,
    dev_info: &mut DeviceMemory<i32>,
) -> Result<()> {
    ctx.bind()?;
    validate_orgtr_inputs(n, a.len(), lda, tau.len())?;
    require_info_buffer(dev_info)?;
    let lwork = cungtr_buffer_size(ctx, fill_mode, n, a, lda, tau)?;
    require_workspace(workspace.len(), lwork)?;
    unsafe {
        try_ffi!(sys::cusolverDnCungtr(
            ctx.as_raw(),
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

pub fn zungtr(
    ctx: &Context,
    fill_mode: FillMode,
    n: usize,
    a: &mut DeviceMemory<Complex64>,
    lda: usize,
    tau: &DeviceMemory<Complex64>,
    workspace: &mut DeviceMemory<Complex64>,
    dev_info: &mut DeviceMemory<i32>,
) -> Result<()> {
    ctx.bind()?;
    validate_orgtr_inputs(n, a.len(), lda, tau.len())?;
    require_info_buffer(dev_info)?;
    let lwork = zungtr_buffer_size(ctx, fill_mode, n, a, lda, tau)?;
    require_workspace(workspace.len(), lwork)?;
    unsafe {
        try_ffi!(sys::cusolverDnZungtr(
            ctx.as_raw(),
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

pub fn sormtr_buffer_size(
    ctx: &Context,
    side: SideMode,
    fill_mode: FillMode,
    operation: Operation,
    m: usize,
    n: usize,
    a: &DeviceMemory<f32>,
    lda: usize,
    tau: &DeviceMemory<f32>,
    c: &DeviceMemory<f32>,
    ldc: usize,
) -> Result<usize> {
    ctx.bind()?;
    validate_ormtr_inputs(side, m, n, a.len(), lda, tau.len(), c.len(), ldc)?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnSormtr_bufferSize(
            ctx.as_raw(),
            side.into(),
            fill_mode.into(),
            operation.into(),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
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

pub fn dormtr_buffer_size(
    ctx: &Context,
    side: SideMode,
    fill_mode: FillMode,
    operation: Operation,
    m: usize,
    n: usize,
    a: &DeviceMemory<f64>,
    lda: usize,
    tau: &DeviceMemory<f64>,
    c: &DeviceMemory<f64>,
    ldc: usize,
) -> Result<usize> {
    ctx.bind()?;
    validate_ormtr_inputs(side, m, n, a.len(), lda, tau.len(), c.len(), ldc)?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnDormtr_bufferSize(
            ctx.as_raw(),
            side.into(),
            fill_mode.into(),
            operation.into(),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
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

pub fn cunmtr_buffer_size(
    ctx: &Context,
    side: SideMode,
    fill_mode: FillMode,
    operation: Operation,
    m: usize,
    n: usize,
    a: &DeviceMemory<Complex32>,
    lda: usize,
    tau: &DeviceMemory<Complex32>,
    c: &DeviceMemory<Complex32>,
    ldc: usize,
) -> Result<usize> {
    ctx.bind()?;
    validate_ormtr_inputs(side, m, n, a.len(), lda, tau.len(), c.len(), ldc)?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnCunmtr_bufferSize(
            ctx.as_raw(),
            side.into(),
            fill_mode.into(),
            operation.into(),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
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

pub fn zunmtr_buffer_size(
    ctx: &Context,
    side: SideMode,
    fill_mode: FillMode,
    operation: Operation,
    m: usize,
    n: usize,
    a: &DeviceMemory<Complex64>,
    lda: usize,
    tau: &DeviceMemory<Complex64>,
    c: &DeviceMemory<Complex64>,
    ldc: usize,
) -> Result<usize> {
    ctx.bind()?;
    validate_ormtr_inputs(side, m, n, a.len(), lda, tau.len(), c.len(), ldc)?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnZunmtr_bufferSize(
            ctx.as_raw(),
            side.into(),
            fill_mode.into(),
            operation.into(),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
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
pub fn sormtr(
    ctx: &Context,
    side: SideMode,
    fill_mode: FillMode,
    operation: Operation,
    m: usize,
    n: usize,
    a: &mut DeviceMemory<f32>,
    lda: usize,
    tau: &mut DeviceMemory<f32>,
    c: &mut DeviceMemory<f32>,
    ldc: usize,
    workspace: &mut DeviceMemory<f32>,
    dev_info: &mut DeviceMemory<i32>,
) -> Result<()> {
    ctx.bind()?;
    validate_ormtr_inputs(side, m, n, a.len(), lda, tau.len(), c.len(), ldc)?;
    require_info_buffer(dev_info)?;
    let lwork = sormtr_buffer_size(ctx, side, fill_mode, operation, m, n, a, lda, tau, c, ldc)?;
    require_workspace(workspace.len(), lwork)?;
    unsafe {
        try_ffi!(sys::cusolverDnSormtr(
            ctx.as_raw(),
            side.into(),
            fill_mode.into(),
            operation.into(),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            a.as_mut_ptr().cast(),
            to_i32(lda, "lda")?,
            tau.as_mut_ptr().cast(),
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
pub fn dormtr(
    ctx: &Context,
    side: SideMode,
    fill_mode: FillMode,
    operation: Operation,
    m: usize,
    n: usize,
    a: &mut DeviceMemory<f64>,
    lda: usize,
    tau: &mut DeviceMemory<f64>,
    c: &mut DeviceMemory<f64>,
    ldc: usize,
    workspace: &mut DeviceMemory<f64>,
    dev_info: &mut DeviceMemory<i32>,
) -> Result<()> {
    ctx.bind()?;
    validate_ormtr_inputs(side, m, n, a.len(), lda, tau.len(), c.len(), ldc)?;
    require_info_buffer(dev_info)?;
    let lwork = dormtr_buffer_size(ctx, side, fill_mode, operation, m, n, a, lda, tau, c, ldc)?;
    require_workspace(workspace.len(), lwork)?;
    unsafe {
        try_ffi!(sys::cusolverDnDormtr(
            ctx.as_raw(),
            side.into(),
            fill_mode.into(),
            operation.into(),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            a.as_mut_ptr().cast(),
            to_i32(lda, "lda")?,
            tau.as_mut_ptr().cast(),
            c.as_mut_ptr().cast(),
            to_i32(ldc, "ldc")?,
            workspace.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast(),
        ))?;
    }
    Ok(())
}

pub fn cunmtr(
    ctx: &Context,
    side: SideMode,
    fill_mode: FillMode,
    operation: Operation,
    m: usize,
    n: usize,
    a: &mut DeviceMemory<Complex32>,
    lda: usize,
    tau: &mut DeviceMemory<Complex32>,
    c: &mut DeviceMemory<Complex32>,
    ldc: usize,
    workspace: &mut DeviceMemory<Complex32>,
    dev_info: &mut DeviceMemory<i32>,
) -> Result<()> {
    ctx.bind()?;
    validate_ormtr_inputs(side, m, n, a.len(), lda, tau.len(), c.len(), ldc)?;
    require_info_buffer(dev_info)?;
    let lwork = cunmtr_buffer_size(ctx, side, fill_mode, operation, m, n, a, lda, tau, c, ldc)?;
    require_workspace(workspace.len(), lwork)?;
    unsafe {
        try_ffi!(sys::cusolverDnCunmtr(
            ctx.as_raw(),
            side.into(),
            fill_mode.into(),
            operation.into(),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            a.as_mut_ptr().cast(),
            to_i32(lda, "lda")?,
            tau.as_mut_ptr().cast(),
            c.as_mut_ptr().cast(),
            to_i32(ldc, "ldc")?,
            workspace.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast(),
        ))?;
    }
    Ok(())
}

pub fn zunmtr(
    ctx: &Context,
    side: SideMode,
    fill_mode: FillMode,
    operation: Operation,
    m: usize,
    n: usize,
    a: &mut DeviceMemory<Complex64>,
    lda: usize,
    tau: &mut DeviceMemory<Complex64>,
    c: &mut DeviceMemory<Complex64>,
    ldc: usize,
    workspace: &mut DeviceMemory<Complex64>,
    dev_info: &mut DeviceMemory<i32>,
) -> Result<()> {
    ctx.bind()?;
    validate_ormtr_inputs(side, m, n, a.len(), lda, tau.len(), c.len(), ldc)?;
    require_info_buffer(dev_info)?;
    let lwork = zunmtr_buffer_size(ctx, side, fill_mode, operation, m, n, a, lda, tau, c, ldc)?;
    require_workspace(workspace.len(), lwork)?;
    unsafe {
        try_ffi!(sys::cusolverDnZunmtr(
            ctx.as_raw(),
            side.into(),
            fill_mode.into(),
            operation.into(),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            a.as_mut_ptr().cast(),
            to_i32(lda, "lda")?,
            tau.as_mut_ptr().cast(),
            c.as_mut_ptr().cast(),
            to_i32(ldc, "ldc")?,
            workspace.as_mut_ptr().cast(),
            to_i32(lwork, "lwork")?,
            dev_info.as_mut_ptr().cast(),
        ))?;
    }
    Ok(())
}
