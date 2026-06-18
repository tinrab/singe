use singe_cuda::{
    memory::DeviceMemory,
    types::{Complex32, Complex64},
};

use crate::{
    context::Context,
    dense::validation::{
        require_info_buffer, require_workspace, validate_bidiagonal_buffers,
        validate_bidiagonal_dims, validate_orgbr_inputs,
    },
    error::Result,
    sys, try_ffi,
    types::SideMode,
    utility::{to_i32, to_usize},
};

pub fn sgebrd_buffer_size(ctx: &Context, m: usize, n: usize) -> Result<usize> {
    ctx.bind()?;
    validate_bidiagonal_dims(m, n)?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnSgebrd_bufferSize(
            ctx.as_raw(),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            &raw mut lwork,
        ))?;
    }
    to_usize(lwork, "lwork")
}

pub fn dgebrd_buffer_size(ctx: &Context, m: usize, n: usize) -> Result<usize> {
    ctx.bind()?;
    validate_bidiagonal_dims(m, n)?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnDgebrd_bufferSize(
            ctx.as_raw(),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            &raw mut lwork,
        ))?;
    }
    to_usize(lwork, "lwork")
}

pub fn cgebrd_buffer_size(ctx: &Context, m: usize, n: usize) -> Result<usize> {
    ctx.bind()?;
    validate_bidiagonal_dims(m, n)?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnCgebrd_bufferSize(
            ctx.as_raw(),
            to_i32(m, "m")?,
            to_i32(n, "n")?,
            &raw mut lwork,
        ))?;
    }
    to_usize(lwork, "lwork")
}

pub fn zgebrd_buffer_size(ctx: &Context, m: usize, n: usize) -> Result<usize> {
    ctx.bind()?;
    validate_bidiagonal_dims(m, n)?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnZgebrd_bufferSize(
            ctx.as_raw(),
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
pub fn sgebrd(
    ctx: &Context,
    m: usize,
    n: usize,
    a: &mut DeviceMemory<f32>,
    lda: usize,
    d: &mut DeviceMemory<f32>,
    e: &mut DeviceMemory<f32>,
    tauq: &mut DeviceMemory<f32>,
    taup: &mut DeviceMemory<f32>,
    workspace: &mut DeviceMemory<f32>,
    dev_info: &mut DeviceMemory<i32>,
) -> Result<()> {
    ctx.bind()?;
    validate_bidiagonal_buffers(m, n, a.len(), lda, d.len(), e.len(), tauq.len(), taup.len())?;
    require_info_buffer(dev_info)?;
    let lwork = sgebrd_buffer_size(ctx, m, n)?;
    require_workspace(workspace.len(), lwork)?;
    unsafe {
        try_ffi!(sys::cusolverDnSgebrd(
            ctx.as_raw(),
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
pub fn dgebrd(
    ctx: &Context,
    m: usize,
    n: usize,
    a: &mut DeviceMemory<f64>,
    lda: usize,
    d: &mut DeviceMemory<f64>,
    e: &mut DeviceMemory<f64>,
    tauq: &mut DeviceMemory<f64>,
    taup: &mut DeviceMemory<f64>,
    workspace: &mut DeviceMemory<f64>,
    dev_info: &mut DeviceMemory<i32>,
) -> Result<()> {
    ctx.bind()?;
    validate_bidiagonal_buffers(m, n, a.len(), lda, d.len(), e.len(), tauq.len(), taup.len())?;
    require_info_buffer(dev_info)?;
    let lwork = dgebrd_buffer_size(ctx, m, n)?;
    require_workspace(workspace.len(), lwork)?;
    unsafe {
        try_ffi!(sys::cusolverDnDgebrd(
            ctx.as_raw(),
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
pub fn cgebrd(
    ctx: &Context,
    m: usize,
    n: usize,
    a: &mut DeviceMemory<Complex32>,
    lda: usize,
    d: &mut DeviceMemory<f32>,
    e: &mut DeviceMemory<f32>,
    tauq: &mut DeviceMemory<Complex32>,
    taup: &mut DeviceMemory<Complex32>,
    workspace: &mut DeviceMemory<Complex32>,
    dev_info: &mut DeviceMemory<i32>,
) -> Result<()> {
    ctx.bind()?;
    validate_bidiagonal_buffers(m, n, a.len(), lda, d.len(), e.len(), tauq.len(), taup.len())?;
    require_info_buffer(dev_info)?;
    let lwork = cgebrd_buffer_size(ctx, m, n)?;
    require_workspace(workspace.len(), lwork)?;
    unsafe {
        try_ffi!(sys::cusolverDnCgebrd(
            ctx.as_raw(),
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
pub fn zgebrd(
    ctx: &Context,
    m: usize,
    n: usize,
    a: &mut DeviceMemory<Complex64>,
    lda: usize,
    d: &mut DeviceMemory<f64>,
    e: &mut DeviceMemory<f64>,
    tauq: &mut DeviceMemory<Complex64>,
    taup: &mut DeviceMemory<Complex64>,
    workspace: &mut DeviceMemory<Complex64>,
    dev_info: &mut DeviceMemory<i32>,
) -> Result<()> {
    ctx.bind()?;
    validate_bidiagonal_buffers(m, n, a.len(), lda, d.len(), e.len(), tauq.len(), taup.len())?;
    require_info_buffer(dev_info)?;
    let lwork = zgebrd_buffer_size(ctx, m, n)?;
    require_workspace(workspace.len(), lwork)?;
    unsafe {
        try_ffi!(sys::cusolverDnZgebrd(
            ctx.as_raw(),
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

pub fn sorgbr_buffer_size(
    ctx: &Context,
    side: SideMode,
    m: usize,
    n: usize,
    k: usize,
    a: &DeviceMemory<f32>,
    lda: usize,
    tau: &DeviceMemory<f32>,
) -> Result<usize> {
    ctx.bind()?;
    validate_orgbr_inputs(side, m, n, k, a.len(), lda, tau.len())?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnSorgbr_bufferSize(
            ctx.as_raw(),
            side.into(),
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

pub fn dorgbr_buffer_size(
    ctx: &Context,
    side: SideMode,
    m: usize,
    n: usize,
    k: usize,
    a: &DeviceMemory<f64>,
    lda: usize,
    tau: &DeviceMemory<f64>,
) -> Result<usize> {
    ctx.bind()?;
    validate_orgbr_inputs(side, m, n, k, a.len(), lda, tau.len())?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnDorgbr_bufferSize(
            ctx.as_raw(),
            side.into(),
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

pub fn cungbr_buffer_size(
    ctx: &Context,
    side: SideMode,
    m: usize,
    n: usize,
    k: usize,
    a: &DeviceMemory<Complex32>,
    lda: usize,
    tau: &DeviceMemory<Complex32>,
) -> Result<usize> {
    ctx.bind()?;
    validate_orgbr_inputs(side, m, n, k, a.len(), lda, tau.len())?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnCungbr_bufferSize(
            ctx.as_raw(),
            side.into(),
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

pub fn zungbr_buffer_size(
    ctx: &Context,
    side: SideMode,
    m: usize,
    n: usize,
    k: usize,
    a: &DeviceMemory<Complex64>,
    lda: usize,
    tau: &DeviceMemory<Complex64>,
) -> Result<usize> {
    ctx.bind()?;
    validate_orgbr_inputs(side, m, n, k, a.len(), lda, tau.len())?;
    let mut lwork = 0;
    unsafe {
        try_ffi!(sys::cusolverDnZungbr_bufferSize(
            ctx.as_raw(),
            side.into(),
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
pub fn sorgbr(
    ctx: &Context,
    side: SideMode,
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
    validate_orgbr_inputs(side, m, n, k, a.len(), lda, tau.len())?;
    require_info_buffer(dev_info)?;
    let lwork = sorgbr_buffer_size(ctx, side, m, n, k, a, lda, tau)?;
    require_workspace(workspace.len(), lwork)?;
    unsafe {
        try_ffi!(sys::cusolverDnSorgbr(
            ctx.as_raw(),
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
pub fn dorgbr(
    ctx: &Context,
    side: SideMode,
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
    validate_orgbr_inputs(side, m, n, k, a.len(), lda, tau.len())?;
    require_info_buffer(dev_info)?;
    let lwork = dorgbr_buffer_size(ctx, side, m, n, k, a, lda, tau)?;
    require_workspace(workspace.len(), lwork)?;
    unsafe {
        try_ffi!(sys::cusolverDnDorgbr(
            ctx.as_raw(),
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

pub fn cungbr(
    ctx: &Context,
    side: SideMode,
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
    validate_orgbr_inputs(side, m, n, k, a.len(), lda, tau.len())?;
    require_info_buffer(dev_info)?;
    let lwork = cungbr_buffer_size(ctx, side, m, n, k, a, lda, tau)?;
    require_workspace(workspace.len(), lwork)?;
    unsafe {
        try_ffi!(sys::cusolverDnCungbr(
            ctx.as_raw(),
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

pub fn zungbr(
    ctx: &Context,
    side: SideMode,
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
    validate_orgbr_inputs(side, m, n, k, a.len(), lda, tau.len())?;
    require_info_buffer(dev_info)?;
    let lwork = zungbr_buffer_size(ctx, side, m, n, k, a, lda, tau)?;
    require_workspace(workspace.len(), lwork)?;
    unsafe {
        try_ffi!(sys::cusolverDnZungbr(
            ctx.as_raw(),
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
