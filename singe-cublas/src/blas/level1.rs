use singe_cuda::view::{DeviceOutput, DeviceRange};

use crate::{
    blas::{ComplexElement, Element, RealElement, shape::ensure_vector},
    context::Context,
    error::Result,
    scalar::{Scalar, ScalarMut},
    try_ffi,
    utility::to_i64,
};

/// Enqueues `x = alpha * x` over `n` elements `incx` apart.
///
/// # Safety
///
/// The caller upholds the [raw execution contract](crate#raw-execution).
///
/// # Errors
///
/// Returns an error if `x` is too small, an increment is zero, a value exceeds the native integer range, an operand belongs to another CUDA context, or cuBLAS rejects the call.
pub unsafe fn scal_async<T: Element>(
    context: &Context,
    n: usize,
    alpha: Scalar<'_, T>,
    x: &mut (impl DeviceOutput<T> + ?Sized),
    incx: usize,
) -> Result<()> {
    context.bind()?;
    ensure_vector("x", x.len(), n, incx)?;
    let (n, incx) = (to_i64(n, "n")?, to_i64(incx, "incx")?);
    context.prepare_operands(&[x.region()], &[alpha.region()])?;
    context.use_pointer_mode(&[alpha.pointer_mode()])?;
    // SAFETY: `x` holds `n` values `incx` apart in this context, ordered after its producers; the caller upholds the raw execution contract.
    unsafe {
        try_ffi!(T::scal(
            context.as_raw(),
            n,
            alpha.ptr(),
            x.as_mut_ptr(),
            incx
        ))
    }
}

/// Computes `x = alpha * x` and waits for it.
///
/// # Errors
///
/// Returns an error if the handle's stream is capturing, waiting fails, or for the reasons [`scal_async`] lists.
pub fn scal<T: Element>(
    context: &Context,
    n: usize,
    alpha: Scalar<'_, T>,
    x: &mut (impl DeviceOutput<T> + ?Sized),
    incx: usize,
) -> Result<()> {
    // SAFETY: the scope waits for the handle's stream before the borrowed operands are released.
    context.sync_scope(|| unsafe { scal_async(context, n, alpha, x, incx) })
}

/// Enqueues `y = alpha * x + y`.
///
/// # Safety
///
/// The caller upholds the [raw execution contract](crate#raw-execution).
///
/// # Errors
///
/// Returns an error if a vector is too small, an increment is zero, a value exceeds the native integer range, an operand belongs to another CUDA context, or cuBLAS rejects the call.
pub unsafe fn axpy_async<T: Element>(
    context: &Context,
    n: usize,
    alpha: Scalar<'_, T>,
    x: &(impl DeviceRange<T> + ?Sized),
    incx: usize,
    y: &mut (impl DeviceOutput<T> + ?Sized),
    incy: usize,
) -> Result<()> {
    context.bind()?;
    ensure_vector("x", x.len(), n, incx)?;
    ensure_vector("y", y.len(), n, incy)?;
    let (n, incx, incy) = (
        to_i64(n, "n")?,
        to_i64(incx, "incx")?,
        to_i64(incy, "incy")?,
    );
    context.prepare_operands(&[x.region(), y.region()], &[alpha.region()])?;
    context.use_pointer_mode(&[alpha.pointer_mode()])?;
    // SAFETY: the vectors hold the strided values in this context, ordered after their producers; the caller upholds the raw execution contract.
    unsafe {
        try_ffi!(T::axpy(
            context.as_raw(),
            n,
            alpha.ptr(),
            x.as_device_ptr(),
            incx,
            y.as_mut_ptr(),
            incy,
        ))
    }
}

/// Computes `y = alpha * x + y` and waits for it.
///
/// # Errors
///
/// Returns an error if the handle's stream is capturing, waiting fails, or for the reasons [`axpy_async`] lists.
pub fn axpy<T: Element>(
    context: &Context,
    n: usize,
    alpha: Scalar<'_, T>,
    x: &(impl DeviceRange<T> + ?Sized),
    incx: usize,
    y: &mut (impl DeviceOutput<T> + ?Sized),
    incy: usize,
) -> Result<()> {
    // SAFETY: the scope waits for the handle's stream before the borrowed operands are released.
    context.sync_scope(|| unsafe { axpy_async(context, n, alpha, x, incx, y, incy) })
}

/// Enqueues `y = x`.
///
/// # Safety
///
/// The caller upholds the [raw execution contract](crate#raw-execution).
///
/// # Errors
///
/// Returns an error if a vector is too small, an increment is zero, a value exceeds the native integer range, an operand belongs to another CUDA context, or cuBLAS rejects the call.
pub unsafe fn copy_async<T: Element>(
    context: &Context,
    n: usize,
    x: &(impl DeviceRange<T> + ?Sized),
    incx: usize,
    y: &mut (impl DeviceOutput<T> + ?Sized),
    incy: usize,
) -> Result<()> {
    context.bind()?;
    ensure_vector("x", x.len(), n, incx)?;
    ensure_vector("y", y.len(), n, incy)?;
    let (n, incx, incy) = (
        to_i64(n, "n")?,
        to_i64(incx, "incx")?,
        to_i64(incy, "incy")?,
    );
    context.prepare_operands(&[x.region(), y.region()], &[])?;
    // SAFETY: the vectors hold the strided values in this context, ordered after their producers; the caller upholds the raw execution contract.
    unsafe {
        try_ffi!(T::copy(
            context.as_raw(),
            n,
            x.as_device_ptr(),
            incx,
            y.as_mut_ptr(),
            incy,
        ))
    }
}

/// Copies `x` into `y` and waits for it.
///
/// # Errors
///
/// Returns an error if the handle's stream is capturing, waiting fails, or for the reasons [`copy_async`] lists.
pub fn copy<T: Element>(
    context: &Context,
    n: usize,
    x: &(impl DeviceRange<T> + ?Sized),
    incx: usize,
    y: &mut (impl DeviceOutput<T> + ?Sized),
    incy: usize,
) -> Result<()> {
    // SAFETY: the scope waits for the handle's stream before the borrowed operands are released.
    context.sync_scope(|| unsafe { copy_async(context, n, x, incx, y, incy) })
}

/// Enqueues an exchange of the values of `x` and `y`.
///
/// # Safety
///
/// The caller upholds the [raw execution contract](crate#raw-execution).
///
/// # Errors
///
/// Returns an error if a vector is too small, an increment is zero, a value exceeds the native integer range, an operand belongs to another CUDA context, or cuBLAS rejects the call.
pub unsafe fn swap_async<T: Element>(
    context: &Context,
    n: usize,
    x: &mut (impl DeviceOutput<T> + ?Sized),
    incx: usize,
    y: &mut (impl DeviceOutput<T> + ?Sized),
    incy: usize,
) -> Result<()> {
    context.bind()?;
    ensure_vector("x", x.len(), n, incx)?;
    ensure_vector("y", y.len(), n, incy)?;
    let (n, incx, incy) = (
        to_i64(n, "n")?,
        to_i64(incx, "incx")?,
        to_i64(incy, "incy")?,
    );
    context.prepare_operands(&[x.region(), y.region()], &[])?;
    // SAFETY: the vectors hold the strided values in this context, ordered after their producers; the caller upholds the raw execution contract.
    unsafe {
        try_ffi!(T::swap(
            context.as_raw(),
            n,
            x.as_mut_ptr(),
            incx,
            y.as_mut_ptr(),
            incy,
        ))
    }
}

/// Exchanges the values of `x` and `y` and waits for it.
///
/// # Errors
///
/// Returns an error if the handle's stream is capturing, waiting fails, or for the reasons [`swap_async`] lists.
pub fn swap<T: Element>(
    context: &Context,
    n: usize,
    x: &mut (impl DeviceOutput<T> + ?Sized),
    incx: usize,
    y: &mut (impl DeviceOutput<T> + ?Sized),
    incy: usize,
) -> Result<()> {
    // SAFETY: the scope waits for the handle's stream before the borrowed operands are released.
    context.sync_scope(|| unsafe { swap_async(context, n, x, incx, y, incy) })
}

/// Enqueues the unconjugated dot product of `x` and `y` into `result`.
///
/// # Safety
///
/// The caller upholds the [raw execution contract](crate#raw-execution).
///
/// # Errors
///
/// Returns an error if a vector is too small, an increment is zero, a value exceeds the native integer range, an operand belongs to another CUDA context, or cuBLAS rejects the call.
pub unsafe fn dot_async<T: Element>(
    context: &Context,
    n: usize,
    x: &(impl DeviceRange<T> + ?Sized),
    incx: usize,
    y: &(impl DeviceRange<T> + ?Sized),
    incy: usize,
    mut result: ScalarMut<'_, T>,
) -> Result<()> {
    context.bind()?;
    ensure_vector("x", x.len(), n, incx)?;
    ensure_vector("y", y.len(), n, incy)?;
    let (n, incx, incy) = (
        to_i64(n, "n")?,
        to_i64(incx, "incx")?,
        to_i64(incy, "incy")?,
    );
    context.prepare_operands(&[x.region(), y.region()], &[result.region()])?;
    context.use_pointer_mode(&[result.pointer_mode()])?;
    // SAFETY: the vectors hold the strided values in this context, ordered after their producers; the caller upholds the raw execution contract.
    unsafe {
        try_ffi!(T::dot(
            context.as_raw(),
            n,
            x.as_device_ptr(),
            incx,
            y.as_device_ptr(),
            incy,
            result.as_mut_ptr(),
        ))
    }
}

/// Computes the unconjugated dot product of `x` and `y` into `result` and waits for it.
///
/// # Errors
///
/// Returns an error if the handle's stream is capturing, waiting fails, or for the reasons [`dot_async`] lists.
pub fn dot<T: Element>(
    context: &Context,
    n: usize,
    x: &(impl DeviceRange<T> + ?Sized),
    incx: usize,
    y: &(impl DeviceRange<T> + ?Sized),
    incy: usize,
    result: ScalarMut<'_, T>,
) -> Result<()> {
    // SAFETY: the scope waits for the handle's stream before the borrowed operands are released.
    context.sync_scope(|| unsafe { dot_async(context, n, x, incx, y, incy, result) })
}

/// Enqueues the dot product of conjugated `x` and `y` into `result`.
///
/// # Safety
///
/// The caller upholds the [raw execution contract](crate#raw-execution).
///
/// # Errors
///
/// Returns an error if a vector is too small, an increment is zero, a value exceeds the native integer range, an operand belongs to another CUDA context, or cuBLAS rejects the call.
pub unsafe fn dotc_async<T: ComplexElement>(
    context: &Context,
    n: usize,
    x: &(impl DeviceRange<T> + ?Sized),
    incx: usize,
    y: &(impl DeviceRange<T> + ?Sized),
    incy: usize,
    mut result: ScalarMut<'_, T>,
) -> Result<()> {
    context.bind()?;
    ensure_vector("x", x.len(), n, incx)?;
    ensure_vector("y", y.len(), n, incy)?;
    let (n, incx, incy) = (
        to_i64(n, "n")?,
        to_i64(incx, "incx")?,
        to_i64(incy, "incy")?,
    );
    context.prepare_operands(&[x.region(), y.region()], &[result.region()])?;
    context.use_pointer_mode(&[result.pointer_mode()])?;
    // SAFETY: the vectors hold the strided values in this context, ordered after their producers; the caller upholds the raw execution contract.
    unsafe {
        try_ffi!(T::dotc(
            context.as_raw(),
            n,
            x.as_device_ptr(),
            incx,
            y.as_device_ptr(),
            incy,
            result.as_mut_ptr(),
        ))
    }
}

/// Computes the dot product of conjugated `x` and `y` into `result` and waits for it.
///
/// # Errors
///
/// Returns an error if the handle's stream is capturing, waiting fails, or for the reasons [`dotc_async`] lists.
pub fn dotc<T: ComplexElement>(
    context: &Context,
    n: usize,
    x: &(impl DeviceRange<T> + ?Sized),
    incx: usize,
    y: &(impl DeviceRange<T> + ?Sized),
    incy: usize,
    result: ScalarMut<'_, T>,
) -> Result<()> {
    // SAFETY: the scope waits for the handle's stream before the borrowed operands are released.
    context.sync_scope(|| unsafe { dotc_async(context, n, x, incx, y, incy, result) })
}

/// Enqueues the Euclidean norm of `x` into `result`.
///
/// # Safety
///
/// The caller upholds the [raw execution contract](crate#raw-execution).
///
/// # Errors
///
/// Returns an error if `x` is too small, `incx` is zero, a value exceeds the native integer range, an operand belongs to another CUDA context, or cuBLAS rejects the call.
pub unsafe fn nrm2_async<T: Element>(
    context: &Context,
    n: usize,
    x: &(impl DeviceRange<T> + ?Sized),
    incx: usize,
    mut result: ScalarMut<'_, T::Real>,
) -> Result<()> {
    context.bind()?;
    ensure_vector("x", x.len(), n, incx)?;
    let (n, incx) = (to_i64(n, "n")?, to_i64(incx, "incx")?);
    context.prepare_operands(&[x.region()], &[result.region()])?;
    context.use_pointer_mode(&[result.pointer_mode()])?;
    // SAFETY: `x` holds the strided values in this context, ordered after its producers, and `result` has the element's real type; the caller upholds the raw execution contract.
    unsafe {
        try_ffi!(T::nrm2(
            context.as_raw(),
            n,
            x.as_device_ptr(),
            incx,
            result.as_mut_ptr().cast(),
        ))
    }
}

/// Computes the Euclidean norm of `x` into `result` and waits for it.
///
/// # Errors
///
/// Returns an error if the handle's stream is capturing, waiting fails, or for the reasons [`nrm2_async`] lists.
pub fn nrm2<T: Element>(
    context: &Context,
    n: usize,
    x: &(impl DeviceRange<T> + ?Sized),
    incx: usize,
    result: ScalarMut<'_, T::Real>,
) -> Result<()> {
    // SAFETY: the scope waits for the handle's stream before the borrowed operands are released.
    context.sync_scope(|| unsafe { nrm2_async(context, n, x, incx, result) })
}

/// Enqueues the sum of the absolute values of `x` into `result`.
///
/// For complex values, it sums the absolute real and imaginary parts.
///
/// # Safety
///
/// The caller upholds the [raw execution contract](crate#raw-execution).
///
/// # Errors
///
/// Returns an error if `x` is too small, `incx` is zero, a value exceeds the native integer range, an operand belongs to another CUDA context, or cuBLAS rejects the call.
pub unsafe fn asum_async<T: Element>(
    context: &Context,
    n: usize,
    x: &(impl DeviceRange<T> + ?Sized),
    incx: usize,
    mut result: ScalarMut<'_, T::Real>,
) -> Result<()> {
    context.bind()?;
    ensure_vector("x", x.len(), n, incx)?;
    let (n, incx) = (to_i64(n, "n")?, to_i64(incx, "incx")?);
    context.prepare_operands(&[x.region()], &[result.region()])?;
    context.use_pointer_mode(&[result.pointer_mode()])?;
    // SAFETY: `x` holds the strided values in this context, ordered after its producers, and `result` has the element's real type; the caller upholds the raw execution contract.
    unsafe {
        try_ffi!(T::asum(
            context.as_raw(),
            n,
            x.as_device_ptr(),
            incx,
            result.as_mut_ptr().cast(),
        ))
    }
}

/// Computes the sum of the absolute values of `x` into `result` and waits for it.
///
/// # Errors
///
/// Returns an error if the handle's stream is capturing, waiting fails, or for the reasons [`asum_async`] lists.
pub fn asum<T: Element>(
    context: &Context,
    n: usize,
    x: &(impl DeviceRange<T> + ?Sized),
    incx: usize,
    result: ScalarMut<'_, T::Real>,
) -> Result<()> {
    // SAFETY: the scope waits for the handle's stream before the borrowed operands are released.
    context.sync_scope(|| unsafe { asum_async(context, n, x, incx, result) })
}

/// Enqueues the one-based index of the first element of `x` with the largest magnitude into `result`.
///
/// For complex values, the magnitude is the sum of the absolute real and imaginary parts.
///
/// # Safety
///
/// The caller upholds the [raw execution contract](crate#raw-execution).
///
/// # Errors
///
/// Returns an error if `x` is too small, `incx` is zero, a value exceeds the native integer range, an operand belongs to another CUDA context, or cuBLAS rejects the call.
pub unsafe fn iamax_async<T: Element>(
    context: &Context,
    n: usize,
    x: &(impl DeviceRange<T> + ?Sized),
    incx: usize,
    mut result: ScalarMut<'_, i64>,
) -> Result<()> {
    context.bind()?;
    ensure_vector("x", x.len(), n, incx)?;
    let (n, incx) = (to_i64(n, "n")?, to_i64(incx, "incx")?);
    context.prepare_operands(&[x.region()], &[result.region()])?;
    context.use_pointer_mode(&[result.pointer_mode()])?;
    // SAFETY: `x` holds the strided values in this context, ordered after its producers; the caller upholds the raw execution contract.
    unsafe {
        try_ffi!(T::iamax(
            context.as_raw(),
            n,
            x.as_device_ptr(),
            incx,
            result.as_mut_ptr(),
        ))
    }
}

/// Computes the one-based index of the first element of `x` with the largest magnitude into `result` and waits for it.
///
/// # Errors
///
/// Returns an error if the handle's stream is capturing, waiting fails, or for the reasons [`iamax_async`] lists.
pub fn iamax<T: Element>(
    context: &Context,
    n: usize,
    x: &(impl DeviceRange<T> + ?Sized),
    incx: usize,
    result: ScalarMut<'_, i64>,
) -> Result<()> {
    // SAFETY: the scope waits for the handle's stream before the borrowed operands are released.
    context.sync_scope(|| unsafe { iamax_async(context, n, x, incx, result) })
}

/// Enqueues the one-based index of the first element of `x` with the smallest magnitude into `result`.
///
/// For complex values, the magnitude is the sum of the absolute real and imaginary parts.
///
/// # Safety
///
/// The caller upholds the [raw execution contract](crate#raw-execution).
///
/// # Errors
///
/// Returns an error if `x` is too small, `incx` is zero, a value exceeds the native integer range, an operand belongs to another CUDA context, or cuBLAS rejects the call.
pub unsafe fn iamin_async<T: Element>(
    context: &Context,
    n: usize,
    x: &(impl DeviceRange<T> + ?Sized),
    incx: usize,
    mut result: ScalarMut<'_, i64>,
) -> Result<()> {
    context.bind()?;
    ensure_vector("x", x.len(), n, incx)?;
    let (n, incx) = (to_i64(n, "n")?, to_i64(incx, "incx")?);
    context.prepare_operands(&[x.region()], &[result.region()])?;
    context.use_pointer_mode(&[result.pointer_mode()])?;
    // SAFETY: `x` holds the strided values in this context, ordered after its producers; the caller upholds the raw execution contract.
    unsafe {
        try_ffi!(T::iamin(
            context.as_raw(),
            n,
            x.as_device_ptr(),
            incx,
            result.as_mut_ptr(),
        ))
    }
}

/// Computes the one-based index of the first element of `x` with the smallest magnitude into `result` and waits for it.
///
/// # Errors
///
/// Returns an error if the handle's stream is capturing, waiting fails, or for the reasons [`iamin_async`] lists.
pub fn iamin<T: Element>(
    context: &Context,
    n: usize,
    x: &(impl DeviceRange<T> + ?Sized),
    incx: usize,
    result: ScalarMut<'_, i64>,
) -> Result<()> {
    // SAFETY: the scope waits for the handle's stream before the borrowed operands are released.
    context.sync_scope(|| unsafe { iamin_async(context, n, x, incx, result) })
}

/// Enqueues the Givens rotation `[x, y] = [c * x + s * y, -conj(s) * x + c * y]`.
///
/// # Safety
///
/// The caller upholds the [raw execution contract](crate#raw-execution).
///
/// # Errors
///
/// Returns an error if a vector is too small, an increment is zero, a value exceeds the native integer range, the scalars use different pointer modes, an operand belongs to another CUDA context, or cuBLAS rejects the call.
pub unsafe fn rot_async<T: Element>(
    context: &Context,
    n: usize,
    x: &mut (impl DeviceOutput<T> + ?Sized),
    incx: usize,
    y: &mut (impl DeviceOutput<T> + ?Sized),
    incy: usize,
    c: Scalar<'_, T::Real>,
    s: Scalar<'_, T>,
) -> Result<()> {
    context.bind()?;
    ensure_vector("x", x.len(), n, incx)?;
    ensure_vector("y", y.len(), n, incy)?;
    let (n, incx, incy) = (
        to_i64(n, "n")?,
        to_i64(incx, "incx")?,
        to_i64(incy, "incy")?,
    );
    context.prepare_operands(&[x.region(), y.region()], &[c.region(), s.region()])?;
    context.use_pointer_mode(&[c.pointer_mode(), s.pointer_mode()])?;
    // SAFETY: the vectors hold the strided values in this context, ordered after their producers, and `c` has the element's real type; the caller upholds the raw execution contract.
    unsafe {
        try_ffi!(T::rot(
            context.as_raw(),
            n,
            x.as_mut_ptr(),
            incx,
            y.as_mut_ptr(),
            incy,
            c.ptr().cast(),
            s.ptr(),
        ))
    }
}

/// Applies the Givens rotation described by `c` and `s` to `x` and `y` and waits for it.
///
/// # Errors
///
/// Returns an error if the handle's stream is capturing, waiting fails, or for the reasons [`rot_async`] lists.
pub fn rot<T: Element>(
    context: &Context,
    n: usize,
    x: &mut (impl DeviceOutput<T> + ?Sized),
    incx: usize,
    y: &mut (impl DeviceOutput<T> + ?Sized),
    incy: usize,
    c: Scalar<'_, T::Real>,
    s: Scalar<'_, T>,
) -> Result<()> {
    // SAFETY: the scope waits for the handle's stream before the borrowed operands are released.
    context.sync_scope(|| unsafe { rot_async(context, n, x, incx, y, incy, c, s) })
}

/// Enqueues the construction of the Givens rotation that zeroes `b`, overwriting `a` with `r` and writing `c` and `s`.
///
/// # Safety
///
/// The caller upholds the [raw execution contract](crate#raw-execution).
///
/// # Errors
///
/// Returns an error if the scalars use different pointer modes, a scalar belongs to another CUDA context, or cuBLAS rejects the call.
pub unsafe fn rotg_async<T: Element>(
    context: &Context,
    mut a: ScalarMut<'_, T>,
    mut b: ScalarMut<'_, T>,
    mut c: ScalarMut<'_, T::Real>,
    mut s: ScalarMut<'_, T>,
) -> Result<()> {
    context.bind()?;
    context.prepare_operands(&[], &[a.region(), b.region(), c.region(), s.region()])?;
    context.use_pointer_mode(&[
        a.pointer_mode(),
        b.pointer_mode(),
        c.pointer_mode(),
        s.pointer_mode(),
    ])?;
    // SAFETY: the scalars share one pointer mode, belong to this context, and `c` has the element's real type; the caller upholds the raw execution contract.
    unsafe {
        try_ffi!(T::rotg(
            context.as_raw(),
            a.as_mut_ptr(),
            b.as_mut_ptr(),
            c.as_mut_ptr().cast(),
            s.as_mut_ptr(),
        ))
    }
}

/// Constructs the Givens rotation that zeroes `b` and waits for it.
///
/// # Errors
///
/// Returns an error if the handle's stream is capturing, waiting fails, or for the reasons [`rotg_async`] lists.
pub fn rotg<T: Element>(
    context: &Context,
    a: ScalarMut<'_, T>,
    b: ScalarMut<'_, T>,
    c: ScalarMut<'_, T::Real>,
    s: ScalarMut<'_, T>,
) -> Result<()> {
    // SAFETY: the scope waits for the handle's stream before the borrowed operands are released.
    context.sync_scope(|| unsafe { rotg_async(context, a, b, c, s) })
}

/// Enqueues the modified Givens transformation that `param` describes on `x` and `y`.
///
/// # Safety
///
/// The caller upholds the [raw execution contract](crate#raw-execution).
///
/// # Errors
///
/// Returns an error if a vector is too small, an increment is zero, a value exceeds the native integer range, an operand belongs to another CUDA context, or cuBLAS rejects the call.
pub unsafe fn rotm_async<T: RealElement>(
    context: &Context,
    n: usize,
    x: &mut (impl DeviceOutput<T> + ?Sized),
    incx: usize,
    y: &mut (impl DeviceOutput<T> + ?Sized),
    incy: usize,
    param: Scalar<'_, [T; 5]>,
) -> Result<()> {
    context.bind()?;
    ensure_vector("x", x.len(), n, incx)?;
    ensure_vector("y", y.len(), n, incy)?;
    let (n, incx, incy) = (
        to_i64(n, "n")?,
        to_i64(incx, "incx")?,
        to_i64(incy, "incy")?,
    );
    context.prepare_operands(&[x.region(), y.region()], &[param.region()])?;
    context.use_pointer_mode(&[param.pointer_mode()])?;
    // SAFETY: the vectors hold the strided values in this context, ordered after their producers, and `param` holds five values; the caller upholds the raw execution contract.
    unsafe {
        try_ffi!(T::rotm(
            context.as_raw(),
            n,
            x.as_mut_ptr(),
            incx,
            y.as_mut_ptr(),
            incy,
            param.ptr().cast(),
        ))
    }
}

/// Applies the modified Givens transformation that `param` describes to `x` and `y` and waits for it.
///
/// # Errors
///
/// Returns an error if the handle's stream is capturing, waiting fails, or for the reasons [`rotm_async`] lists.
pub fn rotm<T: RealElement>(
    context: &Context,
    n: usize,
    x: &mut (impl DeviceOutput<T> + ?Sized),
    incx: usize,
    y: &mut (impl DeviceOutput<T> + ?Sized),
    incy: usize,
    param: Scalar<'_, [T; 5]>,
) -> Result<()> {
    // SAFETY: the scope waits for the handle's stream before the borrowed operands are released.
    context.sync_scope(|| unsafe { rotm_async(context, n, x, incx, y, incy, param) })
}

/// Enqueues the construction of the modified Givens transformation that zeroes the second component of `(sqrt(d1) * x1, sqrt(d2) * y1)`.
///
/// # Safety
///
/// The caller upholds the [raw execution contract](crate#raw-execution).
///
/// # Errors
///
/// Returns an error if the scalars use different pointer modes, a scalar belongs to another CUDA context, or cuBLAS rejects the call.
pub unsafe fn rotmg_async<T: RealElement>(
    context: &Context,
    mut d1: ScalarMut<'_, T>,
    mut d2: ScalarMut<'_, T>,
    mut x1: ScalarMut<'_, T>,
    y1: Scalar<'_, T>,
    mut param: ScalarMut<'_, [T; 5]>,
) -> Result<()> {
    context.bind()?;
    context.prepare_operands(
        &[],
        &[
            d1.region(),
            d2.region(),
            x1.region(),
            y1.region(),
            param.region(),
        ],
    )?;
    context.use_pointer_mode(&[
        d1.pointer_mode(),
        d2.pointer_mode(),
        x1.pointer_mode(),
        y1.pointer_mode(),
        param.pointer_mode(),
    ])?;
    // SAFETY: the scalars share one pointer mode, belong to this context, and `param` holds five values; the caller upholds the raw execution contract.
    unsafe {
        try_ffi!(T::rotmg(
            context.as_raw(),
            d1.as_mut_ptr(),
            d2.as_mut_ptr(),
            x1.as_mut_ptr(),
            y1.ptr(),
            param.as_mut_ptr().cast(),
        ))
    }
}

/// Constructs a modified Givens transformation and waits for it.
///
/// # Errors
///
/// Returns an error if the handle's stream is capturing, waiting fails, or for the reasons [`rotmg_async`] lists.
pub fn rotmg<T: RealElement>(
    context: &Context,
    d1: ScalarMut<'_, T>,
    d2: ScalarMut<'_, T>,
    x1: ScalarMut<'_, T>,
    y1: Scalar<'_, T>,
    param: ScalarMut<'_, [T; 5]>,
) -> Result<()> {
    // SAFETY: the scope waits for the handle's stream before the borrowed operands are released.
    context.sync_scope(|| unsafe { rotmg_async(context, d1, d2, x1, y1, param) })
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use singe_cuda::memory::DeviceMemory;

    use super::*;
    use crate::testing::setup_context;

    #[test]
    fn test_scal_scales_vector() -> Result<()> {
        let test = setup_context()?;
        let context = test.context();

        let mut x = DeviceMemory::from_slice(context.cuda_context(), &[1.0_f32, 2.0, 3.0, 4.0])?;
        scal(context, 4, Scalar::host(&2.5), &mut x, 1)?;

        let result = x.copy_to_host_vec()?;
        assert_eq!(result, vec![2.5, 5.0, 7.5, 10.0]);

        Ok(())
    }

    #[test]
    fn test_rot_rotates_vectors() -> Result<()> {
        let test = setup_context()?;
        let context = test.context();

        let mut x = DeviceMemory::from_slice(context.cuda_context(), &[1.0_f64, 2.0, 3.0, 4.0])?;
        let mut y = DeviceMemory::from_slice(context.cuda_context(), &[5.0_f64, 6.0, 7.0, 8.0])?;
        rot(
            context,
            4,
            &mut x,
            1,
            &mut y,
            1,
            Scalar::host(&2.1),
            Scalar::host(&1.2),
        )?;

        let x_result = x.copy_to_host_vec()?;
        let y_result = y.copy_to_host_vec()?;
        let x_expected = [8.1, 11.4, 14.7, 18.0];
        let y_expected = [9.3, 10.2, 11.1, 12.0];

        for (actual, expected) in x_result.iter().zip(x_expected) {
            assert!((actual - expected).abs() < 1.0e-12);
        }
        for (actual, expected) in y_result.iter().zip(y_expected) {
            assert!((actual - expected).abs() < 1.0e-12);
        }

        Ok(())
    }
}
