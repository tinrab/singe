use std::ptr;

use singe_cuda::{data_type::DataType, memory::DeviceMemory};

use crate::{
    context::Context,
    error::{Error, Result},
    sys, try_ffi,
    utility::to_i32,
};

/// Supports the 64-bit integer interface.
///
/// Scales vector `x` by scalar $\alpha$ and overwrites it with the result.
/// The performed operation is $\mathbf{x}\lbrack j\rbrack = \alpha \times \mathbf{x}\lbrack j\rbrack$ for $i = 1,\ldots,n$ and $j = 1 + \left( {i - 1} \right)\cdot \text{incx}$.
/// The formula uses the BLAS 1-based indexing convention.
///
/// See NETLIB documentation:
///
/// [sscal()](https://www.netlib.org/blas/sscal.f), [dscal()](https://www.netlib.org/blas/dscal.f), [csscal()](https://www.netlib.org/blas/csscal.f), [cscal()](https://www.netlib.org/blas/cscal.f), [zdscal()](https://www.netlib.org/blas/zdscal.f), [zscal()](https://www.netlib.org/blas/zscal.f).
///
/// # Errors
///
/// Returns an error if the context cannot be used with host pointer mode,
/// `incx` is zero or too large for cuBLAS, the vector length is invalid, or
/// cuBLAS rejects the operation.
pub fn sscal(ctx: &Context, alpha: &f32, x: &mut DeviceMemory<f32>, incx: usize) -> Result<()> {
    ctx.bind()?;
    ctx.require_host_pointer_mode()?;
    if incx == 0 {
        return Err(Error::InvalidIncrement);
    }

    let n = vector_len(x.len(), incx)?;
    if n == 0 {
        return Ok(());
    }

    let incx = to_i32(incx, "incx")?;

    unsafe {
        try_ffi!(sys::cublasSscal_v2(
            ctx.as_raw(),
            n,
            alpha,
            x.as_mut_ptr(),
            incx,
        ))?;
    }

    Ok(())
}

/// Supports the 64-bit integer interface.
///
/// Generalizes the typed cuBLAS norm routines so input data, output data, and compute type can
/// be specified independently.
///
/// Computes the Euclidean norm of vector `x`.
/// The code uses a multiphase model of accumulation to avoid intermediate underflow and overflow, with the result being equivalent to $\sqrt{\sum\_{i = 1}^{n}\left( {\mathbf{x}\lbrack j\rbrack \times \mathbf{x}\lbrack j\rbrack} \right)}$ where $j = 1 + \left( {i - 1} \right)\cdot \text{incx}$ in exact arithmetic.
/// The formula uses the BLAS 1-based indexing convention.
///
/// See NETLIB documentation:
///
/// [snrm2()](https://www.netlib.org/blas/snrm2.f90), [dnrm2()](https://www.netlib.org/blas/dnrm2.f90), [scnrm2()](https://www.netlib.org/blas/scnrm2.f90), [dznrm2()](https://www.netlib.org/blas/dznrm2.f90).
///
/// # Errors
///
/// Returns an error if the context cannot be used with host pointer mode,
/// `incx` is zero or too large for cuBLAS, `x` is too small for the requested
/// vector shape, a type combination is unsupported, cuBLAS cannot allocate its
/// reduction buffer, or cuBLAS rejects or fails the operation.
pub fn nrm2_ex<TX, TResult>(
    ctx: &Context,
    x: &DeviceMemory<TX>,
    x_type: DataType,
    incx: usize,
    result: &mut TResult,
    result_type: DataType,
    execution_type: DataType,
) -> Result<()> {
    ctx.bind()?;
    ctx.require_host_pointer_mode()?;
    if incx == 0 {
        return Err(Error::InvalidIncrement);
    }

    let n = vector_len(x.len(), incx)?;
    if n == 0 {
        return Ok(());
    }

    let incx = to_i32(incx, "incx")?;

    unsafe {
        try_ffi!(sys::cublasNrm2Ex(
            ctx.as_raw(),
            n,
            x.as_ptr() as _,
            x_type.into(),
            incx,
            ptr::from_mut(result) as _,
            result_type.into(),
            execution_type.into(),
        ))?;
    }

    Ok(())
}

/// Supports the 64-bit integer interface.
///
/// Generalizes the typed cuBLAS dot-product routines so input data, output data,
/// and compute type can be specified independently.
/// The `dotc` variants compute conjugated dot products, and the `dotu` variants compute
/// unconjugated dot products.
///
/// Computes the dot product of vectors `x` and `y`.
/// The result is $\sum\_{i = 1}^{n}\left( {\mathbf{x}\lbrack k\rbrack \times \mathbf{y}\lbrack j\rbrack} \right)$ where $k = 1 + \left( {i - 1} \right)\cdot \text{incx}$ and $j = 1 + \left( {i - 1} \right)\cdot \text{incy}$.
/// For conjugated dot products, the element of vector `x` is conjugated.
/// The formula uses the BLAS 1-based indexing convention.
///
/// See NETLIB documentation:
///
/// [sdot()](https://www.netlib.org/blas/sdot.f), [ddot()](https://www.netlib.org/blas/ddot.f), [cdotu()](https://www.netlib.org/blas/cdotu.f), [cdotc()](https://www.netlib.org/blas/cdotc.f), [zdotu()](https://www.netlib.org/blas/zdotu.f), [zdotc()](https://www.netlib.org/blas/zdotc.f).
///
/// # Errors
///
/// Returns an error if the context cannot be used with host pointer mode,
/// `incx` or `incy` is zero or too large for cuBLAS, `x` or `y` is too small
/// for the requested vector shape, a type combination is unsupported, cuBLAS
/// cannot allocate its reduction buffer, or cuBLAS rejects or fails the
/// operation.
pub fn dot_ex<TX, TY, TResult>(
    ctx: &Context,
    x: &DeviceMemory<TX>,
    x_type: DataType,
    incx: usize,
    y: &DeviceMemory<TY>,
    y_type: DataType,
    incy: usize,
    result: &mut TResult,
    result_type: DataType,
    execution_type: DataType,
) -> Result<()> {
    ctx.bind()?;
    ctx.require_host_pointer_mode()?;
    if incx == 0 || incy == 0 {
        return Err(Error::InvalidIncrement);
    }

    let n = vector_binary_len(x.len(), incx, y.len(), incy)?;
    if n == 0 {
        return Ok(());
    }

    let incx = to_i32(incx, "incx")?;
    let incy = to_i32(incy, "incy")?;

    unsafe {
        try_ffi!(sys::cublasDotEx(
            ctx.as_raw(),
            n,
            x.as_ptr() as _,
            x_type.into(),
            incx,
            y.as_ptr() as _,
            y_type.into(),
            incy,
            ptr::from_mut(result) as _,
            result_type.into(),
            execution_type.into(),
        ))?;
    }

    Ok(())
}

pub fn dotc_ex<TX, TY, TResult>(
    ctx: &Context,
    x: &DeviceMemory<TX>,
    x_type: DataType,
    incx: usize,
    y: &DeviceMemory<TY>,
    y_type: DataType,
    incy: usize,
    result: &mut TResult,
    result_type: DataType,
    execution_type: DataType,
) -> Result<()> {
    ctx.bind()?;
    ctx.require_host_pointer_mode()?;
    if incx == 0 || incy == 0 {
        return Err(Error::InvalidIncrement);
    }

    let n = vector_binary_len(x.len(), incx, y.len(), incy)?;
    if n == 0 {
        return Ok(());
    }

    let incx = to_i32(incx, "incx")?;
    let incy = to_i32(incy, "incy")?;

    unsafe {
        try_ffi!(sys::cublasDotcEx(
            ctx.as_raw(),
            n,
            x.as_ptr() as _,
            x_type.into(),
            incx,
            y.as_ptr() as _,
            y_type.into(),
            incy,
            ptr::from_mut(result) as _,
            result_type.into(),
            execution_type.into(),
        ))?;
    }

    Ok(())
}

/// Supports the 64-bit integer interface.
///
/// Applies the Givens rotation matrix to vectors `x` and `y`.
///
/// # Errors
///
/// Returns an error if the context cannot be used with host pointer mode,
/// either increment is zero or too large for cuBLAS, the vector lengths are
/// incompatible, or cuBLAS rejects the operation.
pub fn drot(
    ctx: &Context,
    x: &mut DeviceMemory<f64>,
    incx: usize,
    y: &mut DeviceMemory<f64>,
    incy: usize,
    c: &f64,
    s: &f64,
) -> Result<()> {
    ctx.bind()?;
    ctx.require_host_pointer_mode()?;
    if incx == 0 || incy == 0 {
        return Err(Error::InvalidIncrement);
    }

    let n = vector_binary_len(x.len(), incx, y.len(), incy)?;
    if n == 0 {
        return Ok(());
    }

    let incx = to_i32(incx, "incx")?;
    let incy = to_i32(incy, "incy")?;

    unsafe {
        try_ffi!(sys::cublasDrot_v2(
            ctx.as_raw(),
            n,
            x.as_mut_ptr(),
            incx,
            y.as_mut_ptr(),
            incy,
            c,
            s,
        ))?;
    }

    Ok(())
}

/// Supports the 64-bit integer interface.
///
/// Extends the typed cuBLAS rotation routines so vector and scalar types can be chosen independently.
///
/// # Errors
///
/// Returns an error if the context cannot be used with host pointer mode,
/// either increment is zero or too large for cuBLAS, the vector lengths are
/// incompatible, the supplied data type combination is not supported, or
/// cuBLAS rejects the operation.
pub fn rot_ex<TX, TY, TCS>(
    ctx: &Context,
    x: &mut DeviceMemory<TX>,
    x_type: DataType,
    incx: usize,
    y: &mut DeviceMemory<TY>,
    y_type: DataType,
    incy: usize,
    c: &TCS,
    s: &TCS,
    cs_type: DataType,
    execution_type: DataType,
) -> Result<()> {
    ctx.bind()?;
    ctx.require_host_pointer_mode()?;
    if incx == 0 || incy == 0 {
        return Err(Error::InvalidIncrement);
    }

    let n = vector_binary_len(x.len(), incx, y.len(), incy)?;
    if n == 0 {
        return Ok(());
    }

    let incx = to_i32(incx, "incx")?;
    let incy = to_i32(incy, "incy")?;

    unsafe {
        try_ffi!(sys::cublasRotEx(
            ctx.as_raw(),
            n,
            x.as_mut_ptr() as _,
            x_type.into(),
            incx,
            y.as_mut_ptr() as _,
            y_type.into(),
            incy,
            ptr::from_ref(c) as _,
            ptr::from_ref(s) as _,
            cs_type.into(),
            execution_type.into(),
        ))?;
    }

    Ok(())
}

/// Supports the 64-bit integer interface.
///
/// Constructs the Givens rotation matrix parameters for `a` and `b`.
///
/// # Errors
///
/// Returns an error if the context cannot be used with host pointer mode or
/// cuBLAS rejects the operation.
pub fn drotg(ctx: &Context, a: &mut f64, b: &mut f64, c: &mut f64, s: &mut f64) -> Result<()> {
    ctx.bind()?;
    ctx.require_host_pointer_mode()?;

    unsafe {
        try_ffi!(sys::cublasDrotg_v2(
            ctx.as_raw(),
            ptr::from_mut(a),
            ptr::from_mut(b),
            ptr::from_mut(c),
            ptr::from_mut(s),
        ))?;
    }

    Ok(())
}

/// Supports the 64-bit integer interface.
///
/// Applies the modified Givens transformation described by `param` to vectors `x` and `y`.
///
/// # Errors
///
/// Returns an error if the context cannot be used with host pointer mode,
/// either increment is zero or too large for cuBLAS, the vector lengths are
/// incompatible, or cuBLAS rejects the operation.
pub fn drotm(
    ctx: &Context,
    x: &mut DeviceMemory<f64>,
    incx: usize,
    y: &mut DeviceMemory<f64>,
    incy: usize,
    param: &[f64; 5],
) -> Result<()> {
    ctx.bind()?;
    ctx.require_host_pointer_mode()?;
    if incx == 0 || incy == 0 {
        return Err(Error::InvalidIncrement);
    }

    let n = vector_binary_len(x.len(), incx, y.len(), incy)?;
    if n == 0 {
        return Ok(());
    }

    let incx = to_i32(incx, "incx")?;
    let incy = to_i32(incy, "incy")?;

    unsafe {
        try_ffi!(sys::cublasDrotm_v2(
            ctx.as_raw(),
            n,
            x.as_mut_ptr(),
            incx,
            y.as_mut_ptr(),
            incy,
            param.as_ptr(),
        ))?;
    }

    Ok(())
}

/// Supports the 64-bit integer interface.
///
/// Constructs the modified Givens transformation parameters in `param`.
///
/// # Errors
///
/// Returns an error if the context cannot be used with host pointer mode or
/// cuBLAS rejects the operation.
pub fn drotmg(
    ctx: &Context,
    d1: &mut f64,
    d2: &mut f64,
    x1: &mut f64,
    y1: &f64,
    param: &mut [f64; 5],
) -> Result<()> {
    ctx.bind()?;
    ctx.require_host_pointer_mode()?;

    unsafe {
        try_ffi!(sys::cublasDrotmg_v2(
            ctx.as_raw(),
            ptr::from_mut(d1),
            ptr::from_mut(d2),
            ptr::from_mut(x1),
            ptr::from_ref(y1),
            param.as_mut_ptr(),
        ))?;
    }

    Ok(())
}

/// Supports the 64-bit integer interface.
///
/// Scales vector `x` by scalar $\alpha$ and overwrites it with the result.
/// The performed operation is $\mathbf{x}\lbrack j\rbrack = \alpha \times \mathbf{x}\lbrack j\rbrack$ for $i = 1,\ldots,n$ and $j = 1 + \left( {i - 1} \right)\cdot \text{incx}$.
/// The formula uses the BLAS 1-based indexing convention.
///
/// See NETLIB documentation:
///
/// [sscal()](https://www.netlib.org/blas/sscal.f), [dscal()](https://www.netlib.org/blas/dscal.f), [csscal()](https://www.netlib.org/blas/csscal.f), [cscal()](https://www.netlib.org/blas/cscal.f), [zdscal()](https://www.netlib.org/blas/zdscal.f), [zscal()](https://www.netlib.org/blas/zscal.f).
///
/// # Errors
///
/// Returns an error if the context cannot be used with host pointer mode,
/// `incx` is zero or too large for cuBLAS, `x` is too small for the requested
/// vector shape, a type combination is unsupported, or cuBLAS rejects or fails
/// the operation.
pub fn scal_ex<TAlpha, TX>(
    ctx: &Context,
    alpha: &TAlpha,
    alpha_type: DataType,
    x: &mut DeviceMemory<TX>,
    x_type: DataType,
    incx: usize,
    execution_type: DataType,
) -> Result<()> {
    ctx.bind()?;
    ctx.require_host_pointer_mode()?;
    if incx == 0 {
        return Err(Error::InvalidIncrement);
    }

    let n = vector_len(x.len(), incx)?;
    if n == 0 {
        return Ok(());
    }

    let incx = to_i32(incx, "incx")?;

    unsafe {
        try_ffi!(sys::cublasScalEx(
            ctx.as_raw(),
            n,
            ptr::from_ref(alpha) as _,
            alpha_type.into(),
            x.as_mut_ptr() as _,
            x_type.into(),
            incx,
            execution_type.into(),
        ))?;
    }

    Ok(())
}

/// Supports the 64-bit integer interface.
///
/// Generalizes the typed cuBLAS AXPY routines so input data, output data, and compute type can
/// be specified independently.
///
/// Multiplies vector `x` by scalar $\alpha$ and adds it to vector `y`, overwriting
/// `y` with the result.
/// The performed operation is $\mathbf{y}\lbrack j\rbrack = \alpha \times \mathbf{x}\lbrack k\rbrack + \mathbf{y}\lbrack j\rbrack$ for $i = 1,\ldots,n$, $k = 1 + \left( {i - 1} \right)\cdot \text{incx}$ and $j = 1 + \left( {i - 1} \right)\cdot \text{incy}$.
/// The formula uses the BLAS 1-based indexing convention.
///
/// See NETLIB documentation:
///
/// [saxpy()](https://www.netlib.org/blas/saxpy.f), [daxpy()](https://www.netlib.org/blas/daxpy.f), [caxpy()](https://www.netlib.org/blas/caxpy.f), [zaxpy()](https://www.netlib.org/blas/zaxpy.f).
///
/// # Errors
///
/// Returns an error if the context cannot be used with host pointer mode,
/// `incx` or `incy` is zero or too large for cuBLAS, `x` or `y` is too small
/// for the requested vector shape, a type combination is unsupported, or
/// cuBLAS rejects or fails the operation.
pub fn axpy_ex<TAlpha, TX, TY>(
    ctx: &Context,
    alpha: &TAlpha,
    alpha_type: DataType,
    x: &DeviceMemory<TX>,
    x_type: DataType,
    incx: usize,
    y: &mut DeviceMemory<TY>,
    y_type: DataType,
    incy: usize,
    execution_type: DataType,
) -> Result<()> {
    ctx.bind()?;
    ctx.require_host_pointer_mode()?;
    if incx == 0 || incy == 0 {
        return Err(Error::InvalidIncrement);
    }

    let n = vector_binary_len(x.len(), incx, y.len(), incy)?;
    if n == 0 {
        return Ok(());
    }

    let incx = to_i32(incx, "incx")?;
    let incy = to_i32(incy, "incy")?;

    unsafe {
        try_ffi!(sys::cublasAxpyEx(
            ctx.as_raw(),
            n,
            ptr::from_ref(alpha) as _,
            alpha_type.into(),
            x.as_ptr() as _,
            x_type.into(),
            incx,
            y.as_mut_ptr() as _,
            y_type.into(),
            incy,
            execution_type.into(),
        ))?;
    }

    Ok(())
}

pub fn copy_ex<TX, TY>(
    ctx: &Context,
    x: &DeviceMemory<TX>,
    x_type: DataType,
    incx: usize,
    y: &mut DeviceMemory<TY>,
    y_type: DataType,
    incy: usize,
) -> Result<()> {
    ctx.bind()?;
    if incx == 0 || incy == 0 {
        return Err(Error::InvalidIncrement);
    }

    let n = vector_binary_len(x.len(), incx, y.len(), incy)?;
    if n == 0 {
        return Ok(());
    }

    let incx = to_i32(incx, "incx")?;
    let incy = to_i32(incy, "incy")?;

    unsafe {
        try_ffi!(sys::cublasCopyEx(
            ctx.as_raw(),
            n,
            x.as_ptr() as _,
            x_type.into(),
            incx,
            y.as_mut_ptr() as _,
            y_type.into(),
            incy,
        ))?;
    }

    Ok(())
}

pub fn swap_ex<TX, TY>(
    ctx: &Context,
    x: &mut DeviceMemory<TX>,
    x_type: DataType,
    incx: usize,
    y: &mut DeviceMemory<TY>,
    y_type: DataType,
    incy: usize,
) -> Result<()> {
    ctx.bind()?;
    if incx == 0 || incy == 0 {
        return Err(Error::InvalidIncrement);
    }

    let n = vector_binary_len(x.len(), incx, y.len(), incy)?;
    if n == 0 {
        return Ok(());
    }

    let incx = to_i32(incx, "incx")?;
    let incy = to_i32(incy, "incy")?;

    unsafe {
        try_ffi!(sys::cublasSwapEx(
            ctx.as_raw(),
            n,
            x.as_mut_ptr() as _,
            x_type.into(),
            incx,
            y.as_mut_ptr() as _,
            y_type.into(),
            incy,
        ))?;
    }

    Ok(())
}

pub fn iamax_ex<TX>(
    ctx: &Context,
    x: &DeviceMemory<TX>,
    x_type: DataType,
    incx: usize,
) -> Result<i32> {
    ctx.bind()?;
    ctx.require_host_pointer_mode()?;
    if incx == 0 {
        return Err(Error::InvalidIncrement);
    }

    let n = vector_len(x.len(), incx)?;
    if n == 0 {
        return Ok(0);
    }

    let incx = to_i32(incx, "incx")?;
    let mut result = 0;

    unsafe {
        try_ffi!(sys::cublasIamaxEx(
            ctx.as_raw(),
            n,
            x.as_ptr() as _,
            x_type.into(),
            incx,
            &raw mut result,
        ))?;
    }

    Ok(result)
}

pub fn iamin_ex<TX>(
    ctx: &Context,
    x: &DeviceMemory<TX>,
    x_type: DataType,
    incx: usize,
) -> Result<i32> {
    ctx.bind()?;
    ctx.require_host_pointer_mode()?;
    if incx == 0 {
        return Err(Error::InvalidIncrement);
    }

    let n = vector_len(x.len(), incx)?;
    if n == 0 {
        return Ok(0);
    }

    let incx = to_i32(incx, "incx")?;
    let mut result = 0;

    unsafe {
        try_ffi!(sys::cublasIaminEx(
            ctx.as_raw(),
            n,
            x.as_ptr() as _,
            x_type.into(),
            incx,
            &raw mut result,
        ))?;
    }

    Ok(result)
}

pub fn asum_ex<TX, TResult>(
    ctx: &Context,
    x: &DeviceMemory<TX>,
    x_type: DataType,
    incx: usize,
    result: &mut TResult,
    result_type: DataType,
    execution_type: DataType,
) -> Result<()> {
    ctx.bind()?;
    ctx.require_host_pointer_mode()?;
    if incx == 0 {
        return Err(Error::InvalidIncrement);
    }

    let n = vector_len(x.len(), incx)?;
    if n == 0 {
        return Ok(());
    }

    let incx = to_i32(incx, "incx")?;

    unsafe {
        try_ffi!(sys::cublasAsumEx(
            ctx.as_raw(),
            n,
            x.as_ptr() as _,
            x_type.into(),
            incx,
            ptr::from_mut(result) as _,
            result_type.into(),
            execution_type.into(),
        ))?;
    }

    Ok(())
}

fn vector_len(length: usize, inc: usize) -> Result<i32> {
    if length == 0 {
        return Ok(0);
    }

    let n = 1 + (length - 1) / inc;
    i32::try_from(n).map_err(|_| Error::OutOfRange { name: "n".into() })
}

fn vector_binary_len(x_length: usize, incx: usize, y_length: usize, incy: usize) -> Result<i32> {
    let x_n = vector_len(x_length, incx)?;
    let y_n = vector_len(y_length, incy)?;

    if x_n != y_n {
        return Err(Error::InvalidVectorShape);
    }

    Ok(x_n)
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use super::*;
    use crate::testing::setup_context;

    #[test]
    fn test_sscal_scales_vector() -> Result<()> {
        let ctx = setup_context()?;

        let mut x = DeviceMemory::from_slice(&[1.0_f32, 2.0, 3.0, 4.0])?;
        sscal(&ctx, &2.5, &mut x, 1)?;

        let result = x.copy_to_host_vec()?;
        assert_eq!(result, vec![2.5, 5.0, 7.5, 10.0]);

        Ok(())
    }

    #[test]
    fn test_drot_rotates_vectors() -> Result<()> {
        let ctx = setup_context()?;

        let mut x = DeviceMemory::from_slice(&[1.0_f64, 2.0, 3.0, 4.0])?;
        let mut y = DeviceMemory::from_slice(&[5.0_f64, 6.0, 7.0, 8.0])?;
        drot(&ctx, &mut x, 1, &mut y, 1, &2.1, &1.2)?;

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
