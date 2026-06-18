use std::mem::size_of;

use singe_cuda::{
    stream::{Stream, StreamScope},
    view::{DeviceBuffer, DeviceBufferMut, DeviceRepr, HostBuffer, HostBufferMut},
};

use crate::{
    context::Context,
    error::{Error, Result},
    sys, try_ffi,
    utility::{required_matrix_len, required_vector_len, to_i32},
};

/// Supports the 64-bit integer interface.
///
/// Copies `n` elements from host vector `x` to device vector `y`.
///
/// Elements in both vectors have size `size_of::<T>()`. The spacing between
/// consecutive elements is given by `incx` for `x` and by `incy` for `y`.
///
/// In column-major matrices, an increment of `1` accesses a column or partial
/// column, while an increment equal to the leading dimension accesses a row or
/// partial row.
///
/// # Errors
///
/// Returns an error if the context cannot be bound, either increment is zero,
/// the host or device storage is too small, an argument cannot be represented by
/// cuBLAS, or cuBLAS cannot access the GPU memory.
pub fn copy_vector_to_device<T, X, Y>(
    ctx: &Context,
    n: usize,
    x: &X,
    incx: usize,
    y: &mut Y,
    incy: usize,
) -> Result<()>
where
    T: DeviceRepr,
    X: HostBuffer<T> + ?Sized,
    Y: DeviceBufferMut<T> + ?Sized,
{
    ctx.bind()?;
    if n == 0 {
        return Ok(());
    }
    if incx == 0 || incy == 0 {
        return Err(Error::InvalidIncrement);
    }

    let required_host = required_vector_len(n, incx)?;
    let required_device = required_vector_len(n, incy)?;
    if x.len() < required_host || y.len() < required_device {
        return Err(Error::InvalidVectorShape);
    }

    let n = to_i32(n, "n")?;
    let elem_size = to_i32(size_of::<T>(), "elem_size")?;
    let incx = to_i32(incx, "incx")?;
    let incy = to_i32(incy, "incy")?;

    unsafe {
        try_ffi!(sys::cublasSetVector(
            n,
            elem_size,
            x.as_host_ptr().cast(),
            incx,
            y.as_device_mut_ptr().cast(),
            incy,
        ))?;
    }

    Ok(())
}

/// Supports the 64-bit integer interface.
///
/// Copies `n` elements from device vector `x` to host vector `y`.
///
/// Elements in both vectors have size `size_of::<T>()`.
/// The storage spacing between consecutive elements is given by `incx` for the source vector and `incy` for the destination vector `y`.
///
/// In column-major matrices, an increment of `1` accesses a column or partial
/// column, while an increment equal to the leading dimension accesses a row or
/// partial row.
///
/// # Errors
///
/// Returns an error if the context cannot be bound, either increment is zero,
/// the device or host storage is too small, an argument cannot be represented by
/// cuBLAS, or cuBLAS cannot access the GPU memory.
pub fn copy_vector_to_host<T, X, Y>(
    ctx: &Context,
    n: usize,
    x: &X,
    incx: usize,
    y: &mut Y,
    incy: usize,
) -> Result<()>
where
    T: DeviceRepr,
    X: DeviceBuffer<T> + ?Sized,
    Y: HostBufferMut<T> + ?Sized,
{
    ctx.bind()?;
    if n == 0 {
        return Ok(());
    }
    if incx == 0 || incy == 0 {
        return Err(Error::InvalidIncrement);
    }

    let required_device = required_vector_len(n, incx)?;
    let required_host = required_vector_len(n, incy)?;
    if x.len() < required_device || y.len() < required_host {
        return Err(Error::InvalidVectorShape);
    }

    let n = to_i32(n, "n")?;
    let elem_size = to_i32(size_of::<T>(), "elem_size")?;
    let incx = to_i32(incx, "incx")?;
    let incy = to_i32(incy, "incy")?;

    unsafe {
        try_ffi!(sys::cublasGetVector(
            n,
            elem_size,
            x.as_device_ptr().cast(),
            incx,
            y.as_host_mut_ptr().cast(),
            incy,
        ))?;
    }

    Ok(())
}

pub fn copy_vector_to_device_async<'scope, 'env, T, X, Y>(
    ctx: &Context,
    n: usize,
    x: &'env X,
    incx: usize,
    y: &mut Y,
    incy: usize,
    stream: &StreamScope<'scope, 'env>,
) -> Result<()>
where
    T: DeviceRepr,
    X: HostBuffer<T> + ?Sized,
    Y: DeviceBufferMut<T> + ?Sized,
{
    unsafe { copy_vector_to_device_async_unchecked(ctx, n, x, incx, y, incy, stream.stream()) }
}

/// Supports the 64-bit integer interface.
///
/// Asynchronously copies host vector `x` to device vector `y` on `stream`.
///
/// # Safety
///
/// `x` must remain valid until the asynchronous copy on `stream` has completed.
///
/// # Errors
///
/// Returns an error if `stream` does not belong to `ctx`, either increment is
/// zero, the host or device storage is too small, an argument cannot be
/// represented by cuBLAS, or cuBLAS cannot access the GPU memory.
pub unsafe fn copy_vector_to_device_async_unchecked<T, X, Y>(
    ctx: &Context,
    n: usize,
    x: &X,
    incx: usize,
    y: &mut Y,
    incy: usize,
    stream: &Stream,
) -> Result<()>
where
    T: DeviceRepr,
    X: HostBuffer<T> + ?Sized,
    Y: DeviceBufferMut<T> + ?Sized,
{
    ctx.ensure_stream(stream)?;
    if n == 0 {
        return Ok(());
    }
    if incx == 0 || incy == 0 {
        return Err(Error::InvalidIncrement);
    }

    let required_host = required_vector_len(n, incx)?;
    let required_device = required_vector_len(n, incy)?;
    if x.len() < required_host || y.len() < required_device {
        return Err(Error::InvalidVectorShape);
    }

    let n = to_i32(n, "n")?;
    let elem_size = to_i32(size_of::<T>(), "elem_size")?;
    let incx = to_i32(incx, "incx")?;
    let incy = to_i32(incy, "incy")?;

    unsafe {
        try_ffi!(sys::cublasSetVectorAsync(
            n,
            elem_size,
            x.as_host_ptr().cast(),
            incx,
            y.as_device_mut_ptr().cast(),
            incy,
            stream.as_raw(),
        ))?;
    }

    Ok(())
}

pub fn copy_vector_to_host_async<'scope, 'env, T, X, Y>(
    ctx: &Context,
    n: usize,
    x: &X,
    incx: usize,
    y: &'env mut Y,
    incy: usize,
    stream: &StreamScope<'scope, 'env>,
) -> Result<()>
where
    T: DeviceRepr,
    X: DeviceBuffer<T> + ?Sized,
    Y: HostBufferMut<T> + ?Sized,
{
    unsafe { copy_vector_to_host_async_unchecked(ctx, n, x, incx, y, incy, stream.stream()) }
}

/// Supports the 64-bit integer interface.
///
/// Asynchronously copies device vector `x` to host vector `y` on `stream`.
///
/// # Safety
///
/// `y` must remain valid and uniquely writable until the asynchronous copy on
/// `stream` has completed.
///
/// # Errors
///
/// Returns an error if `stream` does not belong to `ctx`, either increment is
/// zero, the device or host storage is too small, an argument cannot be
/// represented by cuBLAS, or cuBLAS cannot access the GPU memory.
pub unsafe fn copy_vector_to_host_async_unchecked<T, X, Y>(
    ctx: &Context,
    n: usize,
    x: &X,
    incx: usize,
    y: &mut Y,
    incy: usize,
    stream: &Stream,
) -> Result<()>
where
    T: DeviceRepr,
    X: DeviceBuffer<T> + ?Sized,
    Y: HostBufferMut<T> + ?Sized,
{
    ctx.ensure_stream(stream)?;
    if n == 0 {
        return Ok(());
    }
    if incx == 0 || incy == 0 {
        return Err(Error::InvalidIncrement);
    }

    let required_device = required_vector_len(n, incx)?;
    let required_host = required_vector_len(n, incy)?;
    if x.len() < required_device || y.len() < required_host {
        return Err(Error::InvalidVectorShape);
    }

    let n = to_i32(n, "n")?;
    let elem_size = to_i32(size_of::<T>(), "elem_size")?;
    let incx = to_i32(incx, "incx")?;
    let incy = to_i32(incy, "incy")?;

    unsafe {
        try_ffi!(sys::cublasGetVectorAsync(
            n,
            elem_size,
            x.as_device_ptr().cast(),
            incx,
            y.as_host_mut_ptr().cast(),
            incy,
            stream.as_raw(),
        ))?;
    }

    Ok(())
}

pub fn copy_matrix_to_device_async<'scope, 'env, T, A, B>(
    ctx: &Context,
    rows: usize,
    cols: usize,
    a: &'env A,
    lda: usize,
    b: &mut B,
    ldb: usize,
    stream: &StreamScope<'scope, 'env>,
) -> Result<()>
where
    T: DeviceRepr,
    A: HostBuffer<T> + ?Sized,
    B: DeviceBufferMut<T> + ?Sized,
{
    unsafe {
        copy_matrix_to_device_async_unchecked(ctx, rows, cols, a, lda, b, ldb, stream.stream())
    }
}

/// Supports the 64-bit integer interface.
///
/// Asynchronously copies a host matrix tile to device memory on `stream`.
///
/// # Safety
///
/// `a` must remain valid until the asynchronous copy on `stream` has completed.
///
/// # Errors
///
/// Returns an error if `stream` does not belong to `ctx`, either leading
/// dimension is invalid, the host or device storage is too small, an argument
/// cannot be represented by cuBLAS, or cuBLAS cannot access the GPU memory.
pub unsafe fn copy_matrix_to_device_async_unchecked<T, A, B>(
    ctx: &Context,
    rows: usize,
    cols: usize,
    a: &A,
    lda: usize,
    b: &mut B,
    ldb: usize,
    stream: &Stream,
) -> Result<()>
where
    T: DeviceRepr,
    A: HostBuffer<T> + ?Sized,
    B: DeviceBufferMut<T> + ?Sized,
{
    ctx.ensure_stream(stream)?;
    if rows == 0 || cols == 0 {
        return Ok(());
    }
    if lda < rows || ldb < rows {
        return Err(Error::InvalidLeadingDimension);
    }
    if a.len() < required_matrix_len(lda, cols)? || b.len() < required_matrix_len(ldb, cols)? {
        return Err(Error::InvalidMatrixShape);
    }

    let rows = to_i32(rows, "rows")?;
    let cols = to_i32(cols, "cols")?;
    let elem_size = to_i32(size_of::<T>(), "elem_size")?;
    let lda = to_i32(lda, "lda")?;
    let ldb = to_i32(ldb, "ldb")?;

    unsafe {
        try_ffi!(sys::cublasSetMatrixAsync(
            rows,
            cols,
            elem_size,
            a.as_host_ptr().cast(),
            lda,
            b.as_device_mut_ptr().cast(),
            ldb,
            stream.as_raw(),
        ))?;
    }

    Ok(())
}

pub fn copy_matrix_to_host_async<'scope, 'env, T, A, B>(
    ctx: &Context,
    rows: usize,
    cols: usize,
    a: &A,
    lda: usize,
    b: &'env mut B,
    ldb: usize,
    stream: &StreamScope<'scope, 'env>,
) -> Result<()>
where
    T: DeviceRepr,
    A: DeviceBuffer<T> + ?Sized,
    B: HostBufferMut<T> + ?Sized,
{
    unsafe { copy_matrix_to_host_async_unchecked(ctx, rows, cols, a, lda, b, ldb, stream.stream()) }
}

/// Supports the 64-bit integer interface.
///
/// Asynchronously copies a device matrix tile to host memory on `stream`.
///
/// # Safety
///
/// `b` must remain valid and uniquely writable until the asynchronous copy on
/// `stream` has completed.
///
/// # Errors
///
/// Returns an error if `stream` does not belong to `ctx`, either leading
/// dimension is invalid, the device or host storage is too small, an argument
/// cannot be represented by cuBLAS, or cuBLAS cannot access the GPU memory.
pub unsafe fn copy_matrix_to_host_async_unchecked<T, A, B>(
    ctx: &Context,
    rows: usize,
    cols: usize,
    a: &A,
    lda: usize,
    b: &mut B,
    ldb: usize,
    stream: &Stream,
) -> Result<()>
where
    T: DeviceRepr,
    A: DeviceBuffer<T> + ?Sized,
    B: HostBufferMut<T> + ?Sized,
{
    ctx.ensure_stream(stream)?;
    if rows == 0 || cols == 0 {
        return Ok(());
    }
    if lda < rows || ldb < rows {
        return Err(Error::InvalidLeadingDimension);
    }
    if a.len() < required_matrix_len(lda, cols)? || b.len() < required_matrix_len(ldb, cols)? {
        return Err(Error::InvalidMatrixShape);
    }

    let rows = to_i32(rows, "rows")?;
    let cols = to_i32(cols, "cols")?;
    let elem_size = to_i32(size_of::<T>(), "elem_size")?;
    let lda = to_i32(lda, "lda")?;
    let ldb = to_i32(ldb, "ldb")?;

    unsafe {
        try_ffi!(sys::cublasGetMatrixAsync(
            rows,
            cols,
            elem_size,
            a.as_device_ptr().cast(),
            lda,
            b.as_host_mut_ptr().cast(),
            ldb,
            stream.as_raw(),
        ))?;
    }

    Ok(())
}

/// Supports the 64-bit integer interface.
///
/// Copies a `rows x cols` tile from host matrix `a` to device matrix `b`.
///
/// Each element has size `size_of::<T>()`. Both matrices are stored in
/// column-major format, with leading dimensions `lda` and `ldb`.
/// The leading dimension indicates the number of rows of the allocated matrix, even if only a submatrix of it is being used.
///
/// # Errors
///
/// Returns an error if the context cannot be bound, either leading dimension is
/// invalid, the host or device storage is too small, an argument cannot be
/// represented by cuBLAS, or cuBLAS cannot access the GPU memory.
pub fn copy_matrix_to_device<T, A, B>(
    ctx: &Context,
    rows: usize,
    cols: usize,
    a: &A,
    lda: usize,
    b: &mut B,
    ldb: usize,
) -> Result<()>
where
    T: DeviceRepr,
    A: HostBuffer<T> + ?Sized,
    B: DeviceBufferMut<T> + ?Sized,
{
    ctx.bind()?;
    if rows == 0 || cols == 0 {
        return Ok(());
    }
    if lda < rows || ldb < rows {
        return Err(Error::InvalidLeadingDimension);
    }
    if a.len() < required_matrix_len(lda, cols)? || b.len() < required_matrix_len(ldb, cols)? {
        return Err(Error::InvalidMatrixShape);
    }

    let rows = to_i32(rows, "rows")?;
    let cols = to_i32(cols, "cols")?;
    let elem_size = to_i32(size_of::<T>(), "elem_size")?;
    let lda = to_i32(lda, "lda")?;
    let ldb = to_i32(ldb, "ldb")?;

    unsafe {
        try_ffi!(sys::cublasSetMatrix(
            rows,
            cols,
            elem_size,
            a.as_host_ptr().cast(),
            lda,
            b.as_device_mut_ptr().cast(),
            ldb,
        ))?;
    }

    Ok(())
}

/// Supports the 64-bit integer interface.
///
/// Copies a `rows x cols` tile from device matrix `a` to host matrix `b`.
///
/// Each element has size `size_of::<T>()`. Both matrices are stored in
/// column-major format, with leading dimensions `lda` and `ldb`.
/// The leading dimension indicates the number of rows of the allocated matrix, even if only a submatrix of it is being used.
///
/// # Errors
///
/// Returns an error if the context cannot be bound, either leading dimension is
/// invalid, the device or host storage is too small, an argument cannot be
/// represented by cuBLAS, or cuBLAS cannot access the GPU memory.
pub fn copy_matrix_to_host<T, A, B>(
    ctx: &Context,
    rows: usize,
    cols: usize,
    a: &A,
    lda: usize,
    b: &mut B,
    ldb: usize,
) -> Result<()>
where
    T: DeviceRepr,
    A: DeviceBuffer<T> + ?Sized,
    B: HostBufferMut<T> + ?Sized,
{
    ctx.bind()?;
    if rows == 0 || cols == 0 {
        return Ok(());
    }
    if lda < rows || ldb < rows {
        return Err(Error::InvalidLeadingDimension);
    }
    if a.len() < required_matrix_len(lda, cols)? || b.len() < required_matrix_len(ldb, cols)? {
        return Err(Error::InvalidMatrixShape);
    }

    let rows = to_i32(rows, "rows")?;
    let cols = to_i32(cols, "cols")?;
    let elem_size = to_i32(size_of::<T>(), "elem_size")?;
    let lda = to_i32(lda, "lda")?;
    let ldb = to_i32(ldb, "ldb")?;

    unsafe {
        try_ffi!(sys::cublasGetMatrix(
            rows,
            cols,
            elem_size,
            a.as_device_ptr().cast(),
            lda,
            b.as_host_mut_ptr().cast(),
            ldb,
        ))?;
    }

    Ok(())
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use super::*;
    use crate::testing::setup_context;
    use singe_cuda::memory::DeviceMemory;

    #[test]
    fn test_vector_transfer_round_trip() -> Result<()> {
        let context = setup_context()?;
        let stream = context.cuda_context().create_stream()?;

        let host = vec![1_i32, 2, 3, 4, 5, 6];
        let mut device = DeviceMemory::<i32>::create(host.len())?;

        copy_vector_to_device(&context, host.len(), &host, 1, &mut device, 1)?;

        let mut sync_result = vec![0_i32; host.len()];
        copy_vector_to_host(&context, host.len(), &device, 1, &mut sync_result, 1)?;
        assert_eq!(sync_result, host);

        let host_async = vec![6_i32, 5, 4, 3, 2, 1];
        let mut async_result = vec![0_i32; host_async.len()];
        let mut transfer_result = Ok(());
        stream.scope(|scope| {
            transfer_result = copy_vector_to_device_async(
                &context,
                host_async.len(),
                &host_async,
                1,
                &mut device,
                1,
                scope,
            )
            .and_then(|()| {
                copy_vector_to_host_async(
                    &context,
                    host_async.len(),
                    &device,
                    1,
                    &mut async_result,
                    1,
                    scope,
                )
            });
            Ok(()).into()
        })?;
        transfer_result?;

        assert_eq!(async_result, host_async);

        Ok(())
    }

    #[test]
    fn test_matrix_transfer_round_trip() -> Result<()> {
        let _context = setup_context()?;

        let rows = 2;
        let cols = 3;
        let lda = 2;
        let ldb = 2;
        let host = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut device = DeviceMemory::<f32>::create(host.len())?;

        copy_matrix_to_device(&_context, rows, cols, &host, lda, &mut device, ldb)?;

        let mut result = vec![0.0_f32; host.len()];
        copy_matrix_to_host(&_context, rows, cols, &device, lda, &mut result, ldb)?;

        assert_eq!(result, host);

        Ok(())
    }

    #[test]
    fn test_matrix_transfer_async_round_trip() -> Result<()> {
        let context = setup_context()?;
        let stream = context.cuda_context().create_stream()?;

        let rows = 2;
        let cols = 3;
        let lda = 2;
        let ldb = 2;
        let host = vec![6.0_f32, 5.0, 4.0, 3.0, 2.0, 1.0];
        let mut device = DeviceMemory::<f32>::create(host.len())?;

        let mut result = vec![0.0_f32; host.len()];
        let mut transfer_result = Ok(());
        stream.scope(|scope| {
            transfer_result = copy_matrix_to_device_async(
                &context,
                rows,
                cols,
                &host,
                lda,
                &mut device,
                ldb,
                scope,
            )
            .and_then(|()| {
                copy_matrix_to_host_async(
                    &context,
                    rows,
                    cols,
                    &device,
                    lda,
                    &mut result,
                    ldb,
                    scope,
                )
            });
            Ok(()).into()
        })?;
        transfer_result?;

        assert_eq!(result, host);

        Ok(())
    }
}
