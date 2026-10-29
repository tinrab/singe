//! Strided host/device copies using the handle-independent cuBLAS transfer API.
#![expect(
    clippy::similar_names,
    reason = "BLAS and factorization operand names follow the standard mathematical API"
)]

use std::{mem::size_of, ptr, sync::Arc};

use singe_cuda::{
    error::Error as CudaError,
    stream::{Stream, StreamBinding, StreamScope},
    view::{DeviceInput, DeviceOutput, DeviceRepr, DeviceSlice, HostSlice, HostSliceMut},
};

use crate::{
    error::{Error, Result},
    sys, try_ffi,
    utility::to_i64,
};

#[derive(Clone, Copy)]
enum Direction {
    ToDevice,
    ToHost,
}

// cuBLAS may form byte pitches even when a tile has only one column/element.
// Check those products before native integer arithmetic.
fn check_byte_spans<T>(element_spans: &[usize]) -> Result<()> {
    for &elements in element_spans {
        let bytes = elements
            .checked_mul(size_of::<T>())
            .ok_or_else(|| Error::out_of_range("copy byte span"))?;
        to_i64(bytes, "copy byte span")?;
    }
    Ok(())
}

struct VectorCopy {
    n: i64,
    element_size: i64,
    source_stride: i64,
    destination_stride: i64,
}

impl VectorCopy {
    fn create<T: DeviceRepr>(
        n: usize,
        source_stride: usize,
        destination_stride: usize,
        source_len: usize,
        destination_len: usize,
    ) -> Result<Option<Self>> {
        let () = T::LAYOUT;
        if n == 0 {
            return Ok(None);
        }
        if source_stride == 0 || destination_stride == 0 {
            return Err(Error::InvalidIncrement);
        }
        if size_of::<T>() == 0 {
            return Err(Error::out_of_range("element size"));
        }
        let span = |stride: usize| {
            (n - 1)
                .checked_mul(stride)
                .and_then(|v| v.checked_add(1))
                .ok_or_else(|| Error::out_of_range("vector span"))
        };
        if span(source_stride)? > source_len || span(destination_stride)? > destination_len {
            return Err(Error::InvalidVectorShape);
        }
        check_byte_spans::<T>(&[
            span(source_stride)?,
            span(destination_stride)?,
            source_stride,
            destination_stride,
        ])?;
        Ok(Some(Self {
            n: to_i64(n, "n")?,
            element_size: to_i64(size_of::<T>(), "element size")?,
            source_stride: to_i64(source_stride, "source stride")?,
            destination_stride: to_i64(destination_stride, "destination stride")?,
        }))
    }

    // SAFETY: the public callers establish exact element spans and queued access lifetimes.
    unsafe fn enqueue<T>(
        &self,
        source: *const T,
        destination: *mut T,
        stream: Option<&Stream>,
        direction: Direction,
    ) -> Result<()> {
        let Self {
            n,
            element_size,
            source_stride,
            destination_stride,
        } = *self;
        let stream = stream.map_or(ptr::null_mut(), Stream::as_raw);
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            match direction {
                Direction::ToDevice => try_ffi!(sys::cublasSetVectorAsync_64(
                    n,
                    element_size,
                    source.cast(),
                    source_stride,
                    destination.cast(),
                    destination_stride,
                    stream
                )),
                Direction::ToHost => try_ffi!(sys::cublasGetVectorAsync_64(
                    n,
                    element_size,
                    source.cast(),
                    source_stride,
                    destination.cast(),
                    destination_stride,
                    stream
                )),
            }
        }
    }
}

struct MatrixCopy {
    rows: i64,
    cols: i64,
    element_size: i64,
    source_leading: i64,
    destination_leading: i64,
}

impl MatrixCopy {
    fn create<T: DeviceRepr>(
        rows: usize,
        cols: usize,
        source_leading: usize,
        destination_leading: usize,
        source_len: usize,
        destination_len: usize,
    ) -> Result<Option<Self>> {
        let () = T::LAYOUT;
        if rows == 0 || cols == 0 {
            return Ok(None);
        }
        if source_leading < rows || destination_leading < rows {
            return Err(Error::InvalidLeadingDimension);
        }
        if size_of::<T>() == 0 {
            return Err(Error::out_of_range("element size"));
        }
        let span = |leading: usize| {
            (cols - 1)
                .checked_mul(leading)
                .and_then(|v| v.checked_add(rows))
                .ok_or_else(|| Error::out_of_range("matrix span"))
        };
        if span(source_leading)? > source_len || span(destination_leading)? > destination_len {
            return Err(Error::InvalidMatrixShape);
        }
        check_byte_spans::<T>(&[
            span(source_leading)?,
            span(destination_leading)?,
            source_leading,
            destination_leading,
        ])?;
        Ok(Some(Self {
            rows: to_i64(rows, "rows")?,
            cols: to_i64(cols, "cols")?,
            element_size: to_i64(size_of::<T>(), "element size")?,
            source_leading: to_i64(source_leading, "source leading dimension")?,
            destination_leading: to_i64(destination_leading, "destination leading dimension")?,
        }))
    }

    // SAFETY: the public callers establish exact column-major spans and queued access lifetimes.
    unsafe fn enqueue<T>(
        &self,
        source: *const T,
        destination: *mut T,
        stream: Option<&Stream>,
        direction: Direction,
    ) -> Result<()> {
        let Self {
            rows,
            cols,
            element_size,
            source_leading,
            destination_leading,
        } = *self;
        let stream = stream.map_or(ptr::null_mut(), Stream::as_raw);
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            match direction {
                Direction::ToDevice => try_ffi!(sys::cublasSetMatrixAsync_64(
                    rows,
                    cols,
                    element_size,
                    source.cast(),
                    source_leading,
                    destination.cast(),
                    destination_leading,
                    stream
                )),
                Direction::ToHost => try_ffi!(sys::cublasGetMatrixAsync_64(
                    rows,
                    cols,
                    element_size,
                    source.cast(),
                    source_leading,
                    destination.cast(),
                    destination_leading,
                    stream
                )),
            }
        }
    }
}

/// Copies a strided vector from host to device.
///
/// Gaps and tails are unchanged. Dimensions select the native 32-bit or 64-bit interface.
/// Completes default-stream work on success, failure and unwind before returning.
///
/// # Errors
///
/// For a nonempty copy, returns an error if an increment is zero or the selected elements exceed an operand's length, or if element sizes, byte spans, or dimensions exceed the native integer range.
/// Returns an error if synchronizing prior accesses, binding the CUDA context, enqueuing the cuBLAS copy, or waiting for completion fails.
pub fn copy_vector_to_device<T, X, Y>(
    n: usize,
    x: &X,
    incx: usize,
    y: &mut Y,
    incy: usize,
) -> Result<()>
where
    T: DeviceRepr,
    X: HostSlice<T> + ?Sized,
    Y: DeviceOutput<T> + ?Sized,
{
    let Some(copy) = VectorCopy::create::<T>(n, incx, incy, x.len(), y.len())? else {
        return Ok(());
    };
    let binding = StreamBinding::Default(Arc::clone(y.region().context()));
    binding.sync_scope(|| {
        y.region().synchronize()?;
        // SAFETY: checked spans and borrowed ranges survive completion on every path.
        unsafe { copy.enqueue(x.as_host_ptr(), y.as_mut_ptr(), None, Direction::ToDevice) }
    })
}

/// Copies a strided vector from host to device.
///
/// Gaps and tails are unchanged. Dimensions select the native 32-bit or 64-bit interface.
/// Borrows both operands until the scope completes, including failure and unwind.
///
/// # Errors
///
/// For a nonempty copy, returns an error if an increment is zero or the selected elements exceed an operand's length, or if element sizes, byte spans, or dimensions exceed the native integer range.
/// Returns an error if the device operand belongs to another stream context, joining prior accesses or binding the context fails, or cuBLAS cannot enqueue the copy.
pub fn copy_vector_to_device_async<'env, T, X, Y>(
    n: usize,
    x: &'env X,
    incx: usize,
    y: &'env mut Y,
    incy: usize,
    scope: &StreamScope<'_, 'env>,
) -> Result<()>
where
    T: DeviceRepr,
    X: HostSlice<T> + ?Sized,
    Y: DeviceOutput<T> + ?Sized,
{
    // SAFETY: both resource borrows are tied to the scope's completion lifetime.
    unsafe { copy_vector_to_device_async_unchecked(n, x, incx, y, incy, scope.stream()) }
}

/// Copies a strided vector from host to device.
/// Gaps and tails are unchanged. Dimensions select the native 32-bit or 64-bit interface.
///
/// # Safety
/// Retain both ranges and the stream through completion and failed submission;
/// preserve valid source elements and exclusive destination access until then.
/// Order source producers before this copy. Captured replays require retaining
/// both addresses and establishing the same access contract on every replay.
///
/// # Errors
///
/// For a nonempty copy, returns an error if an increment is zero or the selected elements exceed an operand's length, or if element sizes, byte spans, or dimensions exceed the native integer range.
/// Returns an error if the device operand belongs to another stream context, joining prior accesses or binding the context fails, or cuBLAS cannot enqueue the copy.
pub unsafe fn copy_vector_to_device_async_unchecked<T, X, Y>(
    n: usize,
    x: &X,
    incx: usize,
    y: &mut Y,
    incy: usize,
    stream: &Stream,
) -> Result<()>
where
    T: DeviceRepr,
    X: HostSlice<T> + ?Sized,
    Y: DeviceOutput<T> + ?Sized,
{
    if y.region().context() != stream.context() {
        return Err(CudaError::MemoryContextMismatch.into());
    }
    let Some(copy) = VectorCopy::create::<T>(n, incx, incy, x.len(), y.len())? else {
        return Ok(());
    };
    stream.join_regions(&[y.region()])?;
    stream.context().bind()?;
    // SAFETY: checked spans and the caller's queued access contract cover native use.
    unsafe {
        copy.enqueue(
            x.as_host_ptr(),
            y.as_mut_ptr(),
            Some(stream),
            Direction::ToDevice,
        )
    }
}

/// Copies a strided vector from device to host.
///
/// Gaps and tails are unchanged. Dimensions select the native 32-bit or 64-bit interface.
/// Completes default-stream work on success, failure and unwind before returning.
///
/// # Errors
///
/// For a nonempty copy, returns an error if an increment is zero or the selected elements exceed an operand's length, or if element sizes, byte spans, or dimensions exceed the native integer range.
/// Returns an error if synchronizing prior accesses, binding the CUDA context, enqueuing the cuBLAS copy, or waiting for completion fails.
pub fn copy_vector_to_host<T, X, Y>(
    n: usize,
    x: &X,
    incx: usize,
    y: &mut Y,
    incy: usize,
) -> Result<()>
where
    T: DeviceRepr,
    X: DeviceSlice<T> + ?Sized,
    Y: HostSliceMut<T> + ?Sized,
{
    let Some(copy) = VectorCopy::create::<T>(n, incx, incy, x.len(), y.len())? else {
        return Ok(());
    };
    let binding = StreamBinding::Default(Arc::clone(x.region().context()));
    binding.sync_scope(|| {
        x.region().synchronize()?;
        // SAFETY: checked spans and borrowed ranges survive completion on every path.
        unsafe {
            copy.enqueue(
                x.as_device_ptr(),
                y.as_host_mut_ptr(),
                None,
                Direction::ToHost,
            )
        }
    })
}

/// Copies a strided vector from device to host.
///
/// Gaps and tails are unchanged. Dimensions select the native 32-bit or 64-bit interface.
/// Borrows both operands until the scope completes, including failure and unwind.
///
/// # Errors
///
/// For a nonempty copy, returns an error if an increment is zero or the selected elements exceed an operand's length, or if element sizes, byte spans, or dimensions exceed the native integer range.
/// Returns an error if the device operand belongs to another stream context, joining prior accesses or binding the context fails, or cuBLAS cannot enqueue the copy.
pub fn copy_vector_to_host_async<'env, T, X, Y>(
    n: usize,
    x: &'env X,
    incx: usize,
    y: &'env mut Y,
    incy: usize,
    scope: &StreamScope<'_, 'env>,
) -> Result<()>
where
    T: DeviceRepr,
    X: DeviceSlice<T> + ?Sized,
    Y: HostSliceMut<T> + ?Sized,
{
    // SAFETY: both resource borrows are tied to the scope's completion lifetime.
    unsafe { copy_vector_to_host_async_unchecked(n, x, incx, y, incy, scope.stream()) }
}

/// Copies a strided vector from device to host.
/// Gaps and tails are unchanged. Dimensions select the native 32-bit or 64-bit interface.
///
/// # Safety
/// Retain both ranges and the stream through completion and failed submission;
/// preserve valid source elements and exclusive destination access until then.
/// Order source producers before this copy. Captured replays require retaining
/// both addresses and establishing the same access contract on every replay.
///
/// # Errors
///
/// For a nonempty copy, returns an error if an increment is zero or the selected elements exceed an operand's length, or if element sizes, byte spans, or dimensions exceed the native integer range.
/// Returns an error if the device operand belongs to another stream context, joining prior accesses or binding the context fails, or cuBLAS cannot enqueue the copy.
pub unsafe fn copy_vector_to_host_async_unchecked<T, X, Y>(
    n: usize,
    x: &X,
    incx: usize,
    y: &mut Y,
    incy: usize,
    stream: &Stream,
) -> Result<()>
where
    T: DeviceRepr,
    X: DeviceInput<T> + ?Sized,
    Y: HostSliceMut<T> + ?Sized,
{
    if x.region().context() != stream.context() {
        return Err(CudaError::MemoryContextMismatch.into());
    }
    let Some(copy) = VectorCopy::create::<T>(n, incx, incy, x.len(), y.len())? else {
        return Ok(());
    };
    stream.join_regions(&[x.region()])?;
    stream.context().bind()?;
    // SAFETY: checked spans and the caller's queued access contract cover native use.
    unsafe {
        copy.enqueue(
            x.as_device_ptr(),
            y.as_host_mut_ptr(),
            Some(stream),
            Direction::ToHost,
        )
    }
}

/// Copies a strided matrix from host to device.
///
/// Matrices use column-major leading dimensions; only the selected rows are copied.
/// Gaps and tails are unchanged. Dimensions select the native 32-bit or 64-bit interface.
/// Completes default-stream work on success, failure and unwind before returning.
///
/// # Errors
///
/// For a nonempty copy, returns an error if a leading dimension is smaller than the row count or the selected matrix exceeds an operand's length, or if element sizes, byte spans, or dimensions exceed the native integer range.
/// Returns an error if synchronizing prior accesses, binding the CUDA context, enqueuing the cuBLAS copy, or waiting for completion fails.
pub fn copy_matrix_to_device<T, X, Y>(
    rows: usize,
    cols: usize,
    a: &X,
    lda: usize,
    b: &mut Y,
    ldb: usize,
) -> Result<()>
where
    T: DeviceRepr,
    X: HostSlice<T> + ?Sized,
    Y: DeviceOutput<T> + ?Sized,
{
    let Some(copy) = MatrixCopy::create::<T>(rows, cols, lda, ldb, a.len(), b.len())? else {
        return Ok(());
    };
    let binding = StreamBinding::Default(Arc::clone(b.region().context()));
    binding.sync_scope(|| {
        b.region().synchronize()?;
        // SAFETY: checked spans and borrowed ranges survive completion on every path.
        unsafe { copy.enqueue(a.as_host_ptr(), b.as_mut_ptr(), None, Direction::ToDevice) }
    })
}

/// Copies a strided matrix from host to device.
///
/// Matrices use column-major leading dimensions; only the selected rows are copied.
/// Gaps and tails are unchanged. Dimensions select the native 32-bit or 64-bit interface.
/// Borrows both operands until the scope completes, including failure and unwind.
///
/// # Errors
///
/// For a nonempty copy, returns an error if a leading dimension is smaller than the row count or the selected matrix exceeds an operand's length, or if element sizes, byte spans, or dimensions exceed the native integer range.
/// Returns an error if the device operand belongs to another stream context, joining prior accesses or binding the context fails, or cuBLAS cannot enqueue the copy.
pub fn copy_matrix_to_device_async<'env, T, X, Y>(
    rows: usize,
    cols: usize,
    a: &'env X,
    lda: usize,
    b: &'env mut Y,
    ldb: usize,
    scope: &StreamScope<'_, 'env>,
) -> Result<()>
where
    T: DeviceRepr,
    X: HostSlice<T> + ?Sized,
    Y: DeviceOutput<T> + ?Sized,
{
    // SAFETY: both resource borrows are tied to the scope's completion lifetime.
    unsafe { copy_matrix_to_device_async_unchecked(rows, cols, a, lda, b, ldb, scope.stream()) }
}

/// Copies a strided matrix from host to device.
///
/// Matrices use column-major leading dimensions; only the selected rows are copied.
/// Gaps and tails are unchanged. Dimensions select the native 32-bit or 64-bit interface.
///
/// # Safety
/// Retain both ranges and the stream through completion and failed submission;
/// preserve valid source elements and exclusive destination access until then.
/// Order source producers before this copy. Captured replays require retaining
/// both addresses and establishing the same access contract on every replay.
///
/// # Errors
///
/// For a nonempty copy, returns an error if a leading dimension is smaller than the row count or the selected matrix exceeds an operand's length, or if element sizes, byte spans, or dimensions exceed the native integer range.
/// Returns an error if the device operand belongs to another stream context, joining prior accesses or binding the context fails, or cuBLAS cannot enqueue the copy.
pub unsafe fn copy_matrix_to_device_async_unchecked<T, X, Y>(
    rows: usize,
    cols: usize,
    a: &X,
    lda: usize,
    b: &mut Y,
    ldb: usize,
    stream: &Stream,
) -> Result<()>
where
    T: DeviceRepr,
    X: HostSlice<T> + ?Sized,
    Y: DeviceOutput<T> + ?Sized,
{
    if b.region().context() != stream.context() {
        return Err(CudaError::MemoryContextMismatch.into());
    }
    let Some(copy) = MatrixCopy::create::<T>(rows, cols, lda, ldb, a.len(), b.len())? else {
        return Ok(());
    };
    stream.join_regions(&[b.region()])?;
    stream.context().bind()?;
    // SAFETY: checked spans and the caller's queued access contract cover native use.
    unsafe {
        copy.enqueue(
            a.as_host_ptr(),
            b.as_mut_ptr(),
            Some(stream),
            Direction::ToDevice,
        )
    }
}

/// Copies a strided matrix from device to host.
///
/// Matrices use column-major leading dimensions; only the selected rows are copied.
/// Gaps and tails are unchanged. Dimensions select the native 32-bit or 64-bit interface.
/// Completes default-stream work on success, failure and unwind before returning.
///
/// # Errors
///
/// For a nonempty copy, returns an error if a leading dimension is smaller than the row count or the selected matrix exceeds an operand's length, or if element sizes, byte spans, or dimensions exceed the native integer range.
/// Returns an error if synchronizing prior accesses, binding the CUDA context, enqueuing the cuBLAS copy, or waiting for completion fails.
pub fn copy_matrix_to_host<T, X, Y>(
    rows: usize,
    cols: usize,
    a: &X,
    lda: usize,
    b: &mut Y,
    ldb: usize,
) -> Result<()>
where
    T: DeviceRepr,
    X: DeviceSlice<T> + ?Sized,
    Y: HostSliceMut<T> + ?Sized,
{
    let Some(copy) = MatrixCopy::create::<T>(rows, cols, lda, ldb, a.len(), b.len())? else {
        return Ok(());
    };
    let binding = StreamBinding::Default(Arc::clone(a.region().context()));
    binding.sync_scope(|| {
        a.region().synchronize()?;
        // SAFETY: checked spans and borrowed ranges survive completion on every path.
        unsafe {
            copy.enqueue(
                a.as_device_ptr(),
                b.as_host_mut_ptr(),
                None,
                Direction::ToHost,
            )
        }
    })
}

/// Copies a strided matrix from device to host.
///
/// Matrices use column-major leading dimensions; only the selected rows are copied.
/// Gaps and tails are unchanged. Dimensions select the native 32-bit or 64-bit interface.
/// Borrows both operands until the scope completes, including failure and unwind.
///
/// # Errors
///
/// For a nonempty copy, returns an error if a leading dimension is smaller than the row count or the selected matrix exceeds an operand's length, or if element sizes, byte spans, or dimensions exceed the native integer range.
/// Returns an error if the device operand belongs to another stream context, joining prior accesses or binding the context fails, or cuBLAS cannot enqueue the copy.
pub fn copy_matrix_to_host_async<'env, T, X, Y>(
    rows: usize,
    cols: usize,
    a: &'env X,
    lda: usize,
    b: &'env mut Y,
    ldb: usize,
    scope: &StreamScope<'_, 'env>,
) -> Result<()>
where
    T: DeviceRepr,
    X: DeviceSlice<T> + ?Sized,
    Y: HostSliceMut<T> + ?Sized,
{
    // SAFETY: both resource borrows are tied to the scope's completion lifetime.
    unsafe { copy_matrix_to_host_async_unchecked(rows, cols, a, lda, b, ldb, scope.stream()) }
}

/// Copies a strided matrix from device to host.
///
/// Matrices use column-major leading dimensions; only the selected rows are copied.
/// Gaps and tails are unchanged. Dimensions select the native 32-bit or 64-bit interface.
///
/// # Safety
/// Retain both ranges and the stream through completion and failed submission;
/// preserve valid source elements and exclusive destination access until then.
/// Order source producers before this copy. Captured replays require retaining
/// both addresses and establishing the same access contract on every replay.
///
/// # Errors
///
/// For a nonempty copy, returns an error if a leading dimension is smaller than the row count or the selected matrix exceeds an operand's length, or if element sizes, byte spans, or dimensions exceed the native integer range.
/// Returns an error if the device operand belongs to another stream context, joining prior accesses or binding the context fails, or cuBLAS cannot enqueue the copy.
pub unsafe fn copy_matrix_to_host_async_unchecked<T, X, Y>(
    rows: usize,
    cols: usize,
    a: &X,
    lda: usize,
    b: &mut Y,
    ldb: usize,
    stream: &Stream,
) -> Result<()>
where
    T: DeviceRepr,
    X: DeviceInput<T> + ?Sized,
    Y: HostSliceMut<T> + ?Sized,
{
    if a.region().context() != stream.context() {
        return Err(CudaError::MemoryContextMismatch.into());
    }
    let Some(copy) = MatrixCopy::create::<T>(rows, cols, lda, ldb, a.len(), b.len())? else {
        return Ok(());
    };
    stream.join_regions(&[a.region()])?;
    stream.context().bind()?;
    // SAFETY: checked spans and the caller's queued access contract cover native use.
    unsafe {
        copy.enqueue(
            a.as_device_ptr(),
            b.as_host_mut_ptr(),
            Some(stream),
            Direction::ToHost,
        )
    }
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use singe_cuda::memory::DeviceMemory;

    use super::*;
    use crate::testing::setup_context;

    #[test]
    fn test_vector_transfer_round_trip() -> Result<()> {
        let context = setup_context()?;
        let stream = context.cuda_context().create_stream()?;

        let host = vec![1_i32, 2, 3, 4, 5, 6];
        let mut device = DeviceMemory::<i32>::create_uninit(context.cuda_context(), host.len())?;

        copy_vector_to_device(host.len(), &host, 1, &mut device, 1)?;
        // SAFETY: the completed contiguous upload wrote every element.
        let mut device = unsafe { device.assume_init() };

        let mut sync_result = vec![0_i32; host.len()];
        copy_vector_to_host(host.len(), &device, 1, &mut sync_result, 1)?;
        assert_eq!(sync_result, host);

        let host_async = vec![6_i32, 5, 4, 3, 2, 1];
        let mut async_result = vec![0_i32; host_async.len()];
        StreamBinding::Borrowed(stream.to_borrowed()).sync_scope(|| -> Result<()> {
            // SAFETY: the upload precedes readback on the same stream; every
            // initialized range and host destination stays live until completion.
            unsafe {
                copy_vector_to_device_async_unchecked(
                    host_async.len(),
                    &host_async,
                    1,
                    &mut device,
                    1,
                    &stream,
                )?;
                copy_vector_to_host_async_unchecked(
                    host_async.len(),
                    &device,
                    1,
                    &mut async_result,
                    1,
                    &stream,
                )?;
            }
            Ok(())
        })?;

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
        let mut device = DeviceMemory::<f32>::create_uninit(_context.cuda_context(), host.len())?;

        copy_matrix_to_device(rows, cols, &host, lda, &mut device, ldb)?;
        // SAFETY: this contiguous tile covers every element after completion.
        let device = unsafe { device.assume_init() };

        let mut result = vec![0.0_f32; host.len()];
        copy_matrix_to_host(rows, cols, &device, lda, &mut result, ldb)?;

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
        let mut device = DeviceMemory::<f32>::create_uninit(context.cuda_context(), host.len())?;

        let mut result = vec![0.0_f32; host.len()];
        StreamBinding::Borrowed(stream.to_borrowed()).sync_scope(|| -> Result<()> {
            // SAFETY: the upload precedes readback on the same stream; every
            // initialized range and host destination stays live until completion.
            unsafe {
                copy_matrix_to_device_async_unchecked(
                    rows,
                    cols,
                    &host,
                    lda,
                    &mut device,
                    ldb,
                    &stream,
                )?;
                copy_matrix_to_host_async_unchecked(
                    rows,
                    cols,
                    &device,
                    lda,
                    &mut result,
                    ldb,
                    &stream,
                )?;
            }
            Ok(())
        })?;

        assert_eq!(result, host);

        Ok(())
    }
}
