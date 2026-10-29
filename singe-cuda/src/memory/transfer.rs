//! Context-aware transfers between typed buffers and borrowed ranges.
//!
//! Borrowed transfers complete before returning. If CUDA cannot drain an operation
//! during failure cleanup, the process aborts to prevent DMA from accessing released storage.

use std::{
    mem::{MaybeUninit, size_of},
    process, ptr, slice,
    sync::Arc,
};

use singe_cuda_sys::runtime;

use crate::{
    error::{Error, Result, Status},
    memory::{DeviceMemory, MemoryCopyKind},
    stream::Stream,
    try_ffi,
    view::{AnyBitPatternDeviceRepr, DeviceOutput, DeviceRepr, DeviceSlice, ranges_overlap},
};

/// Copies host values into an initialized or uninitialized device output range.
///
/// # Errors
///
/// Returns `Error::InvalidMemoryAccess` for unequal element counts, or `Error::InvalidMemoryAllocationRequest` if the byte count overflows. Also propagates access-tracking, readiness synchronization, context binding, capture-state query, native copy, and stream synchronization errors. A nonempty synchronous transfer is rejected during stream capture.
pub fn copy_from_host<T, B>(destination: &mut B, source: &[T]) -> Result<()>
where
    T: DeviceRepr,
    B: DeviceOutput<T> + ?Sized,
{
    let () = T::LAYOUT;
    if destination.len() != source.len() {
        return Err(Error::InvalidMemoryAccess);
    }
    let context = Arc::clone(destination.context());
    let _access = context.acquire_host_access(&[destination.region()])?;
    destination.context().bind()?;
    // SAFETY: the provider grants exclusive writable storage for the full source slice.
    unsafe {
        copy_synchronous(
            destination.as_mut_ptr(),
            source.as_ptr(),
            source.len(),
            MemoryCopyKind::HostToDevice,
            destination.region().stream(),
        )
    }
}

/// Copies initialized device bytes into host values accepting every bit pattern.
///
/// # Errors
///
/// Propagates length, byte-size, access-tracking, capture, and CUDA transfer errors from [`copy_to_host_unchecked`].
pub fn copy_to_host<T, B>(source: &B, destination: &mut [T]) -> Result<()>
where
    T: AnyBitPatternDeviceRepr,
    B: DeviceSlice<T> + ?Sized,
{
    // SAFETY: the source provider contains initialized T values and all bit patterns are valid.
    unsafe { copy_to_host_unchecked(source, destination) }
}

/// Completes a readback into uninitialized host storage and returns its initialized borrow.
/// Errors leave the destination as `MaybeUninit<T>`; no initialized values are exposed.
///
/// # Errors
/// Returns an error for mismatched lengths, capture, context binding, or CUDA transfer failure.
pub fn copy_to_host_uninit<'destination, T, B>(
    source: &B,
    destination: &'destination mut [MaybeUninit<T>],
) -> Result<&'destination mut [T]>
where
    T: AnyBitPatternDeviceRepr,
    B: DeviceSlice<T> + ?Sized,
{
    let () = T::LAYOUT;
    if source.len() != destination.len() {
        return Err(Error::InvalidMemoryAccess);
    }
    let context = Arc::clone(source.region().context());
    let _access = context.acquire_host_access(&[source.region()])?;
    source.region().context().bind()?;
    // SAFETY: the source contains initialized device values; destination owns
    // matching exclusive storage. The completed copy initializes every byte.
    unsafe {
        copy_synchronous(
            destination.as_mut_ptr().cast(),
            source.as_device_ptr(),
            source.len(),
            MemoryCopyKind::DeviceToHost,
            source.region().stream(),
        )?;
        Ok(slice::from_raw_parts_mut(
            destination.as_mut_ptr().cast(),
            destination.len(),
        ))
    }
}

/// Copies device values with caller-established validity into host storage.
///
/// # Safety
///
/// Every source element must contain a valid T value and its producing work
/// must be ordered before the transfer.
///
/// # Errors
///
/// Returns `Error::InvalidMemoryAccess` for unequal element counts, or `Error::InvalidMemoryAllocationRequest` if the byte count overflows. Also propagates access-tracking, readiness synchronization, context binding, capture-state query, native copy, and stream synchronization errors. A nonempty synchronous transfer is rejected during stream capture.
pub unsafe fn copy_to_host_unchecked<T, B>(source: &B, destination: &mut [T]) -> Result<()>
where
    T: DeviceRepr,
    B: DeviceSlice<T> + ?Sized,
{
    let () = T::LAYOUT;
    if source.len() != destination.len() {
        return Err(Error::InvalidMemoryAccess);
    }
    let context = Arc::clone(source.region().context());
    let _access = context.acquire_host_access(&[source.region()])?;
    source.region().context().bind()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        copy_synchronous(
            destination.as_mut_ptr(),
            source.as_device_ptr(),
            source.len(),
            MemoryCopyKind::DeviceToHost,
            source.region().stream(),
        )
    }
}

/// Copies an initialized device range into a newly allocated host vector.
///
/// # Errors
///
/// Propagates byte-size, access-tracking, context binding, capture, and CUDA transfer errors. No vector of initialized values is returned unless the transfer completes successfully.
pub fn copy_to_host_vec<T, B>(source: &B) -> Result<Vec<T>>
where
    T: AnyBitPatternDeviceRepr,
    B: DeviceSlice<T> + ?Sized,
{
    let () = T::LAYOUT;
    source.byte_len()?;
    let context = Arc::clone(source.region().context());
    let _access = context.acquire_host_access(&[source.region()])?;
    source.region().context().bind()?;
    // SAFETY: the source contains initialized values and retains its readiness stream.
    unsafe {
        copy_to_host_vec_unchecked(
            source.as_device_ptr(),
            source.len(),
            source.region().stream(),
        )
    }
}

/// Completes a vector readback with caller-established source value validity.
///
/// # Safety
/// The source must contain `length` valid, transferable values of T.
/// Bind its owning context and order producers before the transfer.
pub(crate) unsafe fn copy_to_host_vec_unchecked<T>(
    source: *const T,
    length: usize,
    stream: Option<&Stream>,
) -> Result<Vec<T>> {
    if size_of::<T>() == 0 {
        return Err(Error::InvalidMemoryAllocationRequest);
    }
    let mut values = Vec::<T>::with_capacity(length);
    // SAFETY: the caller establishes source validity.
    // Completion precedes exposing the vector length.
    unsafe {
        copy_synchronous(
            values.as_mut_ptr(),
            source,
            length,
            MemoryCopyKind::DeviceToHost,
            stream,
        )?;
        values.set_len(length);
    }
    Ok(values)
}

/// Copies equally sized, disjoint ranges within one CUDA context.
///
/// # Errors
///
/// Returns `Error::MemoryContextMismatch` if the ranges belong to different contexts, or `Error::InvalidMemoryAccess` for unequal element counts or overlapping ranges in the same allocation. Also rejects byte-size overflow. Also propagates access-tracking, readiness synchronization, context binding, capture-state query, native copy, and stream synchronization errors. A nonempty synchronous transfer is rejected during stream capture.
pub fn copy_device<T, S, D>(destination: &mut D, source: &S) -> Result<()>
where
    T: DeviceRepr,
    S: DeviceSlice<T> + ?Sized,
    D: DeviceOutput<T> + ?Sized,
{
    let () = T::LAYOUT;
    let destination_pointer = destination.as_mut_ptr();
    let source_region = source.region();
    let destination_region = destination.region();
    if source_region.context() != destination_region.context() {
        return Err(Error::MemoryContextMismatch);
    }
    if source.len() != destination.len() {
        return Err(Error::InvalidMemoryAccess);
    }
    let source_range = source_region.byte_range();
    let destination_range = destination_region.byte_range();
    if source_region.allocation_id() == destination_region.allocation_id()
        && ranges_overlap(&source_range, &destination_range)
    {
        return Err(Error::InvalidMemoryAccess);
    }
    let context = Arc::clone(source_region.context());
    let _access = context.acquire_host_access(&[source_region, destination_region])?;
    context.bind()?;
    // SAFETY: the providers describe disjoint readable and writable ranges in the same context.
    unsafe {
        copy_synchronous(
            destination_pointer,
            source.as_device_ptr(),
            source.len(),
            MemoryCopyKind::DeviceToDevice,
            destination_region.stream(),
        )
    }
}

/// Enqueues a device copy with externally established lifetimes and ordering.
///
/// # Safety
///
/// Both ranges must remain live and exclusively accessible as appropriate
/// until `stream` completes. Allocation and prior producer work on other
/// streams must precede this copy, including stream-ordered allocations.
///
/// # Errors
///
/// Returns `Error::MemoryContextMismatch` for different allocation contexts, `Error::StreamContextMismatch` for a stream from another context, or `Error::InvalidMemoryAccess` for unequal counts or overlapping ranges in the same allocation. Also propagates context binding, byte-size overflow, and native asynchronous copy errors.
pub unsafe fn copy_device_async_unchecked<T, S, D>(
    destination: &mut D,
    source: &S,
    stream: &Stream,
) -> Result<()>
where
    T: DeviceRepr,
    S: DeviceSlice<T> + ?Sized,
    D: DeviceOutput<T> + ?Sized,
{
    let () = T::LAYOUT;
    let destination_pointer = destination.as_mut_ptr();
    let source_region = source.region();
    let destination_region = destination.region();
    if source_region.context() != destination_region.context() {
        return Err(Error::MemoryContextMismatch);
    }
    if stream.context() != destination_region.context() {
        return Err(Error::StreamContextMismatch);
    }
    if source.len() != destination.len() {
        return Err(Error::InvalidMemoryAccess);
    }
    let source_range = source_region.byte_range();
    let destination_range = destination_region.byte_range();
    if source_region.allocation_id() == destination_region.allocation_id()
        && ranges_overlap(&source_range, &destination_range)
    {
        return Err(Error::InvalidMemoryAccess);
    }
    stream.context().bind()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        DeviceMemory::<T>::copy_async(
            destination_pointer,
            source.as_device_ptr(),
            source.len(),
            MemoryCopyKind::DeviceToDevice,
            stream,
        )
    }
}

/// Submits and completes one copy while the host and device borrows remain live.
///
/// # Safety
///
/// The pointers must identify disjoint readable and writable ranges for the
/// copy kind. The caller must bind the context and order external producers.
pub(crate) unsafe fn copy_synchronous<T>(
    destination: *mut T,
    source: *const T,
    length: usize,
    kind: MemoryCopyKind,
    stream: Option<&Stream>,
) -> Result<()> {
    let bytes = length
        .checked_mul(size_of::<T>())
        .ok_or(Error::InvalidMemoryAllocationRequest)?;
    if bytes == 0 {
        return Ok(());
    }
    let handle = stream.map_or(ptr::null_mut(), Stream::as_raw);
    let mut capture_status = runtime::cudaStreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
    // SAFETY: `cudaStreamIsCapturing` receives valid output locations.
    unsafe {
        try_ffi!(runtime::cudaStreamIsCapturing(
            handle,
            &raw mut capture_status
        ))?;
    }
    if capture_status != runtime::cudaStreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE {
        return Err(Status::StreamCaptureUnsupported.into());
    }
    let mut drain = CopyDrain {
        handle,
        drained: false,
    };
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        try_ffi!(runtime::cudaMemcpyAsync(
            destination.cast(),
            source.cast(),
            bytes as _,
            kind.into(),
            handle
        ))?;
        let result = try_ffi!(runtime::cudaStreamSynchronize(handle));
        drain.drained = result.is_ok();
        result
    }
}

struct CopyDrain {
    handle: runtime::cudaStream_t,
    drained: bool,
}

impl Drop for CopyDrain {
    fn drop(&mut self) {
        if !self.drained {
            // Borrowed host/device storage cannot be retained after this call.
            // A failed enqueue can also report earlier asynchronous work.
            // SAFETY: `cudaStreamSynchronize` receives live handles owned by the wrappers.
            let status = unsafe { runtime::cudaStreamSynchronize(self.handle) };
            if status != runtime::cudaError_t::CUDA_SUCCESS {
                process::abort();
            }
        }
    }
}
