//! Typed 16-bit and 32-bit word fills.

use std::{mem::MaybeUninit, sync::Arc};

use singe_cuda_sys::driver as sys;

use crate::{
    context::Context,
    error::{Error, Result},
    graph::owned::{Recordable, Recording},
    memory::DeviceMemory,
    stream::{Stream, StreamBinding, StreamScope},
    submission::{
        Access, AccessMode, DeviceOperation, Execution, Operation, Resources, transfer::DeviceValue,
    },
    try_ffi,
    utility::to_u64,
    view::{AnyBitPatternDeviceRepr, DeviceOutput, DeviceRegionProvider, DeviceViewMut},
};

mod sealed {
    use super::{AnyBitPatternDeviceRepr, Result, sys};

    pub trait Pattern: AnyBitPatternDeviceRepr {
        unsafe fn enqueue(
            pointer: *mut Self,
            count: usize,
            value: Self,
            stream: sys::CUstream,
        ) -> Result<()>;
    }
}

/// Word types supported by native pattern fills: `u16` and `u32`.
pub trait FillPattern: AnyBitPatternDeviceRepr + sealed::Pattern + Send + Sync {}

macro_rules! pattern {
    ($ty:ty, $native:ident) => {
        impl FillPattern for $ty {}
        impl sealed::Pattern for $ty {
            unsafe fn enqueue(
                pointer: *mut Self,
                count: usize,
                value: Self,
                stream: sys::CUstream,
            ) -> Result<()> {
                count
                    .checked_mul(size_of::<Self>())
                    .ok_or(Error::InvalidMemoryAllocationRequest)?;
                if count == 0 {
                    return Ok(());
                }
                if pointer.is_null() || !(pointer as usize).is_multiple_of(align_of::<Self>()) {
                    return Err(Error::InvalidMemoryAccess);
                }
                // SAFETY: the caller upholds this function's safety contract.
                unsafe {
                    try_ffi!(sys::$native(
                        pointer as _,
                        value,
                        to_u64(count, "pattern element count")?,
                        stream
                    ))
                }
            }
        }
    };
}

pattern!(u16, cuMemsetD16Async);

pattern!(u32, cuMemsetD32Async);

/// Fills every word and returns an initialized writable view after completion.
///
/// # Errors
///
/// Returns an error if the destination context or access declarations are invalid, access tracking has failed or is reentrant, or CUDA cannot synchronize conflicting work, bind the context, enqueue the fill, or complete it. Also rejects byte-size overflow and a nonempty null or misaligned destination, and propagates errors from constructing the initialized output view.
pub fn fill<T: FillPattern, B: DeviceOutput<T> + ?Sized>(
    destination: &mut B,
    value: T,
) -> Result<DeviceViewMut<'_, T>> {
    let context = Arc::clone(destination.context());
    let _access = context.acquire_host_access(&[destination.region()])?;
    context.bind()?;
    let binding = StreamBinding::Default(Arc::clone(&context));
    // SAFETY: the destination region is retained by the declared access, and the stream orders the fill.
    binding.sync_scope(|| unsafe {
        T::enqueue(
            destination.as_mut_ptr(),
            destination.len(),
            value,
            binding.as_raw(),
        )
    })?;
    // SAFETY: the full exclusive output contains valid words after the completed fill.
    unsafe { DeviceViewMut::from_raw_parts(destination.region(), destination.len()) }
}

/// Fills borrowed storage and completes before the enclosing stream scope returns.
///
/// # Errors
///
/// Returns `Error::MemoryContextMismatch` if the destination and scope use different contexts. Also returns an error for byte-size overflow, invalid destination alignment or address, failed or nested access tracking, dependency ordering, completion-event creation or recording, or native fill failure. Work may already be queued when a later registration step fails.
pub fn fill_async<'env, T: FillPattern, B: DeviceOutput<T> + ?Sized>(
    destination: &'env mut B,
    value: T,
    scope: &StreamScope<'_, 'env>,
) -> Result<()> {
    if destination.context() != scope.stream().context() {
        return Err(Error::MemoryContextMismatch);
    }
    destination.byte_len()?;
    let accesses = [Access::new(destination.region(), AccessMode::Write)];
    // SAFETY: the destination region is retained by the declared access, and the stream orders the fill.
    unsafe {
        scope.enqueue(&accesses, || {
            T::enqueue(
                destination.as_mut_ptr(),
                destination.len(),
                value,
                scope.stream().as_raw(),
            )
        })
    }
}

/// Enqueues a raw word-pattern fill without retaining storage or declaring accesses.
///
/// # Safety
/// The pointer must identify `count` writable words in this stream's context.
/// Keep it live and exclusive until completion, including every replay if captured.
///
/// # Errors
///
/// Returns an error if the stream context cannot be bound, `count * size_of::<T>()` overflows, a nonempty destination is null or misaligned, the count does not fit the native integer type, or CUDA rejects the fill. A zero count succeeds without submitting a native fill after binding the context.
pub unsafe fn fill_async_unchecked<T: FillPattern>(
    pointer: *mut T,
    count: usize,
    value: T,
    stream: &Stream,
) -> Result<()> {
    stream.context().bind()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::enqueue(pointer, count, value, stream.as_raw()) }
}

/// Prepared owning fill with storage allocated before submission or capture.
#[derive(Debug)]
pub struct PatternFill<T: FillPattern> {
    destination: DeviceMemory<MaybeUninit<T>>,
    value: T,
}

impl<T: FillPattern> PatternFill<T> {
    /// Allocates `count` uninitialized words in `context` and stores the pattern value for deferred filling.
    ///
    /// No fill is submitted until this operation is submitted or recorded.
    ///
    /// # Errors
    ///
    /// Propagates errors from device allocation, including byte-size overflow, context binding, and native allocation failures.
    pub fn create(context: &Arc<Context>, count: usize, value: T) -> Result<Self> {
        Ok(Self {
            destination: DeviceMemory::<T>::create_uninit(context, count)?,
            value,
        })
    }

    /// Borrows the destination allocation without claiming it has been initialized.
    ///
    /// This permits binding graph consumers before recording the fill.
    pub fn destination(&self) -> &DeviceMemory<MaybeUninit<T>> {
        &self.destination
    }

    fn resources(&self) -> Resources {
        let mut resources = Resources::new();
        resources.retain(Arc::clone(self.destination.allocation()));
        resources.access(self.destination.region(), AccessMode::Write);
        resources
    }
}

// SAFETY: the allocation is retained and every output word is written to a valid bit pattern.
unsafe impl<T: FillPattern> Operation for PatternFill<T> {
    type Pending = DeviceValue<T>;

    type Output = DeviceValue<T>;

    fn enqueue(self, execution: &mut Execution) -> Result<Self::Pending> {
        // SAFETY: the declared resources retain every buffer and the stream orders all access to them.
        unsafe {
            execution.enqueue(self.resources(), |stream| {
                fill_async_unchecked(
                    self.destination.as_ptr().cast_mut().cast(),
                    self.destination.len(),
                    self.value,
                    stream,
                )
            })?;
            DeviceValue::from_pending(self.destination, execution.stream())
        }
    }

    unsafe fn resolve(pending: Self::Pending) -> Result<Self::Output> {
        Ok(pending)
    }
}

// SAFETY: resolution preserves the pending device value unchanged.
unsafe impl<T: FillPattern> DeviceOperation for PatternFill<T> {}

// SAFETY: capture only records a fill of preallocated storage retained for every replay.
unsafe impl<T: FillPattern> Recordable for PatternFill<T> {
    type Output = DeviceMemory<MaybeUninit<T>>;

    fn record(self, recording: &mut Recording) -> Result<Self::Output> {
        // SAFETY: the declared resources retain every buffer and the stream orders all access to them.
        unsafe {
            recording.enqueue(self.resources(), |stream| {
                fill_async_unchecked(
                    self.destination.as_ptr().cast_mut().cast(),
                    self.destination.len(),
                    self.value,
                    stream,
                )
            })?;
        }
        Ok(self.destination)
    }
}
