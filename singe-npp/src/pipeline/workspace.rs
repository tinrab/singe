use std::{mem::MaybeUninit, sync::Arc};

use singe_cuda::{
    context::Context as CudaContext,
    memory::DeviceMemory,
    types::{Complex32, Complex64},
};

use crate::{
    context::StreamContext,
    error::Result,
    signal::memory::Signal,
    types::{ComplexI16, ComplexI32, ComplexI64},
    workspace::ScratchBuffer,
};

#[doc(hidden)]
/// Implementations return exactly the requested nonempty range in the supplied
/// context, retaining correct native ownership and the plain device ABI of T.
pub unsafe trait SignalAllocator<T> {
    fn create_signal_uninit(
        context: &Arc<CudaContext>,
        len: usize,
    ) -> Result<Signal<MaybeUninit<T>>>;
}

macro_rules! impl_signal_allocator {
    ($ty:ty) => {
        unsafe impl SignalAllocator<$ty> for Workspace {
            fn create_signal_uninit(
                context: &Arc<CudaContext>,
                len: usize,
            ) -> Result<Signal<MaybeUninit<$ty>>> {
                Signal::<$ty>::create_uninit(context, len)
            }
        }
    };
}

impl_signal_allocator!(u8);

impl_signal_allocator!(i8);

impl_signal_allocator!(u16);

impl_signal_allocator!(u32);

impl_signal_allocator!(i16);

impl_signal_allocator!(i32);

impl_signal_allocator!(i64);

impl_signal_allocator!(f32);

impl_signal_allocator!(f64);

impl_signal_allocator!(ComplexI16);

impl_signal_allocator!(ComplexI32);

impl_signal_allocator!(ComplexI64);

impl_signal_allocator!(Complex32);

impl_signal_allocator!(Complex64);

/// Workspace used by fluent NPP pipelines.
#[derive(Debug, Default)]
pub struct Workspace {
    scratch_pool: Vec<DeviceMemory<MaybeUninit<u8>>>,
}

impl Workspace {
    /// Creates a pipeline workspace with an empty scratch pool.
    #[must_use]
    pub fn create() -> Self {
        Self::default()
    }

    /// Allocates a nonempty uninitialized signal of `len` elements in `context`.
    ///
    /// # Errors
    ///
    /// Returns an error if `len` is zero, the allocation size overflows, or the native element count cannot be represented.
    /// CUDA context binding, NPP allocation, and allocation metadata or alignment validation can also fail.
    pub fn signal_uninit<T>(
        &mut self,
        context: &Arc<CudaContext>,
        len: usize,
    ) -> Result<Signal<MaybeUninit<T>>>
    where
        Self: SignalAllocator<T>,
    {
        <Self as SignalAllocator<T>>::create_signal_uninit(context, len)
    }

    /// Obtains scratch storage with capacity of at least `bytes` in `context`.
    ///
    /// Removes and reuses the first sufficiently large allocation from the same context, or allocates new scratch if no match exists.
    ///
    /// # Errors
    ///
    /// Returns an error if a new allocation is needed and CUDA context binding or device allocation fails.
    pub fn scratch(&mut self, context: &Arc<CudaContext>, bytes: usize) -> Result<ScratchBuffer> {
        if let Some(index) = self.scratch_pool.iter().position(|buffer| {
            buffer.context().as_ref() == context.as_ref() && buffer.len() >= bytes
        }) {
            return Ok(ScratchBuffer::from_memory(
                self.scratch_pool.swap_remove(index),
            ));
        }

        ScratchBuffer::create(context, bytes)
    }

    fn recycle_scratch(&mut self, buffer: ScratchBuffer) {
        self.scratch_pool.push(buffer.into_device_memory());
    }

    /// Waits for `stream_context` and returns `buffer` to the scratch pool.
    ///
    /// The supplied stream must be the stream whose work uses the buffer; this method does not check that association.
    /// The buffer enters the pool even if synchronization reports an error.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding or stream synchronization fails.
    pub fn recycle_scratch_after(
        &mut self,
        stream_context: &StreamContext,
        buffer: ScratchBuffer,
    ) -> Result<()> {
        // Scratch buffers are passed to asynchronous NPP work, so they cannot
        // return to the reusable pool until the issuing stream is complete.
        let synchronize_result = stream_context.synchronize();
        self.recycle_scratch(buffer);
        synchronize_result
    }

    /// Returns the number of scratch allocations currently available in the pool.
    #[must_use]
    pub fn scratch_buffers(&self) -> usize {
        self.scratch_pool.len()
    }
}

/// Runs `operation` with a temporary device scratch allocation of `bytes` bytes.
///
/// Rejects stream capture before allocation and drains the stream before releasing scratch, including errors and unwinding.
/// The callback receives uninitialized byte storage in the stream context.
///
/// # Errors
///
/// Returns an error if capture checks, CUDA context binding, scratch allocation, the callback, or stream synchronization fail.
///
/// # Panics
///
/// Propagates a panic from `operation` after cleanup drains the stream.
pub fn with_temporary_scratch<R>(
    stream_context: &StreamContext,
    bytes: usize,
    operation: impl FnOnce(&mut ScratchBuffer) -> Result<R>,
) -> Result<R> {
    // Keep scratch outside the callback so it outlives the stream drain on
    // error and unwind; the scope rejects capture before allocation.
    let mut scratch = None;
    stream_context.sync_scope(|| {
        scratch = Some(ScratchBuffer::create(stream_context.cuda_context(), bytes)?);
        operation(scratch.as_mut().expect("scratch allocation succeeded"))
    })
}
