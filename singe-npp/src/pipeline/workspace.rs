use singe_cuda::{
    memory::DeviceMemory,
    types::f16,
    types::{Complex32, Complex64},
};

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        memory::Image,
        view::{AC4, C1, C2, C3, C4},
    },
    signal::memory::Signal,
    types::{ComplexI16, ComplexI32, ComplexI64, Size},
    workspace::ScratchBuffer,
};

#[doc(hidden)]
pub trait ImageAllocator<T, L> {
    fn create_image(size: Size) -> Result<Image<T, L>>;
}

#[doc(hidden)]
pub trait SignalAllocator<T> {
    fn create_signal(len: usize) -> Result<Signal<T>>;
}

macro_rules! impl_image_allocator {
    ($ty:ty, $layout:ty) => {
        impl ImageAllocator<$ty, $layout> for Workspace {
            fn create_image(size: Size) -> Result<Image<$ty, $layout>> {
                Image::<$ty, $layout>::create(size)
            }
        }
    };
}

impl_image_allocator!(u8, C1);
impl_image_allocator!(u8, C2);
impl_image_allocator!(u8, C3);
impl_image_allocator!(u8, C4);
impl_image_allocator!(u8, AC4);
impl_image_allocator!(i8, C1);
impl_image_allocator!(i8, C2);
impl_image_allocator!(i8, C3);
impl_image_allocator!(i8, C4);
impl_image_allocator!(i8, AC4);
impl_image_allocator!(u16, C1);
impl_image_allocator!(u16, C2);
impl_image_allocator!(u16, C3);
impl_image_allocator!(u16, C4);
impl_image_allocator!(u16, AC4);
impl_image_allocator!(f16, C1);
impl_image_allocator!(f16, C2);
impl_image_allocator!(f16, C3);
impl_image_allocator!(f16, C4);
impl_image_allocator!(f16, AC4);
impl_image_allocator!(i16, C1);
impl_image_allocator!(i16, C2);
impl_image_allocator!(i16, C3);
impl_image_allocator!(i16, C4);
impl_image_allocator!(i16, AC4);
impl_image_allocator!(u32, C1);
impl_image_allocator!(u32, AC4);
impl_image_allocator!(i32, C1);
impl_image_allocator!(i32, C3);
impl_image_allocator!(i32, C4);
impl_image_allocator!(i32, AC4);
impl_image_allocator!(f32, C1);
impl_image_allocator!(f32, C2);
impl_image_allocator!(f32, C3);
impl_image_allocator!(f32, C4);
impl_image_allocator!(f32, AC4);
impl_image_allocator!(ComplexI16, C1);
impl_image_allocator!(ComplexI16, C2);
impl_image_allocator!(ComplexI16, C3);
impl_image_allocator!(ComplexI16, C4);
impl_image_allocator!(ComplexI16, AC4);
impl_image_allocator!(ComplexI32, C1);
impl_image_allocator!(ComplexI32, C2);
impl_image_allocator!(ComplexI32, C3);
impl_image_allocator!(ComplexI32, C4);
impl_image_allocator!(ComplexI32, AC4);
impl_image_allocator!(Complex32, C1);
impl_image_allocator!(Complex32, C2);
impl_image_allocator!(Complex32, C3);
impl_image_allocator!(Complex32, C4);
impl_image_allocator!(Complex32, AC4);

macro_rules! impl_signal_allocator {
    ($ty:ty) => {
        impl SignalAllocator<$ty> for Workspace {
            fn create_signal(len: usize) -> Result<Signal<$ty>> {
                Signal::<$ty>::create(len)
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
    scratch_pool: Vec<DeviceMemory<u8>>,
}

impl Workspace {
    pub fn create() -> Self {
        Self::default()
    }

    pub fn image<T, L>(&mut self, size: Size) -> Result<Image<T, L>>
    where
        Self: ImageAllocator<T, L>,
    {
        <Self as ImageAllocator<T, L>>::create_image(size)
    }

    pub fn signal<T>(&mut self, len: usize) -> Result<Signal<T>>
    where
        Self: SignalAllocator<T>,
    {
        <Self as SignalAllocator<T>>::create_signal(len)
    }

    pub fn scratch(&mut self, bytes: usize) -> Result<ScratchBuffer> {
        if let Some(index) = self
            .scratch_pool
            .iter()
            .position(|buffer| buffer.len() >= bytes)
        {
            return Ok(ScratchBuffer::from_memory(
                self.scratch_pool.swap_remove(index),
            ));
        }

        ScratchBuffer::create(bytes)
    }

    fn recycle_scratch(&mut self, buffer: ScratchBuffer) {
        self.scratch_pool.push(buffer.into_device_memory());
    }

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

    pub fn scratch_buffers(&self) -> usize {
        self.scratch_pool.len()
    }
}

pub(crate) fn with_temporary_scratch<R>(
    stream_context: &StreamContext,
    bytes: usize,
    operation: impl FnOnce(&mut ScratchBuffer) -> Result<R>,
) -> Result<R> {
    let mut scratch = ScratchBuffer::create(bytes)?;
    let operation_result = operation(&mut scratch);
    // Keep stack-owned scratch alive until stream work that may touch it is done.
    let synchronize_result = stream_context.synchronize();

    match operation_result {
        Ok(value) => {
            synchronize_result?;
            Ok(value)
        }
        Err(err) => {
            let _ = synchronize_result;
            Err(err)
        }
    }
}
