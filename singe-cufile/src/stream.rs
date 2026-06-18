use std::{
    marker::{PhantomData, PhantomPinned},
    mem::size_of,
    pin::Pin,
    ptr::NonNull,
};

use singe_cuda::{
    memory::{DeviceMemory, HostAllocationFlags},
    stream::StreamBinding,
};
use singe_cufile_sys as sys;

use crate::{
    buffer::{IoBuffer, IoBufferMut},
    error::{Error, Result},
    file::{FileHandle, checked_io_size, operation_result},
    try_ffi,
    types::StreamRegisterFlags,
    utility::{to_i64, to_u64},
};

#[derive(Debug)]
pub struct RegisteredStream {
    binding: StreamBinding,
}

impl RegisteredStream {
    pub fn register(binding: StreamBinding, flags: StreamRegisterFlags) -> Result<Self> {
        binding.context().bind()?;
        unsafe {
            try_ffi!(sys::cuFileStreamRegister(binding.as_raw(), flags.bits()))?;
        }
        Ok(Self { binding })
    }

    pub fn binding(&self) -> &StreamBinding {
        &self.binding
    }
}

impl Drop for RegisteredStream {
    fn drop(&mut self) {
        unsafe {
            let _ = sys::cuFileStreamDeregister(self.binding.as_raw());
        }
    }
}

#[derive(Debug)]
pub struct AsyncIo<'a> {
    state: Pin<Box<AsyncIoState>>,
    stream: StreamBinding,
    _file: PhantomData<&'a FileHandle>,
    _buffer: PhantomData<&'a mut ()>,
    _stream: PhantomData<&'a StreamBinding>,
}

#[derive(Debug)]
struct AsyncIoState {
    size: sys::size_t,
    file_offset: sys::off_t,
    buffer_offset: sys::off_t,
    bytes_transferred: HostStatus,
    _pin: PhantomPinned,
}

#[derive(Debug)]
struct HostStatus {
    ptr: NonNull<i64>,
}

impl HostStatus {
    fn create() -> Result<Self> {
        let ptr = unsafe {
            DeviceMemory::<u8>::alloc_pinned(size_of::<i64>(), HostAllocationFlags::DEFAULT)?
        };
        let ptr = NonNull::new(ptr.cast::<i64>()).ok_or(Error::NullHandle)?;
        unsafe {
            ptr.as_ptr().write(0);
        }
        Ok(Self { ptr })
    }

    fn as_mut_ptr(&mut self) -> *mut i64 {
        self.ptr.as_ptr()
    }

    fn get(&self) -> i64 {
        unsafe { self.ptr.as_ptr().read() }
    }
}

impl Drop for HostStatus {
    fn drop(&mut self) {
        unsafe {
            let _ = DeviceMemory::<u8>::free_host(self.ptr.as_ptr().cast());
        }
    }
}

impl<'a> AsyncIo<'a> {
    pub fn read<B>(
        file: &'a FileHandle,
        buffer: &'a mut B,
        size: usize,
        file_offset: u64,
        buffer_offset: u64,
        stream: &'a StreamBinding,
    ) -> Result<Self>
    where
        B: IoBufferMut + ?Sized,
    {
        stream.context().bind()?;
        let size = checked_io_size(buffer.byte_len()?, size, buffer_offset)?;
        let mut operation = Self::create(size, file_offset, buffer_offset, stream)?;
        let state = unsafe { operation.state.as_mut().get_unchecked_mut() };
        unsafe {
            try_ffi!(sys::cuFileReadAsync(
                file.as_raw(),
                buffer.ptr_mut().as_mut() as _,
                &raw mut state.size,
                &raw mut state.file_offset,
                &raw mut state.buffer_offset,
                state.bytes_transferred.as_mut_ptr(),
                stream.as_raw(),
            ))?;
        }
        Ok(operation)
    }

    pub fn write<B>(
        file: &'a FileHandle,
        buffer: &'a B,
        size: usize,
        file_offset: u64,
        buffer_offset: u64,
        stream: &'a StreamBinding,
    ) -> Result<Self>
    where
        B: IoBuffer + ?Sized,
    {
        stream.context().bind()?;
        let size = checked_io_size(buffer.byte_len()?, size, buffer_offset)?;
        let mut operation = Self::create(size, file_offset, buffer_offset, stream)?;
        let state = unsafe { operation.state.as_mut().get_unchecked_mut() };
        unsafe {
            try_ffi!(sys::cuFileWriteAsync(
                file.as_raw(),
                buffer.ptr().as_mut() as _,
                &raw mut state.size,
                &raw mut state.file_offset,
                &raw mut state.buffer_offset,
                state.bytes_transferred.as_mut_ptr(),
                stream.as_raw(),
            ))?;
        }
        Ok(operation)
    }

    pub fn bytes_transferred(&self) -> Result<usize> {
        operation_result(self.state.bytes_transferred.get())
    }

    pub fn requested_size(&self) -> usize {
        self.state.size as usize
    }

    fn create(
        size: usize,
        file_offset: u64,
        buffer_offset: u64,
        stream: &StreamBinding,
    ) -> Result<Self> {
        Ok(Self {
            state: Box::pin(AsyncIoState {
                size: to_u64(size, "size")?,
                file_offset: to_i64(file_offset, "file_offset")?,
                buffer_offset: to_i64(buffer_offset, "buffer_offset")?,
                bytes_transferred: HostStatus::create()?,
                _pin: PhantomPinned,
            }),
            stream: stream.clone(),
            _file: PhantomData,
            _buffer: PhantomData,
            _stream: PhantomData,
        })
    }
}

impl Drop for AsyncIo<'_> {
    fn drop(&mut self) {
        let result = match &self.stream {
            StreamBinding::Default(context) => context.synchronize(),
            StreamBinding::Borrowed(stream) => stream.synchronize(),
            _ => Ok(()),
        };
        if let Err(err) = result {
            #[cfg(debug_assertions)]
            eprintln!("failed to synchronize stream before dropping cufile async io: {err}");
        }
    }
}
