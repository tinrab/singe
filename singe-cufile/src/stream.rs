use std::{
    marker::{PhantomData, PhantomPinned},
    mem,
    pin::Pin,
    process,
    sync::Arc,
};

use singe_cuda::{
    context::Context,
    memory::{HostAllocationFlags, PinnedHostMemory},
    stream::StreamBinding,
};
use singe_cufile_sys as sys;

use crate::{
    buffer::{IoBuffer, IoBufferMut},
    driver::Driver,
    error::{Error, Result},
    file::{FileHandle, checked_io_size, io_pointer, operation_result, prepare_buffer},
    try_ffi,
    types::StreamRegisterFlags,
    utility::{to_i64, to_u64},
};

/// A cuFile stream registration retaining its stream binding and driver session.
#[derive(Debug)]
pub struct RegisteredStream {
    binding: StreamBinding,
    driver: Arc<Driver>,
}

impl RegisteredStream {
    /// Registers a native stream whose lifecycle is coordinated by its caller.
    ///
    /// # Safety
    /// Keep foreign borrowed streams alive through deregistration and do not deregister during capture.
    /// Complete queued work before releasing any resources used by its I/O.
    ///
    /// # Errors
    ///
    /// Returns an error if binding the stream's CUDA context fails or cuFile rejects the stream registration or flags.
    pub unsafe fn register(
        driver: &Arc<Driver>,
        binding: StreamBinding,
        flags: StreamRegisterFlags,
    ) -> Result<Self> {
        binding.context().bind()?;
        // SAFETY: `cuFileStreamRegister` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::cuFileStreamRegister(binding.as_raw(), flags.bits()))?;
        }
        Ok(Self {
            binding,
            driver: Arc::clone(driver),
        })
    }

    /// Returns the retained CUDA stream binding.
    #[must_use]
    pub fn binding(&self) -> &StreamBinding {
        &self.binding
    }
}

impl Drop for RegisteredStream {
    fn drop(&mut self) {
        if self
            .binding
            .context()
            .bind()
            .map_err(Error::from)
            .and_then(|()| {
                // SAFETY: registration retains the binding and its driver through native deregistration.
                unsafe { try_ffi!(sys::cuFileStreamDeregister(self.binding.as_raw())) }
            })
            .is_err()
        {
            // Keep native registration dependencies alive when deregistration cannot be established.
            mem::forget(self.binding.clone());
            mem::forget(Arc::clone(&self.driver));
        }
    }
}

/// A pending I/O operation retaining pinned completion status and its file and buffer borrows.
///
/// Waiting or dropping the operation synchronizes its stream before releasing these resources.
#[derive(Debug)]
pub struct AsyncIo<'a> {
    state: Pin<Box<AsyncIoState>>,
    stream: StreamBinding,
    _file: FileHandle,
    _buffer: PhantomData<&'a mut ()>,
    completed: bool,
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
    memory: PinnedHostMemory<i64>,
}

impl HostStatus {
    fn create(context: &Arc<Context>) -> Result<Self> {
        Ok(Self {
            memory: PinnedHostMemory::zeroed(context, 1, HostAllocationFlags::DEFAULT)?,
        })
    }

    fn as_mut_ptr(&mut self) -> *mut i64 {
        self.memory.as_mut_ptr()
    }

    fn get(&self) -> i64 {
        self.memory.as_slice()[0]
    }
}

impl<'a> AsyncIo<'a> {
    /// Enqueues a borrowed read whose representation and lifetimes are established by its caller.
    ///
    /// # Safety
    /// Retain the buffer and exclusive access until stream completion, even after errors or forgotten handles.
    /// File bytes must leave valid element values or storage that remains explicitly uninitialized.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer length cannot be computed, the requested byte range exceeds the buffer, or a size or offset exceeds its native integer range.
    /// Also returns an error if CUDA readiness, capture checks, or context binding fails, or cuFile reports an I/O error.
    /// The buffer and stream must belong to the same CUDA context; allocating the pinned completion status can also fail.
    pub unsafe fn read<B>(
        file: &FileHandle,
        buffer: &'a mut B,
        size: usize,
        file_offset: u64,
        buffer_offset: u64,
        stream: &StreamBinding,
    ) -> Result<Self>
    where
        B: IoBufferMut + ?Sized,
    {
        prepare_buffer(buffer, Some(stream))?;
        let size = checked_io_size(buffer.byte_len()?, size, buffer_offset)?;
        let pointer = buffer.ptr_mut();
        let (pointer, native_offset) = io_pointer(buffer, pointer, size, buffer_offset)?;
        let mut operation = Self::create(file, size, file_offset, native_offset as u64, stream)?;
        // SAFETY: the caller upholds this function's safety contract.
        let state = unsafe { operation.state.as_mut().get_unchecked_mut() };
        // SAFETY: `cuFileReadAsync` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::cuFileReadAsync(
                file.as_raw(),
                pointer.cast(),
                &raw mut state.size,
                &raw mut state.file_offset,
                &raw mut state.buffer_offset,
                state.bytes_transferred.as_mut_ptr(),
                stream.as_raw(),
            ))?;
        }
        Ok(operation)
    }

    /// Enqueues a borrowed write from externally established initialized bytes.
    ///
    /// # Safety
    /// Input bytes must be initialized when accessed, with prior producers ordered before this I/O.
    /// Retain backing storage and read access until stream completion on every path, including forgotten handles.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer length cannot be computed, the requested byte range exceeds the buffer, or a size or offset exceeds its native integer range.
    /// Also returns an error if CUDA readiness, capture checks, or context binding fails, or cuFile reports an I/O error.
    /// The buffer and stream must belong to the same CUDA context; allocating the pinned completion status can also fail.
    pub unsafe fn write<B>(
        file: &FileHandle,
        buffer: &'a B,
        size: usize,
        file_offset: u64,
        buffer_offset: u64,
        stream: &StreamBinding,
    ) -> Result<Self>
    where
        B: IoBuffer + ?Sized,
    {
        prepare_buffer(buffer, Some(stream))?;
        let size = checked_io_size(buffer.byte_len()?, size, buffer_offset)?;
        let (pointer, native_offset) = io_pointer(buffer, buffer.ptr(), size, buffer_offset)?;
        let mut operation = Self::create(file, size, file_offset, native_offset as u64, stream)?;
        // SAFETY: the caller upholds this function's safety contract.
        let state = unsafe { operation.state.as_mut().get_unchecked_mut() };
        // SAFETY: `cuFileWriteAsync` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::cuFileWriteAsync(
                file.as_raw(),
                pointer.cast(),
                &raw mut state.size,
                &raw mut state.file_offset,
                &raw mut state.buffer_offset,
                state.bytes_transferred.as_mut_ptr(),
                stream.as_raw(),
            ))?;
        }
        Ok(operation)
    }

    /// Reads native completion status after the caller establishes successful stream completion.
    ///
    /// # Safety
    /// The stream must have completed this operation; no native status write may still be in flight.
    ///
    /// # Errors
    ///
    /// Returns a cuFile or operating-system error for a negative native result, or an error if the byte count cannot be represented as `usize`.
    pub unsafe fn bytes_transferred(&self) -> Result<usize> {
        operation_result(self.state.bytes_transferred.get())
    }

    /// Returns the byte count stored in the native submission state.
    ///
    /// # Panics
    ///
    /// Panics if the native size no longer fits `usize`; construction validates the original size as a host byte length.
    #[must_use]
    pub fn requested_size(&self) -> usize {
        usize::try_from(self.state.size)
            .expect("requested size was validated as a host byte length")
    }

    fn create(
        file: &FileHandle,
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
                bytes_transferred: HostStatus::create(stream.context())?,
                _pin: PhantomPinned,
            }),
            stream: stream.clone(),
            _file: file.clone(),
            _buffer: PhantomData,
            completed: false,
        })
    }
}

impl AsyncIo<'_> {
    /// Completes native stream work before inspecting its pinned status.
    ///
    /// # Errors
    ///
    /// Returns an error if stream synchronization fails or the completed operation reports an I/O failure or an unrepresentable byte count.
    pub fn wait(mut self) -> Result<usize> {
        self.stream.synchronize()?;
        self.completed = true;
        // SAFETY: successful stream synchronization includes the native I/O and status write.
        unsafe { self.bytes_transferred() }
    }
}

impl Drop for AsyncIo<'_> {
    fn drop(&mut self) {
        if !self.completed && self.stream.synchronize().is_err() {
            // Retry before releasing pinned status or the caller's borrowed storage.
            if self.stream.synchronize().is_err() {
                process::abort();
            }
        }
    }
}
