use std::{
    marker::{PhantomData, PhantomPinned},
    mem,
    pin::Pin,
    sync::Arc,
};

use singe_core::report_cleanup_failure;
use singe_cuda::{context::Context, memory::PinnedVec, stream::Stream};
use singe_cufile_sys as sys;

use crate::{
    buffer::IoSpan,
    driver::Driver,
    error::Result,
    file::{FileHandle, operation_result, prepare_stream_access},
    try_ffi,
    types::StreamRegisterFlags,
    utility::{to_i64, to_u64},
};

/// A cuFile stream registration retaining its stream and driver session.
#[derive(Debug)]
pub struct RegisteredStream {
    stream: Stream,
    driver: Arc<Driver>,
}

impl RegisteredStream {
    /// Registers `stream` with cuFile, which lets cuFile skip per-call checks that `flags` declares hold.
    ///
    /// # Safety
    ///
    /// Every stream I/O on the registered stream must satisfy the alignment and size guarantees in `flags`.
    /// The registration must not be dropped while the stream is capturing.
    ///
    /// # Errors
    ///
    /// Returns an error if binding the stream's CUDA context fails or cuFile rejects the stream registration or flags.
    pub unsafe fn register(
        driver: &Arc<Driver>,
        stream: &Stream,
        flags: StreamRegisterFlags,
    ) -> Result<Self> {
        stream.context().bind()?;
        // SAFETY: the stream is live in its bound context, and the registration retains it until deregistration.
        unsafe {
            try_ffi!(sys::cuFileStreamRegister(stream.as_raw(), flags.bits()))?;
        }
        Ok(Self {
            stream: stream.clone(),
            driver: Arc::clone(driver),
        })
    }

    /// Returns the registered stream.
    #[must_use]
    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    fn retain_registration_dependencies(&self) {
        // Native registration may still use these when deregistration cannot be established.
        mem::forget(self.stream.clone());
        mem::forget(Arc::clone(&self.driver));
    }
}

impl Drop for RegisteredStream {
    fn drop(&mut self) {
        if let Err(error) = self.stream.context().bind() {
            report_cleanup_failure(
                "bind the cuda context to deregister",
                "cufile stream",
                &error,
            );
            self.retain_registration_dependencies();
            return;
        }

        // SAFETY: registration retains the stream and its driver through native deregistration.
        if let Err(error) = unsafe { try_ffi!(sys::cuFileStreamDeregister(self.stream.as_raw())) } {
            report_cleanup_failure("deregister", "cufile stream", &error);
            self.retain_registration_dependencies();
        }
    }
}

/// A pending stream I/O operation retaining its pinned completion status and its file and buffer borrows.
///
/// Waiting or dropping the operation synchronizes its stream before releasing these resources.
#[derive(Debug)]
pub struct AsyncIo<'a> {
    state: Pin<Box<AsyncIoState>>,
    stream: Stream,
    byte_len: usize,
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
    memory: PinnedVec<i64>,
}

impl HostStatus {
    fn create(context: &Arc<Context>) -> Result<Self> {
        Ok(Self {
            memory: context.pinned_host().zeroed(1)?,
        })
    }

    fn as_mut_ptr(&mut self) -> *mut i64 {
        self.memory.as_mut_ptr()
    }

    fn get(&self) -> i64 {
        self.memory[0]
    }
}

impl<'a> AsyncIo<'a> {
    // The caller upholds the raw execution contract for the exclusively borrowed `span`.
    pub(crate) unsafe fn read_async(
        file: &FileHandle,
        span: IoSpan<'a>,
        file_offset: u64,
        stream: &Stream,
    ) -> Result<Self> {
        prepare_stream_access(&span, stream)?;
        let (pointer, buffer_offset) = span.native_address()?;
        let mut operation = Self::create(file, &span, file_offset, buffer_offset, stream)?;
        // SAFETY: the state is never moved out of its pinned box.
        let state = unsafe { operation.state.as_mut().get_unchecked_mut() };
        // SAFETY: the file handle and stream are live, the buffer range is borrowed for `'a`, and the pinned state outlives the stream work because dropping the operation waits for the stream.
        unsafe {
            try_ffi!(sys::cuFileReadAsync(
                file.as_raw(),
                pointer,
                &raw mut state.size,
                &raw mut state.file_offset,
                &raw mut state.buffer_offset,
                state.bytes_transferred.as_mut_ptr(),
                stream.as_raw(),
            ))?;
        }
        Ok(operation)
    }

    // The caller upholds the raw execution contract for the borrowed `span`.
    pub(crate) unsafe fn write_async(
        file: &FileHandle,
        span: IoSpan<'a>,
        file_offset: u64,
        stream: &Stream,
    ) -> Result<Self> {
        prepare_stream_access(&span, stream)?;
        let (pointer, buffer_offset) = span.native_address()?;
        let mut operation = Self::create(file, &span, file_offset, buffer_offset, stream)?;
        // SAFETY: the state is never moved out of its pinned box.
        let state = unsafe { operation.state.as_mut().get_unchecked_mut() };
        // SAFETY: the file handle and stream are live, the buffer range is borrowed for `'a`, and the pinned state outlives the stream work because dropping the operation waits for the stream.
        unsafe {
            try_ffi!(sys::cuFileWriteAsync(
                file.as_raw(),
                pointer,
                &raw mut state.size,
                &raw mut state.file_offset,
                &raw mut state.buffer_offset,
                state.bytes_transferred.as_mut_ptr(),
                stream.as_raw(),
            ))?;
        }
        Ok(operation)
    }

    /// Reads the native completion status after the caller establishes stream completion.
    ///
    /// # Safety
    ///
    /// The stream must have completed this operation; no native status write may still be in flight.
    ///
    /// # Errors
    ///
    /// Returns an error if the native result is negative, as a cuFile or operating-system error, or if the byte count cannot be represented as `usize`.
    pub unsafe fn bytes_transferred(&self) -> Result<usize> {
        operation_result(self.state.bytes_transferred.get())
    }

    /// Returns the requested length of the transfer, in bytes.
    #[must_use]
    pub fn requested_byte_len(&self) -> usize {
        self.byte_len
    }

    fn create(
        file: &FileHandle,
        span: &IoSpan<'a>,
        file_offset: u64,
        buffer_offset: i64,
        stream: &Stream,
    ) -> Result<Self> {
        Ok(Self {
            state: Box::pin(AsyncIoState {
                size: to_u64(span.byte_len(), "byte length")?,
                file_offset: to_i64(file_offset, "file_offset")?,
                buffer_offset,
                bytes_transferred: HostStatus::create(stream.context())?,
                _pin: PhantomPinned,
            }),
            stream: stream.clone(),
            byte_len: span.byte_len(),
            _file: file.clone(),
            _buffer: PhantomData,
            completed: false,
        })
    }
}

impl AsyncIo<'_> {
    /// Waits for the stream, then returns the number of bytes transferred.
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
        // Pinned status and the caller's borrowed storage cannot be released while native work may write them.
        if !self.completed {
            self.stream.synchronize_or_abort();
        }
    }
}
