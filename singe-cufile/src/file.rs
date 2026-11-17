use std::{
    io,
    mem::ManuallyDrop,
    os::fd::{AsFd, AsRawFd, OwnedFd},
    ptr,
    result::Result as StdResult,
    sync::Arc,
};

use singe_cuda::{
    context::Context,
    error::{Error as CudaError, Status as CudaStatus},
    stream::{AsStream, Stream, StreamBinding, StreamCaptureStatus, StreamScope},
    submission::{Access, AccessMode},
    view::DeviceRegion,
};
use singe_cufile_sys as sys;

use crate::{
    buffer::{IoSlice, IoSliceMut, IoSpan},
    driver::Driver,
    error::{Error, Result, Status},
    stream::AsyncIo,
    try_ffi,
    utility::{to_i64, to_u64, to_usize},
};

/// A shared cuFile registration that retains its driver session and its file descriptor.
///
/// The last clone deregisters the native handle.
#[derive(Debug, Clone)]
pub struct FileHandle {
    inner: Arc<FileOwner>,
}

#[derive(Debug)]
struct FileOwner {
    handle: sys::CUfileHandle_t,
    file: Option<OwnedFd>,
    driver: Arc<Driver>,
}

/// Transferred native ownership of a file registration and the owners it depends on.
#[derive(Debug)]
pub struct RawFileHandle {
    /// The registered native handle.
    pub handle: sys::CUfileHandle_t,
    /// The registered file descriptor when the wrapper owned it; `None` when the caller keeps the descriptor open.
    pub file: Option<OwnedFd>,
    /// The driver session the handle was registered in.
    pub driver: Arc<Driver>,
}

// SAFETY: NVIDIA documents every cuFile API as thread-safe, and the owner's handle and descriptor never change after registration, so it may move between threads.
unsafe impl Send for FileOwner {}

// SAFETY: NVIDIA documents every cuFile API as thread-safe, and `&FileOwner` exposes only the immutable handle, so concurrent I/O through shared clones is allowed.
unsafe impl Sync for FileOwner {}

impl FileHandle {
    /// Takes ownership of a registered native file handle; the last clone deregisters it.
    ///
    /// # Safety
    ///
    /// `raw.handle` must be a registered native file handle, such as one from [`Self::into_raw`], in the session kept open by `raw.driver`, and no other owner may deregister it.
    /// When `raw.file` is set, it must be the descriptor the handle was registered with; otherwise the caller must keep that descriptor open for the handle's lifetime.
    ///
    /// # Errors
    ///
    /// Returns an error if `raw.handle` is null.
    pub unsafe fn from_raw(raw: RawFileHandle) -> Result<Self> {
        if raw.handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            inner: Arc::new(FileOwner {
                handle: raw.handle,
                file: raw.file,
                driver: raw.driver,
            }),
        })
    }

    /// Releases ownership of the native handle, its file descriptor, and its driver session when this is the last clone.
    ///
    /// # Errors
    ///
    /// Returns an error holding this wrapper unchanged if other clones still share the native registration.
    pub fn into_raw(self) -> StdResult<RawFileHandle, Self> {
        match Arc::try_unwrap(self.inner) {
            Ok(owner) => {
                let owner = ManuallyDrop::new(owner);
                // SAFETY: `owner` is never dropped, so each owning field moves out exactly once.
                let (file, driver) = unsafe {
                    (
                        ptr::read(&raw const owner.file),
                        ptr::read(&raw const owner.driver),
                    )
                };
                Ok(RawFileHandle {
                    handle: owner.handle,
                    file,
                    driver,
                })
            }
            Err(inner) => Err(Self { inner }),
        }
    }

    /// Registers a duplicate of an open file descriptor with cuFile.
    ///
    /// The wrapper keeps the duplicate open until the last clone is dropped.
    ///
    /// # Errors
    ///
    /// Returns an error if duplicating the descriptor fails, cuFile rejects the file registration, or the native handle is null.
    pub fn create(driver: &Arc<Driver>, file: &impl AsFd) -> Result<Self> {
        let file = file.as_fd().try_clone_to_owned()?;
        let mut descriptor = sys::CUfileDescr_t {
            type_: sys::CUfileFileHandleType::CU_FILE_HANDLE_TYPE_OPAQUE_FD,
            handle: sys::CUfileDescr_t__bindgen_ty_1 {
                fd: file.as_raw_fd(),
            },
            fs_ops: ptr::null(),
        };
        let mut handle = ptr::null_mut();
        // SAFETY: `descriptor` names the open descriptor `file`, which the owner below keeps open until deregistration, and `handle` is a valid output location.
        unsafe {
            try_ffi!(sys::cuFileHandleRegister(
                &raw mut handle,
                &raw mut descriptor
            ))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            inner: Arc::new(FileOwner {
                handle,
                file: Some(file),
                driver: Arc::clone(driver),
            }),
        })
    }

    /// Returns the borrowed native file handle.
    #[must_use]
    pub fn as_raw(&self) -> sys::CUfileHandle_t {
        self.inner.handle
    }

    /// Reads file bytes starting at `file_offset` into `buffer`, and waits for the read.
    ///
    /// Returns the number of bytes read, which may be less than the buffer's length.
    ///
    /// # Errors
    ///
    /// Returns an error if `file_offset` exceeds the native offset range, the buffer's producers cannot be waited for, its context is capturing or cannot be bound, or cuFile reports an I/O error.
    pub fn read(&self, buffer: IoSliceMut<'_>, file_offset: u64) -> Result<usize> {
        let span = buffer.into_span();
        let file_offset = to_i64(file_offset, "file_offset")?;
        if span.byte_len() == 0 {
            return Ok(0);
        }

        prepare_host_access(span.region, span.context)?;
        let (pointer, buffer_offset) = span.native_address()?;
        // SAFETY: `buffer` exclusively borrows the addressed range for this call, and `cuFileRead` completes before returning.
        let result = unsafe {
            sys::cuFileRead(
                self.inner.handle,
                pointer,
                to_u64(span.byte_len(), "byte length")?,
                file_offset,
                buffer_offset,
            )
        };
        synchronous_result(result)
    }

    /// Writes the bytes of `buffer` to the file starting at `file_offset`, and waits for the write.
    ///
    /// Returns the number of bytes written, which may be less than the buffer's length.
    ///
    /// # Errors
    ///
    /// Returns an error if `file_offset` exceeds the native offset range, the buffer's producers cannot be waited for, its context is capturing or cannot be bound, or cuFile reports an I/O error.
    pub fn write(&self, buffer: IoSlice<'_>, file_offset: u64) -> Result<usize> {
        let span = buffer.span;
        let file_offset = to_i64(file_offset, "file_offset")?;
        if span.byte_len() == 0 {
            return Ok(0);
        }

        prepare_host_access(span.region, span.context)?;
        let (pointer, buffer_offset) = span.native_address()?;
        // SAFETY: `buffer` borrows the initialized range for this call, and `cuFileWrite` completes before returning.
        let result = unsafe {
            sys::cuFileWrite(
                self.inner.handle,
                pointer,
                to_u64(span.byte_len(), "byte length")?,
                file_offset,
                buffer_offset,
            )
        };
        synchronous_result(result)
    }

    /// Reads file bytes starting at `file_offset` into several buffers in order, and waits for the read.
    ///
    /// Returns the number of bytes read.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffers belong to different CUDA contexts, a length or the total exceeds the native range, the buffers' producers cannot be waited for, the context is capturing or cannot be bound, or cuFile reports an I/O error.
    pub fn read_vectored(&self, buffers: &mut [IoSliceMut<'_>], file_offset: u64) -> Result<usize> {
        let spans: Vec<_> = buffers.iter().map(|buffer| buffer.span).collect();
        self.transfer_vectored(&spans, file_offset, sys::cuFileReadv)
    }

    /// Writes several buffers in order to the file starting at `file_offset`, and waits for the write.
    ///
    /// Returns the number of bytes written.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffers belong to different CUDA contexts, a length or the total exceeds the native range, the buffers' producers cannot be waited for, the context is capturing or cannot be bound, or cuFile reports an I/O error.
    pub fn write_vectored(&self, buffers: &[IoSlice<'_>], file_offset: u64) -> Result<usize> {
        let spans: Vec<_> = buffers.iter().map(|buffer| buffer.span).collect();
        self.transfer_vectored(&spans, file_offset, sys::cuFileWritev)
    }

    /// Enqueues a read of file bytes starting at `file_offset` into `buffer` on `stream`.
    ///
    /// The returned [`AsyncIo`] reports the byte count once the stream completes; dropping it waits for the stream.
    ///
    /// # Safety
    ///
    /// See the crate's raw execution contract: if the returned operation is forgotten, the buffer must stay alive and exclusively borrowed until the stream completes the read.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer and stream belong to different CUDA contexts, the stream is capturing, an offset or length exceeds its native range, the pinned completion status cannot be allocated, or cuFile rejects the request.
    pub unsafe fn read_async<'a>(
        &self,
        buffer: IoSliceMut<'a>,
        file_offset: u64,
        stream: &Stream,
    ) -> Result<AsyncIo<'a>> {
        // SAFETY: the caller guarantees that, if the operation is forgotten, the buffer stays alive and exclusively borrowed until the stream completes the read.
        unsafe { AsyncIo::read_async(self, buffer.into_span(), file_offset, stream) }
    }

    /// Enqueues a write of the bytes of `buffer` to the file starting at `file_offset` on `stream`.
    ///
    /// The returned [`AsyncIo`] reports the byte count once the stream completes; dropping it waits for the stream.
    ///
    /// # Safety
    ///
    /// See the crate's raw execution contract: if the returned operation is forgotten, the buffer must stay alive and unmodified until the stream completes the write.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer and stream belong to different CUDA contexts, the stream is capturing, an offset or length exceeds its native range, the pinned completion status cannot be allocated, or cuFile rejects the request.
    pub unsafe fn write_async<'a>(
        &self,
        buffer: IoSlice<'a>,
        file_offset: u64,
        stream: &Stream,
    ) -> Result<AsyncIo<'a>> {
        // SAFETY: the caller guarantees that, if the operation is forgotten, the buffer stays alive and unmodified until the stream completes the write.
        unsafe { AsyncIo::write_async(self, buffer.span, file_offset, stream) }
    }

    /// Enqueues a read into `buffer` on the scope's stream; the scope waits for it before `buffer`'s borrow ends.
    ///
    /// # Errors
    ///
    /// Returns an error if access tracking fails, or the errors of [`Self::read_async`].
    pub fn read_scoped<'env>(
        &self,
        buffer: IoSliceMut<'env>,
        file_offset: u64,
        scope: &StreamScope<'_, 'env>,
    ) -> Result<AsyncIo<'env>> {
        let access = buffer
            .span
            .region
            .map(|region| Access::new(region, AccessMode::Write));
        // SAFETY: cuFile writes only the declared buffer on the scope's stream, `buffer` is an `'env` borrow, and the scope drains its stream before `'env` ends, even if the operation is forgotten.
        unsafe {
            scope.enqueue(access.as_slice(), || {
                Ok(self.read_async(buffer, file_offset, scope.stream()))
            })
        }?
    }

    /// Enqueues a write from `buffer` on the scope's stream; the scope waits for it before `buffer`'s borrow ends.
    ///
    /// # Errors
    ///
    /// Returns an error if access tracking fails, or the errors of [`Self::write_async`].
    pub fn write_scoped<'env>(
        &self,
        buffer: IoSlice<'env>,
        file_offset: u64,
        scope: &StreamScope<'_, 'env>,
    ) -> Result<AsyncIo<'env>> {
        let access = buffer
            .span
            .region
            .map(|region| Access::new(region, AccessMode::Read));
        // SAFETY: cuFile reads only the declared buffer on the scope's stream, `buffer` is an `'env` borrow, and the scope drains its stream before `'env` ends, even if the operation is forgotten.
        unsafe {
            scope.enqueue(access.as_slice(), || {
                Ok(self.write_async(buffer, file_offset, scope.stream()))
            })
        }?
    }

    fn transfer_vectored(
        &self,
        spans: &[IoSpan<'_>],
        file_offset: u64,
        transfer: unsafe extern "C" fn(
            sys::CUfileHandle_t,
            *const sys::CUfileIOVec_t,
            sys::size_t,
            sys::off_t,
            u32,
        ) -> sys::ssize_t,
    ) -> Result<usize> {
        prepare_vectors(spans)?;
        let vectors = spans
            .iter()
            .map(IoSpan::io_vector)
            .collect::<Result<Vec<_>>>()?;
        check_vectors(&vectors)?;
        // SAFETY: the vectors address ranges whose borrows outlive this call, and the vectored call completes before returning.
        let result = unsafe {
            transfer(
                self.inner.handle,
                vectors.as_ptr(),
                to_u64(vectors.len(), "vector count")?,
                to_i64(file_offset, "file_offset")?,
                0,
            )
        };
        synchronous_result(result)
    }
}

impl Drop for FileOwner {
    fn drop(&mut self) {
        // SAFETY: the owner holds the only registration of this live handle, and every operation on it has completed or retains a clone.
        unsafe {
            if !self.handle.is_null() {
                sys::cuFileHandleDeregister(self.handle);
            }
        }
    }
}

fn check_vectors(vectors: &[sys::CUfileIOVec_t]) -> Result<()> {
    vectors.iter().try_fold(0i64, |total, vector| {
        total
            .checked_add(to_i64(vector.len, "vector length")?)
            .ok_or_else(|| Error::out_of_range("vectored io size"))
    })?;
    Ok(())
}

fn synchronous_result(result: i64) -> Result<usize> {
    if result == -1 {
        return Err(io::Error::last_os_error().into());
    }
    operation_result(result)
}

pub(crate) fn operation_result(result: i64) -> Result<usize> {
    if result < 0 {
        if result == -1 {
            return Err(io::Error::other("asynchronous cufile io failed").into());
        }

        let code = u32::try_from(result.unsigned_abs()).map_err(|_| io_failure(result))?;
        let status = Status::from(code);
        if code > sys::CUFILEOP_BASE_ERR && !matches!(status, Status::Unknown(_)) {
            return Err(status.into());
        }

        let code = i32::try_from(code).map_err(|_| io_failure(result))?;
        return Err(io::Error::from_raw_os_error(code).into());
    }
    to_usize(result, "result")
}

fn io_failure(code: i64) -> Error {
    Error::OperationFailed {
        operation: "cufile io".into(),
        code,
    }
}

// Waits for the producers of device storage and binds its context before cuFile accesses it from the host thread.
pub(crate) fn prepare_host_access(
    region: Option<DeviceRegion<'_>>,
    context: Option<&Arc<Context>>,
) -> Result<()> {
    if let Some(region) = region {
        if let Some(ready) = region.stream() {
            ready.ensure_not_capturing()?;
        }

        region.synchronize()?;
    }

    if let Some(context) = context {
        let binding = StreamBinding::Default(Arc::clone(context));
        if binding.capture_status()? != StreamCaptureStatus::None {
            return Err(CudaError::from(CudaStatus::StreamCaptureUnsupported).into());
        }

        context.bind()?;
    }
    Ok(())
}

// Checks the buffer against the stream, rejects capture, and orders the stream after the buffer's producers.
pub(crate) fn prepare_stream_access(span: &IoSpan<'_>, stream: &Stream) -> Result<()> {
    if span
        .context
        .is_some_and(|context| context != stream.context())
    {
        return Err(CudaError::MemoryContextMismatch.into());
    }

    if stream.capture_status()? != StreamCaptureStatus::None {
        return Err(CudaError::from(CudaStatus::StreamCaptureUnsupported).into());
    }

    stream.context().bind()?;
    if let Some(region) = span.region {
        stream.join_regions(&[region])?;
    }
    Ok(())
}

fn prepare_vectors(spans: &[IoSpan<'_>]) -> Result<()> {
    let mut selected: Option<&Arc<Context>> = None;
    for span in spans {
        if let Some(context) = span.context {
            if let Some(selected) = selected
                && selected != context
            {
                return Err(CudaError::MemoryContextMismatch.into());
            }

            selected = Some(context);
        }

        prepare_host_access(span.region, None)?;
    }

    prepare_host_access(None, selected)
}
