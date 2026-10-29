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
    stream::{StreamBinding, StreamCaptureStatus},
    view::DeviceRegion,
};
use singe_cufile_sys as sys;

use crate::{
    buffer::{BufferPointer, IoBuffer, IoBufferMut, IoInput, IoOutput, IoSlice, IoSliceMut},
    driver::Driver,
    error::{Error, Result, Status},
    stream::AsyncIo,
    try_ffi,
    utility::{null_fs_ops, to_i64, to_u64, to_usize},
};

/// A shared cuFile registration that retains its driver session and any owned file descriptor.
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

// SAFETY: NVIDIA documents all cuFile APIs as thread-safe; the owner has immutable state and a uniquely owned file descriptor.
unsafe impl Send for FileOwner {}

unsafe impl Sync for FileOwner {}

impl FileHandle {
    /// Takes ownership of a registered native file handle; the last clone deregisters it.
    ///
    /// # Safety
    /// `handle` must come from `cuFileHandleRegister` in the session kept open by `driver`,
    /// its file descriptor must stay open for the handle's lifetime, and no other owner may
    /// deregister it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NullHandle`] if `handle` is null.
    pub unsafe fn from_raw(driver: &Arc<Driver>, handle: sys::CUfileHandle_t) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            inner: Arc::new(FileOwner {
                handle,
                file: None,
                driver: Arc::clone(driver),
            }),
        })
    }

    /// Releases ownership of the native handle when this is the last clone.
    /// Returns `self` unchanged while other clones share the registration.
    /// A file descriptor owned by the wrapper is closed; the caller keeps the handle valid
    /// only while its own descriptor stays open.
    ///
    /// # Errors
    ///
    /// Returns this wrapper unchanged if other clones still share the native registration.
    pub fn into_raw(self) -> StdResult<sys::CUfileHandle_t, Self> {
        match Arc::try_unwrap(self.inner) {
            Ok(owner) => {
                let mut owner = ManuallyDrop::new(owner);
                let handle = owner.handle;
                // SAFETY: `owner` is never dropped, so each field is dropped exactly once here.
                unsafe {
                    ptr::drop_in_place(&raw mut owner.file);
                    ptr::drop_in_place(&raw mut owner.driver);
                }
                Ok(handle)
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
            fs_ops: null_fs_ops(),
        };
        // SAFETY: `descriptor` describes an open file that this owner keeps alive until deregistration.
        let mut owner = unsafe { Self::register_descriptor_owner(driver, &mut descriptor) }?;
        owner.file = Some(file);
        let handle = Self {
            inner: Arc::new(owner),
        };
        Ok(handle)
    }

    /// Registers a raw cuFile descriptor and takes ownership of the returned handle.
    ///
    /// # Safety
    ///
    /// `descriptor` must describe a file and filesystem operations accepted by `cuFileHandleRegister`.
    /// Resources referenced by the descriptor must stay valid for the duration of the call.
    /// The returned wrapper deregisters the cuFile handle on drop but only owns the source file when `file` is set.
    unsafe fn register_descriptor_owner(
        driver: &Arc<Driver>,
        descriptor: &mut sys::CUfileDescr_t,
    ) -> Result<FileOwner> {
        let mut handle = ptr::null_mut();
        // SAFETY: `cuFileHandleRegister` receives valid output locations.
        unsafe {
            try_ffi!(sys::cuFileHandleRegister(
                &raw mut handle,
                ptr::from_mut(descriptor),
            ))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(FileOwner {
            handle,
            file: None,
            driver: Arc::clone(driver),
        })
    }

    /// Registers an externally managed file descriptor or filesystem callback table.
    ///
    /// # Safety
    /// The descriptor must be accepted by cuFile, and all external resources and thread-safe callbacks must
    /// remain valid until the last cloned file handle and queued operation are released.
    ///
    /// # Errors
    ///
    /// Returns an error if cuFile rejects the descriptor or returns a null handle.
    pub unsafe fn register_descriptor(
        driver: &Arc<Driver>,
        descriptor: &mut sys::CUfileDescr_t,
    ) -> Result<Self> {
        Ok(Self {
            // SAFETY: the caller upholds this function's safety contract.
            inner: Arc::new(unsafe { Self::register_descriptor_owner(driver, descriptor)? }),
        })
    }

    /// Returns the borrowed native file handle.
    #[must_use]
    pub fn as_raw(&self) -> sys::CUfileHandle_t {
        self.inner.handle
    }

    /// Reads up to `size` bytes from the file into the selected buffer range.
    ///
    /// `file_offset` and `buffer_offset` are byte offsets.
    /// Returns the actual number of bytes transferred, which may be less than `size`.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer length cannot be computed, the requested byte range exceeds the buffer, or a size or offset exceeds its native integer range.
    /// Also returns an error if CUDA readiness, capture checks, or context binding fails, or cuFile reports an I/O error.
    pub fn read<B>(
        &self,
        buffer: &mut B,
        size: usize,
        file_offset: u64,
        buffer_offset: u64,
    ) -> Result<usize>
    where
        B: IoOutput + ?Sized,
    {
        // SAFETY: the role trait establishes initialization or arbitrary-byte validity and borrowed I/O completes here.
        unsafe { self.read_unchecked(buffer, size, file_offset, buffer_offset) }
    }

    /// Reads file bytes into exclusively borrowed storage and completes before returning.
    ///
    /// # Safety
    /// File bytes, including partial element writes, must leave valid values or explicitly uninitialized storage.
    /// Preserve exclusive access, backing liveness, and prior access ordering through completion.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer length cannot be computed, the requested byte range exceeds the buffer, or a size or offset exceeds its native integer range.
    /// Also returns an error if CUDA readiness, capture checks, or context binding fails, or cuFile reports an I/O error.
    pub unsafe fn read_unchecked<B>(
        &self,
        buffer: &mut B,
        size: usize,
        file_offset: u64,
        buffer_offset: u64,
    ) -> Result<usize>
    where
        B: IoBufferMut + ?Sized,
    {
        let size = checked_io_size(buffer.byte_len()?, size, buffer_offset)?;
        to_i64(file_offset, "file_offset")?;
        if size == 0 {
            return Ok(0);
        }
        prepare_buffer(buffer, None)?;
        let pointer = buffer.ptr_mut();
        let (pointer, native_offset) = io_pointer(buffer, pointer, size, buffer_offset)?;
        // SAFETY: `cuFileRead` receives only plain values.
        let result = unsafe {
            sys::cuFileRead(
                self.inner.handle,
                pointer.cast(),
                to_u64(size, "size")?,
                to_i64(file_offset, "file_offset")?,
                native_offset,
            )
        };
        synchronous_result(result)
    }

    /// Writes up to `size` bytes from the selected buffer range to the file.
    ///
    /// `file_offset` and `buffer_offset` are byte offsets.
    /// Returns the actual number of bytes transferred, which may be less than `size`.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer length cannot be computed, the requested byte range exceeds the buffer, or a size or offset exceeds its native integer range.
    /// Also returns an error if CUDA readiness, capture checks, or context binding fails, or cuFile reports an I/O error.
    pub fn write<B>(
        &self,
        buffer: &B,
        size: usize,
        file_offset: u64,
        buffer_offset: u64,
    ) -> Result<usize>
    where
        B: IoInput + ?Sized,
    {
        // SAFETY: the role trait establishes initialization or arbitrary-byte validity and borrowed I/O completes here.
        unsafe { self.write_unchecked(buffer, size, file_offset, buffer_offset) }
    }

    /// Writes borrowed bytes to the file and completes before returning.
    ///
    /// # Safety
    /// The selected input bytes, including padding, must be initialized when accessed by cuFile.
    /// Preserve backing liveness and prevent conflicting writes through completion.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer length cannot be computed, the requested byte range exceeds the buffer, or a size or offset exceeds its native integer range.
    /// Also returns an error if CUDA readiness, capture checks, or context binding fails, or cuFile reports an I/O error.
    pub unsafe fn write_unchecked<B>(
        &self,
        buffer: &B,
        size: usize,
        file_offset: u64,
        buffer_offset: u64,
    ) -> Result<usize>
    where
        B: IoBuffer + ?Sized,
    {
        let size = checked_io_size(buffer.byte_len()?, size, buffer_offset)?;
        to_i64(file_offset, "file_offset")?;
        if size == 0 {
            return Ok(0);
        }
        prepare_buffer(buffer, None)?;
        let (pointer, native_offset) = io_pointer(buffer, buffer.ptr(), size, buffer_offset)?;
        // SAFETY: `cuFileWrite` receives only plain values.
        let result = unsafe {
            sys::cuFileWrite(
                self.inner.handle,
                pointer.cast(),
                to_u64(size, "size")?,
                to_i64(file_offset, "file_offset")?,
                native_offset,
            )
        };
        synchronous_result(result)
    }

    /// Reads data from an offset in a registered file handle to one or more buffers
    /// in device or host memory. Returns the number of bytes successfully read.
    ///
    /// Unregistered buffers may be buffered through preallocated pinned buffers.
    /// This is useful for unaligned file offsets and sizes, or when BAR1 memory
    /// is smaller than the allocated memory.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA-associated buffers belong to different contexts, a vector length or total length exceeds the signed native range, or the file offset or vector count exceeds its native range.
    /// Also returns an error if readiness synchronization, capture checks, or context binding fails, or cuFile reports an I/O error.
    pub fn read_vectored(&self, buffers: &mut [IoSliceMut<'_>], file_offset: u64) -> Result<usize> {
        prepare_vectors(buffers.iter().map(|b| (b.region, b.context)))?;
        let vectors: Vec<_> = buffers.iter().map(|buffer| buffer.raw).collect();
        check_vectors(&vectors)?;
        // SAFETY: `cuFileReadv` receives pointers into buffers borrowed for the call.
        let result = unsafe {
            sys::cuFileReadv(
                self.inner.handle,
                vectors.as_ptr(),
                to_u64(vectors.len(), "iovcnt")?,
                to_i64(file_offset, "file_offset")?,
                0,
            )
        };
        synchronous_result(result)
    }

    /// Writes data to an offset in a registered file handle from one or more
    /// buffers in device or host memory. Returns the number of bytes written.
    ///
    /// Unregistered buffers may be buffered through preallocated pinned buffers.
    /// This is useful for unaligned file offsets and sizes, or when BAR1 memory
    /// is smaller than the allocated memory.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA-associated buffers belong to different contexts, a vector length or total length exceeds the signed native range, or the file offset or vector count exceeds its native range.
    /// Also returns an error if readiness synchronization, capture checks, or context binding fails, or cuFile reports an I/O error.
    pub fn write_vectored(&self, buffers: &[IoSlice<'_>], file_offset: u64) -> Result<usize> {
        prepare_vectors(buffers.iter().map(|b| (b.region, b.context)))?;
        let vectors: Vec<_> = buffers.iter().map(|buffer| buffer.raw).collect();
        check_vectors(&vectors)?;
        // SAFETY: `cuFileWritev` receives pointers into buffers borrowed for the call.
        let result = unsafe {
            sys::cuFileWritev(
                self.inner.handle,
                vectors.as_ptr(),
                to_u64(vectors.len(), "iovcnt")?,
                to_i64(file_offset, "file_offset")?,
                0,
            )
        };
        synchronous_result(result)
    }

    /// Enqueues a read into exclusively borrowed storage on the selected stream.
    ///
    /// # Safety
    /// Retain buffer storage and order accesses until stream completion on every path, including forgotten operations.
    /// File bytes, including partial element writes, must leave valid values or explicitly uninitialized storage.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer length cannot be computed, the requested byte range exceeds the buffer, or a size or offset exceeds its native integer range.
    /// Also returns an error if CUDA readiness, capture checks, or context binding fails, or cuFile reports an I/O error.
    /// The buffer and stream must belong to the same CUDA context; allocating the pinned completion status can also fail.
    pub unsafe fn read_async<'a, B>(
        &self,
        buffer: &'a mut B,
        size: usize,
        file_offset: u64,
        buffer_offset: u64,
        stream: &StreamBinding,
    ) -> Result<AsyncIo<'a>>
    where
        B: IoBufferMut + ?Sized,
    {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { AsyncIo::read(self, buffer, size, file_offset, buffer_offset, stream) }
    }

    /// Completes stream-ordered I/O before returning the borrowed buffer.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer length cannot be computed, the requested byte range exceeds the buffer, or a size or offset exceeds its native integer range.
    /// Also returns an error if CUDA readiness, capture checks, or context binding fails, or cuFile reports an I/O error.
    /// The buffer and stream must belong to the same CUDA context; allocating the pinned completion status can also fail.
    /// Stream synchronization and completion-status decoding errors are propagated.
    pub fn read_on<B: IoOutput + ?Sized>(
        &self,
        buffer: &mut B,
        size: usize,
        file_offset: u64,
        buffer_offset: u64,
        stream: &StreamBinding,
    ) -> Result<usize> {
        // SAFETY: the role trait establishes representation validity, and wait drains before releasing the borrow.
        unsafe { self.read_async(buffer, size, file_offset, buffer_offset, stream) }?.wait()
    }

    /// Enqueues a write from borrowed storage on the selected stream.
    ///
    /// # Safety
    /// Retain buffer storage and order accesses until stream completion on every path, including forgotten operations.
    /// Input bytes, including padding, must be initialized when accessed by cuFile.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer length cannot be computed, the requested byte range exceeds the buffer, or a size or offset exceeds its native integer range.
    /// Also returns an error if CUDA readiness, capture checks, or context binding fails, or cuFile reports an I/O error.
    /// The buffer and stream must belong to the same CUDA context; allocating the pinned completion status can also fail.
    pub unsafe fn write_async<'a, B>(
        &self,
        buffer: &'a B,
        size: usize,
        file_offset: u64,
        buffer_offset: u64,
        stream: &StreamBinding,
    ) -> Result<AsyncIo<'a>>
    where
        B: IoBuffer + ?Sized,
    {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { AsyncIo::write(self, buffer, size, file_offset, buffer_offset, stream) }
    }

    /// Completes stream-ordered I/O before returning the borrowed buffer.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer length cannot be computed, the requested byte range exceeds the buffer, or a size or offset exceeds its native integer range.
    /// Also returns an error if CUDA readiness, capture checks, or context binding fails, or cuFile reports an I/O error.
    /// The buffer and stream must belong to the same CUDA context; allocating the pinned completion status can also fail.
    /// Stream synchronization and completion-status decoding errors are propagated.
    pub fn write_on<B: IoInput + ?Sized>(
        &self,
        buffer: &B,
        size: usize,
        file_offset: u64,
        buffer_offset: u64,
        stream: &StreamBinding,
    ) -> Result<usize> {
        // SAFETY: the role trait establishes representation validity, and wait drains before releasing the borrow.
        unsafe { self.write_async(buffer, size, file_offset, buffer_offset, stream) }?.wait()
    }
}

impl Drop for FileOwner {
    fn drop(&mut self) {
        // SAFETY: `cuFileHandleDeregister` receives live handles owned by the wrappers.
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
        let code =
            u32::try_from(result.unsigned_abs()).map_err(|_| Error::OperationFailed(result))?;
        let status = Status::from(code);
        if code > sys::CUFILEOP_BASE_ERR && !matches!(status, Status::Unknown(_)) {
            return Err(status.into());
        }

        let code = i32::try_from(code).map_err(|_| Error::OperationFailed(result))?;
        return Err(io::Error::from_raw_os_error(code).into());
    }
    to_usize(result, "result")
}

pub(crate) fn checked_io_size(
    buffer_size: usize,
    size: usize,
    buffer_offset: u64,
) -> Result<usize> {
    let offset = to_usize(buffer_offset, "buffer_offset")?;
    let available = buffer_size
        .checked_sub(offset)
        .ok_or_else(|| Error::out_of_range("buffer_offset"))?;
    if size > available {
        return Err(Error::out_of_range("size"));
    }
    Ok(size)
}

pub(crate) fn prepare_buffer<B: IoBuffer + ?Sized>(
    buffer: &B,
    stream: Option<&StreamBinding>,
) -> Result<()> {
    if let (Some(context), Some(stream)) = (buffer.context(), stream)
        && context != stream.context()
    {
        return Err(CudaError::MemoryContextMismatch.into());
    }
    if let Some(region) = buffer.device_region() {
        if let Some(ready) = region.stream() {
            ready.ensure_not_capturing()?;
        }
        region.synchronize()?;
    }
    if let Some(stream) = stream {
        if stream.capture_status()? != StreamCaptureStatus::None {
            return Err(CudaError::from(CudaStatus::StreamCaptureUnsupported).into());
        }
        stream.context().bind()?;
    } else if let Some(context) = buffer.context() {
        let binding = StreamBinding::Default(Arc::clone(context));
        if binding.capture_status()? != StreamCaptureStatus::None {
            return Err(CudaError::from(CudaStatus::StreamCaptureUnsupported).into());
        }
        context.bind()?;
    }
    Ok(())
}

pub(crate) fn io_pointer<B: IoBuffer + ?Sized>(
    buffer: &B,
    pointer: BufferPointer,
    size: usize,
    offset: u64,
) -> Result<(*mut (), i64)> {
    checked_io_size(buffer.byte_len()?, size, offset)?;
    if buffer.is_registered() {
        Ok((pointer.as_mut(), to_i64(offset, "buffer_offset")?))
    } else {
        Ok((
            pointer
                .as_mut()
                .cast::<u8>()
                .wrapping_add(to_usize(offset, "buffer_offset")?)
                .cast(),
            0,
        ))
    }
}

fn prepare_vectors<'a>(
    buffers: impl Iterator<Item = (Option<DeviceRegion<'a>>, Option<&'a Arc<Context>>)>,
) -> Result<()> {
    let mut selected = None;
    for (region, context) in buffers {
        if let Some(context) = context {
            if let Some(selected) = selected
                && selected != context
            {
                return Err(CudaError::MemoryContextMismatch.into());
            }
            selected = Some(context);
        }
        if let Some(region) = region {
            if let Some(ready) = region.stream() {
                ready.ensure_not_capturing()?;
            }
            region.synchronize()?;
        }
    }
    if let Some(context) = selected {
        let binding = StreamBinding::Default(Arc::clone(context));
        if binding.capture_status()? != StreamCaptureStatus::None {
            return Err(CudaError::from(CudaStatus::StreamCaptureUnsupported).into());
        }
        context.bind()?;
    }
    Ok(())
}
