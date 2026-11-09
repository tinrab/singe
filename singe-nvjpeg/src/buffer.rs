//! Pinned host and device buffers that decoders reuse for intermediate data.

use std::{ffi::c_void, mem, ptr, sync::Arc};

use singe_cuda::{context::Context as CudaContext, stream::Stream as CudaStream};
use singe_nvjpeg_sys as sys;

use crate::{
    context::{Context, JpegState},
    error::{Error, Result},
    try_ffi,
    utility::{complete, to_u64, to_usize},
};

/// Pinned host buffer that a decoder uses for intermediate data.
///
/// Attach it to a [`JpegState`] with [`JpegState::attach_pinned_buffer`].
#[derive(Debug)]
pub struct PinnedBuffer {
    raw: sys::nvjpegBufferPinned_t,
    pub(crate) handle: Context,
}

/// Device buffer that a decoder uses for intermediate data.
///
/// Attach it to a [`JpegState`] with [`JpegState::attach_device_buffer`].
#[derive(Debug)]
pub struct DeviceBuffer {
    raw: sys::nvjpegBufferDevice_t,
    pub(crate) handle: Context,
}

/// Memory and size of a buffer, from [`PinnedBuffer::retrieve`] or [`DeviceBuffer::retrieve`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RawBuffer {
    ptr: *mut c_void,
    byte_len: usize,
}

impl RawBuffer {
    fn create(ptr: *mut c_void, byte_len: usize) -> Result<Self> {
        if ptr.is_null() && byte_len != 0 {
            return Err(Error::NullPointer {
                name: "buffer".into(),
            });
        }
        Ok(Self { ptr, byte_len })
    }

    /// Returns a pointer to the memory.
    #[must_use]
    pub const fn as_ptr(&self) -> *const c_void {
        self.ptr.cast_const()
    }

    /// Returns a mutable pointer to the memory.
    #[must_use]
    pub const fn as_mut_ptr(&self) -> *mut c_void {
        self.ptr
    }

    /// Returns the size in bytes.
    #[must_use]
    pub const fn byte_len(&self) -> usize {
        self.byte_len
    }

    /// Returns whether the size is zero.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.byte_len == 0
    }
}

impl PinnedBuffer {
    /// Takes ownership of a native pinned host buffer created with `handle`.
    ///
    /// # Safety
    ///
    /// `raw` must be valid, created with `handle`, and owned by nobody else.
    /// It is destroyed when the returned value drops, so all work that used it must have finished.
    ///
    /// # Errors
    ///
    /// Returns an error if `raw` is null.
    pub unsafe fn from_raw(raw: sys::nvjpegBufferPinned_t, handle: &Context) -> Result<Self> {
        if raw.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            raw,
            handle: handle.clone(),
        })
    }

    /// Releases the native handle without destroying it.
    ///
    /// Destroy it, or pass it back to [`from_raw`](Self::from_raw), while its [`Context`] is alive.
    #[must_use]
    pub fn into_raw(mut self) -> sys::nvjpegBufferPinned_t {
        mem::replace(&mut self.raw, ptr::null_mut())
    }

    /// Returns the CUDA context of the buffer.
    #[must_use]
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        self.handle.cuda_context()
    }

    /// Creates an empty pinned host buffer that uses `handle`'s pinned host allocator.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or nvJPEG fails to create the buffer.
    pub fn create(handle: &Context) -> Result<Self> {
        handle.bind()?;

        let mut raw = ptr::null_mut();
        // SAFETY: `nvjpegBufferPinnedCreate` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::nvjpegBufferPinnedCreate(
                handle.as_raw(),
                ptr::null_mut(),
                &raw mut raw
            ))?;
        }

        if raw.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            raw,
            handle: handle.clone(),
        })
    }

    /// Resizes the buffer to `size` bytes and waits for `stream`.
    ///
    /// Resizing ahead of time avoids allocations during decoding.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to another CUDA context, the context cannot be bound, `size` does not fit nvJPEG's integer types, nvJPEG fails to resize the buffer, or waiting for `stream` fails.
    pub fn resize(&mut self, size: usize, stream: &CudaStream) -> Result<()> {
        // SAFETY: `complete` drains the stream before returning, so every borrowed buffer outlives the queued work.
        complete(stream, || unsafe { self.resize_async(size, stream) })
    }

    /// Resizes the buffer to `size` bytes, without waiting for `stream`.
    ///
    /// Resizing ahead of time avoids allocations during decoding.
    /// `stream` is used when the buffer comes from a stream-ordered allocator.
    ///
    /// # Safety
    ///
    /// Nothing else may use the buffer until the work on `stream` finishes, even if this fails.
    /// All earlier uses of the buffer must have finished, because its memory may be reallocated.
    /// The [raw execution rules](crate#raw-execution) apply.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to another CUDA context, the context cannot be bound, `size` does not fit nvJPEG's integer types, or nvJPEG fails to resize the buffer.
    pub unsafe fn resize_async(&mut self, size: usize, stream: &CudaStream) -> Result<()> {
        self.handle.ensure_stream(stream)?;

        // SAFETY: `nvjpegBufferPinnedResize` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::nvjpegBufferPinnedResize(
                self.raw,
                to_u64(size, "size")?,
                stream.as_raw(),
            ))
        }
    }

    /// Returns the buffer's pinned host memory and size.
    ///
    /// Use it to reuse the memory after decoding finishes.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, nvJPEG fails the query, the size does not fit `usize`, or a nonempty buffer has a null pointer.
    pub fn retrieve(&self) -> Result<RawBuffer> {
        self.handle.bind()?;

        let mut size = 0;
        let mut ptr = ptr::null_mut();
        // SAFETY: `nvjpegBufferPinnedRetrieve` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::nvjpegBufferPinnedRetrieve(
                self.raw,
                &raw mut size,
                &raw mut ptr
            ))?;
        }
        RawBuffer::create(ptr, to_usize(size, "size")?)
    }

    /// Returns the native handle, which this buffer still owns.
    #[must_use]
    pub const fn as_raw(&self) -> sys::nvjpegBufferPinned_t {
        self.raw
    }
}

impl DeviceBuffer {
    /// Takes ownership of a native device buffer created with `handle`.
    ///
    /// # Safety
    ///
    /// `raw` must be valid, created with `handle`, and owned by nobody else.
    /// It is destroyed when the returned value drops, so all work that used it must have finished.
    ///
    /// # Errors
    ///
    /// Returns an error if `raw` is null.
    pub unsafe fn from_raw(raw: sys::nvjpegBufferDevice_t, handle: &Context) -> Result<Self> {
        if raw.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            raw,
            handle: handle.clone(),
        })
    }

    /// Releases the native handle without destroying it.
    ///
    /// Destroy it, or pass it back to [`from_raw`](Self::from_raw), while its [`Context`] is alive.
    #[must_use]
    pub fn into_raw(mut self) -> sys::nvjpegBufferDevice_t {
        mem::replace(&mut self.raw, ptr::null_mut())
    }

    /// Returns the CUDA context of the buffer.
    #[must_use]
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        self.handle.cuda_context()
    }

    /// Creates an empty device buffer that uses `handle`'s device allocator.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or nvJPEG fails to create the buffer.
    pub fn create(handle: &Context) -> Result<Self> {
        handle.bind()?;

        let mut raw = ptr::null_mut();
        // SAFETY: `nvjpegBufferDeviceCreate` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::nvjpegBufferDeviceCreate(
                handle.as_raw(),
                ptr::null_mut(),
                &raw mut raw
            ))?;
        }

        if raw.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            raw,
            handle: handle.clone(),
        })
    }

    /// Resizes the buffer to `size` bytes and waits for `stream`.
    ///
    /// Resizing ahead of time avoids allocations during decoding.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to another CUDA context, the context cannot be bound, `size` does not fit nvJPEG's integer types, nvJPEG fails to resize the buffer, or waiting for `stream` fails.
    pub fn resize(&mut self, size: usize, stream: &CudaStream) -> Result<()> {
        // SAFETY: `complete` drains the stream before returning, so every borrowed buffer outlives the queued work.
        complete(stream, || unsafe { self.resize_async(size, stream) })
    }

    /// Resizes the buffer to `size` bytes, without waiting for `stream`.
    ///
    /// Resizing ahead of time avoids allocations during decoding.
    /// `stream` is used when the buffer comes from a stream-ordered allocator.
    ///
    /// # Safety
    ///
    /// Nothing else may use the buffer until the work on `stream` finishes, even if this fails.
    /// All earlier uses of the buffer must have finished, because its memory may be reallocated.
    /// The [raw execution rules](crate#raw-execution) apply.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to another CUDA context, the context cannot be bound, `size` does not fit nvJPEG's integer types, or nvJPEG fails to resize the buffer.
    pub unsafe fn resize_async(&mut self, size: usize, stream: &CudaStream) -> Result<()> {
        self.handle.ensure_stream(stream)?;

        // SAFETY: `nvjpegBufferDeviceResize` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::nvjpegBufferDeviceResize(
                self.raw,
                to_u64(size, "size")?,
                stream.as_raw()
            ))
        }
    }

    /// Returns the buffer's device memory and size.
    ///
    /// Use it to reuse the memory after decoding finishes.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, nvJPEG fails the query, the size does not fit `usize`, or a nonempty buffer has a null pointer.
    pub fn retrieve(&self) -> Result<RawBuffer> {
        self.handle.bind()?;

        let mut size = 0;
        let mut ptr = ptr::null_mut();
        // SAFETY: `nvjpegBufferDeviceRetrieve` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::nvjpegBufferDeviceRetrieve(
                self.raw,
                &raw mut size,
                &raw mut ptr
            ))?;
        }
        RawBuffer::create(ptr, to_usize(size, "size")?)
    }

    /// Returns the native handle, which this buffer still owns.
    #[must_use]
    pub const fn as_raw(&self) -> sys::nvjpegBufferDevice_t {
        self.raw
    }
}

impl JpegState {
    /// Attaches `buffer` to this state, which keeps it.
    ///
    /// The decoder stores intermediate data in it between decoding stages.
    /// A buffer can be moved between decoder states to switch implementations without allocating more memory.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, `buffer` was created with another [`Context`], or nvJPEG rejects it.
    /// The state keeps `buffer` even if nvJPEG rejects it.
    pub fn attach_pinned_buffer(&mut self, buffer: PinnedBuffer) -> Result<()> {
        self.handle.bind()?;
        self.handle.ensure_handle(&buffer.handle)?;
        let raw = buffer.raw;
        self.pinned_buffers.push(buffer);

        // SAFETY: `nvjpegStateAttachPinnedBuffer` receives live handles owned by the wrappers.
        unsafe { try_ffi!(sys::nvjpegStateAttachPinnedBuffer(self.as_raw(), raw)) }
    }

    /// Attaches `buffer` to this state, which keeps it.
    ///
    /// The decoder stores intermediate data in it between decoding stages.
    /// A buffer can be moved between decoder states to switch implementations without allocating more memory.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, `buffer` was created with another [`Context`], or nvJPEG rejects it.
    /// The state keeps `buffer` even if nvJPEG rejects it.
    pub fn attach_device_buffer(&mut self, buffer: DeviceBuffer) -> Result<()> {
        self.handle.bind()?;
        self.handle.ensure_handle(&buffer.handle)?;
        let raw = buffer.raw;
        self.device_buffers.push(buffer);

        // SAFETY: `nvjpegStateAttachDeviceBuffer` receives live handles owned by the wrappers.
        unsafe { try_ffi!(sys::nvjpegStateAttachDeviceBuffer(self.as_raw(), raw)) }
    }
}

impl Drop for PinnedBuffer {
    fn drop(&mut self) {
        if self.handle.bind().is_err() {
            return;
        }
        if !self.raw.is_null() {
            // SAFETY: `nvjpegBufferPinnedDestroy` receives live handles owned by the wrappers.
            unsafe {
                if let Err(error) = try_ffi!(sys::nvjpegBufferPinnedDestroy(self.raw)) {
                    eprintln!(
                        "failed to destroy nvjpeg pinned buffer: {}",
                        singe_core::ErrorChain(&error)
                    );
                }
            }
        }
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if self.handle.bind().is_err() {
            return;
        }
        if !self.raw.is_null() {
            // SAFETY: `nvjpegBufferDeviceDestroy` receives live handles owned by the wrappers.
            unsafe {
                if let Err(error) = try_ffi!(sys::nvjpegBufferDeviceDestroy(self.raw)) {
                    eprintln!(
                        "failed to destroy nvjpeg device buffer: {}",
                        singe_core::ErrorChain(&error)
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_raw_buffer_accessors_report_pointer_and_byte_len() {
        let mut value = 42u8;
        let buffer = RawBuffer::create((&raw mut value).cast(), 1).unwrap();

        assert_eq!(buffer.as_ptr(), (&raw const value).cast());
        assert_eq!(buffer.as_mut_ptr(), (&raw mut value).cast());
        assert_eq!(buffer.byte_len(), 1);
        assert!(!buffer.is_empty());
    }

    #[test]
    fn test_raw_buffer_is_empty_when_byte_len_is_zero() {
        let buffer = RawBuffer::create(ptr::null_mut(), 0).unwrap();

        assert!(buffer.is_empty());
    }
}
