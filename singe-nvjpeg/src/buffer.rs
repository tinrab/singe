//! Pinned host and device buffers that decoders reuse for intermediate data.

use std::{ffi::c_void, mem::ManuallyDrop, ptr, sync::Arc};

use singe_core::report_cleanup_failure;
use singe_cuda::{context::Context as CudaContext, stream::Stream as CudaStream};
use singe_nvjpeg_sys as sys;

use crate::{
    context::{Context, JpegState},
    error::{Error, Result},
    try_ffi,
    utility::{to_u64, to_usize},
};

/// Pinned host buffer that a decoder uses for intermediate data.
///
/// Attach it to a [`JpegState`] with [`JpegState::attach_pinned_buffer`].
#[derive(Debug)]
pub struct PinnedBuffer {
    raw: sys::nvjpegBufferPinned_t,
    pub(crate) handle: Context,
}

/// Native pinned host buffer with its library handle, from [`PinnedBuffer::into_raw`].
///
/// Dropping it leaks the native handle.
#[derive(Debug)]
pub struct RawPinnedBuffer {
    /// Native handle, owned exclusively.
    pub handle: sys::nvjpegBufferPinned_t,
    /// nvJPEG library handle the native handle was created with.
    pub context: Context,
}

/// Device buffer that a decoder uses for intermediate data.
///
/// Attach it to a [`JpegState`] with [`JpegState::attach_device_buffer`].
#[derive(Debug)]
pub struct DeviceBuffer {
    raw: sys::nvjpegBufferDevice_t,
    pub(crate) handle: Context,
}

/// Native device buffer with its library handle, from [`DeviceBuffer::into_raw`].
///
/// Dropping it leaks the native handle.
#[derive(Debug)]
pub struct RawDeviceBuffer {
    /// Native handle, owned exclusively.
    pub handle: sys::nvjpegBufferDevice_t,
    /// nvJPEG library handle the native handle was created with.
    pub context: Context,
}

/// Memory and size of a buffer, from [`PinnedBuffer::retrieve`] or [`DeviceBuffer::retrieve`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferMemory {
    ptr: *mut c_void,
    byte_len: usize,
}

// SAFETY: nvJPEG requires a pinned buffer to be used by one thread at a time, not by a fixed thread, and every resize takes `&mut PinnedBuffer`.
unsafe impl Send for PinnedBuffer {}

// SAFETY: nvJPEG requires a device buffer to be used by one thread at a time, not by a fixed thread, and every resize takes `&mut DeviceBuffer`.
unsafe impl Send for DeviceBuffer {}

impl BufferMemory {
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
    /// Takes ownership of a native pinned host buffer created with `raw.context`.
    ///
    /// # Safety
    ///
    /// `raw.handle` must be valid, created with `raw.context`, and owned by nobody else.
    /// It is destroyed when the returned value drops, so all work that used it must have finished.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NullHandle`] if `raw.handle` is null.
    pub unsafe fn from_raw(raw: RawPinnedBuffer) -> Result<Self> {
        if raw.handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            raw: raw.handle,
            handle: raw.context,
        })
    }

    /// Releases the native handle with its [`Context`], without destroying it.
    ///
    /// Destroy the handle, or pass it back to [`from_raw`](Self::from_raw), while the context is alive.
    #[must_use]
    pub fn into_raw(self) -> RawPinnedBuffer {
        let this = ManuallyDrop::new(self);
        RawPinnedBuffer {
            handle: this.raw,
            // SAFETY: `this` is never dropped, so its context moves out exactly once.
            context: unsafe { ptr::read(&raw const this.handle) },
        }
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
        // SAFETY: `handle` owns a live library handle whose context was bound above, null selects its allocator, and `raw` is a local that receives the buffer.
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
        // SAFETY: `complete` waits for `stream` on every path, so the buffer is not used or dropped while the resize is queued.
        stream.complete(|| unsafe { self.resize_async(size, stream) })
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

        // SAFETY: `&mut self` owns the live buffer and `stream` belongs to its context, as checked above; the caller keeps the buffer unused until the resize finishes.
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
    pub fn retrieve(&self) -> Result<BufferMemory> {
        self.handle.bind()?;

        let mut size = 0;
        let mut ptr = ptr::null_mut();
        // SAFETY: `self` owns the live buffer, and `size` and `ptr` are locals that receive its memory.
        unsafe {
            try_ffi!(sys::nvjpegBufferPinnedRetrieve(
                self.raw,
                &raw mut size,
                &raw mut ptr
            ))?;
        }
        BufferMemory::create(ptr, to_usize(size, "size")?)
    }

    /// Returns the native handle, which this buffer still owns.
    #[must_use]
    pub const fn as_raw(&self) -> sys::nvjpegBufferPinned_t {
        self.raw
    }
}

impl DeviceBuffer {
    /// Takes ownership of a native device buffer created with `raw.context`.
    ///
    /// # Safety
    ///
    /// `raw.handle` must be valid, created with `raw.context`, and owned by nobody else.
    /// It is destroyed when the returned value drops, so all work that used it must have finished.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NullHandle`] if `raw.handle` is null.
    pub unsafe fn from_raw(raw: RawDeviceBuffer) -> Result<Self> {
        if raw.handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            raw: raw.handle,
            handle: raw.context,
        })
    }

    /// Releases the native handle with its [`Context`], without destroying it.
    ///
    /// Destroy the handle, or pass it back to [`from_raw`](Self::from_raw), while the context is alive.
    #[must_use]
    pub fn into_raw(self) -> RawDeviceBuffer {
        let this = ManuallyDrop::new(self);
        RawDeviceBuffer {
            handle: this.raw,
            // SAFETY: `this` is never dropped, so its context moves out exactly once.
            context: unsafe { ptr::read(&raw const this.handle) },
        }
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
        // SAFETY: `handle` owns a live library handle whose context was bound above, null selects its allocator, and `raw` is a local that receives the buffer.
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
        // SAFETY: `complete` waits for `stream` on every path, so the buffer is not used or dropped while the resize is queued.
        stream.complete(|| unsafe { self.resize_async(size, stream) })
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

        // SAFETY: `&mut self` owns the live buffer and `stream` belongs to its context, as checked above; the caller keeps the buffer unused until the resize finishes.
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
    pub fn retrieve(&self) -> Result<BufferMemory> {
        self.handle.bind()?;

        let mut size = 0;
        let mut ptr = ptr::null_mut();
        // SAFETY: `self` owns the live buffer, and `size` and `ptr` are locals that receive its memory.
        unsafe {
            try_ffi!(sys::nvjpegBufferDeviceRetrieve(
                self.raw,
                &raw mut size,
                &raw mut ptr
            ))?;
        }
        BufferMemory::create(ptr, to_usize(size, "size")?)
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
        self.handle.ensure_handle(&buffer.handle, "buffer")?;
        let raw = buffer.raw;
        self.pinned_buffers.push(buffer);

        // SAFETY: the state and the buffer are live handles of the same context, and the state now keeps the buffer alive for as long as it can use it.
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
        self.handle.ensure_handle(&buffer.handle, "buffer")?;
        let raw = buffer.raw;
        self.device_buffers.push(buffer);

        // SAFETY: the state and the buffer are live handles of the same context, and the state now keeps the buffer alive for as long as it can use it.
        unsafe { try_ffi!(sys::nvjpegStateAttachDeviceBuffer(self.as_raw(), raw)) }
    }
}

impl Drop for PinnedBuffer {
    fn drop(&mut self) {
        if let Err(error) = self.handle.bind() {
            report_cleanup_failure(
                "bind the cuda context to destroy",
                "nvjpeg pinned buffer",
                &error,
            );
            return;
        }

        if !self.raw.is_null() {
            // SAFETY: the buffer owns its handle exclusively and is destroyed once; a state it was attached to holds it, so it drops after the state stops using it.
            if let Err(error) = unsafe { try_ffi!(sys::nvjpegBufferPinnedDestroy(self.raw)) } {
                report_cleanup_failure("destroy", "nvjpeg pinned buffer", &error);
            }
        }
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if let Err(error) = self.handle.bind() {
            report_cleanup_failure(
                "bind the cuda context to destroy",
                "nvjpeg device buffer",
                &error,
            );
            return;
        }

        if !self.raw.is_null() {
            // SAFETY: the buffer owns its handle exclusively and is destroyed once; a state it was attached to holds it, so it drops after the state stops using it.
            if let Err(error) = unsafe { try_ffi!(sys::nvjpegBufferDeviceDestroy(self.raw)) } {
                report_cleanup_failure("destroy", "nvjpeg device buffer", &error);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_memory_is_empty_when_byte_len_is_zero() {
        let buffer = BufferMemory::create(ptr::null_mut(), 0).unwrap();

        assert!(buffer.is_empty());
    }
}
