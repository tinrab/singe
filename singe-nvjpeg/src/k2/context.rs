//! The nvJPEG 2000 library handle and raw decoding.

use std::{ptr, sync::Arc};

use singe_core::report_cleanup_failure;
use singe_cuda::{context::Context as CudaContext, stream::Stream as CudaStream};
use singe_nvjpeg_sys::k2 as sys;

use crate::{
    allocator::AllocatorSet,
    error::{Error, Result},
    k2::{Backend, CodeStream, DecodeParams, DecodeState, Image, TileDecode},
    try_ffi_k2,
    utility::{to_u64, to_usize},
};

/// nvJPEG 2000 library handle, which decoders, encoders, and their state are created from.
///
/// It keeps its CUDA context alive.
/// nvJPEG 2000 documents the handle as shareable by several threads, so clones share one handle.
#[derive(Debug, Clone)]
pub struct Context {
    inner: Arc<HandleInner>,
}

#[derive(Debug)]
struct HandleInner {
    raw: sys::nvjpeg2kHandle_t,
    cuda_context: Arc<CudaContext>,
    _allocators: Option<AllocatorSet>,
}

/// Settings for creating a JPEG 2000 [`Context`].
#[derive(Debug, Clone, Copy)]
pub struct ContextOptions {
    /// Decoding implementation; nvJPEG 2000 currently has only [`Backend::Default`].
    pub backend: Backend,
    /// Custom device and pinned host allocators, or `None` for nvJPEG 2000's own.
    pub allocators: Option<AllocatorSet>,
    /// Multiple that every device allocation is rounded up to, in bytes, which reduces reallocations.
    pub device_memory_padding: Option<usize>,
    /// Multiple that every pinned host allocation is rounded up to, in bytes, which reduces reallocations.
    pub pinned_memory_padding: Option<usize>,
}

impl Default for ContextOptions {
    fn default() -> Self {
        Self {
            backend: Backend::Default,
            allocators: None,
            device_memory_padding: None,
            pinned_memory_padding: None,
        }
    }
}

// SAFETY: nvJPEG 2000 documents the library handle as shareable by several threads.
// The unsafe allocator constructors require callbacks that may be called from any thread, and every native call binds the retained CUDA context.
unsafe impl Send for HandleInner {}

// SAFETY: the padding, which the library reads without synchronization, is set in `Context::create` before the handle is shared and never changed afterwards.
unsafe impl Sync for HandleInner {}

impl Context {
    /// Returns the CUDA context of the handle.
    #[must_use]
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        &self.inner.cuda_context
    }

    /// Makes the handle's CUDA context current on this thread.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot make the context current.
    pub fn bind(&self) -> Result<()> {
        Ok(self.cuda_context().bind()?)
    }

    /// Checks that `stream` belongs to the handle's CUDA context and makes the context current.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to another CUDA context or the context cannot be made current.
    pub fn ensure_stream(&self, stream: &CudaStream) -> Result<()> {
        if stream.context() != self.cuda_context() {
            return Err(Error::StreamContextMismatch);
        }

        self.bind()
    }

    pub(crate) fn ensure_handle(&self, other: &Self, name: &'static str) -> Result<()> {
        if !Arc::ptr_eq(&self.inner, &other.inner) {
            return Err(Error::ContextMismatch { name: name.into() });
        }
        Ok(())
    }

    pub(crate) fn ensure_context(
        &self,
        context: &Arc<CudaContext>,
        name: &'static str,
    ) -> Result<()> {
        if self.cuda_context() != context {
            return Err(Error::ContextMismatch { name: name.into() });
        }
        Ok(())
    }

    /// Creates a handle in `cuda_context` with `options`.
    ///
    /// The padding is set before the handle is shared, since nvJPEG 2000 reads it without synchronization.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, a padding does not fit nvJPEG 2000's integer type, or nvJPEG 2000 rejects the settings or fails to create the handle.
    pub fn create(cuda_context: &Arc<CudaContext>, options: &ContextOptions) -> Result<Self> {
        cuda_context.bind()?;
        let mut raw = ptr::null_mut();
        if let Some(allocators) = options.allocators {
            let mut device_allocator = allocators.device().to_nvjpeg2k_raw();
            let mut pinned_allocator = allocators.pinned().to_nvjpeg2k_raw();
            // SAFETY: the allocator structs and `raw` are locals that outlive the call, and the `unsafe` allocator constructors made their callers promise callbacks that meet the native contract.
            unsafe {
                try_ffi_k2!(sys::nvjpeg2kCreateV2(
                    options.backend.into(),
                    &raw mut device_allocator,
                    &raw mut pinned_allocator,
                    &raw mut raw
                ))?;
            }
        } else {
            // SAFETY: null allocators select nvJPEG 2000's own, and `raw` is a local that receives the new handle.
            unsafe {
                try_ffi_k2!(sys::nvjpeg2kCreate(
                    options.backend.into(),
                    ptr::null_mut(),
                    ptr::null_mut(),
                    &raw mut raw
                ))?;
            }
        }

        if raw.is_null() {
            return Err(Error::NullHandle);
        }

        let context = Self {
            inner: Arc::new(HandleInner {
                raw,
                cuda_context: Arc::clone(cuda_context),
                _allocators: options.allocators,
            }),
        };

        if let Some(padding) = options.device_memory_padding {
            // SAFETY: the handle is live, its context is bound, and no other thread can reach it yet.
            unsafe {
                try_ffi_k2!(sys::nvjpeg2kSetDeviceMemoryPadding(
                    to_u64(padding, "device memory padding")?,
                    context.inner.raw
                ))?;
            }
        }

        if let Some(padding) = options.pinned_memory_padding {
            // SAFETY: the handle is live, its context is bound, and no other thread can reach it yet.
            unsafe {
                try_ffi_k2!(sys::nvjpeg2kSetPinnedMemoryPadding(
                    to_u64(padding, "pinned memory padding")?,
                    context.inner.raw
                ))?;
            }
        }

        Ok(context)
    }

    /// Returns the native handle, which this value still owns.
    #[must_use]
    pub fn as_raw(&self) -> sys::nvjpeg2kHandle_t {
        self.inner.raw
    }

    /// Creates a decoder state.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or nvJPEG 2000 fails to create the state.
    pub fn create_decode_state(&self) -> Result<DecodeState> {
        self.bind()?;

        let mut raw = ptr::null_mut();
        // SAFETY: `self` owns a live handle whose context was bound above, and `raw` is a local that receives the state.
        unsafe {
            try_ffi_k2!(sys::nvjpeg2kDecodeStateCreate(self.inner.raw, &raw mut raw))?;
        }

        if raw.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(DecodeState {
            raw,
            handle: self.clone(),
        })
    }

    /// Returns the multiple that device allocations are rounded up to, in bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, nvJPEG 2000 fails the query, or the value does not fit `usize`.
    pub fn device_memory_padding(&self) -> Result<usize> {
        self.bind()?;

        let mut padding = 0;
        // SAFETY: `self` owns a live handle whose context was bound above, and `padding` is a local that receives the value.
        unsafe {
            try_ffi_k2!(sys::nvjpeg2kGetDeviceMemoryPadding(
                &raw mut padding,
                self.inner.raw
            ))?;
        }
        to_usize(padding, "padding")
    }

    /// Returns the multiple that pinned host allocations are rounded up to, in bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, nvJPEG 2000 fails the query, or the value does not fit `usize`.
    pub fn pinned_memory_padding(&self) -> Result<usize> {
        self.bind()?;

        let mut padding = 0;
        // SAFETY: `self` owns a live handle whose context was bound above, and `padding` is a local that receives the value.
        unsafe {
            try_ffi_k2!(sys::nvjpeg2kGetPinnedMemoryPadding(
                &raw mut padding,
                self.inner.raw
            ))?;
        }
        to_usize(padding, "padding")
    }

    /// Enqueues decoding of the parsed image in `code_stream` into `output` on `stream`, using `state`.
    ///
    /// # Safety
    ///
    /// `output` must describe device-accessible buffers large enough for the image, valid and used by nothing else until the work on `stream` finishes.
    /// The [raw execution rules](crate#raw-execution) apply, including on failure and for every replay of a captured graph.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream`, `code_stream`, or the settings belong to another CUDA context, the context cannot be bound, `state` was created with another [`Context`], or nvJPEG 2000 fails.
    pub unsafe fn decode_async(
        &self,
        state: &mut DecodeState,
        code_stream: &CodeStream,
        output: &mut Image,
        stream: &CudaStream,
    ) -> Result<()> {
        self.ensure_stream(stream)?;
        self.ensure_handle(&state.handle, "state")?;
        self.ensure_context(code_stream.cuda_context(), "code_stream")?;

        // SAFETY: `state`, `code_stream`, and `stream` belong to this live handle's context, as checked above; the caller guarantees `output` is large enough and that the operands outlive the queued work.
        unsafe {
            try_ffi_k2!(sys::nvjpeg2kDecode(
                self.inner.raw,
                state.raw,
                code_stream.raw,
                output.as_raw_mut(),
                stream.as_raw()
            ))
        }
    }

    /// Enqueues decoding of the parsed image in `code_stream` into `output` on `stream`, using `state` and the settings in `params`.
    ///
    /// # Safety
    ///
    /// `output` must describe device-accessible buffers large enough for the image, valid and used by nothing else until the work on `stream` finishes.
    /// The [raw execution rules](crate#raw-execution) apply, including on failure and for every replay of a captured graph.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream`, `code_stream`, or the settings belong to another CUDA context, the context cannot be bound, `state` was created with another [`Context`], or nvJPEG 2000 fails.
    pub unsafe fn decode_image_async(
        &self,
        state: &mut DecodeState,
        code_stream: &CodeStream,
        params: &DecodeParams,
        output: &mut Image,
        stream: &CudaStream,
    ) -> Result<()> {
        self.ensure_stream(stream)?;
        self.ensure_handle(&state.handle, "state")?;
        self.ensure_context(code_stream.cuda_context(), "code_stream")?;
        self.ensure_context(params.cuda_context(), "params")?;

        // SAFETY: `state`, `code_stream`, `params`, and `stream` belong to this live handle's context, as checked above; the caller guarantees `output` is large enough and that the operands outlive the queued work.
        unsafe {
            try_ffi_k2!(sys::nvjpeg2kDecodeImage(
                self.inner.raw,
                state.raw,
                code_stream.raw,
                params.raw,
                output.as_raw_mut(),
                stream.as_raw()
            ))
        }
    }

    /// Enqueues decoding of one tile of the parsed image in `code_stream` into `output` on `stream`, using `state` and optional settings.
    ///
    /// # Safety
    ///
    /// `output` must describe device-accessible buffers large enough for the image, valid and used by nothing else until the work on `stream` finishes.
    /// The [raw execution rules](crate#raw-execution) apply, including on failure and for every replay of a captured graph.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream`, `code_stream`, or the settings belong to another CUDA context, the context cannot be bound, `state` was created with another [`Context`], or nvJPEG 2000 fails.
    pub unsafe fn decode_tile_async(
        &self,
        state: &mut DecodeState,
        code_stream: &CodeStream,
        params: Option<&DecodeParams>,
        tile: TileDecode,
        output: &mut Image,
        stream: &CudaStream,
    ) -> Result<()> {
        self.ensure_stream(stream)?;
        self.ensure_handle(&state.handle, "state")?;
        self.ensure_context(code_stream.cuda_context(), "code_stream")?;
        if let Some(params) = params {
            self.ensure_context(params.cuda_context(), "params")?;
        }

        // SAFETY: `state`, `code_stream`, `stream`, and any `params` belong to this live handle's context, as checked above, and null selects default settings; the caller guarantees `output` is large enough and that the operands outlive the queued work.
        unsafe {
            try_ffi_k2!(sys::nvjpeg2kDecodeTile(
                self.inner.raw,
                state.raw,
                code_stream.raw,
                params.map_or(ptr::null_mut(), |params| params.raw),
                tile.tile_id(),
                tile.resolution_levels(),
                output.as_raw_mut(),
                stream.as_raw()
            ))
        }
    }
}

impl Drop for HandleInner {
    fn drop(&mut self) {
        if let Err(error) = self.cuda_context.bind() {
            report_cleanup_failure(
                "bind the cuda context to destroy",
                "nvjpeg2k handle",
                &error,
            );
            return;
        }

        if !self.raw.is_null() {
            // SAFETY: the last owner drops `HandleInner`, so no state, stream, or encoder still uses the handle, and it is destroyed once.
            if let Err(error) = unsafe { try_ffi_k2!(sys::nvjpeg2kDestroy(self.raw)) } {
                report_cleanup_failure("destroy", "nvjpeg2k handle", &error);
            }
        }
    }
}
