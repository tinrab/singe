//! The nvJPEG 2000 library handle and raw decoding.

use std::{
    ptr,
    sync::{Arc, Mutex},
};

use singe_cuda::{
    context::Context as CudaContext, error::Error as CudaError, stream::Stream as CudaStream,
};
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
#[derive(Debug, Clone)]
pub struct Context {
    inner: Arc<HandleInner>,
}

#[derive(Debug)]
struct HandleInner {
    pub(crate) raw: sys::nvjpeg2kHandle_t,
    cuda_context: Arc<CudaContext>,
    streams: Mutex<Vec<CudaStream>>,
    pub(crate) _allocators: Option<AllocatorSet>,
}

/// Builder for a [`Context`] with a custom backend or allocators.
#[derive(Debug, Clone, Copy)]
pub struct ContextBuilder {
    pub(crate) backend: Backend,
    pub(crate) allocators: Option<AllocatorSet>,
}

impl ContextBuilder {
    /// Creates a builder with nvJPEG 2000's defaults.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            backend: Backend::Default,
            allocators: None,
        }
    }

    /// Sets the backend.
    #[must_use]
    pub const fn backend(mut self, backend: Backend) -> Self {
        self.backend = backend;
        self
    }

    /// Sets custom device and pinned host allocators.
    #[must_use]
    pub const fn allocators(mut self, allocators: AllocatorSet) -> Self {
        self.allocators = Some(allocators);
        self
    }

    /// Creates the handle in `cuda_context`.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, or nvJPEG 2000 rejects the backend or allocators or fails to create the handle.
    pub fn create(self, cuda_context: &Arc<CudaContext>) -> Result<Context> {
        Context::create_with_builder(self, cuda_context)
    }
}

impl Default for ContextBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// SAFETY: native library handles support sharing; decoder state and parsed streams
// remain separate per task, and native calls bind their retained actual context.
// Custom allocator callbacks must uphold their threading contract.
unsafe impl Send for HandleInner {}

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

    /// Checks that `stream` belongs to the handle's CUDA context, makes the context current, and keeps `stream` alive until the handle and its resources are destroyed.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to another CUDA context or the context cannot be made current.
    pub fn ensure_stream(&self, stream: &CudaStream) -> Result<()> {
        if stream.context() != self.cuda_context() {
            return Err(Error::StreamContextMismatch);
        }

        self.bind()?;
        let mut streams = self.inner.streams.lock().map_err(|_| Error::LockPoisoned {
            name: "nvjpeg streams",
        })?;
        if !streams
            .iter()
            .any(|owner| owner.as_raw() == stream.as_raw())
        {
            streams.push(stream.clone());
        }

        drop(streams);
        Ok(())
    }

    pub(crate) fn ensure_handle(&self, other: &Self) -> Result<()> {
        if !Arc::ptr_eq(&self.inner, &other.inner) {
            return Err(Error::HandleMismatch);
        }
        Ok(())
    }

    pub(crate) fn ensure_context(&self, context: &Arc<CudaContext>) -> Result<()> {
        if self.cuda_context() != context {
            return Err(CudaError::MemoryContextMismatch.into());
        }
        Ok(())
    }

    /// Returns a builder for a handle with a custom backend or allocators.
    #[must_use]
    pub const fn builder() -> ContextBuilder {
        ContextBuilder::new()
    }

    /// Creates a handle in `cuda_context` with the default backend and allocators.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, or nvJPEG 2000 rejects the backend or fails to create the handle.
    pub fn create_simple(cuda_context: &Arc<CudaContext>) -> Result<Self> {
        cuda_context.bind()?;
        let mut raw = ptr::null_mut();
        // SAFETY: `nvjpeg2kCreateSimple` receives valid output locations.
        unsafe {
            try_ffi_k2!(sys::nvjpeg2kCreateSimple(&raw mut raw))?;
        }

        if raw.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            inner: Arc::new(HandleInner {
                raw,
                cuda_context: Arc::clone(cuda_context),
                streams: Mutex::new(Vec::new()),
                _allocators: None,
            }),
        })
    }

    /// Creates a handle in `cuda_context` that uses `backend` and the default allocators.
    ///
    /// nvJPEG 2000 currently has only [`Backend::Default`].
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, or nvJPEG 2000 rejects the backend or fails to create the handle.
    pub fn create(cuda_context: &Arc<CudaContext>, backend: Backend) -> Result<Self> {
        Self::builder().backend(backend).create(cuda_context)
    }

    fn create_with_builder(
        builder: ContextBuilder,
        cuda_context: &Arc<CudaContext>,
    ) -> Result<Self> {
        cuda_context.bind()?;
        let mut raw = ptr::null_mut();
        if let Some(allocators) = builder.allocators {
            let mut device_allocator = allocators.device().to_nvjpeg2k_raw();
            let mut pinned_allocator = allocators.pinned().to_nvjpeg2k_raw();
            // SAFETY: `nvjpeg2kCreateV2` receives valid output locations.
            unsafe {
                try_ffi_k2!(sys::nvjpeg2kCreateV2(
                    builder.backend.into(),
                    &raw mut device_allocator,
                    &raw mut pinned_allocator,
                    &raw mut raw
                ))?;
            }

            if raw.is_null() {
                return Err(Error::NullHandle);
            }
            Ok(Self {
                inner: Arc::new(HandleInner {
                    raw,
                    cuda_context: Arc::clone(cuda_context),
                    streams: Mutex::new(Vec::new()),
                    _allocators: Some(allocators),
                }),
            })
        } else {
            // SAFETY: `nvjpeg2kCreate` receives valid output locations.
            unsafe {
                try_ffi_k2!(sys::nvjpeg2kCreate(
                    builder.backend.into(),
                    ptr::null_mut(),
                    ptr::null_mut(),
                    &raw mut raw
                ))?;
            }

            if raw.is_null() {
                return Err(Error::NullHandle);
            }
            Ok(Self {
                inner: Arc::new(HandleInner {
                    raw,
                    cuda_context: Arc::clone(cuda_context),
                    streams: Mutex::new(Vec::new()),
                    _allocators: None,
                }),
            })
        }
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
        // SAFETY: `nvjpeg2kDecodeStateCreate` receives valid output locations.
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

    /// Rounds every later device allocation up to a multiple of `padding` bytes, which reduces reallocations.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, `padding` does not fit nvJPEG 2000's integer types, or nvJPEG 2000 rejects it.
    pub fn set_device_memory_padding(&self, padding: usize) -> Result<()> {
        self.bind()?;

        // SAFETY: `nvjpeg2kSetDeviceMemoryPadding` receives only plain values.
        unsafe {
            try_ffi_k2!(sys::nvjpeg2kSetDeviceMemoryPadding(
                to_u64(padding, "padding")?,
                self.inner.raw
            ))
        }
    }

    /// Returns the multiple that device allocations are rounded up to, in bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, nvJPEG 2000 fails the query, or the value does not fit `usize`.
    pub fn device_memory_padding(&self) -> Result<usize> {
        self.bind()?;

        let mut padding = 0;
        // SAFETY: `nvjpeg2kGetDeviceMemoryPadding` receives valid output locations.
        unsafe {
            try_ffi_k2!(sys::nvjpeg2kGetDeviceMemoryPadding(
                &raw mut padding,
                self.inner.raw
            ))?;
        }
        to_usize(padding, "padding")
    }

    /// Rounds every later pinned host allocation up to a multiple of `padding` bytes, which reduces reallocations.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, `padding` does not fit nvJPEG 2000's integer types, or nvJPEG 2000 rejects it.
    pub fn set_pinned_memory_padding(&self, padding: usize) -> Result<()> {
        self.bind()?;

        // SAFETY: `nvjpeg2kSetPinnedMemoryPadding` receives only plain values.
        unsafe {
            try_ffi_k2!(sys::nvjpeg2kSetPinnedMemoryPadding(
                to_u64(padding, "padding")?,
                self.inner.raw
            ))
        }
    }

    /// Returns the multiple that pinned host allocations are rounded up to, in bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, nvJPEG 2000 fails the query, or the value does not fit `usize`.
    pub fn pinned_memory_padding(&self) -> Result<usize> {
        self.bind()?;

        let mut padding = 0;
        // SAFETY: `nvjpeg2kGetPinnedMemoryPadding` receives valid output locations.
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
        self.ensure_handle(&state.handle)?;
        self.ensure_context(code_stream.cuda_context())?;

        // SAFETY: `nvjpeg2kDecode` receives live handles owned by the wrappers.
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
        self.ensure_handle(&state.handle)?;
        self.ensure_context(code_stream.cuda_context())?;
        self.ensure_context(params.cuda_context())?;

        // SAFETY: `nvjpeg2kDecodeImage` receives live handles owned by the wrappers.
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
        self.ensure_handle(&state.handle)?;
        self.ensure_context(code_stream.cuda_context())?;
        if let Some(params) = params {
            self.ensure_context(params.cuda_context())?;
        }

        // SAFETY: `nvjpeg2kDecodeTile` receives live handles owned by the wrappers.
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
        if self.cuda_context.bind().is_err() {
            return;
        }
        if !self.raw.is_null() {
            // SAFETY: `nvjpeg2kDestroy` receives live handles owned by the wrappers.
            unsafe {
                if let Err(error) = try_ffi_k2!(sys::nvjpeg2kDestroy(self.raw)) {
                    eprintln!(
                        "failed to destroy nvjpeg2k handle: {}",
                        singe_core::ErrorChain(&error)
                    );
                }
            }
        }
    }
}
