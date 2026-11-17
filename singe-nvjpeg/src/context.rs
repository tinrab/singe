//! The nvJPEG library handle and decoder state.

use std::{mem::ManuallyDrop, ptr, result::Result as StdResult, sync::Arc};

use singe_core::report_cleanup_failure;
use singe_cuda::{context::Context as CudaContext, stream::Stream as CudaStream};
use singe_nvjpeg_sys as sys;

use crate::{
    allocator::AllocatorSet,
    buffer::{DeviceBuffer, PinnedBuffer},
    decoder::DecoderOwner,
    error::{Error, Result},
    try_ffi,
    types::{Backend, CreateFlags, HardwareDecoderInfo, HardwareEncoderInfo, JpegInfo},
    utility::{to_non_zero_u64, to_u64, to_usize},
};

/// nvJPEG library handle, which states, buffers, decoders, and encoders are created from.
///
/// It keeps its CUDA context alive.
/// nvJPEG documents the handle as usable by several threads at once, so clones share one handle.
#[derive(Debug, Clone)]
pub struct Context {
    inner: Arc<Inner>,
}

#[derive(Debug)]
struct Inner {
    raw: sys::nvjpegHandle_t,
    cuda_context: Arc<CudaContext>,
    backend: Backend,
    allocators: Option<AllocatorSet>,
}

/// Settings for creating a [`Context`].
#[derive(Debug, Clone, Copy)]
pub struct ContextOptions {
    /// Decoding implementation; with [`Backend::Default`], nvJPEG chooses it.
    pub backend: Backend,
    /// Creation flags.
    pub flags: CreateFlags,
    /// Custom device and pinned host allocators, or `None` for nvJPEG's own.
    pub allocators: Option<AllocatorSet>,
    /// Extra bytes added to each device allocation, which reduces reallocations.
    pub device_memory_padding: Option<usize>,
    /// Extra bytes added to each pinned host allocation, which reduces reallocations.
    pub pinned_memory_padding: Option<usize>,
}

/// Native nvJPEG library handle with what it was created with, for [`Context::from_raw`] and from [`Context::into_raw`].
#[derive(Debug)]
pub struct RawContext {
    /// Native library handle, owned exclusively.
    pub handle: sys::nvjpegHandle_t,
    /// CUDA context the handle was created in.
    pub cuda_context: Arc<CudaContext>,
    /// Backend the handle was created with.
    pub backend: Backend,
    /// Custom allocators the handle was created with, whose callbacks must outlive it.
    pub allocators: Option<AllocatorSet>,
}

/// State that a decoder keeps across the stages of decoding one image.
#[derive(Debug)]
pub struct JpegState {
    raw: sys::nvjpegJpegState_t,
    pub(crate) handle: Context,
    pub(crate) pinned_buffers: Vec<PinnedBuffer>,
    pub(crate) device_buffers: Vec<DeviceBuffer>,
    // nvJPEG ties a decoder state to its decoder, so the decoder must outlive it.
    pub(crate) decoder: Option<Arc<DecoderOwner>>,
}

/// Native decoder state with its library handle, for [`JpegState::from_raw`].
#[derive(Debug)]
pub struct RawJpegState {
    /// Native state handle, owned exclusively.
    pub handle: sys::nvjpegJpegState_t,
    /// nvJPEG library handle the state was created with.
    pub context: Context,
}

// SAFETY: nvJPEG documents the library handle as usable by several threads at once.
// The unsafe allocator constructors require callbacks that may be called from any thread, and every native call binds the retained CUDA context.
unsafe impl Send for Inner {}

// SAFETY: the padding is set in `Context::create` before the handle is shared, and nothing changes the handle's configuration through `&Inner` afterwards.
unsafe impl Sync for Inner {}

// SAFETY: nvJPEG requires a state to be used by one thread at a time, not by a fixed thread, and every mutating call takes `&mut JpegState`.
unsafe impl Send for JpegState {}

impl Default for ContextOptions {
    fn default() -> Self {
        Self {
            backend: Backend::Default,
            flags: CreateFlags::DEFAULT,
            allocators: None,
            device_memory_padding: None,
            pinned_memory_padding: None,
        }
    }
}

impl Context {
    /// Creates a handle in `cuda_context` with `options`.
    ///
    /// The padding is set before the handle is shared, since nvJPEG reads it without synchronization.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, a padding does not fit nvJPEG's integer type, or nvJPEG rejects the settings or fails to create the handle.
    pub fn create(cuda_context: &Arc<CudaContext>, options: &ContextOptions) -> Result<Self> {
        cuda_context.bind()?;
        let mut raw = ptr::null_mut();
        if let Some(allocators) = options.allocators {
            let mut device_allocator = allocators.device().to_nvjpeg_raw();
            let mut pinned_allocator = allocators.pinned().to_nvjpeg_raw();
            // SAFETY: the allocator structs and `raw` are locals that outlive the call, and the `unsafe` allocator constructors made their callers promise callbacks that meet the native contract.
            unsafe {
                try_ffi!(sys::nvjpegCreateExV2(
                    options.backend.into(),
                    &raw mut device_allocator,
                    &raw mut pinned_allocator,
                    options.flags.bits(),
                    &raw mut raw
                ))?;
            }
        } else {
            // The V2 API requires custom allocator callbacks, so nvJPEG's own allocators are selected through `nvjpegCreateEx`.
            // SAFETY: null allocators select nvJPEG's own, and `raw` is a local that receives the new handle.
            unsafe {
                try_ffi!(sys::nvjpegCreateEx(
                    options.backend.into(),
                    ptr::null_mut(),
                    ptr::null_mut(),
                    options.flags.bits(),
                    &raw mut raw
                ))?;
            }
        }

        let context = Self::from_parts(RawContext {
            handle: raw,
            cuda_context: Arc::clone(cuda_context),
            backend: options.backend,
            allocators: options.allocators,
        })?;
        context.apply_padding(options)?;
        Ok(context)
    }

    // Runs before the handle is shared, since nvJPEG reads the padding without synchronization.
    fn apply_padding(&self, options: &ContextOptions) -> Result<()> {
        if let Some(padding) = options.device_memory_padding {
            // SAFETY: the handle is live, its context is bound, and no other thread can reach it yet.
            unsafe {
                try_ffi!(sys::nvjpegSetDeviceMemoryPadding(
                    to_u64(padding, "device memory padding")?,
                    self.as_raw()
                ))?;
            }
        }

        if let Some(padding) = options.pinned_memory_padding {
            // SAFETY: the handle is live, its context is bound, and no other thread can reach it yet.
            unsafe {
                try_ffi!(sys::nvjpegSetPinnedMemoryPadding(
                    to_u64(padding, "pinned memory padding")?,
                    self.as_raw()
                ))?;
            }
        }

        Ok(())
    }

    fn from_parts(raw: RawContext) -> Result<Self> {
        if raw.handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            inner: Arc::new(Inner {
                raw: raw.handle,
                cuda_context: raw.cuda_context,
                backend: raw.backend,
                allocators: raw.allocators,
            }),
        })
    }

    /// Takes ownership of a native handle created outside this crate.
    ///
    /// # Safety
    ///
    /// `raw.handle` must be valid, created in `raw.cuda_context` with `raw.backend` and `raw.allocators`, and owned by nobody else; it is destroyed when the returned value drops.
    /// Any custom allocator callbacks and their data must stay valid until then.
    /// All work that used the handle must have finished.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NullHandle`] if `raw.handle` is null.
    pub unsafe fn from_raw(raw: RawContext) -> Result<Self> {
        Self::from_parts(raw)
    }

    /// Releases the native handle with its CUDA context, backend, and allocators, without destroying it.
    ///
    /// # Errors
    ///
    /// Returns the context unchanged if a clone or an object created from it is still alive.
    pub fn into_raw(self) -> StdResult<RawContext, Self> {
        match Arc::try_unwrap(self.inner) {
            Ok(inner) => {
                let inner = ManuallyDrop::new(inner);
                Ok(RawContext {
                    handle: inner.raw,
                    // SAFETY: `inner` is never dropped, so its CUDA context moves out exactly once.
                    cuda_context: unsafe { ptr::read(&raw const inner.cuda_context) },
                    backend: inner.backend,
                    allocators: inner.allocators,
                })
            }
            Err(inner) => Err(Self { inner }),
        }
    }

    /// Returns the native handle, which this value still owns.
    #[must_use]
    pub fn as_raw(&self) -> sys::nvjpegHandle_t {
        self.inner.raw
    }

    /// Returns the CUDA context of the handle.
    #[must_use]
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        &self.inner.cuda_context
    }

    /// Returns the backend the handle was created with, which the staged decoders of a [`Decoder`](crate::decode::Decoder) use.
    #[must_use]
    pub fn backend(&self) -> Backend {
        self.inner.backend
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

    /// Creates a decoder state.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or nvJPEG fails to create the state.
    pub fn create_jpeg_state(&self) -> Result<JpegState> {
        self.bind()?;

        let mut raw = ptr::null_mut();
        // SAFETY: `self` owns a live handle whose context was bound above, and `raw` is a local that receives the new state.
        unsafe {
            try_ffi!(sys::nvjpegJpegStateCreate(self.as_raw(), &raw mut raw))?;
        }

        JpegState::from_created_raw(raw, self, None)
    }

    /// Returns the extra bytes added to each device allocation.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, nvJPEG fails the query, or the value does not fit `usize`.
    pub fn device_memory_padding(&self) -> Result<usize> {
        self.bind()?;

        let mut padding = 0;
        // SAFETY: `self` owns a live handle whose context was bound above, and `padding` is a local that receives the value.
        unsafe {
            try_ffi!(sys::nvjpegGetDeviceMemoryPadding(
                &raw mut padding,
                self.as_raw()
            ))?;
        }
        to_usize(padding, "padding")
    }

    /// Returns the extra bytes added to each pinned host allocation.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, nvJPEG fails the query, or the value does not fit `usize`.
    pub fn pinned_memory_padding(&self) -> Result<usize> {
        self.bind()?;

        let mut padding = 0;
        // SAFETY: `self` owns a live handle whose context was bound above, and `padding` is a local that receives the value.
        unsafe {
            try_ffi!(sys::nvjpegGetPinnedMemoryPadding(
                &raw mut padding,
                self.as_raw()
            ))?;
        }
        to_usize(padding, "padding")
    }

    /// Returns the number of hardware decoder engines and the number of cores in each.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or nvJPEG fails the query.
    pub fn hardware_decoder_info(&self) -> Result<HardwareDecoderInfo> {
        self.bind()?;

        let mut engines = 0;
        let mut cores_per_engine = 0;
        // SAFETY: `self` owns a live handle whose context was bound above, and both counts are locals that receive the values.
        unsafe {
            try_ffi!(sys::nvjpegGetHardwareDecoderInfo(
                self.as_raw(),
                &raw mut engines,
                &raw mut cores_per_engine
            ))?;
        }
        Ok(HardwareDecoderInfo::new(engines, cores_per_engine))
    }

    /// Returns the number of hardware encoder engines.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or nvJPEG fails the query.
    pub fn hardware_encoder_info(&self) -> Result<HardwareEncoderInfo> {
        self.bind()?;

        let mut engines = 0;
        // SAFETY: `self` owns a live handle whose context was bound above, and `engines` is a local that receives the count.
        unsafe {
            try_ffi!(sys::nvjpegGetHardwareEncoderInfo(
                self.as_raw(),
                &raw mut engines
            ))?;
        }
        Ok(HardwareEncoderInfo::new(engines))
    }

    /// Reads the JPEG header in `data` and returns the chroma subsampling and the size of each component.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, `data` is empty or too long for nvJPEG, nvJPEG cannot parse the header, or it reports an invalid component count or size.
    pub fn image_info(&self, data: &[u8]) -> Result<JpegInfo> {
        self.bind()?;

        let mut components = 0;
        let mut subsampling = sys::nvjpegChromaSubsampling_t::NVJPEG_CSS_UNKNOWN;
        let mut widths = [0; sys::NVJPEG_MAX_COMPONENT as usize];
        let mut heights = [0; sys::NVJPEG_MAX_COMPONENT as usize];
        // SAFETY: `self` owns a live handle, `data` is borrowed for the synchronous call with its checked length, and `widths` and `heights` hold the `NVJPEG_MAX_COMPONENT` entries nvJPEG writes.
        unsafe {
            try_ffi!(sys::nvjpegGetImageInfo(
                self.as_raw(),
                data.as_ptr(),
                to_non_zero_u64(data.len(), "data length")?,
                &raw mut components,
                &raw mut subsampling,
                widths.as_mut_ptr(),
                heights.as_mut_ptr(),
            ))?;
        }
        JpegInfo::from_raw(
            to_usize(components, "components")?,
            subsampling.try_into()?,
            widths,
            heights,
        )
    }
}

impl JpegState {
    /// Returns the CUDA context of the state.
    #[must_use]
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        self.handle.cuda_context()
    }

    /// Returns the native state handle, which this value still owns.
    #[must_use]
    pub const fn as_raw(&self) -> sys::nvjpegJpegState_t {
        self.raw
    }

    /// Takes ownership of a native decoder state created with `raw.context`, and keeps `raw.context` alive.
    ///
    /// # Safety
    ///
    /// `raw.handle` must be valid, created with `raw.context`, and owned by nobody else; it is destroyed when the returned value drops.
    /// If it was created for a decoder, that decoder must stay alive until then.
    /// Buffers attached to it outside this crate must stay valid until then.
    /// All work that used it must have finished.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NullHandle`] if `raw.handle` is null.
    pub unsafe fn from_raw(raw: RawJpegState) -> Result<Self> {
        let RawJpegState { handle, context } = raw;
        Self::from_created_raw(handle, &context, None)
    }

    pub(crate) fn from_created_raw(
        raw: sys::nvjpegJpegState_t,
        handle: &Context,
        decoder: Option<Arc<DecoderOwner>>,
    ) -> Result<Self> {
        if raw.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            raw,
            handle: handle.clone(),
            pinned_buffers: Vec::new(),
            device_buffers: Vec::new(),
            decoder,
        })
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        if let Err(error) = self.cuda_context.bind() {
            report_cleanup_failure("bind the cuda context to destroy", "nvjpeg handle", &error);
            return;
        }

        if !self.raw.is_null() {
            // SAFETY: the last owner drops `Inner`, so no state, buffer, or coder still uses the handle, and it is destroyed once.
            if let Err(error) = unsafe { try_ffi!(sys::nvjpegDestroy(self.raw)) } {
                report_cleanup_failure("destroy", "nvjpeg handle", &error);
            }
        }
    }
}

impl Drop for JpegState {
    fn drop(&mut self) {
        if let Err(error) = self.handle.bind() {
            report_cleanup_failure(
                "bind the cuda context to destroy",
                "nvjpeg jpeg state",
                &error,
            );
            return;
        }

        if !self.raw.is_null() {
            // SAFETY: the state owns its handle exclusively and is destroyed once; the raw execution rules keep queued work from outliving it.
            if let Err(error) = unsafe { try_ffi!(sys::nvjpegJpegStateDestroy(self.raw)) } {
                report_cleanup_failure("destroy", "nvjpeg jpeg state", &error);
            }
        }
    }
}
