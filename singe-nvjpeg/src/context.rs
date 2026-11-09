//! The nvJPEG library handle and decoder state.

use std::{
    ptr,
    sync::{Arc, Mutex},
};

use singe_cuda::{context::Context as CudaContext, stream::Stream as CudaStream};
use singe_nvjpeg_sys as sys;

use crate::{
    allocator::AllocatorSet,
    buffer::{DeviceBuffer, PinnedBuffer},
    error::{Error, Result},
    try_ffi,
    types::{Backend, CreateFlags, HardwareDecoderInfo, HardwareEncoderInfo, JpegInfo},
    utility::{to_non_zero_u64, to_u64, to_usize},
};

/// nvJPEG library handle, which states, buffers, decoders, and encoders are created from.
///
/// It keeps its CUDA context alive.
#[derive(Debug, Clone)]
pub struct Context {
    inner: Arc<Inner>,
}

#[derive(Debug)]
pub(crate) struct Inner {
    raw: sys::nvjpegHandle_t,
    cuda_context: Arc<CudaContext>,
    streams: Mutex<Vec<CudaStream>>,
    _allocators: Option<AllocatorSet>,
}

/// State that a decoder keeps across the stages of decoding one image.
#[derive(Debug)]
pub struct JpegState {
    raw: sys::nvjpegJpegState_t,
    pub(crate) handle: Context,
    pub(crate) pinned_buffers: Vec<PinnedBuffer>,
    pub(crate) device_buffers: Vec<DeviceBuffer>,
}

// SAFETY: the native library handle may be shared; per-call decode/encode state
// stays exclusive and every native boundary binds the retained CUDA context.
// Custom allocator callbacks must uphold their declared threading contract.
unsafe impl Send for Inner {}

unsafe impl Sync for Inner {}

impl Context {
    /// Returns a builder for a handle with a custom backend, flags, or allocators.
    #[must_use]
    pub const fn builder() -> ContextBuilder {
        ContextBuilder::new()
    }

    /// Creates a handle in `cuda_context` with the default backend and allocators.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or nvJPEG fails to create the handle.
    pub fn create_simple(cuda_context: &Arc<CudaContext>) -> Result<Self> {
        cuda_context.bind()?;
        let mut raw = ptr::null_mut();
        // SAFETY: `nvjpegCreateSimple` receives valid output locations.
        unsafe {
            try_ffi!(sys::nvjpegCreateSimple(&raw mut raw))?;
        }
        Self::from_created_raw(raw, cuda_context, None)
    }

    /// Creates a handle in `cuda_context` that uses `backend`.
    ///
    /// With [`Backend::Default`], nvJPEG chooses the implementation.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or nvJPEG fails to create the handle.
    pub fn create(cuda_context: &Arc<CudaContext>, backend: Backend) -> Result<Self> {
        Self::create_with_flags(cuda_context, backend, CreateFlags::DEFAULT)
    }

    /// Creates a handle in `cuda_context` that uses `backend` and `flags`.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or nvJPEG fails to create the handle.
    pub fn create_with_flags(
        cuda_context: &Arc<CudaContext>,
        backend: Backend,
        flags: CreateFlags,
    ) -> Result<Self> {
        Self::builder()
            .backend(backend)
            .flags(flags)
            .create(cuda_context)
    }

    fn create_with_builder(
        builder: ContextBuilder,
        cuda_context: &Arc<CudaContext>,
    ) -> Result<Self> {
        cuda_context.bind()?;
        let mut raw = ptr::null_mut();
        if let Some(allocators) = builder.allocators {
            let mut device_allocator = allocators.device().to_nvjpeg_raw();
            let mut pinned_allocator = allocators.pinned().to_nvjpeg_raw();
            // SAFETY: `nvjpegCreateExV2` receives valid output locations.
            unsafe {
                try_ffi!(sys::nvjpegCreateExV2(
                    builder.backend.into(),
                    &raw mut device_allocator,
                    &raw mut pinned_allocator,
                    builder.flags.bits(),
                    &raw mut raw
                ))?;
            }
            Self::from_created_raw(raw, cuda_context, Some(allocators))
        } else {
            // SAFETY: `nvjpegCreateEx` receives valid output locations.
            unsafe {
                // The V2 API requires custom allocator callbacks. The default
                // CUDA allocators are selected through nvjpegCreateEx.
                try_ffi!(sys::nvjpegCreateEx(
                    builder.backend.into(),
                    ptr::null_mut(),
                    ptr::null_mut(),
                    builder.flags.bits(),
                    &raw mut raw
                ))?;
            }
            Self::from_created_raw(raw, cuda_context, None)
        }
    }

    fn from_created_raw(
        raw: sys::nvjpegHandle_t,
        cuda_context: &Arc<CudaContext>,
        allocators: Option<AllocatorSet>,
    ) -> Result<Self> {
        if raw.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            inner: Arc::new(Inner {
                raw,
                cuda_context: Arc::clone(cuda_context),
                streams: Mutex::new(Vec::new()),
                _allocators: allocators,
            }),
        })
    }

    /// Takes ownership of a native handle created outside this crate.
    ///
    /// # Safety
    ///
    /// `raw` must be valid, created in `cuda_context`, and owned by nobody else; it is destroyed with `nvjpegDestroy`.
    /// Any custom allocator callbacks and their data must stay valid until then.
    /// All work that used `raw` must have finished.
    ///
    /// # Errors
    ///
    /// Returns an error if `raw` is null.
    pub unsafe fn from_raw(
        raw: sys::nvjpegHandle_t,
        cuda_context: &Arc<CudaContext>,
    ) -> Result<Self> {
        Self::from_created_raw(raw, cuda_context, None)
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

    /// Makes the handle's CUDA context current on this thread.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot make the context current.
    pub fn bind(&self) -> Result<()> {
        Ok(self.cuda_context().bind()?)
    }

    /// Checks that `stream` belongs to the handle's CUDA context, keeps it alive until the handle and its resources are destroyed, and makes the context current.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to another CUDA context or the context cannot be made current.
    pub fn ensure_stream(&self, stream: &CudaStream) -> Result<()> {
        if stream.context() != self.cuda_context() {
            return Err(Error::StreamContextMismatch);
        }

        self.bind()?;
        // Register the owner before any native call can enqueue work or fail.
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

    /// Creates a decoder state.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or nvJPEG fails to create the state.
    pub fn create_jpeg_state(&self) -> Result<JpegState> {
        self.bind()?;

        let mut raw = ptr::null_mut();
        // SAFETY: `nvjpegJpegStateCreate` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::nvjpegJpegStateCreate(self.as_raw(), &raw mut raw))?;
        }

        if raw.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(JpegState {
            raw,
            handle: self.clone(),
            pinned_buffers: Vec::new(),
            device_buffers: Vec::new(),
        })
    }

    /// Sets the extra bytes added to each later device allocation.
    ///
    /// A large padding reduces how often device memory has to be reallocated.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, `padding` does not fit nvJPEG's integer types, or nvJPEG rejects it.
    pub fn set_device_memory_padding(&self, padding: usize) -> Result<()> {
        self.bind()?;

        // SAFETY: `nvjpegSetDeviceMemoryPadding` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::nvjpegSetDeviceMemoryPadding(
                to_u64(padding, "padding")?,
                self.as_raw()
            ))
        }
    }

    /// Returns the extra bytes added to each device allocation.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, nvJPEG fails the query, or the value does not fit `usize`.
    pub fn device_memory_padding(&self) -> Result<usize> {
        self.bind()?;

        let mut padding = 0;
        // SAFETY: `nvjpegGetDeviceMemoryPadding` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::nvjpegGetDeviceMemoryPadding(
                &raw mut padding,
                self.as_raw()
            ))?;
        }
        to_usize(padding, "padding")
    }

    /// Sets the extra bytes added to each later pinned host allocation.
    ///
    /// A large padding reduces how often pinned host memory has to be reallocated.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, `padding` does not fit nvJPEG's integer types, or nvJPEG rejects it.
    pub fn set_pinned_memory_padding(&self, padding: usize) -> Result<()> {
        self.bind()?;

        // SAFETY: `nvjpegSetPinnedMemoryPadding` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::nvjpegSetPinnedMemoryPadding(
                to_u64(padding, "padding")?,
                self.as_raw()
            ))
        }
    }

    /// Returns the extra bytes added to each pinned host allocation.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, nvJPEG fails the query, or the value does not fit `usize`.
    pub fn pinned_memory_padding(&self) -> Result<usize> {
        self.bind()?;

        let mut padding = 0;
        // SAFETY: `nvjpegGetPinnedMemoryPadding` receives live handles owned by the wrappers and valid output locations.
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
        // SAFETY: `nvjpegGetHardwareDecoderInfo` receives live handles owned by the wrappers and valid output locations.
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
        // SAFETY: `nvjpegGetHardwareEncoderInfo` receives live handles owned by the wrappers and valid output locations.
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
        // SAFETY: `nvjpegGetImageInfo` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
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
            subsampling.into(),
            widths,
            heights,
        )
    }
}

/// Builder for a [`Context`] with a custom backend, flags, or allocators.
#[derive(Debug, Clone, Copy)]
pub struct ContextBuilder {
    backend: Backend,
    flags: CreateFlags,
    allocators: Option<AllocatorSet>,
}

impl ContextBuilder {
    /// Creates a builder with nvJPEG's defaults.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            backend: Backend::Default,
            flags: CreateFlags::DEFAULT,
            allocators: None,
        }
    }

    /// Sets the backend.
    #[must_use]
    pub const fn backend(mut self, backend: Backend) -> Self {
        self.backend = backend;
        self
    }

    /// Sets the creation flags.
    #[must_use]
    pub const fn flags(mut self, flags: CreateFlags) -> Self {
        self.flags = flags;
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
    /// Returns an error if the CUDA context cannot be bound, nvJPEG rejects the backend, flags, or allocators, or nvJPEG fails to create the handle.
    pub fn create(self, cuda_context: &Arc<CudaContext>) -> Result<Context> {
        Context::create_with_builder(self, cuda_context)
    }
}

impl Default for ContextBuilder {
    fn default() -> Self {
        Self::new()
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

    /// Takes ownership of a native decoder state created with `handle`, and keeps `handle` alive.
    ///
    /// # Safety
    ///
    /// `raw` must be valid, created with `handle`, and owned by nobody else; it is destroyed with `nvjpegJpegStateDestroy`.
    /// Buffers attached to it outside this crate must stay valid until then.
    /// All work that used it must have finished.
    ///
    /// # Errors
    ///
    /// Returns an error if `raw` is null.
    pub unsafe fn from_raw_with_handle(
        raw: sys::nvjpegJpegState_t,
        handle: &Context,
    ) -> Result<Self> {
        if raw.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            raw,
            handle: handle.clone(),
            pinned_buffers: Vec::new(),
            device_buffers: Vec::new(),
        })
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        if self.cuda_context.bind().is_err() {
            return;
        }
        if !self.raw.is_null() {
            // SAFETY: `nvjpegDestroy` receives live handles owned by the wrappers.
            unsafe {
                if let Err(error) = try_ffi!(sys::nvjpegDestroy(self.raw)) {
                    eprintln!(
                        "failed to destroy nvjpeg handle: {}",
                        singe_core::ErrorChain(&error)
                    );
                }
            }
        }
    }
}

impl Drop for JpegState {
    fn drop(&mut self) {
        if self.handle.bind().is_err() {
            return;
        }
        if !self.raw.is_null() {
            // SAFETY: `nvjpegJpegStateDestroy` receives live handles owned by the wrappers.
            unsafe {
                if let Err(error) = try_ffi!(sys::nvjpegJpegStateDestroy(self.raw)) {
                    eprintln!(
                        "failed to destroy nvjpeg jpeg state: {}",
                        singe_core::ErrorChain(&error)
                    );
                }
            }
        }
    }
}
