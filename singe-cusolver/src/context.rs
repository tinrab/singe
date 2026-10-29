use std::{mem, ptr, sync::Arc};

use singe_cuda::{
    context::Context as CudaContext,
    stream::{BorrowedStream as CudaBorrowedStream, Stream, StreamBinding as CudaStreamBinding},
    types::EmulationStrategy,
    view::DeviceRegionProvider,
};
use singe_cuda_sys::runtime;

use crate::{
    error::{Error, Result},
    sys, try_ffi,
    types::{DeterministicMode, MathMode},
};

/// A stateful cuSOLVER handle.
///
/// Use one context per host thread or concurrent task. The handle is movable
/// between threads, but it is intentionally not `Clone` or `Sync`.
#[derive(Debug)]
pub struct Context {
    handle: Handle,
}

#[derive(Debug)]
struct Handle {
    raw: sys::cusolverDnHandle_t,
    cuda_context: Arc<CudaContext>,
    stream: Option<Stream>,
}

// SAFETY: cuSOLVER handles are stateful and stream-bound. The owner may move between
// threads, but callers need exclusive access to mutate handle state.
unsafe impl Send for Handle {}

/// The stream bound to a cuSOLVER handle.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum StreamBinding {
    /// The CUDA default stream.
    Default,
    /// An owned reference to a stream installed through [`Context::set_stream`].
    Retained(Stream),
    /// An externally bound stream whose owner must be retained by its importer.
    Borrowed(BorrowedStream),
}

/// A stream borrowed from a CUDA context, associated with a cuSOLVER handle.
#[derive(Debug, Clone)]
pub struct BorrowedStream {
    handle: runtime::cudaStream_t,
    cuda_context: Arc<CudaContext>,
}

impl BorrowedStream {
    /// Returns the raw CUDA stream handle.
    #[must_use]
    pub const fn as_raw(&self) -> runtime::cudaStream_t {
        self.handle
    }

    /// Returns a reference to the CUDA context this stream belongs to.
    #[must_use]
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.cuda_context
    }
}

/// A transferred native handle with its actual CUDA context and selected stream owner.
///
/// Destroy `handle` before releasing these owners, or reclaim it with `Context::from_raw`.
/// External workspaces, callbacks and queued buffer lifetimes remain caller obligations.
#[derive(Debug)]
pub struct RawContext {
    /// Exclusively owned native cuSOLVER handle; destroy it before releasing the retained context and stream.
    pub handle: sys::cusolverDnHandle_t,
    /// Actual CUDA context required for native use and destruction.
    pub cuda_context: Arc<CudaContext>,
    /// Retained installed stream, if one was selected through this wrapper. External stream ownership remains a caller obligation.
    pub stream: Option<Stream>,
}

// SAFETY: transfer carries exclusive native ownership and the resources needed for cross-thread cleanup.
unsafe impl Send for RawContext {}

impl Context {
    /// Creates a cuSOLVER dense handle for the given CUDA context.
    /// Call this before invoking other cuSOLVER operations through this wrapper.
    ///
    /// cuSOLVER allocates the GPU-side resources it needs here. On the first
    /// application-defined stream passed to [`Context::set_stream`], cuSOLVER may also
    /// allocate an internal workspace.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, if cuSOLVER cannot
    /// create a handle, or if cuSOLVER returns a null handle.
    pub fn create(cuda_context: &Arc<CudaContext>) -> Result<Self> {
        cuda_context.bind()?;

        let mut handle = ptr::null_mut();
        // SAFETY: `cusolverDnCreate` receives valid output locations.
        unsafe {
            try_ffi!(sys::cusolverDnCreate(&raw mut handle))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle: Handle {
                raw: handle,
                cuda_context: Arc::clone(cuda_context),
                stream: None,
            },
        })
    }

    /// Wraps an existing cuSOLVER dense handle and takes ownership of it.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid cuSOLVER dense handle associated with
    /// `cuda_context`. Ownership of `handle` is transferred to the returned
    /// context, and the handle must not be destroyed elsewhere after calling
    /// this function. Retain any externally bound stream until the handle is
    /// destroyed or a replacement stream is installed through [`Self::set_stream`].
    ///
    /// # Errors
    ///
    /// Returns an error if `handle` is null, a supplied stream belongs to another CUDA context, CUDA binding fails, or cuSOLVER rejects installing the supplied stream.
    pub unsafe fn from_raw_parts(
        handle: sys::cusolverDnHandle_t,
        cuda_context: Arc<CudaContext>,
        stream: Option<Stream>,
    ) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        if stream
            .as_ref()
            .is_some_and(|stream| stream.context() != &cuda_context)
        {
            return Err(Error::StreamContextMismatch);
        }
        let mut context = Self {
            handle: Handle {
                raw: handle,
                cuda_context,
                stream: None,
            },
        };
        if let Some(stream) = stream {
            context.set_stream(Some(&stream))?;
        }
        Ok(context)
    }

    /// Returns the underlying CUDA context used by this cuSOLVER handle.
    #[must_use]
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        &self.handle.cuda_context
    }

    /// Binds the underlying CUDA context associated with this handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound.
    pub fn bind(&self) -> Result<()> {
        Ok(self.cuda_context().bind()?)
    }

    fn stream_binding(&self) -> Result<CudaStreamBinding> {
        Ok(match self.stream()? {
            StreamBinding::Default => CudaStreamBinding::Default(Arc::clone(self.cuda_context())),
            StreamBinding::Retained(stream) => CudaStreamBinding::Borrowed(stream.to_borrowed()),
            StreamBinding::Borrowed(stream) => {
                // SAFETY: imported handles must retain their external stream;
                // the cuSOLVER context retains the associated CUDA context.
                CudaStreamBinding::Borrowed(unsafe {
                    CudaBorrowedStream::from_raw(stream.as_raw(), Arc::clone(self.cuda_context()))
                })
            }
        })
    }

    /// Completes borrowed solver work on the installed stream on every path.
    /// Capture is rejected before `f` runs. If CUDA cannot establish completion,
    /// the process aborts before returning borrowed resources to their callers.
    ///
    /// # Errors
    ///
    /// Returns an error if the bound stream cannot be queried, stream capture is active or invalidated, CUDA binding or synchronization fails, or `f` returns an error.
    pub fn sync_scope<R>(&self, f: impl FnOnce() -> Result<R>) -> Result<R> {
        self.stream_binding()?.sync_scope(f)
    }

    pub(crate) fn validate_region(
        &self,
        value: &(impl DeviceRegionProvider + ?Sized),
    ) -> Result<()> {
        if value.region().context() != self.cuda_context() {
            return Err(Error::MemoryContextMismatch);
        }
        Ok(())
    }

    pub(crate) fn prepare_region(
        &self,
        value: &(impl DeviceRegionProvider + ?Sized),
    ) -> Result<()> {
        self.validate_region(value)?;
        if let Some(producer) = value.region().stream()
            && producer.as_raw() != self.stream_binding()?.as_raw()
        {
            producer.ensure_not_capturing()?;
            producer.synchronize()?;
        }
        Ok(())
    }

    /// Ensures `stream` belongs to the same CUDA context as this handle.
    ///
    /// Returns an error if the stream belongs to a different context.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to another CUDA context or that context cannot be bound.
    pub fn ensure_stream(&self, stream: &Stream) -> Result<()> {
        if self.cuda_context() != stream.context() {
            return Err(Error::StreamContextMismatch);
        }

        self.bind()
    }

    /// Returns the stream currently used by this cuSOLVER handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or if cuSOLVER
    /// cannot report the current stream.
    pub fn stream(&self) -> Result<StreamBinding> {
        self.bind()?;

        let mut stream = ptr::null_mut();
        // SAFETY: `cusolverDnGetStream` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::cusolverDnGetStream(self.as_raw(), &raw mut stream))?;
        }

        Ok(if stream.is_null() {
            StreamBinding::Default
        } else if let Some(retained) = &self.handle.stream
            && retained.as_raw() == stream
        {
            StreamBinding::Retained(retained.clone())
        } else {
            StreamBinding::Borrowed(BorrowedStream {
                handle: stream,
                cuda_context: Arc::clone(self.cuda_context()),
            })
        })
    }

    /// Sets the stream used by this cuSOLVER handle.
    ///
    /// Passing `None` restores the CUDA default stream.
    /// The handle retains the selected stream until it is replaced or destroyed.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to another CUDA context, if the CUDA
    /// context cannot be bound, or if cuSOLVER rejects the stream.
    pub fn set_stream(&mut self, stream: Option<&Stream>) -> Result<()> {
        if let Some(stream) = stream {
            self.ensure_stream(stream)?;
        } else {
            self.bind()?;
        }

        // SAFETY: `cusolverDnSetStream` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::cusolverDnSetStream(
                self.as_raw(),
                stream.map_or_else(ptr::null_mut, Stream::as_raw),
            ))?;
        }
        self.handle.stream = stream.cloned();
        Ok(())
    }

    /// Returns the deterministic mode currently configured on this handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or if cuSOLVER
    /// cannot report the deterministic mode.
    pub fn deterministic_mode(&self) -> Result<DeterministicMode> {
        self.bind()?;

        let mut mode = sys::cusolverDeterministicMode_t::CUSOLVER_DETERMINISTIC_RESULTS;
        // SAFETY: `cusolverDnGetDeterministicMode` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::cusolverDnGetDeterministicMode(
                self.as_raw(),
                &raw mut mode,
            ))?;
        }
        Ok(mode.into())
    }

    /// Sets the deterministic mode for operations executed through this handle.
    ///
    /// Allowing non-deterministic results may improve performance for some
    /// operations, including [`xgeqrf`](crate::dense::legacy::qr::xgeqrf), [`Gesvd`](crate::svd::Gesvd), [`Gesvdr`](crate::svd::Gesvdr), and [`Gesvdp`](crate::svd::Gesvdp).
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or if cuSOLVER
    /// rejects the deterministic mode.
    pub fn set_deterministic_mode(&self, mode: DeterministicMode) -> Result<()> {
        self.bind()?;
        // SAFETY: `cusolverDnSetDeterministicMode` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::cusolverDnSetDeterministicMode(
                self.as_raw(),
                mode.into(),
            ))?;
        }
        Ok(())
    }

    /// Returns the math mode currently configured on this handle.
    ///
    /// See [`MathMode`] for the supported wrapper values.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or if cuSOLVER
    /// cannot report the math mode.
    pub fn math_mode(&self) -> Result<MathMode> {
        self.bind()?;

        let mut mode = sys::cusolverMathMode_t::CUSOLVER_DEFAULT_MATH;
        // SAFETY: `cusolverDnGetMathMode` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::cusolverDnGetMathMode(self.as_raw(), &raw mut mode))?;
        }
        Ok(mode.into())
    }

    /// Sets the math mode for operations executed through this handle.
    ///
    /// See [`MathMode`] for the supported wrapper values and combinations.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or if cuSOLVER
    /// rejects the math mode.
    pub fn set_math_mode(&self, mode: MathMode) -> Result<()> {
        self.bind()?;
        // SAFETY: `cusolverDnSetMathMode` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::cusolverDnSetMathMode(self.as_raw(), mode.into()))?;
        }
        Ok(())
    }

    /// Returns the emulation strategy configured on this handle.
    ///
    /// This only affects operations that use one of the emulated math modes
    /// described by [`MathMode`].
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or if cuSOLVER
    /// cannot report the emulation strategy.
    pub fn emulation_strategy(&self) -> Result<EmulationStrategy> {
        self.bind()?;

        let mut strategy = EmulationStrategy::Default.into();
        // SAFETY: `cusolverDnGetEmulationStrategy` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::cusolverDnGetEmulationStrategy(
                self.as_raw(),
                &raw mut strategy,
            ))?;
        }
        Ok(strategy.into())
    }

    /// Sets the emulation strategy for operations executed through this handle.
    ///
    /// This only affects operations that use one of the emulated math modes
    /// described by [`MathMode`].
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or if cuSOLVER
    /// rejects the emulation strategy.
    pub fn set_emulation_strategy(&self, strategy: EmulationStrategy) -> Result<()> {
        self.bind()?;
        // SAFETY: `cusolverDnSetEmulationStrategy` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::cusolverDnSetEmulationStrategy(
                self.as_raw(),
                strategy.into(),
            ))?;
        }
        Ok(())
    }

    /// Returns the raw cuSOLVER dense handle.
    ///
    /// The returned handle is borrowed and remains valid only while this
    /// context and its underlying CUDA context are alive.
    #[must_use]
    pub fn as_raw(&self) -> sys::cusolverDnHandle_t {
        self.handle.raw
    }

    /// Reclaims a native handle transferred together with its context and selected stream.
    ///
    /// # Safety
    ///
    /// `handle` must remain valid and uniquely owned in the retained CUDA context.
    /// The retained stream must match the native stream if one is installed.
    /// Complete all borrowed work before releasing or replacing its resources.
    ///
    /// # Errors
    ///
    /// Returns an error if the native handle is null or the retained stream belongs to another CUDA context.
    pub unsafe fn from_raw(raw: RawContext) -> Result<Self> {
        if raw.handle.is_null() {
            return Err(Error::NullHandle);
        }
        if raw
            .stream
            .as_ref()
            .is_some_and(|stream| stream.context() != &raw.cuda_context)
        {
            return Err(Error::StreamContextMismatch);
        }
        Ok(Self {
            handle: Handle {
                raw: raw.handle,
                cuda_context: raw.cuda_context,
                stream: raw.stream,
            },
        })
    }

    /// Transfers the native handle together with its actual CUDA context and selected stream.
    #[must_use]
    pub fn into_raw(mut self) -> RawContext {
        let handle = mem::replace(&mut self.handle.raw, ptr::null_mut());
        RawContext {
            handle,
            cuda_context: Arc::clone(&self.handle.cuda_context),
            stream: self.handle.stream.take(),
        }
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        if self.raw.is_null() {
            return;
        }
        if let Err(error) = self.cuda_context.bind() {
            #[cfg(debug_assertions)]
            eprintln!("failed to bind cuda context before destroying cusolver handle: {error}");
        }

        // SAFETY: `cusolverDnDestroy` receives live handles owned by the wrappers.
        unsafe {
            if let Err(error) = try_ffi!(sys::cusolverDnDestroy(self.raw)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cusolver context: {error}");
            }
        }
    }
}
