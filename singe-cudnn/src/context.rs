use std::{mem::replace, ptr, result::Result as StdResult, sync::Arc};

use singe_cuda::{
    context::Context as CudaContext,
    stream::{BorrowedStream, Stream, StreamBinding},
};
use singe_cudnn_sys as sys;

use crate::{
    error::{Error, Result},
    try_ffi,
};

/// A stateful cuDNN handle.
///
/// Use one context per host thread or concurrent task. The handle is movable
/// between threads, but it is intentionally not `Clone` or `Sync`.
/// Streams installed through `set_stream` are retained until replacement or destruction.
#[derive(Debug)]
pub struct Context {
    handle: Handle,
    stream: Option<Stream>,
}

#[derive(Debug)]
struct Handle {
    raw: sys::cudnnHandle_t,
    cuda_context: Arc<CudaContext>,
}

struct ScopedStream<'a> {
    context: &'a mut Context,
    previous: StreamBinding,
    previous_owner: Option<Stream>,
    restored: bool,
}

impl ScopedStream<'_> {
    fn restore(&mut self) -> Result<()> {
        if self.restored {
            return Ok(());
        }
        self.context.bind()?;
        // SAFETY: the saved binding retains an owned stream or borrows the context's original external stream.
        unsafe {
            try_ffi!(sys::cudnnSetStream(
                self.context.as_raw(),
                self.previous.as_raw(),
            ))?;
        }
        self.context.stream = self.previous_owner.take();
        self.restored = true;
        Ok(())
    }
}

impl Drop for ScopedStream<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.restore() {
            #[cfg(debug_assertions)]
            eprintln!("failed to restore cudnn stream: {error}");
        }
    }
}

// SAFETY: cuDNN handles carry mutable execution state. Ownership can move between
// threads, but the wrapper deliberately does not expose shared concurrent use.
unsafe impl Send for Handle {}

/// A transferred native handle with its actual CUDA context and selected stream owner.
/// Destroy `handle` before releasing these owners, or reclaim it with `Context::from_raw`.
/// External workspaces, callbacks and queued buffer lifetimes remain caller obligations.
#[derive(Debug)]
pub struct RawContext {
    /// Exclusively owned native cuDNN handle; destroy it before releasing the retained stream and CUDA context.
    pub handle: sys::cudnnHandle_t,
    /// Actual CUDA context needed for native handle use and destruction.
    pub cuda_context: Arc<CudaContext>,
    /// Retained installed stream, if one was selected through this wrapper. External stream ownership remains a caller obligation.
    pub stream: Option<Stream>,
}

// SAFETY: transfer carries exclusive native ownership and the resources needed for cross-thread cleanup.
unsafe impl Send for RawContext {}

impl Context {
    /// Initializes cuDNN and creates a library handle.
    /// Creating the handle allocates host and device resources and must happen
    /// before using other cuDNN operations through this context.
    ///
    /// The cuDNN handle is tied to the current CUDA context.
    /// To use the library on multiple devices, create one cuDNN handle for each device.
    ///
    /// For a given device, multiple cuDNN handles with different configurations
    /// (for example, different current CUDA streams) may be created.
    /// Because [`Context::create`] allocates some internal resources, destroying the handle may implicitly synchronize the current CUDA device. Create and destroy handles outside performance-critical code paths.
    ///
    /// For multithreaded applications that use the same device from different threads, create one or a few cuDNN handles per thread and reuse each handle for the thread lifetime.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDNN cannot allocate host or CUDA resources, if the
    /// GPU architecture is unsupported, if cuDNN rejects the handle creation
    /// inputs, if license validation fails, or if CUDA/cuDNN initialization
    /// cannot find a compatible enabled GPU.
    pub fn create(cuda_context: &Arc<CudaContext>) -> Result<Self> {
        cuda_context.bind()?;

        let mut handle = ptr::null_mut();
        // SAFETY: `cudnnCreate` receives valid output locations.
        unsafe {
            try_ffi!(sys::cudnnCreate(&raw mut handle))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle: Handle {
                raw: handle,
                cuda_context: Arc::clone(cuda_context),
            },
            stream: None,
        })
    }

    /// Wraps an existing cuDNN handle and takes ownership of it.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid cuDNN handle associated with `cuda_context`.
    /// Ownership of `handle` is transferred to the returned context, and the
    /// handle must not be destroyed elsewhere after calling this function.
    /// A supplied stream is installed and retained. If no owned stream is
    /// supplied, externally configured streams must remain valid until this
    /// handle is destroyed or another stream is installed.
    ///
    /// # Errors
    ///
    /// Returns an error if the handle is null, the supplied stream belongs to another CUDA context, CUDA binding fails, or cuDNN rejects installing the supplied stream.
    pub unsafe fn from_raw_parts(
        handle: sys::cudnnHandle_t,
        cuda_context: Arc<CudaContext>,
        stream: Option<Stream>,
    ) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        let mut context = Self {
            handle: Handle {
                raw: handle,
                cuda_context,
            },
            stream: None,
        };
        if let Some(stream) = stream {
            context.set_stream(Some(&stream))?;
        }
        Ok(context)
    }

    /// Returns the actual CUDA context retained by the cuDNN handle.
    #[must_use]
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        &self.handle.cuda_context
    }

    /// Makes the retained CUDA context current on this host thread.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot bind the context.
    pub fn bind(&self) -> Result<()> {
        Ok(self.cuda_context().bind()?)
    }

    /// Returns the CUDA stream configured on this cuDNN handle.
    /// When no stream is configured, this reports CUDA's default stream.
    /// Bindings for streams installed through Rust retain their stream owner.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDNN rejects the handle.
    pub fn stream(&self) -> Result<StreamBinding> {
        self.bind()?;

        let mut stream = ptr::null_mut();
        // SAFETY: `cudnnGetStream` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::cudnnGetStream(self.as_raw(), &raw mut stream))?;
        }

        Ok(if stream.is_null() {
            StreamBinding::Default(Arc::clone(self.cuda_context()))
        } else if let Some(owner) = &self.stream
            && owner.as_raw() == stream
        {
            StreamBinding::Borrowed(owner.to_borrowed())
        } else {
            // SAFETY: the library reports the stream bound to this handle, which the handle keeps alive.
            StreamBinding::Borrowed(unsafe {
                BorrowedStream::from_raw(stream.cast(), Arc::clone(self.cuda_context()))
            })
        })
    }

    /// Sets the CUDA stream used by subsequent cuDNN calls on this handle.
    /// Retains the supplied stream and requires exclusive configuration access.
    /// cuDNN uses the new stream to launch GPU kernels or to synchronize when cuDNN kernels run on internal streams.
    /// If the cuDNN library stream is not set, all kernels use the default stream.
    /// Setting the stream in the cuDNN handle guarantees issue-order execution of cuDNN calls and other GPU kernels launched in the same stream.
    ///
    /// With CUDA 11.x or later, internal streams have the same priority as the stream set by the last call to this method.
    /// In CUDA graph capture mode, CUDA 11.8 or later is required in order for the stream priorities to match.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDNN rejects the handle, if `stream` belongs to a
    /// different context than the cuDNN handle, if CUDA stream APIs fail, or if
    /// the stream priority is out of range.
    pub fn set_stream(&mut self, stream: Option<&Stream>) -> Result<()> {
        if stream.is_some_and(|stream| self.cuda_context() != stream.context()) {
            return Err(Error::StreamContextMismatch);
        }
        self.bind()?;

        // SAFETY: `cudnnSetStream` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::cudnnSetStream(
                self.as_raw(),
                stream.map_or_else(ptr::null_mut, Stream::as_raw),
            ))?;
        }
        self.stream = stream.cloned();
        Ok(())
    }

    /// Uses a temporary stream for `f`, then restores the original native stream and its owner.
    /// Restoration also runs when `f` returns an error or unwinds.
    /// Work enqueued by `f` retains the ordinary execution and completion contracts.
    /// The closure may use any error type that can represent a cuDNN error.
    /// Its error takes precedence over a simultaneous restoration failure.
    ///
    /// # Errors
    /// Returns stream configuration, closure or restoration errors.
    pub fn with_stream<T, E>(
        &mut self,
        stream: Option<&Stream>,
        f: impl FnOnce(&Self) -> StdResult<T, E>,
    ) -> StdResult<T, E>
    where
        E: From<Error>,
    {
        let previous = self.stream()?;
        let previous_owner = self.stream.clone();
        self.set_stream(stream)?;
        let mut scope = ScopedStream {
            context: self,
            previous,
            previous_owner,
            restored: false,
        };
        let result = f(scope.context);
        let restoration = scope.restore().map_err(E::from);
        match (result, restoration) {
            (Ok(value), Ok(())) => Ok(value),
            (Err(error), _) | (_, Err(error)) => Err(error),
        }
    }

    /// Returns the raw cuDNN handle.
    ///
    /// The returned handle is borrowed and remains valid only while this
    /// context and its underlying CUDA context are alive.
    #[must_use]
    pub fn as_raw(&self) -> sys::cudnnHandle_t {
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
            },
            stream: raw.stream,
        })
    }

    /// Transfers the native handle together with its actual CUDA context and selected stream.
    #[must_use]
    pub fn into_raw(mut self) -> RawContext {
        let handle = replace(&mut self.handle.raw, ptr::null_mut());
        RawContext {
            handle,
            cuda_context: Arc::clone(&self.handle.cuda_context),
            stream: self.stream.take(),
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
            eprintln!("failed to bind cuda context before destroying cudnn handle: {error}");
        }

        // SAFETY: `cudnnDestroy` receives live handles owned by the wrappers.
        unsafe {
            if let Err(error) = try_ffi!(sys::cudnnDestroy(self.raw)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cudnn context: {error}");
            }
        }
    }
}
