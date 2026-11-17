//! Native cuDNN handle ownership and CUDA context and stream binding.

use std::{convert::Infallible, result::Result as StdResult, sync::Arc};

use singe_cuda::{
    context::Context as CudaContext,
    handle::{CreateHandle, Handle, NativeHandle, RawHandle, StreamHandle},
    stream::{Stream, StreamBinding},
    sys::runtime,
    view::DeviceRegion,
};
use singe_cudnn_sys as sys;

use crate::{
    error::{Error, Result},
    try_ffi,
};

/// Marker type for the native cuDNN handle owned by [`Context`].
#[derive(Debug)]
pub struct CudnnHandle(Infallible);

// SAFETY: cuDNN handles may be used and destroyed from any thread, one call at a time.
unsafe impl NativeHandle for CudnnHandle {
    type Raw = sys::cudnnHandle_t;

    type State = ();

    type Error = Error;

    const OBJECT: &'static str = "cudnn context";

    fn null_handle() -> Error {
        Error::NullHandle
    }

    fn is_null(raw: Self::Raw) -> bool {
        raw.is_null()
    }

    unsafe fn destroy(raw: Self::Raw, (): ()) -> Result<()> {
        // SAFETY: the caller passes a live, uniquely owned handle with its context bound.
        unsafe { try_ffi!(sys::cudnnDestroy(raw)) }
    }
}

// SAFETY: `cudnnCreate` binds to the current context.
unsafe impl CreateHandle for CudnnHandle {
    unsafe fn create(raw: *mut Self::Raw) -> Result<()> {
        // SAFETY: the caller provides a writable output location and binds the context.
        unsafe { try_ffi!(sys::cudnnCreate(raw)) }
    }
}

// SAFETY: `cudnnSetStream` only installs the stream; `Handle` retains it.
unsafe impl StreamHandle for CudnnHandle {
    unsafe fn set_stream(raw: Self::Raw, stream: runtime::cudaStream_t) -> Result<()> {
        // SAFETY: the caller passes a live handle and a null or live stream of its bound context.
        unsafe { try_ffi!(sys::cudnnSetStream(raw, stream)) }
    }
}

/// Raw ownership of a cuDNN handle, its CUDA context, and its installed stream.
pub type RawContext = RawHandle<CudnnHandle>;

/// A stateful cuDNN handle.
///
/// Use one context per host thread or concurrent task.
/// The handle is movable between threads, but it is intentionally not `Clone` or `Sync`.
/// Streams installed through `set_stream` are retained until replacement or destruction.
#[derive(Debug)]
pub struct Context {
    handle: Handle<CudnnHandle>,
}

impl Context {
    /// Initializes cuDNN and creates a library handle on the legacy default stream.
    ///
    /// The cuDNN handle is tied to `cuda_context`; create one handle for each device.
    /// Creating and destroying a handle allocates resources and may synchronize the device, so keep handles out of performance-critical code paths.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, cuDNN cannot allocate host or CUDA resources, the GPU architecture is unsupported, license validation fails, or cuDNN returns a null handle.
    pub fn create(cuda_context: &Arc<CudaContext>) -> Result<Self> {
        Ok(Self {
            handle: Handle::create(cuda_context)?,
        })
    }

    /// Adopts a cuDNN handle with its CUDA context and installed stream.
    ///
    /// # Safety
    ///
    /// `raw.handle` must be a uniquely owned cuDNN handle created in `raw.cuda_context`, with `raw.stream` installed on it, or the legacy default stream when `raw.stream` is `None`.
    ///
    /// # Errors
    ///
    /// Returns an error if the handle is null or the stream belongs to another CUDA context.
    pub unsafe fn from_raw(raw: RawContext) -> Result<Self> {
        Ok(Self {
            // SAFETY: the caller guarantees a uniquely owned live handle of `raw.cuda_context` with `raw.stream` installed.
            handle: unsafe { Handle::from_raw(raw)? },
        })
    }

    /// Returns the CUDA context retained by the cuDNN handle.
    #[must_use]
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        self.handle.cuda_context()
    }

    /// Makes the retained CUDA context current on this host thread.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot bind the context.
    pub fn bind(&self) -> Result<()> {
        self.handle.bind()
    }

    /// Returns the stream installed on this handle, or `None` for the legacy default stream.
    #[must_use]
    pub fn stream(&self) -> Option<&Stream> {
        self.handle.stream()
    }

    /// Returns the stream this handle's operations run on.
    #[must_use]
    pub fn stream_binding(&self) -> StreamBinding {
        self.handle.stream_binding()
    }

    /// Installs `stream` for cuDNN operations on this handle and retains it, or the legacy default stream for `None`.
    ///
    /// cuDNN uses the stream to launch its kernels, and to synchronize when its kernels run on internal streams.
    /// On an error the handle keeps its previous stream.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to another CUDA context, the CUDA context cannot be bound, or cuDNN rejects the stream.
    pub fn set_stream(&mut self, stream: Option<&Stream>) -> Result<()> {
        self.handle.set_stream(stream)
    }

    /// Runs `f`, then waits for this handle's stream, even when `f` fails.
    ///
    /// # Errors
    ///
    /// Returns an error if the stream is capturing, `f` fails, or the wait fails.
    pub fn sync_scope<R>(&self, f: impl FnOnce() -> Result<R>) -> Result<R> {
        self.handle.sync_scope(f)
    }

    /// Orders later work on this handle's stream after the producers of these regions.
    ///
    /// # Errors
    ///
    /// Returns an error if a region belongs to another CUDA context or the dependencies cannot be enqueued.
    pub fn prepare_regions(&self, regions: &[DeviceRegion<'_>]) -> Result<()> {
        self.handle.prepare_regions(regions)
    }

    // Installs `stream` for `f`, then restores the previous stream, also when `f` fails.
    // The closure's error takes precedence over a restoration failure.
    pub(crate) fn with_stream<T, E>(
        &mut self,
        stream: Option<&Stream>,
        f: impl FnOnce(&Self) -> StdResult<T, E>,
    ) -> StdResult<T, E>
    where
        E: From<Error>,
    {
        let previous = self.handle.stream().cloned();
        self.set_stream(stream)?;
        let result = f(self);
        let restoration = self.set_stream(previous.as_ref()).map_err(E::from);
        match (result, restoration) {
            (Ok(value), Ok(())) => Ok(value),
            (Err(error), _) | (_, Err(error)) => Err(error),
        }
    }

    /// Returns the raw cuDNN handle.
    ///
    /// The returned handle is borrowed and remains valid only while this context and its underlying CUDA context are alive.
    #[must_use]
    pub fn as_raw(&self) -> sys::cudnnHandle_t {
        self.handle.as_raw()
    }

    /// Consumes the context and returns the raw cuDNN handle, its CUDA context, and its stream without destroying the handle.
    ///
    /// Destroy the transferred handle before releasing its context and stream, or re-adopt the complete [`RawContext`].
    #[must_use]
    pub fn into_raw(self) -> RawContext {
        self.handle.into_raw()
    }
}
