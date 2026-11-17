//! Owned native library handles with their CUDA context and stream.

use std::{
    cell::Cell,
    error::Error as StdError,
    fmt,
    marker::PhantomData,
    mem::{self, ManuallyDrop, MaybeUninit},
    ptr,
    sync::Arc,
};

use singe_core::report_cleanup_failure;
use singe_cuda_sys::runtime;

use crate::{
    context::Context,
    error::Error as CudaError,
    stream::{AsStream, Stream, StreamBinding},
    view::DeviceRegion,
};

/// A native library handle type that [`Handle`] owns.
///
/// Wrappers implement it on an uninhabited marker type.
///
/// # Safety
///
/// The handle may be used and destroyed from any thread, one call at a time.
/// `destroy` must release the handle, and must not release a part of its state that the library may still use.
pub unsafe trait NativeHandle: 'static {
    /// Native handle type.
    type Raw: Copy + fmt::Debug;

    /// Owned state that [`Handle`] keeps with the native handle and passes to `destroy`, or `()`.
    type State: fmt::Debug + Send;

    /// Error type of the wrapper crate.
    type Error: From<CudaError> + StdError + 'static;

    /// Name of the object in cleanup reports, such as `"cusparse context"`.
    const OBJECT: &'static str;

    /// Returns the error for a null native handle.
    fn null_handle() -> Self::Error;

    /// Returns whether `raw` is the null handle.
    fn is_null(raw: Self::Raw) -> bool;

    /// Destroys a native handle and releases its state.
    ///
    /// # Safety
    ///
    /// `raw` must be a live handle with no other owner, `state` must be the state it was adopted with, and its CUDA context must be bound.
    ///
    /// # Errors
    ///
    /// Returns an error if the library cannot destroy the handle.
    unsafe fn destroy(raw: Self::Raw, state: Self::State) -> Result<(), Self::Error>;
}

/// A native library handle that [`Handle::create`] creates without arguments.
///
/// # Safety
///
/// `create` must create a handle in the bound CUDA context.
pub unsafe trait CreateHandle: NativeHandle<State = ()> {
    /// Creates a native handle in the bound CUDA context and writes it to `raw`.
    ///
    /// # Safety
    ///
    /// `raw` must be valid for writes.
    ///
    /// # Errors
    ///
    /// Returns an error if the library cannot create the handle.
    unsafe fn create(raw: *mut Self::Raw) -> Result<(), Self::Error>;
}

/// A native library handle that owns the stream its operations run on.
///
/// # Safety
///
/// `set_stream` must only install the stream on the handle; [`Handle`] retains it.
pub unsafe trait StreamHandle: NativeHandle {
    /// Installs `stream` on the handle, or the legacy default stream for null.
    ///
    /// # Safety
    ///
    /// `raw` must be a live handle whose CUDA context is bound, and `stream` must be null or a live stream of that context.
    ///
    /// # Errors
    ///
    /// Returns an error if the library rejects the stream.
    unsafe fn set_stream(raw: Self::Raw, stream: runtime::cudaStream_t) -> Result<(), Self::Error>;
}

/// An owned native library handle, its CUDA context, and the stream installed on it.
///
/// The handle is `Send` and not `Sync`.
/// Dropping it destroys the native handle in its context; if the context cannot be bound, the handle and its state leak.
pub struct Handle<H: NativeHandle> {
    raw: H::Raw,
    state: ManuallyDrop<H::State>,
    stream: Option<Stream>,
    cuda_context: Arc<Context>,
    _not_sync: PhantomData<Cell<()>>,
}

/// Raw ownership of a native library handle, its CUDA context, and its installed stream.
///
/// Destroy `handle` before releasing these owners, or re-adopt the value with [`Handle::from_raw`].
pub struct RawHandle<H: NativeHandle> {
    /// Native handle.
    pub handle: H::Raw,
    /// CUDA context the handle was created in.
    pub cuda_context: Arc<Context>,
    /// Stream installed on the handle, or `None` for the legacy default stream.
    pub stream: Option<Stream>,
    /// State kept with the handle.
    pub state: H::State,
}

// SAFETY: `NativeHandle` allows using and destroying the handle from any thread, one call at a time, and its state is `Send`; `Handle` is not `Sync`.
unsafe impl<H: NativeHandle> Send for Handle<H> {}

// SAFETY: `NativeHandle` allows using and destroying the handle from any thread, and the context and stream owners are `Send`; the value only carries them to their next owner.
unsafe impl<H: NativeHandle> Send for RawHandle<H> {}

impl<H: NativeHandle> fmt::Debug for Handle<H> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Handle")
            .field("object", &H::OBJECT)
            .field("raw", &self.raw)
            .field("state", &*self.state)
            .field("stream", &self.stream)
            .field("cuda_context", &self.cuda_context)
            .finish()
    }
}

impl<H: NativeHandle> fmt::Debug for RawHandle<H> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RawHandle")
            .field("object", &H::OBJECT)
            .field("handle", &self.handle)
            .field("state", &self.state)
            .field("stream", &self.stream)
            .field("cuda_context", &self.cuda_context)
            .finish()
    }
}

impl<H: CreateHandle> Handle<H> {
    /// Creates a native handle in `cuda_context`, on the legacy default stream.
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound, the library cannot create the handle, or it returns a null handle.
    pub fn create(cuda_context: &Arc<Context>) -> Result<Self, H::Error> {
        cuda_context.bind()?;
        let mut raw = MaybeUninit::uninit();
        // SAFETY: the output location is valid for writes and the context is bound.
        unsafe { H::create(raw.as_mut_ptr())? };
        // SAFETY: successful creation initialized `raw`.
        let raw = unsafe { raw.assume_init() };
        if H::is_null(raw) {
            return Err(H::null_handle());
        }

        Ok(Self {
            raw,
            state: ManuallyDrop::new(()),
            stream: None,
            cuda_context: Arc::clone(cuda_context),
            _not_sync: PhantomData,
        })
    }
}

impl<H: NativeHandle> Handle<H> {
    /// Adopts a native handle with its context and installed stream.
    ///
    /// # Safety
    ///
    /// `raw.handle` must be live and uniquely owned, created in `raw.cuda_context`, with `raw.stream` installed on it and `raw.state` as its state.
    ///
    /// # Errors
    ///
    /// Returns an error if the handle is null or the stream belongs to another context.
    pub unsafe fn from_raw(raw: RawHandle<H>) -> Result<Self, H::Error> {
        if H::is_null(raw.handle) {
            return Err(H::null_handle());
        }

        if raw
            .stream
            .as_ref()
            .is_some_and(|stream| stream.context() != &raw.cuda_context)
        {
            return Err(CudaError::StreamContextMismatch.into());
        }
        Ok(Self {
            raw: raw.handle,
            state: ManuallyDrop::new(raw.state),
            stream: raw.stream,
            cuda_context: raw.cuda_context,
            _not_sync: PhantomData,
        })
    }

    /// Transfers the native handle, its context, its installed stream, and its state to the caller.
    #[must_use]
    pub fn into_raw(self) -> RawHandle<H> {
        let this = ManuallyDrop::new(self);
        // SAFETY: `this` is never dropped, so each owner moves out exactly once.
        unsafe {
            RawHandle {
                handle: this.raw,
                cuda_context: ptr::read(&raw const this.cuda_context),
                stream: ptr::read(&raw const this.stream),
                state: ManuallyDrop::into_inner(ptr::read(&raw const this.state)),
            }
        }
    }

    /// Destroys the native handle in its context, returning the failure that dropping it would only report.
    ///
    /// If the context cannot be bound, the handle and its state leak.
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound or the library cannot destroy the handle.
    pub fn destroy(self) -> Result<(), H::Error> {
        let raw = self.into_raw();
        if let Err(error) = raw.cuda_context.bind() {
            mem::forget(raw.state);
            return Err(error.into());
        }

        // SAFETY: the handle is live, uniquely owned, adopted with this state, and its context is bound.
        unsafe { H::destroy(raw.handle, raw.state) }
    }

    /// Returns the state kept with the handle.
    #[must_use]
    pub fn state(&self) -> &H::State {
        &self.state
    }

    /// Returns the native handle without transferring ownership.
    #[must_use]
    pub fn as_raw(&self) -> H::Raw {
        self.raw
    }

    /// Returns the CUDA context the handle was created in.
    #[must_use]
    pub fn cuda_context(&self) -> &Arc<Context> {
        &self.cuda_context
    }

    /// Makes the handle's CUDA context current on the calling thread.
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound.
    pub fn bind(&self) -> Result<(), H::Error> {
        Ok(self.cuda_context.bind()?)
    }

    /// Returns the stream installed on the handle, or `None` for the legacy default stream.
    #[must_use]
    pub fn stream(&self) -> Option<&Stream> {
        self.stream.as_ref()
    }

    /// Returns the stream the handle's operations run on.
    #[must_use]
    pub fn stream_binding(&self) -> StreamBinding {
        self.stream.as_ref().map_or_else(
            || StreamBinding::Default(Arc::clone(&self.cuda_context)),
            |stream| StreamBinding::Borrowed(stream.to_borrowed()),
        )
    }

    /// Runs `f`, then waits for the handle's stream, even when `f` fails.
    ///
    /// # Errors
    ///
    /// Returns an error if the stream is capturing, `f` fails, or the wait fails.
    pub fn sync_scope<R>(&self, f: impl FnOnce() -> Result<R, H::Error>) -> Result<R, H::Error> {
        self.stream_binding().sync_scope(f)
    }

    /// Orders later work on the handle's stream after the producers of these regions.
    ///
    /// # Errors
    ///
    /// Returns `CudaError::MemoryContextMismatch` if a region belongs to another context, or an error if the dependencies cannot be enqueued.
    pub fn prepare_regions(&self, regions: &[DeviceRegion<'_>]) -> Result<(), H::Error> {
        Ok(self.stream_binding().join_regions(regions)?)
    }
}

impl<H: StreamHandle> Handle<H> {
    /// Installs `stream` on the handle and retains it, or the legacy default stream for `None`.
    ///
    /// On an error the handle keeps its previous stream.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to another context, the context cannot be bound, or the library rejects the stream.
    pub fn set_stream(&mut self, stream: Option<&Stream>) -> Result<(), H::Error> {
        if stream.is_some_and(|stream| stream.context() != &self.cuda_context) {
            return Err(CudaError::StreamContextMismatch.into());
        }

        self.bind()?;
        // SAFETY: the handle is live in the bound context, and the stream is retained below.
        unsafe { H::set_stream(self.raw, stream.map_or(ptr::null_mut(), Stream::as_raw))? };
        self.stream = stream.cloned();
        Ok(())
    }
}

impl<H: NativeHandle> Drop for Handle<H> {
    fn drop(&mut self) {
        // SAFETY: drop runs once, and nothing reads `self.state` after this.
        let state = unsafe { ManuallyDrop::take(&mut self.state) };
        if H::is_null(self.raw) {
            return;
        }

        if let Err(error) = self.cuda_context.bind() {
            report_cleanup_failure("bind the cuda context to destroy", H::OBJECT, &error);
            mem::forget(state);
            return;
        }

        // SAFETY: the handle is live, uniquely owned, adopted with this state, and its context is bound.
        if let Err(error) = unsafe { H::destroy(self.raw, state) } {
            report_cleanup_failure("destroy", H::OBJECT, &error);
        }
    }
}
