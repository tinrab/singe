use std::{convert::Infallible, ptr, sync::Arc};

use singe_cublas_sys as sys;
use singe_cuda::{
    context::Context as CudaContext,
    handle::{CreateHandle, Handle, NativeHandle, RawHandle},
    stream::{Stream, StreamBinding},
};
use singe_cuda_sys::runtime;

use crate::error::{Error, Result, try_ffi};

/// Marker type for the native cuBLASLt handle owned by [`Context`].
#[derive(Debug)]
pub struct CublasLtHandle(Infallible);

// SAFETY: cuBLASLt handles may be used and destroyed from any thread, one call at a time.
unsafe impl NativeHandle for CublasLtHandle {
    type Raw = sys::cublasLtHandle_t;

    type State = ();

    type Error = Error;

    const OBJECT: &'static str = "cublaslt context";

    fn null_handle() -> Error {
        Error::NullHandle
    }

    fn is_null(raw: Self::Raw) -> bool {
        raw.is_null()
    }

    unsafe fn destroy(raw: Self::Raw, (): ()) -> Result<()> {
        // SAFETY: `NativeHandle::destroy` requires a live, uniquely owned handle with its context bound, so nothing uses it afterwards.
        unsafe { try_ffi!(sys::cublasLtDestroy(raw)) }
    }
}

// SAFETY: `cublasLtCreate` binds to the current context.
unsafe impl CreateHandle for CublasLtHandle {
    unsafe fn create(raw: *mut Self::Raw) -> Result<()> {
        // SAFETY: `CreateHandle::create` requires a writable `raw` and a bound CUDA context, which is all handle creation needs.
        unsafe { try_ffi!(sys::cublasLtCreate(raw)) }
    }
}

/// Raw ownership of a cuBLASLt handle and its CUDA context.
///
/// cuBLASLt takes a stream per call, so the transferred `stream` is always `None`.
pub type RawContext = RawHandle<CublasLtHandle>;

/// An owned cuBLASLt handle tied to a retained CUDA context.
///
/// Operations take their stream per call.
/// The handle is movable between threads, but it is not `Clone` or `Sync`.
#[derive(Debug)]
pub struct Context {
    handle: Handle<CublasLtHandle>,
}

impl Context {
    /// Creates a cuBLASLt handle in `cuda_context`.
    ///
    /// To use cuBLASLt on several devices, create one context per device.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, cuBLASLt cannot create the handle, or it returns a null handle.
    pub fn create(cuda_context: &Arc<CudaContext>) -> Result<Self> {
        Ok(Self {
            handle: Handle::create(cuda_context)?,
        })
    }

    /// Adopts a cuBLASLt handle with its CUDA context.
    ///
    /// # Safety
    ///
    /// `raw.handle` must be a uniquely owned cuBLASLt handle created in `raw.cuda_context`.
    ///
    /// # Errors
    ///
    /// Returns an error if the handle is null or `raw.stream` belongs to another CUDA context.
    pub unsafe fn from_raw(raw: RawContext) -> Result<Self> {
        Ok(Self {
            // SAFETY: the caller guarantees `raw.handle` is a uniquely owned cuBLASLt handle created in `raw.cuda_context`.
            handle: unsafe { Handle::from_raw(raw)? },
        })
    }

    /// Transfers the native handle and its CUDA context to the caller.
    ///
    /// Destroy the transferred handle before releasing its context, or re-adopt the complete [`RawContext`].
    #[must_use]
    pub fn into_raw(self) -> RawContext {
        self.handle.into_raw()
    }

    /// Returns the raw cuBLASLt handle.
    ///
    /// The returned handle is borrowed and remains valid only while this context is alive.
    #[must_use]
    pub fn as_raw(&self) -> sys::cublasLtHandle_t {
        self.handle.as_raw()
    }

    /// Returns the CUDA context of this handle.
    #[must_use]
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        self.handle.cuda_context()
    }

    /// Makes the handle's CUDA context current on the calling thread.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound.
    pub fn bind(&self) -> Result<()> {
        self.handle.bind()
    }

    /// Checks that `stream` belongs to this handle's CUDA context, then binds the context.
    ///
    /// # Errors
    ///
    /// Returns [`Error::StreamContextMismatch`] if `stream` belongs to another CUDA context, or an error if binding the context fails.
    pub fn ensure_stream(&self, stream: &Stream) -> Result<()> {
        if self.cuda_context() != stream.context() {
            return Err(Error::StreamContextMismatch);
        }

        self.bind()
    }

    // Binds the context and returns the native stream for `stream`, or the legacy default stream for `None`.
    pub(crate) fn native_stream(&self, stream: Option<&Stream>) -> Result<runtime::cudaStream_t> {
        if let Some(stream) = stream {
            self.ensure_stream(stream)?;
            Ok(stream.as_raw())
        } else {
            self.bind()?;
            Ok(ptr::null_mut())
        }
    }

    // Returns the selection that work on `stream`, or on the legacy default stream for `None`, is ordered on.
    pub(crate) fn stream_binding(&self, stream: Option<&Stream>) -> StreamBinding {
        stream.map_or_else(
            || StreamBinding::Default(Arc::clone(self.cuda_context())),
            |stream| StreamBinding::Borrowed(stream.to_borrowed()),
        )
    }
}
