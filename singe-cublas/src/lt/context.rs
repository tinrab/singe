use std::{mem, ptr, sync::Arc};

use singe_core::LibraryVersion;
use singe_cublas_sys as sys;
use singe_cuda::{context::Context as CudaContext, stream::Stream};
use singe_cuda_sys::runtime;

use crate::{
    error::{Error, Result},
    lt::version,
    try_ffi,
};

/// An owned cuBLASLt handle tied to a retained CUDA context.
#[derive(Debug)]
pub struct Context {
    handle: Handle,
}

#[derive(Debug)]
struct Handle {
    raw: sys::cublasLtHandle_t,
    cuda_context: Arc<CudaContext>,
}

// SAFETY: cuBLASLt handles are stateful library contexts. They may move between
// threads, but shared concurrent use is not exposed by this wrapper.
unsafe impl Send for Handle {}

/// A transferred native context with its actual CUDA context.
///
/// Destroy `handle` before releasing these owners, or reclaim it with `Context::from_raw`.
/// External workspaces, callbacks, queued buffers and graph replays remain caller obligations.
#[derive(Debug)]
pub struct RawContext {
    /// The owned native cuBLASLt handle.
    pub handle: sys::cublasLtHandle_t,
    /// The CUDA context associated with the native handle.
    pub cuda_context: Arc<CudaContext>,
}

// SAFETY: transfer carries exclusive native ownership and the resources needed for cross-thread cleanup.
unsafe impl Send for RawContext {}

impl Context {
    /// Creates a cuBLASLt context for the current CUDA device.
    ///
    /// The handle is tied to the device associated with `cuda_context`. To use cuBLASLt on multiple
    /// devices, create one [`Context`] per device.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt initialization fails, if binding `cuda_context` fails, or if the
    /// library does not return a valid handle.
    pub fn create(cuda_context: &Arc<CudaContext>) -> Result<Self> {
        cuda_context.bind()?;

        let mut handle = ptr::null_mut();
        // SAFETY: `cublasLtCreate` receives valid output locations.
        unsafe {
            try_ffi!(sys::cublasLtCreate(&raw mut handle))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle: Handle {
                raw: handle,
                cuda_context: Arc::clone(cuda_context),
            },
        })
    }

    /// Wraps an existing cuBLASLt handle.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid cuBLASLt handle associated with `cuda_context`.
    /// The returned context takes ownership of `handle` and destroys it on drop.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NullHandle`] if `handle` is null.
    pub unsafe fn from_raw_parts(
        handle: sys::cublasLtHandle_t,
        cuda_context: Arc<CudaContext>,
    ) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle: Handle {
                raw: handle,
                cuda_context,
            },
        })
    }

    /// Returns the underlying CUDA context used by this cuBLASLt handle.
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

    /// Ensures `stream` belongs to the same CUDA context as this handle.
    ///
    /// # Errors
    ///
    /// Returns [`Error::StreamContextMismatch`] if `stream` belongs to another CUDA context, or a CUDA error if binding the context fails.
    pub fn ensure_stream(&self, stream: &Stream) -> Result<()> {
        if self.cuda_context() != stream.context() {
            return Err(Error::StreamContextMismatch);
        }

        self.bind()
    }

    /// Returns the loaded cuBLASLt library version.
    ///
    /// # Errors
    ///
    /// Returns an error if binding the CUDA context fails or the native version cannot be represented as `u64`.
    pub fn version(&self) -> Result<LibraryVersion> {
        self.bind()?;
        version()
    }

    pub(crate) unsafe fn stream_raw(
        &self,
        stream: Option<&Stream>,
    ) -> Result<runtime::cudaStream_t> {
        if let Some(stream) = stream {
            self.ensure_stream(stream)?;
            Ok(stream.as_raw())
        } else {
            self.bind()?;
            Ok(ptr::null_mut())
        }
    }

    /// Returns the raw cuBLASLt handle.
    ///
    /// The returned handle is borrowed and remains valid only while this
    /// context and its underlying CUDA context are alive.
    #[must_use]
    pub fn as_raw(&self) -> sys::cublasLtHandle_t {
        self.handle.raw
    }

    /// Reclaims a native context transferred together with its dependent owners.
    ///
    /// # Safety
    ///
    /// `handle` must remain valid, uniquely owned, and associated with `cuda_context`.
    /// Retained streams must cover all streams still referenced by native state or queued work.
    /// Complete all borrowed work before releasing or replacing its resources.
    ///
    /// # Errors
    ///
    /// Returns an error if the transferred handle is null.
    pub unsafe fn from_raw(raw: RawContext) -> Result<Self> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { Self::from_raw_parts(raw.handle, raw.cuda_context) }
    }

    /// Transfers the native handle together with its actual CUDA context.
    #[must_use]
    pub fn into_raw(mut self) -> RawContext {
        let handle = mem::replace(&mut self.handle.raw, ptr::null_mut());
        RawContext {
            handle,
            cuda_context: Arc::clone(&self.handle.cuda_context),
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
            eprintln!("failed to bind cuda context before destroying cublaslt handle: {error}");
        }

        // SAFETY: `cublasLtDestroy` receives live handles owned by the wrappers.
        unsafe {
            if let Err(error) = try_ffi!(sys::cublasLtDestroy(self.raw)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cublaslt context: {error}");
            }
        }
    }
}
