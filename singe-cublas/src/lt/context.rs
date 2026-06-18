use std::{mem::ManuallyDrop, ptr, sync::Arc};

use singe_core::LibraryVersion;
use singe_cublas_sys as sys;
use singe_cuda::{context::Context as CudaContext, stream::Stream};
use singe_cuda_sys::runtime;

use crate::{
    error::{Error, Result},
    lt::version,
    try_ffi,
};

#[derive(Debug)]
pub struct Context {
    handle: Handle,
}

#[derive(Debug)]
struct Handle {
    raw: sys::cublasLtHandle_t,
    cuda_ctx: Arc<CudaContext>,
}

// cuBLASLt handles are stateful library contexts. They may move between
// threads, but shared concurrent use is not exposed by this wrapper.
unsafe impl Send for Handle {}

impl Context {
    /// Creates a cuBLASLt context for the current CUDA device.
    ///
    /// The handle is tied to the device associated with `cuda_ctx`. To use cuBLASLt on multiple
    /// devices, create one [`Context`] per device.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt initialization fails, if binding `cuda_ctx` fails, or if the
    /// library does not return a valid handle.
    pub fn create(cuda_ctx: &Arc<CudaContext>) -> Result<Self> {
        cuda_ctx.bind()?;

        let mut handle = ptr::null_mut();
        unsafe {
            try_ffi!(sys::cublasLtCreate(&raw mut handle))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle: Handle {
                raw: handle,
                cuda_ctx: Arc::clone(cuda_ctx),
            },
        })
    }

    /// Wraps an existing cuBLASLt handle.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid cuBLASLt handle associated with `cuda_ctx`.
    /// The returned context takes ownership of `handle` and destroys it on drop.
    pub unsafe fn from_raw(
        handle: sys::cublasLtHandle_t,
        cuda_ctx: Arc<CudaContext>,
    ) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle: Handle {
                raw: handle,
                cuda_ctx,
            },
        })
    }

    /// Returns the underlying CUDA context used by this cuBLASLt handle.
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        &self.handle.cuda_ctx
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
    /// Returns an error if the stream belongs to a different context.
    pub fn ensure_stream(&self, stream: &Stream) -> Result<()> {
        if self.cuda_context().as_ref() != stream.context() {
            return Err(Error::StreamContextMismatch);
        }

        self.bind()
    }

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
    pub fn as_raw(&self) -> sys::cublasLtHandle_t {
        self.handle.raw
    }

    /// Consumes this context and returns the owned raw cuBLASLt handle.
    ///
    /// The caller becomes responsible for eventually destroying the handle.
    pub fn into_raw(self) -> sys::cublasLtHandle_t {
        let this = ManuallyDrop::new(self);
        this.handle.raw
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        if let Err(err) = self.cuda_ctx.bind() {
            #[cfg(debug_assertions)]
            eprintln!("failed to bind cuda context before destroying cublasLt handle: {err}");
        }

        unsafe {
            if let Err(err) = try_ffi!(sys::cublasLtDestroy(self.raw)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cublasLt context: {err}");
            }
        }
    }
}
