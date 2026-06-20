use std::{mem::ManuallyDrop, ptr, sync::Arc};

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
#[derive(Debug)]
pub struct Context {
    handle: Handle,
}

#[derive(Debug)]
struct Handle {
    raw: sys::cudnnHandle_t,
    cuda_ctx: Arc<CudaContext>,
}

// cuDNN handles carry mutable execution state. Ownership can move between
// threads, but the wrapper deliberately does not expose shared concurrent use.
unsafe impl Send for Handle {}

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
    pub fn create(cuda_ctx: &Arc<CudaContext>) -> Result<Self> {
        cuda_ctx.bind()?;

        let mut handle = ptr::null_mut();
        unsafe {
            try_ffi!(sys::cudnnCreate(&raw mut handle))?;
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

    /// Wraps an existing cuDNN handle and takes ownership of it.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid cuDNN handle associated with `cuda_ctx`.
    /// Ownership of `handle` is transferred to the returned context, and the
    /// handle must not be destroyed elsewhere after calling this function.
    pub unsafe fn from_raw(handle: sys::cudnnHandle_t, cuda_ctx: Arc<CudaContext>) -> Result<Self> {
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

    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        &self.handle.cuda_ctx
    }

    pub fn bind(&self) -> Result<()> {
        Ok(self.cuda_context().bind()?)
    }

    fn set_stream_checked(&self, stream: &Stream) -> Result<()> {
        if self.cuda_context().as_ref() != stream.context() {
            return Err(Error::StreamContextMismatch);
        }

        self.bind()?;
        unsafe {
            try_ffi!(sys::cudnnSetStream(self.as_raw(), stream.as_raw()))?;
        }
        Ok(())
    }

    /// Returns the CUDA stream configured on this cuDNN handle.
    /// When no stream is configured, this reports CUDA's default stream.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDNN rejects the handle.
    pub fn stream(&self) -> Result<StreamBinding> {
        self.bind()?;

        let mut stream = ptr::null_mut();
        unsafe {
            try_ffi!(sys::cudnnGetStream(self.as_raw(), &raw mut stream))?;
        }

        Ok(if stream.is_null() {
            StreamBinding::Default(Arc::clone(self.cuda_context()))
        } else {
            StreamBinding::Borrowed(unsafe {
                BorrowedStream::from_raw(stream.cast(), Arc::clone(self.cuda_context()))
            })
        })
    }

    /// Sets the CUDA stream used by subsequent cuDNN calls on this handle.
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
    pub fn set_stream(&self, stream: Option<&Stream>) -> Result<()> {
        if let Some(stream) = stream {
            return self.set_stream_checked(stream);
        } else {
            self.bind()?;
        }

        unsafe {
            try_ffi!(sys::cudnnSetStream(
                self.as_raw(),
                match stream {
                    Some(stream) => stream.as_raw(),
                    None => ptr::null_mut(),
                },
            ))?;
        }
        Ok(())
    }

    /// Returns the raw cuDNN handle.
    ///
    /// The returned handle is borrowed and remains valid only while this
    /// context and its underlying CUDA context are alive.
    pub fn as_raw(&self) -> sys::cudnnHandle_t {
        self.handle.raw
    }

    /// Consumes the context and returns the raw cuDNN handle without
    /// destroying it.
    ///
    /// The caller becomes responsible for eventually destroying the returned
    /// handle with cuDNN.
    pub fn into_raw(self) -> sys::cudnnHandle_t {
        let context = ManuallyDrop::new(self);
        context.handle.raw
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        if let Err(err) = self.cuda_ctx.bind() {
            #[cfg(debug_assertions)]
            eprintln!("failed to bind cuda context before destroying cudnn handle: {err}");
        }

        unsafe {
            if let Err(err) = try_ffi!(sys::cudnnDestroy(self.raw)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cudnn context: {err}");
            }
        }
    }
}
