use std::{mem::ManuallyDrop, path::Path, ptr, sync::Arc};

use singe_core::path_to_cstring;
use singe_cuda::{
    context::Context as CudaContext,
    stream::{BorrowedStream, Stream, StreamBinding},
};

use crate::{
    error::{Error, Result},
    scalar::Scalar,
    sys, try_ffi,
    types::PointerMode,
};

/// A stateful cuSPARSE handle.
///
/// Use one context per host thread or concurrent task. The handle is movable
/// between threads, but it is intentionally not `Clone` or `Sync`.
#[derive(Debug)]
pub struct Context {
    handle: Handle,
}

#[derive(Debug)]
struct Handle {
    raw: sys::cusparseHandle_t,
    cuda_ctx: Arc<CudaContext>,
}

// cuSPARSE handles carry mutable pointer-mode and stream state. Ownership can
// move between threads, but the wrapper requires exclusive access for mutation.
unsafe impl Send for Handle {}

impl Context {
    /// Initializes the cuSPARSE library and creates a cuSPARSE handle.
    /// The handle must be created before using other cuSPARSE operations through this wrapper.
    /// It allocates hardware resources needed to access the GPU.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, if cuSPARSE cannot
    /// create a handle, or if cuSPARSE returns a null handle.
    pub fn create(cuda_ctx: &Arc<CudaContext>) -> Result<Self> {
        cuda_ctx.bind()?;

        let mut handle = ptr::null_mut();
        unsafe {
            try_ffi!(sys::cusparseCreate(&raw mut handle))?;
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

    /// Wraps an existing cuSPARSE handle and takes ownership of it.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid cuSPARSE handle associated with `cuda_ctx`.
    /// Ownership of `handle` is transferred to the returned context, and the
    /// handle must not be destroyed elsewhere after calling this function.
    pub unsafe fn from_raw(
        handle: sys::cusparseHandle_t,
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

    /// Returns the underlying CUDA context used by this cuSPARSE handle.
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

    /// Returns the version number of the cuSPARSE library.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or if cuSPARSE
    /// cannot report the version for this handle.
    pub fn version(&self) -> Result<i32> {
        self.bind()?;

        let mut version = 0;
        unsafe {
            try_ffi!(sys::cusparseGetVersion(self.as_raw(), &raw mut version))?;
        }
        Ok(version)
    }

    /// Returns the stream used for cuSPARSE operations on this handle.
    /// If no explicit stream has been set, cuSPARSE uses the CUDA default stream.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or if cuSPARSE
    /// cannot report the current stream.
    pub fn stream(&self) -> Result<StreamBinding> {
        self.bind()?;

        let mut stream = ptr::null_mut();
        unsafe {
            try_ffi!(sys::cusparseGetStream(self.as_raw(), &raw mut stream))?;
        }

        Ok(if stream.is_null() {
            StreamBinding::Default(Arc::clone(self.cuda_context()))
        } else {
            StreamBinding::Borrowed(BorrowedStream::from_raw(
                stream,
                Arc::clone(self.cuda_context()),
            ))
        })
    }

    /// Sets the stream used by cuSPARSE operations on this handle.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to another CUDA context, if the CUDA
    /// context cannot be bound, or if cuSPARSE rejects the stream.
    pub fn set_stream(&self, stream: Option<&Stream>) -> Result<()> {
        if let Some(stream) = stream {
            self.ensure_stream(stream)?;
        } else {
            self.bind()?;
        }

        unsafe {
            try_ffi!(sys::cusparseSetStream(
                self.as_raw(),
                match stream {
                    Some(stream) => stream.as_raw(),
                    None => ptr::null_mut(),
                },
            ))?;
        }
        Ok(())
    }

    /// Returns the context-global scalar pointer mode used by cuSPARSE operations on this handle.
    /// See [`PointerMode`] for scalar pointer semantics.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or if cuSPARSE
    /// cannot report the pointer mode.
    pub fn scalar_pointer_mode(&self) -> Result<PointerMode> {
        self.bind()?;

        let mut mode = sys::cusparsePointerMode_t::CUSPARSE_POINTER_MODE_HOST;
        unsafe {
            try_ffi!(sys::cusparseGetPointerMode(self.as_raw(), &raw mut mode))?;
        }
        Ok(mode.into())
    }

    /// Sets the context-global scalar pointer mode used by cuSPARSE operations on this handle.
    /// The default mode reads scalar values from host memory.
    /// See [`PointerMode`] for scalar pointer semantics.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or if cuSPARSE
    /// rejects the pointer mode.
    pub fn set_scalar_pointer_mode(&self, mode: PointerMode) -> Result<()> {
        self.bind()?;
        unsafe {
            try_ffi!(sys::cusparseSetPointerMode(self.as_raw(), mode.into()))?;
        }
        Ok(())
    }

    pub(crate) fn require_scalar_pointer_mode<T>(
        &self,
        alpha: Scalar<'_, T>,
        beta: Scalar<'_, T>,
    ) -> Result<()> {
        let alpha_mode = alpha.pointer_mode();
        let beta_mode = beta.pointer_mode();
        if alpha_mode != beta_mode {
            return Err(Error::ScalarPointerModeMismatch);
        }
        if self.scalar_pointer_mode()? != alpha_mode {
            self.set_scalar_pointer_mode(alpha_mode)?;
        }
        Ok(())
    }

    /// Experimental: sets the logging callback function.
    ///
    /// The callback must use the cuSPARSE logger callback ABI.
    ///
    /// # Safety
    ///
    /// `callback`, if present, must remain valid for use by cuSPARSE and must
    /// follow the callback ABI expected by the library.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSPARSE rejects the callback.
    pub unsafe fn set_logger_callback(callback: sys::cusparseLoggerCallback_t) -> Result<()> {
        unsafe {
            try_ffi!(sys::cusparseLoggerSetCallback(callback))?;
        }
        Ok(())
    }

    /// Experimental: sets the logging level.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSPARSE rejects the logging level.
    pub fn set_logger_level(level: i32) -> Result<()> {
        unsafe {
            try_ffi!(sys::cusparseLoggerSetLevel(level))?;
        }
        Ok(())
    }

    /// Experimental: sets the logging mask.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSPARSE rejects the logging mask.
    pub fn set_logger_mask(mask: i32) -> Result<()> {
        unsafe {
            try_ffi!(sys::cusparseLoggerSetMask(mask))?;
        }
        Ok(())
    }

    /// Experimental: sets the logging output file.
    /// Once registered, the provided file handle must remain open until another
    /// file handle is registered.
    ///
    /// # Safety
    ///
    /// `file` must be a valid `FILE` handle for as long as cuSPARSE may write to it.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSPARSE rejects the file handle.
    pub unsafe fn set_logger_file(file: *mut sys::FILE) -> Result<()> {
        unsafe {
            try_ffi!(sys::cusparseLoggerSetFile(file))?;
        }
        Ok(())
    }

    /// Experimental: sets the logging output file by path.
    ///
    /// # Errors
    ///
    /// Returns an error if `path` cannot be converted to a C string or if
    /// cuSPARSE cannot open the log file.
    pub fn set_logger_path(path: impl AsRef<Path>) -> Result<()> {
        let path = path_to_cstring(path.as_ref())?;
        unsafe {
            try_ffi!(sys::cusparseLoggerOpenFile(path.as_ptr()))?;
        }
        Ok(())
    }

    /// Disables cuSPARSE logging.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSPARSE cannot disable logging.
    pub fn disable_logger() -> Result<()> {
        unsafe {
            try_ffi!(sys::cusparseLoggerForceDisable())?;
        }
        Ok(())
    }

    /// Returns the raw cuSPARSE handle.
    ///
    /// The returned handle is borrowed and remains valid only while this
    /// context and its underlying CUDA context are alive.
    pub fn as_raw(&self) -> sys::cusparseHandle_t {
        self.handle.raw
    }

    /// Consumes the context and returns the raw cuSPARSE handle without
    /// destroying it.
    ///
    /// The caller becomes responsible for eventually destroying the returned
    /// handle with cuSPARSE.
    pub fn into_raw(self) -> sys::cusparseHandle_t {
        let context = ManuallyDrop::new(self);
        context.handle.raw
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        if let Err(err) = self.cuda_ctx.bind() {
            #[cfg(debug_assertions)]
            eprintln!("failed to bind cuda context before destroying cusparse handle: {err}");
        }

        unsafe {
            if let Err(err) = try_ffi!(sys::cusparseDestroy(self.raw)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cusparse context: {err}");
            }
        }
    }
}
