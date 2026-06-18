#[allow(unused_imports)]
use crate::{
    dense::legacy::qr::xgeqrf,
    irs::xgesv,
    svd::{Gesvd, Gesvdp, Gesvdr},
};

use std::{mem::ManuallyDrop, path::Path, ptr, sync::Arc};

use singe_core::path_to_cstring;
use singe_cuda::{context::Context as CudaContext, stream::Stream, types::EmulationStrategy};
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
    cuda_ctx: Arc<CudaContext>,
}

// cuSOLVER handles are stateful and stream-bound. The owner may move between
// threads, but callers need exclusive access to mutate handle state.
unsafe impl Send for Handle {}

/// The stream bound to a cuSOLVER handle.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum StreamBinding {
    /// The CUDA default stream.
    Default,
    /// A borrowed stream associated with the same CUDA context.
    Borrowed(BorrowedStream),
}

/// A stream borrowed from a CUDA context, associated with a cuSOLVER handle.
#[derive(Debug, Clone)]
pub struct BorrowedStream {
    handle: runtime::cudaStream_t,
    cuda_ctx: Arc<CudaContext>,
}

impl BorrowedStream {
    /// Returns the raw CUDA stream handle.
    pub const fn as_raw(&self) -> runtime::cudaStream_t {
        self.handle
    }

    /// Returns a reference to the CUDA context this stream belongs to.
    pub fn context(&self) -> &CudaContext {
        self.cuda_ctx.as_ref()
    }
}

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
    pub fn create(cuda_ctx: &Arc<CudaContext>) -> Result<Self> {
        cuda_ctx.bind()?;

        let mut handle = ptr::null_mut();
        unsafe {
            try_ffi!(sys::cusolverDnCreate(&raw mut handle))?;
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

    /// Wraps an existing cuSOLVER dense handle and takes ownership of it.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid cuSOLVER dense handle associated with
    /// `cuda_ctx`. Ownership of `handle` is transferred to the returned
    /// context, and the handle must not be destroyed elsewhere after calling
    /// this function.
    pub unsafe fn from_raw(
        handle: sys::cusolverDnHandle_t,
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

    /// Returns the underlying CUDA context used by this cuSOLVER handle.
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

    /// Returns the stream currently used by this cuSOLVER handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or if cuSOLVER
    /// cannot report the current stream.
    pub fn stream(&self) -> Result<StreamBinding> {
        self.bind()?;

        let mut stream = ptr::null_mut();
        unsafe {
            try_ffi!(sys::cusolverDnGetStream(self.as_raw(), &raw mut stream))?;
        }

        Ok(if stream.is_null() {
            StreamBinding::Default
        } else {
            StreamBinding::Borrowed(BorrowedStream {
                handle: stream,
                cuda_ctx: Arc::clone(self.cuda_context()),
            })
        })
    }

    /// Sets the stream used by this cuSOLVER handle.
    ///
    /// Passing `None` restores the CUDA default stream.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to another CUDA context, if the CUDA
    /// context cannot be bound, or if cuSOLVER rejects the stream.
    pub fn set_stream(&self, stream: Option<&Stream>) -> Result<()> {
        if let Some(stream) = stream {
            self.ensure_stream(stream)?;
        } else {
            self.bind()?;
        }

        unsafe {
            try_ffi!(sys::cusolverDnSetStream(
                self.as_raw(),
                match stream {
                    Some(stream) => stream.as_raw(),
                    None => ptr::null_mut(),
                },
            ))?;
        }
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
    /// operations, including [`xgeqrf`], [`Gesvd`], [`Gesvdr`], and [`Gesvdp`].
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or if cuSOLVER
    /// rejects the deterministic mode.
    pub fn set_deterministic_mode(&self, mode: DeterministicMode) -> Result<()> {
        self.bind()?;
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
        unsafe {
            try_ffi!(sys::cusolverDnSetEmulationStrategy(
                self.as_raw(),
                strategy.into(),
            ))?;
        }
        Ok(())
    }

    /// Installs the cuSOLVER logger callback.
    ///
    /// # Safety
    ///
    /// `callback`, if present, must remain valid for use by cuSOLVER and must
    /// follow the callback ABI expected by the library.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSOLVER rejects the callback.
    pub unsafe fn set_logger_callback(callback: sys::cusolverDnLoggerCallback_t) -> Result<()> {
        unsafe {
            try_ffi!(sys::cusolverDnLoggerSetCallback(callback))?;
        }
        Ok(())
    }

    /// Sets the cuSOLVER logger verbosity level.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSOLVER rejects the logging level.
    pub fn set_logger_level(level: i32) -> Result<()> {
        unsafe {
            try_ffi!(sys::cusolverDnLoggerSetLevel(level))?;
        }
        Ok(())
    }

    /// Sets the cuSOLVER logger mask.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSOLVER rejects the logging mask.
    pub fn set_logger_mask(mask: i32) -> Result<()> {
        unsafe {
            try_ffi!(sys::cusolverDnLoggerSetMask(mask))?;
        }
        Ok(())
    }

    /// Sets the FILE handle used for cuSOLVER logging.
    ///
    /// Once registered, the file handle must remain open until another handle is
    /// installed or logging is disabled.
    ///
    /// # Safety
    ///
    /// `file` must be a valid `FILE` handle for as long as cuSOLVER may write to it.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSOLVER rejects the file handle.
    pub unsafe fn set_logger_file(file: *mut sys::FILE) -> Result<()> {
        unsafe {
            try_ffi!(sys::cusolverDnLoggerSetFile(file))?;
        }
        Ok(())
    }

    /// Sets the cuSOLVER logging output file by path.
    ///
    /// # Errors
    ///
    /// Returns an error if `path` cannot be converted to a C string or if
    /// cuSOLVER cannot open the log file.
    pub fn set_logger_path(path: impl AsRef<Path>) -> Result<()> {
        let path = path_to_cstring(path.as_ref())?;
        unsafe {
            try_ffi!(sys::cusolverDnLoggerOpenFile(path.as_ptr()))?;
        }
        Ok(())
    }

    /// Disables cuSOLVER logging for the current process.
    ///
    /// # Errors
    ///
    /// Returns an error if cuSOLVER cannot disable logging.
    pub fn disable_logger() -> Result<()> {
        unsafe {
            try_ffi!(sys::cusolverDnLoggerForceDisable())?;
        }
        Ok(())
    }

    /// Returns the raw cuSOLVER dense handle.
    ///
    /// The returned handle is borrowed and remains valid only while this
    /// context and its underlying CUDA context are alive.
    pub fn as_raw(&self) -> sys::cusolverDnHandle_t {
        self.handle.raw
    }

    /// Consumes the context and returns the raw cuSOLVER dense handle without
    /// destroying it.
    ///
    /// The caller becomes responsible for eventually destroying the returned
    /// handle with cuSOLVER.
    pub fn into_raw(self) -> sys::cusolverDnHandle_t {
        let context = ManuallyDrop::new(self);
        context.handle.raw
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        if let Err(err) = self.cuda_ctx.bind() {
            #[cfg(debug_assertions)]
            eprintln!("failed to bind cuda context before destroying cusolver handle: {err}");
        }

        unsafe {
            if let Err(err) = try_ffi!(sys::cusolverDnDestroy(self.raw)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cusolver context: {err}");
            }
        }
    }
}
