use std::{mem::ManuallyDrop, path::Path, ptr, sync::Arc};

use singe_core::path_to_cstring;
use singe_cuda::{context::Context as CudaContext, stream::Stream};

use crate::{
    error::{Error, Result},
    sys, try_ffi,
    types::{LoggerLevel, LoggerMask},
    utility::to_u32,
};

/// A stateful cuTENSOR handle.
///
/// Use one context per host thread or concurrent task. The handle is movable
/// between threads, but it is intentionally not `Clone` or `Sync`.
#[derive(Debug)]
pub struct Context {
    handle: Handle,
}

#[derive(Debug)]
struct Handle {
    raw: sys::cutensorHandle_t,
    cuda_ctx: Arc<CudaContext>,
}

#[derive(Debug, Clone)]
pub(crate) struct ContextRef {
    raw: sys::cutensorHandle_t,
    cuda_ctx: Arc<CudaContext>,
}

// cuTENSOR handles own mutable plan-cache state. The owning wrapper and
// lightweight references may move between threads, but are not shared as Sync.
unsafe impl Send for Handle {}
unsafe impl Send for ContextRef {}

impl Context {
    /// Initializes the cuTENSOR library and allocates the memory for the library context.
    ///
    /// The device associated with a particular cuTENSOR handle is assumed to remain unchanged after the [`Context::create`] call.
    /// To use a different device, make that device current through the CUDA wrapper crate and create a new cuTENSOR handle with [`Context::create`].
    ///
    /// Each handle has a plan cache that stores least-recently-used cuTENSOR plans.
    /// Its default capacity is 64, but it can be changed with [`Context::resize_plan_cache`].
    /// See the Plan Cache Guide for more information.
    ///
    /// The returned [`Context`] frees the cuTENSOR handle when dropped.
    ///
    /// This blocking call is thread-safe but not reentrant.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, if cuTENSOR cannot
    /// create a handle, or if cuTENSOR returns a null handle.
    pub fn create(cuda_ctx: &Arc<CudaContext>) -> Result<Self> {
        cuda_ctx.bind()?;

        let mut handle = ptr::null_mut();
        unsafe {
            try_ffi!(sys::cutensorCreate(&raw mut handle))?;
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

    /// Wraps an existing cuTENSOR handle and takes ownership of it.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid cuTENSOR handle associated with `cuda_ctx`.
    /// Ownership of `handle` is transferred to the returned context, and the
    /// handle must not be destroyed elsewhere after calling this function.
    pub unsafe fn from_raw(
        handle: sys::cutensorHandle_t,
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

    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        &self.handle.cuda_ctx
    }

    pub fn bind(&self) -> Result<()> {
        Ok(self.cuda_context().bind()?)
    }

    pub(crate) fn as_context_ref(&self) -> ContextRef {
        ContextRef {
            raw: self.handle.raw,
            cuda_ctx: Arc::clone(&self.handle.cuda_ctx),
        }
    }

    /// Resizes the plan cache.
    ///
    /// Changes the number of plans that can be stored in the plan cache of the handle.
    ///
    /// Resizing invalidates the cache.
    ///
    /// The call is not thread-safe, but the resulting cache can be shared across threads safely.
    ///
    /// This non-blocking call is neither reentrant nor thread-safe.
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound, `entry_count` cannot
    /// be represented for cuTENSOR, or cuTENSOR rejects the new cache size.
    pub fn resize_plan_cache(&self, entry_count: usize) -> Result<()> {
        self.bind()?;

        let entry_count = to_u32(entry_count, "entry_count")?;

        unsafe {
            try_ffi!(sys::cutensorHandleResizePlanCache(
                self.as_raw(),
                entry_count,
            ))?;
        }
        Ok(())
    }

    /// Writes the plan cache that belongs to this handle to file.
    ///
    /// This non-blocking call is thread-safe but not reentrant.
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound, `path` cannot be
    /// converted to a C string, no plan cache is attached, or cuTENSOR cannot
    /// write the file.
    pub fn write_plan_cache_to_file(&self, path: impl AsRef<Path>) -> Result<()> {
        self.bind()?;

        let path = path_to_cstring(path.as_ref())?;
        unsafe {
            try_ffi!(sys::cutensorHandleWritePlanCacheToFile(
                self.as_raw(),
                path.as_ptr(),
            ))?;
        }
        Ok(())
    }

    /// Reads a plan cache from file and overwrites this handle's cache lines.
    ///
    /// A cache is only valid for the same cuTENSOR version, CUDA version, and GPU
    /// architecture, including multiprocessor count.
    ///
    /// This non-blocking call is thread-safe but not reentrant.
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound, `path` cannot be
    /// converted to a C string, the file cannot be read, the stored cache is
    /// incompatible with this cuTENSOR/CUDA/device configuration, the current
    /// cache is too small for the stored data, or cuTENSOR rejects the cache.
    pub fn read_plan_cache_from_file(&self, path: impl AsRef<Path>) -> Result<u32> {
        self.bind()?;

        let path = path_to_cstring(path.as_ref())?;
        let mut cachelines_read = 0;
        unsafe {
            try_ffi!(sys::cutensorHandleReadPlanCacheFromFile(
                self.as_raw(),
                path.as_ptr(),
                &raw mut cachelines_read,
            ))?;
        }
        Ok(cachelines_read)
    }

    /// Writes the per-library kernel cache to file.
    ///
    /// Writes the just-in-time compiled kernels to the provided file.
    /// These kernels belong to the library, not to the handle.
    ///
    /// This non-blocking call is thread-safe but not reentrant.
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound, `path` cannot be
    /// converted to a C string, JIT kernel caching is unsupported for this
    /// operating system, CUDA Toolkit, or device, or cuTENSOR cannot write the
    /// file.
    pub fn write_kernel_cache_to_file(&self, path: impl AsRef<Path>) -> Result<()> {
        self.bind()?;

        let path = path_to_cstring(path.as_ref())?;
        unsafe {
            try_ffi!(sys::cutensorWriteKernelCacheToFile(
                self.as_raw(),
                path.as_ptr()
            ))?;
        }
        Ok(())
    }

    /// Reads a kernel cache from file and adds all non-existing JIT compiled kernels to the kernel cache.
    ///
    /// A cache is only valid for the same cuTENSOR version, CUDA version, and GPU
    /// architecture, including multiprocessor count.
    ///
    /// This non-blocking call is thread-safe but not reentrant.
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound, `path` cannot be
    /// converted to a C string, the file cannot be read, the stored cache is
    /// incompatible with this cuTENSOR/CUDA/device configuration, JIT kernel
    /// caching is unsupported for this operating system, CUDA Toolkit, or
    /// device, or cuTENSOR rejects the cache.
    pub fn read_kernel_cache_from_file(&self, path: impl AsRef<Path>) -> Result<()> {
        self.bind()?;

        let path = path_to_cstring(path.as_ref())?;
        unsafe {
            try_ffi!(sys::cutensorReadKernelCacheFromFile(
                self.as_raw(),
                path.as_ptr()
            ))?;
        }
        Ok(())
    }

    pub fn ensure_stream(&self, stream: &Stream) -> Result<()> {
        if self.cuda_context().as_ref() != stream.context() {
            return Err(Error::StreamContextMismatch);
        }

        self.bind()
    }

    /// Sets the logging callback.
    ///
    /// # Safety
    ///
    /// `callback` must remain valid for every later cuTENSOR log event that may
    /// call it and must follow cuTENSOR's callback requirements.
    ///
    /// # Errors
    ///
    /// Returns an error if cuTENSOR rejects the callback.
    pub unsafe fn set_logger_callback(callback: sys::cutensorLoggerCallback_t) -> Result<()> {
        unsafe {
            try_ffi!(sys::cutensorLoggerSetCallback(callback))?;
        }
        Ok(())
    }

    /// Sets the logging level.
    ///
    /// # Errors
    ///
    /// Returns an error if cuTENSOR rejects the logging level.
    pub fn set_logger_level(level: LoggerLevel) -> Result<()> {
        unsafe {
            try_ffi!(sys::cutensorLoggerSetLevel(level.as_raw()))?;
        }
        Ok(())
    }

    /// Sets the log mask.
    ///
    /// # Errors
    ///
    /// Returns an error if cuTENSOR rejects the logging mask.
    pub fn set_logger_mask(mask: LoggerMask) -> Result<()> {
        unsafe {
            try_ffi!(sys::cutensorLoggerSetMask(mask.bits()))?;
        }
        Ok(())
    }

    /// Sets the logging output file.
    ///
    /// # Safety
    ///
    /// `file` must be a valid C `FILE` pointer for every later cuTENSOR log write
    /// that may use it, or a null pointer if cuTENSOR accepts that for the active
    /// logger configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if cuTENSOR rejects the file pointer.
    pub unsafe fn set_logger_file(file: *mut sys::FILE) -> Result<()> {
        unsafe {
            try_ffi!(sys::cutensorLoggerSetFile(file))?;
        }
        Ok(())
    }

    /// Sets the logging output file by path.
    ///
    /// # Errors
    ///
    /// Returns an error if `path` cannot be converted to a C string or if
    /// cuTENSOR cannot open the log file.
    pub fn set_logger_path(path: impl AsRef<Path>) -> Result<()> {
        let path = path_to_cstring(path.as_ref())?;
        unsafe {
            try_ffi!(sys::cutensorLoggerOpenFile(path.as_ptr()))?;
        }
        Ok(())
    }

    /// Disables logging for the entire run.
    ///
    /// # Errors
    ///
    /// Returns an error if cuTENSOR cannot disable logging.
    pub fn disable_logger() -> Result<()> {
        unsafe {
            try_ffi!(sys::cutensorLoggerForceDisable())?;
        }
        Ok(())
    }

    /// Returns the raw cuTENSOR handle.
    ///
    /// The returned handle is borrowed and remains valid only while this
    /// context and its underlying CUDA context are alive.
    pub fn as_raw(&self) -> sys::cutensorHandle_t {
        self.handle.raw
    }

    /// Consumes the context and returns the raw cuTENSOR handle without
    /// destroying it.
    ///
    /// The caller becomes responsible for eventually destroying the returned
    /// handle with cuTENSOR.
    pub fn into_raw(self) -> sys::cutensorHandle_t {
        let context = ManuallyDrop::new(self);
        context.handle.raw
    }
}

impl ContextRef {
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        &self.cuda_ctx
    }

    pub fn bind(&self) -> Result<()> {
        Ok(self.cuda_ctx.bind()?)
    }

    pub fn ensure_stream(&self, stream: &Stream) -> Result<()> {
        if self.cuda_context().as_ref() != stream.context() {
            return Err(Error::StreamContextMismatch);
        }

        self.bind()
    }

    pub(crate) fn same_handle(&self, other: &Self) -> bool {
        self.raw == other.raw
    }

    /// Returns the raw cuTENSOR handle.
    ///
    /// The returned handle is borrowed and remains valid only while this
    /// context reference and its underlying CUDA context are alive.
    pub fn as_raw(&self) -> sys::cutensorHandle_t {
        self.raw
    }
}

pub(crate) fn validate_same_context(
    expected: &ContextRef,
    actual: &ContextRef,
    name: &str,
) -> Result<()> {
    if !expected.same_handle(actual) {
        return Err(Error::ContextMismatch { name: name.into() });
    }
    Ok(())
}

impl Drop for Handle {
    fn drop(&mut self) {
        if let Err(err) = self.cuda_ctx.bind() {
            #[cfg(debug_assertions)]
            eprintln!("failed to bind cuda context before destroying cutensor handle: {err}");
        }

        unsafe {
            if let Err(err) = try_ffi!(sys::cutensorDestroy(self.raw)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cutensor context: {err}");
            }
        }
    }
}
