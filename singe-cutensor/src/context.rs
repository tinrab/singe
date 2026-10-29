use std::{
    cell::Cell, marker::PhantomData, mem, path::Path, ptr, result::Result as StdResult, sync::Arc,
};

use singe_core::path_to_cstring;
use singe_cuda::{context::Context as CudaContext, stream::Stream};

use crate::{
    error::{Error, Result},
    sys, try_ffi,
    utility::to_u32,
};

/// A stateful cuTENSOR handle.
///
/// Use one context per host thread or concurrent task. The handle is movable
/// between threads, but it is intentionally not `Clone` or `Sync`.
#[derive(Debug)]
pub struct Context {
    handle: Arc<Handle>,
    _not_sync: PhantomData<Cell<()>>,
}

#[derive(Debug)]
struct Handle {
    raw: sys::cutensorHandle_t,
    cuda_context: Arc<CudaContext>,
}

#[derive(Debug, Clone)]
pub(crate) struct ContextRef {
    handle: Arc<Handle>,
    _not_sync: PhantomData<Cell<()>>,
}

// SAFETY: cuTENSOR handles own mutable plan-cache state. The owning wrapper and
// lightweight references may move between threads, but are not shared as Sync.
unsafe impl Send for Handle {}

// The private ownership record is immutable and native shared operations are thread-safe.
// Cache resizing requires unique ownership; public wrappers remain !Sync through their markers.
unsafe impl Sync for Handle {}

/// A uniquely transferred cuTENSOR handle and the CUDA context needed to destroy it.
/// Destroy `handle` before dropping this value, or reclaim it with `Context::from_raw`.
#[derive(Debug)]
pub struct RawContext {
    /// Transferred native cuTENSOR handle; destroy it before releasing its CUDA context.
    pub handle: sys::cutensorHandle_t,
    /// Actual CUDA context required for using and destroying the transferred handle.
    pub cuda_context: Arc<CudaContext>,
}

// SAFETY: transfer carries unique native ownership and its actual CUDA context.
unsafe impl Send for RawContext {}

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
    pub fn create(cuda_context: &Arc<CudaContext>) -> Result<Self> {
        cuda_context.bind()?;

        let mut handle = ptr::null_mut();
        // SAFETY: `cutensorCreate` receives valid output locations.
        unsafe {
            try_ffi!(sys::cutensorCreate(&raw mut handle))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle: Arc::new(Handle {
                raw: handle,
                cuda_context: Arc::clone(cuda_context),
            }),
            _not_sync: PhantomData,
        })
    }

    /// Wraps an existing cuTENSOR handle and takes ownership of it.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid cuTENSOR handle associated with `cuda_context`.
    /// Ownership of `handle` is transferred to the returned context, and the
    /// handle must not be destroyed elsewhere after calling this function.
    ///
    /// # Errors
    ///
    /// Returns an error if the native handle is null.
    pub unsafe fn from_raw_parts(
        handle: sys::cutensorHandle_t,
        cuda_context: Arc<CudaContext>,
    ) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle: Arc::new(Handle {
                raw: handle,
                cuda_context,
            }),
            _not_sync: PhantomData,
        })
    }

    /// Returns the actual CUDA context retained by this library handle.
    #[must_use]
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        &self.handle.cuda_context
    }

    /// Makes this handle's retained CUDA context current.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot bind the context.
    pub fn bind(&self) -> Result<()> {
        Ok(self.cuda_context().bind()?)
    }

    pub(crate) fn as_context_ref(&self) -> ContextRef {
        ContextRef {
            handle: Arc::clone(&self.handle),
            _not_sync: PhantomData,
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
    pub fn resize_plan_cache(&mut self, entry_count: usize) -> Result<()> {
        if Arc::get_mut(&mut self.handle).is_none() {
            return Err(Error::OwnershipShared);
        }
        self.bind()?;

        let entry_count = to_u32(entry_count, "entry_count")?;

        // SAFETY: `cutensorHandleResizePlanCache` receives live handles owned by the wrappers.
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
        // SAFETY: `cutensorHandleWritePlanCacheToFile` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
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
        // SAFETY: `cutensorHandleReadPlanCacheFromFile` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
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
        // SAFETY: `cutensorWriteKernelCacheToFile` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
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
        // SAFETY: `cutensorReadKernelCacheFromFile` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::cutensorReadKernelCacheFromFile(
                self.as_raw(),
                path.as_ptr()
            ))?;
        }
        Ok(())
    }

    /// Checks that `stream` belongs to this handle's actual CUDA context and binds that context.
    ///
    /// # Errors
    ///
    /// Returns an error if the stream context differs or CUDA cannot bind the context.
    pub fn ensure_stream(&self, stream: &Stream) -> Result<()> {
        if self.cuda_context() != stream.context() {
            return Err(Error::StreamContextMismatch);
        }

        self.bind()
    }

    /// Returns the raw cuTENSOR handle.
    ///
    /// The returned handle is borrowed and remains valid only while this
    /// context and its underlying CUDA context are alive.
    #[must_use]
    pub fn as_raw(&self) -> sys::cutensorHandle_t {
        self.handle.raw
    }

    /// Reclaims a uniquely owned native handle and its actual CUDA context.
    ///
    /// # Safety
    ///
    /// `handle` must remain valid and uniquely owned in the retained CUDA context.
    ///
    /// # Errors
    ///
    /// Returns an error if the native handle is null.
    pub unsafe fn from_raw(raw: RawContext) -> Result<Self> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { Self::from_raw_parts(raw.handle, raw.cuda_context) }
    }

    /// Transfers the native handle and context if no dependent resources retain the handle.
    /// Returns the original context when descriptors or plans still share ownership.
    ///
    /// # Errors
    ///
    /// Returns the original context as `Err` if descriptors, operations, plans, or other dependent resources still retain the library handle.
    pub fn into_raw(self) -> StdResult<RawContext, Self> {
        match Arc::try_unwrap(self.handle) {
            Ok(mut owner) => {
                let handle = mem::replace(&mut owner.raw, ptr::null_mut());
                Ok(RawContext {
                    handle,
                    cuda_context: Arc::clone(&owner.cuda_context),
                })
            }
            Err(handle) => Err(Self {
                handle,
                _not_sync: PhantomData,
            }),
        }
    }
}

impl ContextRef {
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        &self.handle.cuda_context
    }

    pub fn bind(&self) -> Result<()> {
        Ok(self.handle.cuda_context.bind()?)
    }

    pub fn ensure_stream(&self, stream: &Stream) -> Result<()> {
        if self.cuda_context() != stream.context() {
            return Err(Error::StreamContextMismatch);
        }

        self.bind()
    }

    pub(crate) fn same_handle(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.handle, &other.handle)
    }

    /// Returns the raw cuTENSOR handle.
    ///
    /// The returned handle is borrowed and remains valid only while this
    /// context reference and its underlying CUDA context are alive.
    pub fn as_raw(&self) -> sys::cutensorHandle_t {
        self.handle.raw
    }
}

pub(crate) fn validate_same_context(
    expected: &ContextRef,
    actual: &ContextRef,
    name: &str,
) -> Result<()> {
    if !expected.same_handle(actual) {
        return Err(Error::ContextMismatch {
            name: name.to_owned().into(),
        });
    }
    Ok(())
}

impl Drop for Handle {
    fn drop(&mut self) {
        if self.raw.is_null() {
            return;
        }
        if let Err(error) = self.cuda_context.bind() {
            #[cfg(debug_assertions)]
            eprintln!("failed to bind cuda context before destroying cutensor handle: {error}");
        }

        // SAFETY: `cutensorDestroy` receives live handles owned by the wrappers.
        unsafe {
            if let Err(error) = try_ffi!(sys::cutensorDestroy(self.raw)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cutensor context: {error}");
            }
        }
    }
}
