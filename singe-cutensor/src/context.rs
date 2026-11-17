use std::{
    cell::Cell, convert::Infallible, marker::PhantomData, path::Path, result::Result as StdResult,
    sync::Arc,
};

use singe_core::path_to_cstring;
use singe_cuda::{
    context::Context as CudaContext,
    handle::{CreateHandle, Handle, NativeHandle, RawHandle},
    stream::Stream,
};

use crate::{
    error::{Error, Result},
    sys, try_ffi,
    utility::to_u32,
};

/// Marker type for the native cuTENSOR handle owned by [`Context`].
#[derive(Debug)]
pub struct CutensorHandle(Infallible);

// SAFETY: cuTENSOR handles have no thread affinity.
unsafe impl NativeHandle for CutensorHandle {
    type Raw = sys::cutensorHandle_t;

    type State = ();

    type Error = Error;

    const OBJECT: &'static str = "cutensor context";

    fn null_handle() -> Error {
        Error::NullHandle
    }

    fn is_null(raw: Self::Raw) -> bool {
        raw.is_null()
    }

    unsafe fn destroy(raw: Self::Raw, (): ()) -> Result<()> {
        // SAFETY: the caller passes a live, uniquely owned handle with its context bound.
        unsafe { try_ffi!(sys::cutensorDestroy(raw)) }
    }
}

// SAFETY: `cutensorCreate` creates the handle for the device of the bound context.
unsafe impl CreateHandle for CutensorHandle {
    unsafe fn create(raw: *mut Self::Raw) -> Result<()> {
        // SAFETY: the caller provides a writable output location and binds the context.
        unsafe { try_ffi!(sys::cutensorCreate(raw)) }
    }
}

/// Raw ownership of a cuTENSOR handle and its CUDA context.
pub type RawContext = RawHandle<CutensorHandle>;

/// A stateful cuTENSOR handle.
///
/// Use one context per host thread or concurrent task.
/// The handle is movable between threads, but it is intentionally not `Clone` or `Sync`.
#[derive(Debug)]
pub struct Context {
    handle: Arc<SharedHandle>,
    _not_sync: PhantomData<Cell<()>>,
}

#[derive(Debug)]
struct SharedHandle(Handle<CutensorHandle>);

// SAFETY: descriptors and plans on several threads may share one handle, since cuTENSOR documents its descriptor, plan, and execution calls as thread-safe.
// Resizing the plan cache, the one call that is not, requires unique ownership in `Context::resize_plan_cache`.
unsafe impl Sync for SharedHandle {}

/// A shared reference to a cuTENSOR handle that keeps the handle alive.
///
/// Descriptors and plans hold one, and their raw records carry it.
#[derive(Debug, Clone)]
pub struct ContextRef {
    handle: Arc<SharedHandle>,
    _not_sync: PhantomData<Cell<()>>,
}

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
    /// Returns an error if the CUDA context cannot be bound, if cuTENSOR cannot create a handle, or if cuTENSOR returns a null handle.
    pub fn create(cuda_context: &Arc<CudaContext>) -> Result<Self> {
        Ok(Self {
            handle: Arc::new(SharedHandle(Handle::create(cuda_context)?)),
            _not_sync: PhantomData,
        })
    }

    /// Returns the actual CUDA context retained by this library handle.
    #[must_use]
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        self.handle.0.cuda_context()
    }

    pub(crate) fn bind(&self) -> Result<()> {
        self.handle.0.bind()
    }

    /// Returns a shared reference to this handle, for building raw descriptor and plan records.
    #[must_use]
    pub fn as_context_ref(&self) -> ContextRef {
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
    /// Returns an error if the context cannot be bound, `entry_count` cannot be represented for cuTENSOR, or cuTENSOR rejects the new cache size.
    pub fn resize_plan_cache(&mut self, entry_count: usize) -> Result<()> {
        if Arc::get_mut(&mut self.handle).is_none() {
            return Err(Error::OwnershipShared {
                name: "cutensor context".into(),
            });
        }

        self.bind()?;

        let entry_count = to_u32(entry_count, "entry_count")?;

        // SAFETY: the handle is live, and the `Arc::get_mut` check above gives this call the exclusive use that resizing requires.
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
    /// Returns an error if the context cannot be bound, `path` cannot be converted to a C string, no plan cache is attached, or cuTENSOR cannot write the file.
    pub fn write_plan_cache_to_file(&self, path: impl AsRef<Path>) -> Result<()> {
        self.bind()?;

        let path = path_to_cstring(path.as_ref())?;
        // SAFETY: `self.handle` keeps the handle live, and `path` is a nul-terminated string that outlives the call.
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
    /// A cache is only valid for the same cuTENSOR version, CUDA version, and GPU architecture, including multiprocessor count.
    ///
    /// This non-blocking call is thread-safe but not reentrant.
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound, `path` cannot be converted to a C string, the file cannot be read, the stored cache is incompatible with this cuTENSOR/CUDA/device configuration, the current cache is too small for the stored data, or cuTENSOR rejects the cache.
    pub fn read_plan_cache_from_file(&self, path: impl AsRef<Path>) -> Result<u32> {
        self.bind()?;

        let path = path_to_cstring(path.as_ref())?;
        let mut cachelines_read = 0;
        // SAFETY: `self.handle` keeps the handle live, `path` is a nul-terminated string that outlives the call, and `cachelines_read` is a local output.
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
    /// Returns an error if the context cannot be bound, `path` cannot be converted to a C string, JIT kernel caching is unsupported for this operating system, CUDA Toolkit, or device, or cuTENSOR cannot write the file.
    pub fn write_kernel_cache_to_file(&self, path: impl AsRef<Path>) -> Result<()> {
        self.bind()?;

        let path = path_to_cstring(path.as_ref())?;
        // SAFETY: `self.handle` keeps the handle live, and `path` is a nul-terminated string that outlives the call.
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
    /// A cache is only valid for the same cuTENSOR version, CUDA version, and GPU architecture, including multiprocessor count.
    ///
    /// This non-blocking call is thread-safe but not reentrant.
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound, `path` cannot be converted to a C string, the file cannot be read, the stored cache is incompatible with this cuTENSOR/CUDA/device configuration, JIT kernel caching is unsupported for this operating system, CUDA Toolkit, or device, or cuTENSOR rejects the cache.
    pub fn read_kernel_cache_from_file(&self, path: impl AsRef<Path>) -> Result<()> {
        self.bind()?;

        let path = path_to_cstring(path.as_ref())?;
        // SAFETY: `self.handle` keeps the handle live, and `path` is a nul-terminated string that outlives the call.
        unsafe {
            try_ffi!(sys::cutensorReadKernelCacheFromFile(
                self.as_raw(),
                path.as_ptr()
            ))?;
        }
        Ok(())
    }

    /// Returns the raw cuTENSOR handle.
    ///
    /// The returned handle is borrowed and remains valid only while this context and its underlying CUDA context are alive.
    #[must_use]
    pub fn as_raw(&self) -> sys::cutensorHandle_t {
        self.handle.0.as_raw()
    }

    /// Adopts a native cuTENSOR handle and its CUDA context.
    ///
    /// # Safety
    ///
    /// `raw.handle` must be a uniquely owned cuTENSOR handle created in `raw.cuda_context`, and `raw.stream` must be `None`.
    ///
    /// # Errors
    ///
    /// Returns an error if `raw.handle` is null or `raw.stream` belongs to another CUDA context.
    pub unsafe fn from_raw(raw: RawContext) -> Result<Self> {
        Ok(Self {
            // SAFETY: the caller guarantees a uniquely owned handle created in `raw.cuda_context`.
            handle: Arc::new(SharedHandle(unsafe { Handle::from_raw(raw)? })),
            _not_sync: PhantomData,
        })
    }

    /// Transfers the native handle and its CUDA context if no descriptors or plans retain the handle.
    ///
    /// # Errors
    ///
    /// Returns the original context as `Err` if descriptors, operations, plans, or other dependent resources still retain the library handle.
    pub fn into_raw(self) -> StdResult<RawContext, Self> {
        match Arc::try_unwrap(self.handle) {
            Ok(shared) => Ok(shared.0.into_raw()),
            Err(handle) => Err(Self {
                handle,
                _not_sync: PhantomData,
            }),
        }
    }
}

impl ContextRef {
    /// Returns the actual CUDA context retained by the library handle.
    #[must_use]
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        self.handle.0.cuda_context()
    }

    pub(crate) fn bind(&self) -> Result<()> {
        self.handle.0.bind()
    }

    pub(crate) fn ensure_stream(&self, stream: &Stream) -> Result<()> {
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
    /// The returned handle is borrowed and remains valid only while this context reference and its underlying CUDA context are alive.
    #[must_use]
    pub fn as_raw(&self) -> sys::cutensorHandle_t {
        self.handle.0.as_raw()
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
