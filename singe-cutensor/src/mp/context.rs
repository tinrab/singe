use std::{
    marker::PhantomData,
    mem::ManuallyDrop,
    ptr,
    sync::{Arc, Mutex, MutexGuard, PoisonError},
};

use singe_cuda::{context::Context as CudaContext, stream::Stream};
use singe_nccl::communicator::Communicator;

use crate::{
    error::{Error, Result},
    sys, try_ffi,
};

/// Shared multi-process cuTENSOR handle tied to a borrowed NCCL communicator lifetime.
///
/// The handle retains its actual CUDA context and optional stream until dependent resources are released.
#[derive(Debug, Clone)]
pub struct Context<'comm> {
    handle: Arc<Handle<'comm>>,
}

#[derive(Debug)]
pub(crate) struct Handle<'comm> {
    raw: sys::cutensorMpHandle_t,
    cuda_context: Arc<CudaContext>,
    local_device_id: i32,
    stream: Option<Stream>,
    execution_lock: Mutex<()>,
    _communicator: PhantomData<&'comm Communicator>,
}

#[derive(Debug, Clone)]
pub(crate) struct ContextRef<'comm> {
    handle: Arc<Handle<'comm>>,
}

// SAFETY: cuTENSORMp contexts are created for a fixed communicator, CUDA context, and
// stream binding. The Arc-backed handle can be shared immutably, while wrappers
// remain movable.
unsafe impl Send for Handle<'_> {}

unsafe impl Sync for Handle<'_> {}

unsafe impl Send for Context<'_> {}

unsafe impl Send for ContextRef<'_> {}

impl<'comm> Context<'comm> {
    /// Synchronizes this CUDA context and clears communication resources cached by the handle.
    ///
    /// Call before freeing or reallocating externally managed workspace regions registered by communication operations. The wrapper serializes this operation with execution through the shared context lock.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding or synchronization fails, or the native cache-clear call fails.
    pub fn clear_operation_cache(&self) -> Result<()> {
        self.as_context_ref().clear_operation_cache()
    }

    /// Creates a context and retains its stream until all plans and allocations are released.
    /// `None` selects the default CUDA stream.
    ///
    /// # Errors
    ///
    /// Returns an error if the stream and communicator CUDA contexts differ, communicator binding fails, native handle creation fails, or the returned handle is null.
    pub fn create(
        communicator: &'comm Communicator,
        local_device_id: i32,
        stream: Option<&Stream>,
    ) -> Result<Self> {
        if stream.is_some_and(|stream| communicator.cuda_context() != stream.context()) {
            return Err(Error::StreamContextMismatch);
        }
        communicator.bind()?;

        let mut handle = ptr::null_mut();
        // SAFETY: `cutensorMpCreate` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::cutensorMpCreate(
                &raw mut handle,
                communicator.as_raw(),
                local_device_id,
                stream.map_or(ptr::null_mut(), Stream::as_raw),
            ))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle: Arc::new(Handle {
                raw: handle,
                cuda_context: Arc::clone(communicator.cuda_context()),
                local_device_id,
                stream: stream.cloned(),
                execution_lock: Mutex::new(()),
                _communicator: PhantomData,
            }),
        })
    }

    /// Wraps an existing cuTENSORMp handle.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid cuTENSORMp handle associated with `cuda_context`,
    /// `local_device_id`, and `stream`. The returned context takes ownership
    /// of `handle` and destroys it when the last clone is dropped.
    ///
    /// # Errors
    ///
    /// Returns an error if `handle` is null.
    pub unsafe fn from_raw(
        handle: sys::cutensorMpHandle_t,
        cuda_context: Arc<CudaContext>,
        local_device_id: i32,
        stream: Option<Stream>,
    ) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle: Arc::new(Handle {
                raw: handle,
                cuda_context,
                local_device_id,
                stream,
                execution_lock: Mutex::new(()),
                _communicator: PhantomData,
            }),
        })
    }

    /// Returns the actual CUDA context retained by this multi-process handle.
    #[must_use]
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        &self.handle.cuda_context
    }

    /// Makes the retained CUDA context current.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot bind the retained context.
    pub fn bind(&self) -> Result<()> {
        Ok(self.cuda_context().bind()?)
    }

    /// Returns the local CUDA device ordinal recorded when the handle was created.
    #[must_use]
    pub fn local_device_id(&self) -> i32 {
        self.handle.local_device_id
    }

    /// Returns the retained handle stream, or `None` for its default CUDA stream.
    #[must_use]
    pub fn stream(&self) -> Option<&Stream> {
        self.handle.stream.as_ref()
    }

    pub(crate) fn as_context_ref(&self) -> ContextRef<'comm> {
        ContextRef {
            handle: Arc::clone(&self.handle),
        }
    }

    /// Returns the borrowed native multi-process library handle.
    #[must_use]
    pub fn as_raw(&self) -> sys::cutensorMpHandle_t {
        self.handle.raw
    }

    /// Consumes this context and returns the owned raw cuTENSORMp handle.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OwnershipShared`] if cloned contexts or context references
    /// still point at the same handle.
    pub fn into_raw(self) -> Result<sys::cutensorMpHandle_t> {
        let handle = Arc::try_unwrap(self.handle).map_err(|_| Error::OwnershipShared)?;
        let handle = ManuallyDrop::new(handle);
        Ok(handle.raw)
    }
}

impl ContextRef<'_> {
    pub(crate) fn lock_execution(&self) -> MutexGuard<'_, ()> {
        self.handle
            .execution_lock
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    pub(crate) fn clear_operation_cache(&self) -> Result<()> {
        let _lock = self.lock_execution();
        self.bind()?;
        self.cuda_context().synchronize()?;
        // SAFETY: `cutensorMpClearOperationCache` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::cutensorMpClearOperationCache(self.as_raw()))?;
        }
        Ok(())
    }

    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        &self.handle.cuda_context
    }

    pub fn bind(&self) -> Result<()> {
        Ok(self.cuda_context().bind()?)
    }

    pub(crate) fn same_handle(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.handle, &other.handle)
    }

    pub fn as_raw(&self) -> sys::cutensorMpHandle_t {
        self.handle.raw
    }
}

impl Drop for Handle<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.cuda_context.bind() {
            #[cfg(debug_assertions)]
            eprintln!("failed to bind cuda context before destroying cutensormp handle: {error}");
        }

        // SAFETY: `cutensorMpDestroy` receives live handles owned by the wrappers.
        unsafe {
            if let Err(error) = try_ffi!(sys::cutensorMpDestroy(self.raw)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cutensormp context: {error}");
            }
        }
    }
}

pub(crate) fn validate_same_context(
    expected: &ContextRef<'_>,
    actual: &ContextRef<'_>,
    name: &str,
) -> Result<()> {
    if !expected.same_handle(actual) {
        return Err(Error::MpContextMismatch {
            name: name.to_owned().into(),
        });
    }
    Ok(())
}
