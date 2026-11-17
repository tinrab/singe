use std::{
    marker::PhantomData,
    mem, ptr,
    rc::Rc,
    result::Result as StdResult,
    sync::{Arc, Mutex, MutexGuard},
};

use singe_core::report_cleanup_failure;
use singe_cuda::{context::Context as CudaContext, device::Device, stream::Stream};
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
    handle: Rc<Handle<'comm>>,
}

#[derive(Debug)]
pub(crate) struct Handle<'comm> {
    raw: NativeHandle,
    cuda_context: Arc<CudaContext>,
    local_device: Device,
    stream: Option<Stream>,
    execution_lock: Mutex<()>,
    _communicator: PhantomData<&'comm Communicator>,
}

#[derive(Debug, Clone)]
pub(crate) struct ContextRef<'comm> {
    handle: Rc<Handle<'comm>>,
}

// Holds the native pointer so that `Handle`'s auto traits come from its other fields, including the borrowed communicator.
#[derive(Debug)]
struct NativeHandle(sys::cutensorMpHandle_t);

// SAFETY: the pointer has no thread affinity, and `Handle` serializes executions and cache clears through `execution_lock`.
// Whether a `Handle` crosses threads is left to its `PhantomData<&'comm Communicator>`, since NCCL forbids using one communicator from two threads at once.
unsafe impl Send for NativeHandle {}

// SAFETY: as for `Send`; shared access only reads the pointer.
unsafe impl Sync for NativeHandle {}

/// A uniquely transferred cuTENSORMp handle and the owners needed to use and destroy it.
///
/// Destroy `handle` before dropping this value, or reclaim it with [`Context::from_raw`].
#[derive(Debug)]
pub struct RawContext<'comm> {
    /// Transferred native cuTENSORMp handle; destroy it before releasing its CUDA context and stream.
    pub handle: sys::cutensorMpHandle_t,
    /// CUDA context required for using and destroying the transferred handle.
    pub cuda_context: Arc<CudaContext>,
    /// Local CUDA device the handle was created for.
    pub local_device: Device,
    /// Stream the handle executes on, or `None` for the default CUDA stream.
    pub stream: Option<Stream>,
    /// Borrow of the NCCL communicator the handle was created with.
    pub communicator: PhantomData<&'comm Communicator>,
}

impl<'comm> Context<'comm> {
    /// Synchronizes this CUDA context and clears communication resources cached by the handle.
    ///
    /// Call before freeing or reallocating externally managed workspace regions registered by communication operations.
    /// The wrapper serializes this operation with execution through the shared context lock.
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
        local_device: Device,
        stream: Option<&Stream>,
    ) -> Result<Self> {
        if stream.is_some_and(|stream| communicator.cuda_context() != stream.context()) {
            return Err(Error::StreamContextMismatch);
        }

        communicator.bind()?;

        let mut handle = ptr::null_mut();
        // SAFETY: `communicator` is live for `'comm`, `stream` was checked to share its CUDA context, and `handle` is a local output.
        unsafe {
            try_ffi!(sys::cutensorMpCreate(
                &raw mut handle,
                communicator.as_raw(),
                local_device.id(),
                stream.map_or(ptr::null_mut(), Stream::as_raw),
            ))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle: Rc::new(Handle {
                raw: NativeHandle(handle),
                cuda_context: Arc::clone(communicator.cuda_context()),
                local_device,
                stream: stream.cloned(),
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

    /// Returns the local CUDA device recorded when the handle was created.
    #[must_use]
    pub fn local_device(&self) -> Device {
        self.handle.local_device
    }

    /// Returns the retained handle stream, or `None` for its default CUDA stream.
    #[must_use]
    pub fn stream(&self) -> Option<&Stream> {
        self.handle.stream.as_ref()
    }

    pub(crate) fn as_context_ref(&self) -> ContextRef<'comm> {
        ContextRef {
            handle: Rc::clone(&self.handle),
        }
    }

    /// Returns the borrowed native multi-process library handle.
    #[must_use]
    pub fn as_raw(&self) -> sys::cutensorMpHandle_t {
        self.handle.raw.0
    }

    /// Transfers the native handle with its CUDA context and stream if nothing else retains it.
    ///
    /// # Errors
    ///
    /// Returns the original context as `Err` if clones, descriptors, plans, or workspaces still retain the handle.
    pub fn into_raw(self) -> StdResult<RawContext<'comm>, Self> {
        match Rc::try_unwrap(self.handle) {
            Ok(mut owner) => {
                let handle = mem::replace(&mut owner.raw.0, ptr::null_mut());
                Ok(RawContext {
                    handle,
                    cuda_context: Arc::clone(&owner.cuda_context),
                    local_device: owner.local_device,
                    stream: owner.stream.take(),
                    communicator: PhantomData,
                })
            }
            Err(handle) => Err(Self { handle }),
        }
    }

    /// Takes ownership of a native cuTENSORMp handle with its CUDA context and stream.
    ///
    /// # Safety
    ///
    /// `raw.handle` must be a valid cuTENSORMp handle created with the borrowed communicator for `raw.cuda_context`, `raw.local_device`, and `raw.stream`, and uniquely owned.
    /// The returned context destroys the handle when the last clone and every resource created from it are dropped.
    ///
    /// # Errors
    ///
    /// Returns an error if `raw.handle` is null.
    pub unsafe fn from_raw(raw: RawContext<'comm>) -> Result<Self> {
        if raw.handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle: Rc::new(Handle {
                raw: NativeHandle(raw.handle),
                cuda_context: raw.cuda_context,
                local_device: raw.local_device,
                stream: raw.stream,
                execution_lock: Mutex::new(()),
                _communicator: PhantomData,
            }),
        })
    }

    pub(crate) fn into_context_ref(self) -> ContextRef<'comm> {
        ContextRef {
            handle: self.handle,
        }
    }
}

impl<'comm> ContextRef<'comm> {
    pub(crate) fn into_context(self) -> Context<'comm> {
        Context {
            handle: self.handle,
        }
    }

    pub(crate) fn lock_execution(&self) -> Result<MutexGuard<'_, ()>> {
        self.handle
            .execution_lock
            .lock()
            .map_err(|_| Error::LockPoisoned {
                name: "cutensor execution".into(),
            })
    }

    pub(crate) fn clear_operation_cache(&self) -> Result<()> {
        let _lock = self.lock_execution()?;
        self.bind()?;
        self.cuda_context().synchronize()?;
        // SAFETY: the `Rc` keeps the handle live, `execution_lock` excludes executions, and the CUDA context was synchronized above.
        unsafe {
            try_ffi!(sys::cutensorMpClearOperationCache(self.as_raw()))?;
        }
        Ok(())
    }

    pub(crate) fn cuda_context(&self) -> &Arc<CudaContext> {
        &self.handle.cuda_context
    }

    pub(crate) fn bind(&self) -> Result<()> {
        Ok(self.cuda_context().bind()?)
    }

    pub(crate) fn same_handle(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.handle, &other.handle)
    }

    pub(crate) fn stream(&self) -> Option<&Stream> {
        self.handle.stream.as_ref()
    }

    pub(crate) fn as_raw(&self) -> sys::cutensorMpHandle_t {
        self.handle.raw.0
    }
}

impl Drop for Handle<'_> {
    fn drop(&mut self) {
        if self.raw.0.is_null() {
            return;
        }

        if let Err(error) = self.cuda_context.bind() {
            report_cleanup_failure(
                "bind the cuda context to destroy",
                "cutensormp context",
                &error,
            );
            return;
        }

        // SAFETY: this is the last `Rc` owner of the non-null handle, so no descriptor, plan, or workspace still uses it.
        unsafe {
            if let Err(error) = try_ffi!(sys::cutensorMpDestroy(self.raw.0)) {
                report_cleanup_failure("destroy", "cutensormp context", &error);
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
        return Err(Error::ContextMismatch {
            name: name.to_owned().into(),
        });
    }
    Ok(())
}
