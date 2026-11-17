//! Imported external semaphores and timeline operations.

use std::{mem::ManuallyDrop, os::fd::OwnedFd, ptr, result::Result as StdResult, sync::Arc};

use singe_core::report_cleanup_failure;
use singe_cuda_sys::driver as sys;

use crate::{
    context::Context,
    error::{Error, Result},
    stream::{Stream, StreamCaptureStatus},
    submission::Resources,
    try_ffi,
};

/// An imported external semaphore.
#[derive(Debug)]
pub struct ExternalSemaphore {
    handle: sys::CUexternalSemaphore,
    context: Arc<Context>,
}

/// Raw ownership of an imported external semaphore and its context.
///
/// Destroy `handle` while `context` is alive, or re-adopt the value with [`ExternalSemaphore::from_raw`].
#[derive(Debug)]
pub struct RawExternalSemaphore {
    /// Native external semaphore handle.
    pub handle: sys::CUexternalSemaphore,
    /// Context the semaphore was imported into.
    pub context: Arc<Context>,
}

// SAFETY: CUDA driver external-semaphore handles may be used and destroyed from any thread once their context is bound, and the wrapper has no interior mutability.
unsafe impl Send for ExternalSemaphore {}

// SAFETY: every method takes `&self` and only passes the immutable handle to thread-safe driver calls, so shared references on several threads cannot race.
unsafe impl Sync for ExternalSemaphore {}

impl ExternalSemaphore {
    /// Imports an opaque or timeline semaphore file descriptor.
    /// Ownership of the descriptor transfers to CUDA on success.
    ///
    /// # Errors
    ///
    /// Returns an error if `context` cannot be bound or CUDA cannot import the descriptor with the selected opaque or timeline handle type.
    /// Returns `Error::NullHandle` for a null imported handle.
    /// A descriptor not successfully imported is closed when its `OwnedFd` is dropped; after a successful native import CUDA owns it.
    #[cfg(unix)]
    pub fn import_fd(context: &Arc<Context>, fd: OwnedFd, timeline: bool) -> Result<Arc<Self>> {
        use std::os::fd::{AsRawFd, IntoRawFd};
        context.bind()?;
        let kind = if timeline {
            sys::CUexternalSemaphoreHandleType::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD
        } else {
            sys::CUexternalSemaphoreHandleType::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD
        };
        let mut desc = sys::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC {
            type_: kind,
            ..Default::default()
        };
        desc.handle.fd = fd.as_raw_fd();
        let mut handle = ptr::null_mut();
        // SAFETY: `handle` is a writable local, and `desc` is a local descriptor holding the still-owned descriptor `fd` for the call.
        unsafe {
            try_ffi!(sys::cuImportExternalSemaphore(
                &raw mut handle,
                &raw const desc
            ))?;
        }

        let _ = fd.into_raw_fd();
        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Arc::new(Self {
            handle,
            context: Arc::clone(context),
        }))
    }

    /// Takes ownership of an imported CUDA external semaphore.
    ///
    /// # Safety
    ///
    /// `raw.handle` must be live, uniquely owned, and imported into `raw.context`.
    ///
    /// # Errors
    ///
    /// Returns `Error::NullHandle` if `raw.handle` is null.
    pub unsafe fn from_raw(raw: RawExternalSemaphore) -> Result<Arc<Self>> {
        if raw.handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Arc::new(Self {
            handle: raw.handle,
            context: raw.context,
        }))
    }

    /// Returns the borrowed native semaphore handle without transferring ownership.
    #[must_use]
    pub fn as_raw(&self) -> sys::CUexternalSemaphore {
        self.handle
    }

    /// Transfers the semaphore handle and context to the caller.
    ///
    /// # Errors
    ///
    /// Returns the semaphore unchanged as `Err` if another clone or queued operation still retains it.
    pub fn into_raw(this: Arc<Self>) -> StdResult<RawExternalSemaphore, Arc<Self>> {
        let owner = ManuallyDrop::new(Arc::try_unwrap(this)?);
        Ok(RawExternalSemaphore {
            handle: owner.handle,
            // SAFETY: the owner is wrapped in `ManuallyDrop`, so this field moves out exactly once.
            context: unsafe { ptr::read(&raw const owner.context) },
        })
    }

    // Keeps the semaphore alive until the operation enqueued by `enqueue` completes, or until capture ends.
    unsafe fn enqueue_retained(
        self: &Arc<Self>,
        stream: &Stream,
        enqueue: impl FnOnce() -> Result<()>,
    ) -> Result<()> {
        if stream.context() != &self.context {
            return Err(Error::StreamContextMismatch);
        }

        if stream.capture_status()? != StreamCaptureStatus::None {
            stream.defer_drop_until_capture_end(Arc::clone(self))?;
            return enqueue();
        }

        let mut resources = Resources::new();
        resources.retain(Arc::clone(self));
        // SAFETY: the retained semaphore is the only native owner the enqueued operation uses.
        unsafe { stream.enqueue_async(resources, |_| enqueue()) }?;
        Ok(())
    }

    /// Enqueues a signal of timeline fence `value` on `stream`, retaining the semaphore until the signal completes.
    ///
    /// During capture, the semaphore is retained until capture ends.
    ///
    /// # Safety
    ///
    /// During capture, keep the semaphore alive through every replay of the captured graph.
    /// The fence value must follow the semaphore's protocol with its other users.
    ///
    /// # Errors
    ///
    /// Returns `Error::StreamContextMismatch` if `stream` belongs to another context.
    /// Also returns an error if CUDA rejects the semaphore, fence parameters, or asynchronous signal operation, or reports previous asynchronous failures.
    pub unsafe fn signal_async(self: &Arc<Self>, value: u64, stream: &Stream) -> Result<()> {
        let mut params = sys::CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS::default();
        params.params.fence.value = value;
        let enqueue = || {
            self.context.bind()?;
            // SAFETY: the semaphore and stream are live in the bound context, and `params` outlives the call.
            unsafe {
                try_ffi!(sys::cuSignalExternalSemaphoresAsync(
                    &raw const self.handle,
                    &raw const params,
                    1,
                    stream.as_raw()
                ))
            }
        };
        // SAFETY: the semaphore is retained until the signal completes.
        unsafe { self.enqueue_retained(stream, enqueue) }
    }

    /// Enqueues a wait for timeline fence `value` on `stream`, retaining the semaphore until the wait completes.
    ///
    /// During capture, the semaphore is retained until capture ends.
    ///
    /// # Safety
    ///
    /// During capture, keep the semaphore alive through every replay of the captured graph.
    /// Some user of the semaphore must eventually signal `value`, or the stream never progresses.
    ///
    /// # Errors
    ///
    /// Returns `Error::StreamContextMismatch` if `stream` belongs to another context.
    /// Also returns an error if CUDA rejects the semaphore, fence parameters, or asynchronous wait operation, or reports previous asynchronous failures.
    pub unsafe fn wait_async(self: &Arc<Self>, value: u64, stream: &Stream) -> Result<()> {
        let mut params = sys::CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS::default();
        params.params.fence.value = value;
        let enqueue = || {
            self.context.bind()?;
            // SAFETY: the semaphore and stream are live in the bound context, and `params` outlives the call.
            unsafe {
                try_ffi!(sys::cuWaitExternalSemaphoresAsync(
                    &raw const self.handle,
                    &raw const params,
                    1,
                    stream.as_raw()
                ))
            }
        };
        // SAFETY: the semaphore is retained until the wait completes.
        unsafe { self.enqueue_retained(stream, enqueue) }
    }
}

impl Drop for ExternalSemaphore {
    fn drop(&mut self) {
        if let Err(error) = self.context.bind() {
            report_cleanup_failure(
                "bind the cuda context to destroy",
                "cuda external semaphore",
                &error,
            );
            return;
        }

        // SAFETY: `self.handle` is the live semaphore this owner imported into `self.context`, bound above, and dropping the last owner leaves no other user.
        if let Err(error) = unsafe { try_ffi!(sys::cuDestroyExternalSemaphore(self.handle)) } {
            report_cleanup_failure("destroy", "cuda external semaphore", &error);
        }
    }
}
