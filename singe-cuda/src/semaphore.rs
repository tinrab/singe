//! Imported external semaphores and timeline operations.

use std::{mem::ManuallyDrop, os::fd::OwnedFd, ptr, rc::Rc, sync::Arc};

use singe_cuda_sys::driver as sys;

use crate::{
    context::Context,
    error::{Error, Result},
    stream::Stream,
    try_ffi,
};

/// An imported external semaphore.
#[derive(Debug)]
pub struct ExternalSemaphore {
    handle: sys::CUexternalSemaphore,
    context: Arc<Context>,
}

impl ExternalSemaphore {
    /// Imports an opaque or timeline semaphore file descriptor.
    /// Ownership of the descriptor transfers to CUDA on success.
    ///
    /// # Errors
    ///
    /// Returns an error if `context` cannot be bound or CUDA cannot import the descriptor with the selected opaque or timeline handle type. Returns `Error::NullHandle` for a null imported handle. A descriptor not successfully imported is closed when its `OwnedFd` is dropped; after a successful native import CUDA owns it.
    #[cfg(unix)]
    pub fn import_fd(context: &Arc<Context>, fd: OwnedFd, timeline: bool) -> Result<Rc<Self>> {
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
        // SAFETY: `cuImportExternalSemaphore` receives valid output locations and pointers into buffers borrowed for the call.
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
        Ok(Rc::new(Self {
            handle,
            context: Arc::clone(context),
        }))
    }

    /// Takes ownership of an imported CUDA external semaphore.
    ///
    /// # Safety
    /// The handle must be live, uniquely owned and associated with `context`.
    ///
    /// # Errors
    ///
    /// Returns `Error::NullHandle` if `handle` is null. The caller must satisfy the remaining handle validity and ownership requirements.
    pub unsafe fn from_raw(
        context: &Arc<Context>,
        handle: sys::CUexternalSemaphore,
    ) -> Result<Rc<Self>> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Rc::new(Self {
            handle,
            context: Arc::clone(context),
        }))
    }

    /// Returns the borrowed native semaphore handle without transferring ownership.
    #[must_use]
    pub fn as_raw(&self) -> sys::CUexternalSemaphore {
        self.handle
    }

    /// Transfers the semaphore handle and retained context to the caller.
    #[must_use]
    pub fn into_raw(self) -> (sys::CUexternalSemaphore, Arc<Context>) {
        let owner = ManuallyDrop::new(self);
        // SAFETY: the owner is wrapped in `ManuallyDrop`, so this field moves out exactly once.
        (owner.handle, unsafe { ptr::read(&raw const owner.context) })
    }

    /// Signals a timeline fence value in the stream.
    ///
    /// # Safety
    /// The semaphore must remain alive until the signal completes.
    ///
    /// # Errors
    ///
    /// Returns `Error::StreamContextMismatch` if `stream` belongs to another context. Also returns an error if CUDA rejects the semaphore, fence parameters, or asynchronous signal operation, or reports previous asynchronous failures.
    pub unsafe fn signal(&self, value: u64, stream: &Stream) -> Result<()> {
        if stream.context() != &self.context {
            return Err(Error::StreamContextMismatch);
        }
        let mut params = sys::CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS::default();
        params.params.fence.value = value;
        // SAFETY: `cuSignalExternalSemaphoresAsync` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::cuSignalExternalSemaphoresAsync(
                &raw const self.handle,
                &raw const params,
                1,
                stream.as_raw()
            ))
        }
    }

    /// Waits for a timeline fence value in the stream.
    ///
    /// # Safety
    /// The semaphore must remain alive until the wait completes.
    ///
    /// # Errors
    ///
    /// Returns `Error::StreamContextMismatch` if `stream` belongs to another context. Also returns an error if CUDA rejects the semaphore, fence parameters, or asynchronous wait operation, or reports previous asynchronous failures.
    pub unsafe fn wait(&self, value: u64, stream: &Stream) -> Result<()> {
        if stream.context() != &self.context {
            return Err(Error::StreamContextMismatch);
        }
        let mut params = sys::CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS::default();
        params.params.fence.value = value;
        // SAFETY: `cuWaitExternalSemaphoresAsync` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::cuWaitExternalSemaphoresAsync(
                &raw const self.handle,
                &raw const params,
                1,
                stream.as_raw()
            ))
        }
    }
}

impl Drop for ExternalSemaphore {
    fn drop(&mut self) {
        let _ = self.context.bind();
        // SAFETY: `cuDestroyExternalSemaphore` receives live handles owned by the wrappers.
        unsafe {
            if let Err(error) = try_ffi!(sys::cuDestroyExternalSemaphore(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cuda external semaphore: {error}");
            }
        }
    }
}
