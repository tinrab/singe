use std::{marker::PhantomData, mem::ManuallyDrop, ptr, sync::Arc};

use singe_cuda::{context::Context as CudaContext, stream::StreamBinding};
use singe_nccl::communicator::Communicator;

use crate::{
    error::{Error, Result},
    sys, try_ffi,
};

#[derive(Debug, Clone)]
pub struct Context<'comm> {
    handle: Arc<Handle<'comm>>,
}

#[derive(Debug)]
pub(crate) struct Handle<'comm> {
    raw: sys::cutensorMpHandle_t,
    cuda_ctx: Arc<CudaContext>,
    local_device_id: i32,
    stream: StreamBinding,
    _communicator: PhantomData<&'comm Communicator>,
}

#[derive(Debug, Clone)]
pub(crate) struct ContextRef<'comm> {
    handle: Arc<Handle<'comm>>,
}

// cuTENSORMp contexts are created for a fixed communicator, CUDA context, and
// stream binding. The Arc-backed handle can be shared immutably, while wrappers
// remain movable.
unsafe impl Send for Handle<'_> {}
unsafe impl Sync for Handle<'_> {}
unsafe impl Send for Context<'_> {}
unsafe impl Send for ContextRef<'_> {}

impl<'comm> Context<'comm> {
    pub fn create(
        communicator: &'comm Communicator,
        local_device_id: i32,
        stream: StreamBinding,
    ) -> Result<Self> {
        if communicator.cuda_context().as_ref() != stream.context() {
            return Err(Error::StreamContextMismatch);
        }
        communicator.bind()?;

        let mut handle = ptr::null_mut();
        unsafe {
            try_ffi!(sys::cutensorMpCreate(
                &raw mut handle,
                communicator.as_raw(),
                local_device_id,
                stream.as_raw(),
            ))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle: Arc::new(Handle {
                raw: handle,
                cuda_ctx: Arc::clone(communicator.cuda_context()),
                local_device_id,
                stream,
                _communicator: PhantomData,
            }),
        })
    }

    /// Wraps an existing cuTENSORMp handle.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid cuTENSORMp handle associated with `cuda_ctx`,
    /// `local_device_id`, and `stream`. The returned context takes ownership
    /// of `handle` and destroys it when the last clone is dropped.
    pub unsafe fn from_raw(
        handle: sys::cutensorMpHandle_t,
        cuda_ctx: Arc<CudaContext>,
        local_device_id: i32,
        stream: StreamBinding,
    ) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle: Arc::new(Handle {
                raw: handle,
                cuda_ctx,
                local_device_id,
                stream,
                _communicator: PhantomData,
            }),
        })
    }

    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        &self.handle.cuda_ctx
    }

    pub fn bind(&self) -> Result<()> {
        Ok(self.cuda_context().bind()?)
    }

    pub fn local_device_id(&self) -> i32 {
        self.handle.local_device_id
    }

    pub fn stream(&self) -> &StreamBinding {
        &self.handle.stream
    }

    pub(crate) fn as_context_ref(&self) -> ContextRef<'comm> {
        ContextRef {
            handle: Arc::clone(&self.handle),
        }
    }

    pub fn as_raw(&self) -> sys::cutensorMpHandle_t {
        self.handle.raw
    }

    /// Consumes this context and returns the owned raw cuTENSORMp handle.
    ///
    /// # Errors
    ///
    /// Returns [`Error::HandleShared`] if cloned contexts or context references
    /// still point at the same handle.
    pub fn into_raw(self) -> Result<sys::cutensorMpHandle_t> {
        let handle = Arc::try_unwrap(self.handle).map_err(|_| Error::HandleShared)?;
        let handle = ManuallyDrop::new(handle);
        Ok(handle.raw)
    }
}

impl ContextRef<'_> {
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        &self.handle.cuda_ctx
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
        if let Err(err) = self.cuda_ctx.bind() {
            #[cfg(debug_assertions)]
            eprintln!("failed to bind cuda context before destroying cutensormp handle: {err}");
        }

        unsafe {
            if let Err(err) = try_ffi!(sys::cutensorMpDestroy(self.raw)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cutensormp context: {err}");
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
        return Err(Error::MpContextMismatch { name: name.into() });
    }
    Ok(())
}
