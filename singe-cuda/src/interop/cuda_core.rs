//! CUDA Core handles retaining their actual Singe context and stream owners.

use std::{ffi::c_void, sync::Arc};

use ::cuda_core::{Device as ForeignDevice, Stream as ForeignStream};

use crate::{context::Context, error::Result, stream::Stream, utility::to_usize};

impl Context {
    /// Binds this context and creates a foreign device retaining its ownership.
    ///
    /// The adapter never acquires responsibility for releasing the CUDA context.
    ///
    /// # Errors
    ///
    /// Returns an error if this context cannot be bound to the calling thread or its device ordinal is negative.
    pub fn to_cuda_core(self: &Arc<Self>) -> Result<Arc<ForeignDevice>> {
        self.bind()?;
        let ordinal = to_usize(self.device().id(), "device ordinal")?;
        let owner = Arc::clone(self);
        // SAFETY: `owner` retains this live context, which belongs to the named device, for as long as the foreign device exists.
        Ok(unsafe {
            ForeignDevice::borrow_with_owner(
                self.as_raw().cast::<c_void>(),
                self.device().id(),
                ordinal,
                owner,
            )
        })
    }
}

impl Stream {
    /// Binds the stream's context and creates a foreign stream retaining both owners.
    ///
    /// The foreign handle borrows destruction responsibility.
    /// Its liveness token owns a clone of this stream and therefore retains its actual context.
    ///
    /// # Errors
    ///
    /// Returns an error if [`Context::to_cuda_core`] fails for the stream's context.
    pub fn to_cuda_core(&self) -> Result<Arc<ForeignStream>> {
        let device = self.context().to_cuda_core()?;
        let owner = Arc::new(self.clone());
        // SAFETY: `owner` is a clone of this stream, which keeps the native stream and its context alive for as long as the foreign stream exists, and `device` wraps that context.
        Ok(unsafe {
            ForeignStream::borrow_with_owner(self.as_raw().cast::<c_void>(), &device, owner)
        })
    }
}
