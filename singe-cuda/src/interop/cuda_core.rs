//! CUDA Core handles retaining their actual Singe context and stream owners.

use std::{ffi::c_void, sync::Arc};

use ::cuda_core::{Device as ForeignDevice, Stream as ForeignStream};

use crate::{context::Context, error::Result, stream::Stream};

impl Context {
    /// Binds this context and creates a foreign device retaining its ownership.
    ///
    /// The adapter never acquires responsibility for releasing the CUDA context.
    ///
    /// # Errors
    ///
    /// Propagates CUDA errors when binding this context to the calling thread.
    pub fn to_cuda_core(self: &Arc<Self>) -> Result<Arc<ForeignDevice>> {
        self.bind()?;
        let owner = Arc::clone(self);
        // SAFETY: the retained context owns this live handle and its associated device.
        Ok(unsafe {
            ForeignDevice::borrow_with_owner(
                self.as_raw().cast::<c_void>(),
                self.device().id(),
                self.device().id() as usize,
                owner,
            )
        })
    }
}

impl Stream {
    /// Binds the stream's context and creates a foreign stream retaining both owners.
    ///
    /// The foreign handle borrows destruction responsibility. Its liveness
    /// token owns a clone of this stream and therefore retains its actual context.
    ///
    /// # Errors
    ///
    /// Propagates CUDA context-binding errors from [`Context::to_cuda_core`].
    pub fn to_cuda_core(&self) -> Result<Arc<ForeignStream>> {
        let device = self.context().to_cuda_core()?;
        let owner = Arc::new(self.clone());
        // SAFETY: the retained stream clone owns this live handle in the adapter's context.
        Ok(unsafe {
            ForeignStream::borrow_with_owner(self.as_raw().cast::<c_void>(), &device, owner)
        })
    }
}
