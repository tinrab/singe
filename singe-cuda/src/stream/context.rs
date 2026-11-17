use std::{ptr, sync::Arc};

use singe_cuda_sys::runtime;

use super::{RawStream, RawStreamOwnership, Stream, StreamOptions};
use crate::{
    context::Context,
    error::{Error, Result},
    try_ffi,
};

impl Context {
    /// Creates an asynchronous stream in this context and makes this context current on the calling thread.
    ///
    /// [`StreamOptions::flags`] sets the stream's synchronization with the legacy default stream:
    ///
    /// * [`StreamFlags::DEFAULT`](super::StreamFlags::DEFAULT): default stream creation behavior.
    /// * [`StreamFlags::NON_BLOCKING`](super::StreamFlags::NON_BLOCKING): allows the created stream to run concurrently with the legacy default stream without implicit synchronization.
    ///
    /// [`StreamOptions::priority`] sets the scheduling priority of work in the stream.
    /// Priorities are a hint to run higher-priority work first when possible; they do not preempt running work or otherwise order execution.
    /// Lower numbers are higher priorities, and `0` is the default priority.
    /// [`Device::stream_priority_range`](crate::device::Device::stream_priority_range) reports the meaningful range, and CUDA clamps a priority outside it to the nearest end.
    /// Priorities affect only compute kernels, not host-to-device or device-to-host memory operations.
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound, CUDA cannot create the stream, CUDA returns a null stream handle, a previous asynchronous launch reports an error, or CUDA reports runtime initialization diagnostics.
    pub fn create_stream(self: &Arc<Self>, options: StreamOptions) -> Result<Stream> {
        self.bind()?;
        let mut handle = ptr::null_mut();
        // SAFETY: `handle` is a writable local, and the context is bound above.
        unsafe {
            try_ffi!(runtime::cudaStreamCreateWithPriority(
                &raw mut handle,
                options.flags.bits(),
                options.priority,
            ))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        // SAFETY: CUDA just created this handle in this context, and nothing else owns it.
        unsafe {
            Stream::from_raw(RawStream {
                handle,
                context: Arc::clone(self),
                ownership: RawStreamOwnership::Owned,
            })
        }
    }
}
