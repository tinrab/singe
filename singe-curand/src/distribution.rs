//! Context-retaining discrete distribution descriptors.

use std::{process, ptr, sync::Arc};

use singe_cuda::{context::Context, stream::Stream};
use singe_cuda_sys::runtime::cudaStream_t;
use singe_curand_sys as sys;

use crate::{
    error::{Error, Result},
    try_ffi,
    utility::{complete_context, drain_stream, require_eager},
};

/// A completed Poisson histogram retaining its actual context and cleanup stream.
/// Raw kernels must retain it through completion, including captured graph replay.
#[derive(Debug)]
pub struct DiscreteDistribution {
    owner: Arc<Owner>,
    stream: Arc<Stream>,
}

#[derive(Debug)]
struct Owner {
    handle: sys::curandDiscreteDistribution_t,
    context: Arc<Context>,
    stream: cudaStream_t,
}

// SAFETY: typed descriptors expose no mutation. Final native cleanup binds the context;
// the wrapper or capture registry keeps its cleanup stream alive.
unsafe impl Send for Owner {}

unsafe impl Sync for Owner {}

impl Drop for Owner {
    fn drop(&mut self) {
        // SAFETY: the wrapper or capture registry keeps this stream live.
        unsafe {
            drain_stream(&self.context, self.stream);
        }
        // SAFETY: `curandDestroyDistribution` receives live handles owned by the wrappers.
        if let Err(error) = unsafe { try_ffi!(sys::curandDestroyDistribution(self.handle)) } {
            #[cfg(debug_assertions)]
            eprintln!("failed to destroy curand distribution: {error}");
        }
        // SAFETY: native destruction may enqueue stream-ordered frees; the stream is still live.
        unsafe {
            drain_stream(&self.context, self.stream);
        }
    }
}

impl DiscreteDistribution {
    /// Creates a completed Poisson histogram for a positive lambda up to 400,000.
    /// Native construction has no stream parameter and may use the default stream;
    /// this call completes the actual context before publishing the histogram.
    ///
    /// # Errors
    ///
    /// Returns an error if the stream is being captured, its capture state or
    /// context cannot be accessed, or context synchronization fails. cuRAND
    /// errors include a lambda outside `(0, 400_000]`, allocation or GPU
    /// initialization failure, missing double-precision support, and an earlier
    /// kernel failure. A null handle returned after successful creation is also
    /// rejected.
    pub fn create_poisson(lambda: f64, stream: &Arc<Stream>) -> Result<Self> {
        require_eager(stream)?;
        // No new native work exists yet, so prior context errors can return normally.
        stream.context().synchronize()?;
        let mut handle = ptr::null_mut();
        // SAFETY: `curandCreatePoissonDistribution` receives valid output locations.
        let result = unsafe {
            try_ffi!(sys::curandCreatePoissonDistribution(
                lambda,
                &raw mut handle
            ))
        };
        // SAFETY: a successful native call transfers exclusive ownership.
        let owner = result.and_then(|()| unsafe { Self::from_raw(handle, stream) });
        // Always complete construction before releasing the host output slot,
        // including after native failures. No user callback can unwind this FFI.
        let completion = complete_context(stream.context());
        let owner = owner?;
        completion?;
        Ok(owner)
    }

    /// Adopts an exclusively owned completed device histogram.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NullHandle`] if `handle` is null.
    ///
    /// # Safety
    /// The handle must be live, exclusively owned, and allocated in this actual
    /// context. All prior use must have completed. Transfer ownership once.
    pub unsafe fn from_raw(
        handle: sys::curandDiscreteDistribution_t,
        stream: &Arc<Stream>,
    ) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            owner: Arc::new(Owner {
                handle,
                context: Arc::clone(stream.context()),
                stream: stream.as_raw(),
            }),
            stream: Arc::clone(stream),
        })
    }

    /// Borrows the histogram handle without transferring destruction ownership.
    #[must_use]
    pub fn as_raw(&self) -> sys::curandDiscreteDistribution_t {
        self.owner.handle
    }

    /// Context in which the histogram was allocated.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        self.stream.context()
    }

    /// Retained stream used to order the histogram's cleanup.
    #[must_use]
    pub fn stream(&self) -> &Arc<Stream> {
        &self.stream
    }
}

impl Drop for DiscreteDistribution {
    fn drop(&mut self) {
        if self
            .stream
            .defer_drop_until_capture_end(Arc::clone(&self.owner))
            .is_err()
        {
            process::abort();
        }
    }
}
