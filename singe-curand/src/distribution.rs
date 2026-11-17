//! Context-retaining discrete distribution descriptors.

use std::{mem::ManuallyDrop, ptr, sync::Arc};

use singe_core::{abort_after_cleanup_failure, report_cleanup_failure};
use singe_cuda::{
    context::Context,
    stream::{AsStream, BorrowedStream, Stream},
};
use singe_curand_sys as sys;

use crate::{
    error::{Error, Result},
    try_ffi,
};

/// A completed Poisson histogram retaining its actual context and cleanup stream.
/// Raw kernels must retain it through completion, including captured graph replay.
#[derive(Debug)]
pub struct DiscreteDistribution {
    owner: ManuallyDrop<Owner>,
    stream: Stream,
}

/// Unmanaged Poisson histogram ownership, including its cleanup stream.
/// Dropping it does not destroy the native histogram.
#[derive(Debug)]
pub struct RawDiscreteDistribution {
    /// Exclusively owned native histogram handle.
    pub handle: sys::curandDiscreteDistribution_t,
    /// Stream that orders the histogram's cleanup, retaining its context.
    pub stream: Stream,
}

#[derive(Debug)]
struct Owner {
    handle: sys::curandDiscreteDistribution_t,
    stream: BorrowedStream,
}

#[expect(
    clippy::non_send_fields_in_send_ty,
    reason = "CUDA stream handles may be used from any thread, and the owner only destroys its handle"
)]
// SAFETY: cuRAND does not tie a histogram to its creating thread, and the owner exposes no methods; it only destroys the histogram, after binding its context, from whichever thread drops it.
unsafe impl Send for Owner {}

// SAFETY: shared references to the owner reach no native state.
unsafe impl Sync for Owner {}

impl Owner {
    fn into_handle(self) -> sys::curandDiscreteDistribution_t {
        let owner = ManuallyDrop::new(self);
        // SAFETY: `owner` is never dropped, so its stream moves out exactly once.
        drop(unsafe { ptr::read(&raw const owner.stream) });
        owner.handle
    }
}

impl Drop for Owner {
    fn drop(&mut self) {
        // Queued kernels may still read the histogram.
        self.stream.synchronize_or_abort();
        // SAFETY: the histogram is live and exclusively owned, and synchronizing bound its context.
        if let Err(error) = unsafe { try_ffi!(sys::curandDestroyDistribution(self.handle)) } {
            report_cleanup_failure("destroy", "curand distribution", &error);
        }
    }
}

impl DiscreteDistribution {
    /// Creates a completed Poisson histogram for a positive lambda up to 400,000.
    ///
    /// Native construction has no stream parameter and may use the default stream; this call completes the actual context before publishing the histogram.
    ///
    /// # Errors
    ///
    /// Returns an error if the stream is being captured, its capture state or context cannot be accessed, or context synchronization fails.
    /// cuRAND errors include a lambda outside `(0, 400_000]`, allocation or GPU initialization failure, missing double-precision support, and an earlier kernel failure.
    /// A null handle returned after successful creation is also rejected.
    pub fn create_poisson(lambda: f64, stream: &Stream) -> Result<Self> {
        stream.ensure_not_capturing()?;
        // No new native work exists yet, so prior context errors can return normally.
        stream.context().synchronize()?;
        let mut handle = ptr::null_mut();
        // SAFETY: `handle` is a valid output location, and the stream's context is bound.
        let result = unsafe {
            try_ffi!(sys::curandCreatePoissonDistribution(
                lambda,
                &raw mut handle
            ))
        };
        // SAFETY: a successful native call transfers exclusive ownership of a histogram in the stream's context.
        let owner = result.and_then(|()| unsafe {
            Self::from_raw(RawDiscreteDistribution {
                handle,
                stream: stream.clone(),
            })
        });
        // Construction completes before the histogram is published, including after native failures.
        stream.context().synchronize_or_abort();
        owner
    }

    /// Adopts an exclusively owned completed device histogram.
    ///
    /// # Safety
    ///
    /// The handle must be live, exclusively owned, and allocated in the stream's actual context.
    /// All prior use must have completed.
    /// Transfer ownership once.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NullHandle`] if `raw.handle` is null.
    pub unsafe fn from_raw(raw: RawDiscreteDistribution) -> Result<Self> {
        if raw.handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            owner: ManuallyDrop::new(Owner {
                handle: raw.handle,
                // SAFETY: the wrapper retains `raw.stream` until this owner drops, directly or through the stream's capture cleanup.
                stream: unsafe { raw.stream.to_unretained() },
            }),
            stream: raw.stream,
        })
    }

    /// Transfers native ownership with the cleanup stream.
    #[must_use]
    pub fn into_raw(self) -> RawDiscreteDistribution {
        let mut this = ManuallyDrop::new(self);
        // SAFETY: `this` is never dropped, so the owner and the stream each move out exactly once.
        let (owner, stream) = unsafe {
            (
                ManuallyDrop::take(&mut this.owner),
                ptr::read(&raw const this.stream),
            )
        };
        RawDiscreteDistribution {
            handle: owner.into_handle(),
            stream,
        }
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
    pub fn stream(&self) -> &Stream {
        &self.stream
    }
}

impl Drop for DiscreteDistribution {
    fn drop(&mut self) {
        // SAFETY: `drop` runs once, and `self.owner` is not used afterwards.
        let owner = Arc::new(unsafe { ManuallyDrop::take(&mut self.owner) });
        // Destroying the histogram is not allowed during capture, so an active capture keeps it until capture ends.
        if let Err(error) = self.stream.defer_drop_until_capture_end(owner) {
            abort_after_cleanup_failure("defer the drop of", "curand distribution", &error);
        }
    }
}
