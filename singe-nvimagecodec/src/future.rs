use std::{marker::PhantomData, process, ptr};

use singe_cuda::stream::Stream;
use singe_nvimagecodec_sys as sys;

use crate::{
    error::{Error, Result},
    instance::{Instance, InstanceRef},
    try_ffi,
    types::ProcessingStatus,
    utility::{to_u64, to_usize, validate_processing_statuses},
};

/// Handle to a submission from [`Decoder::decode_raw_async`](crate::decode::Decoder::decode_raw_async) or [`Encoder::encode_raw_async`](crate::encode::Encoder::encode_raw_async).
///
/// When the future is ready, nvImageCodec has finished its CPU work, but depending on the codec, GPU work may still be queued on the images' CUDA stream.
/// Use [`check_on`](Self::check_on) to also wait for that stream before reading GPU output.
/// The safe decode and encode calls wait for both before returning.
#[derive(Debug)]
#[must_use = "call check_host(), check_on(), or wait_for_host() to observe completion and statuses"]
pub struct Future<'a> {
    raw: sys::nvimgcodecFuture_t,
    _instance: InstanceRef,
    status_count: usize,
    host_completed: bool,
    _resources: PhantomData<&'a ()>,
}

/// Native future handle with the values that keep it valid, from [`Future::into_raw`].
///
/// Destroy the handle before dropping this value, or pass it back to [`Future::from_raw`].
#[derive(Debug)]
pub struct RawFuture<'a> {
    /// Native future handle.
    pub handle: sys::nvimgcodecFuture_t,
    owner: Future<'a>,
}

impl<'a> Future<'a> {
    pub(crate) fn create(
        raw: sys::nvimgcodecFuture_t,
        instance: InstanceRef,
        status_count: usize,
    ) -> Result<Self> {
        if raw.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            raw,
            _instance: instance,
            status_count,
            host_completed: false,
            _resources: PhantomData,
        })
    }

    /// Takes ownership of a native future created outside this crate.
    ///
    /// # Safety
    ///
    /// `raw` must be valid, created with `instance`, and owned by nobody else.
    /// `status_count` must be the number of images in the submission.
    ///
    /// # Errors
    ///
    /// Returns an error if `raw` is null.
    pub unsafe fn from_raw_parts(
        instance: &Instance,
        raw: sys::nvimgcodecFuture_t,
        status_count: usize,
    ) -> Result<Self> {
        Self::create(raw, instance.as_instance_ref(), status_count)
    }

    /// Returns the native handle, which this future still owns.
    #[must_use]
    pub fn as_raw(&self) -> sys::nvimgcodecFuture_t {
        self.raw
    }

    /// Takes back ownership of a handle from [`into_raw`](Self::into_raw).
    ///
    /// # Safety
    ///
    /// The handle must still be valid and owned by nobody else, and everything the submission uses must still be alive.
    pub unsafe fn from_raw(mut raw: RawFuture<'a>) -> Self {
        raw.owner.raw = raw.handle;
        raw.owner
    }

    /// Releases the native handle with the values that keep it valid.
    ///
    /// # Safety
    ///
    /// Everything the submission uses must stay alive until its CPU and GPU work finish.
    /// Destroy the handle or pass it back to [`from_raw`](Self::from_raw).
    #[must_use]
    pub unsafe fn into_raw(mut self) -> RawFuture<'a> {
        let handle = self.raw;
        self.raw = ptr::null_mut();
        RawFuture {
            handle,
            owner: self,
        }
    }

    /// Waits until nvImageCodec has finished its CPU work for the submission.
    ///
    /// GPU work may still be queued on the images' CUDA stream afterwards.
    ///
    /// # Errors
    ///
    /// Returns an error if the wait fails.
    /// Failures of individual images are reported by [`processing_statuses`](Self::processing_statuses).
    pub fn wait_for_host(&mut self) -> Result<()> {
        // SAFETY: `nvimgcodecFutureWaitForAll` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::nvimgcodecFutureWaitForAll(self.raw))?;
        }

        self.host_completed = true;
        Ok(())
    }

    /// Returns the processing status of each image in the submission.
    ///
    /// # Errors
    ///
    /// Returns an error if the status count does not fit nvImageCodec's integer types or nvImageCodec fails the query.
    pub fn processing_statuses(&self) -> Result<Vec<ProcessingStatus>> {
        let mut count = to_u64(self.status_count, "processing_status_count")?;
        let mut statuses = vec![0; self.status_count];
        // SAFETY: `nvimgcodecFutureGetProcessingStatus` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::nvimgcodecFutureGetProcessingStatus(
                self.raw,
                statuses.as_mut_ptr(),
                &raw mut count,
            ))?;
        }

        statuses.truncate(to_usize(count, "processing_status_count")?);
        Ok(statuses.into_iter().map(ProcessingStatus::from).collect())
    }

    /// Waits for the CPU work and returns the statuses if every image succeeded.
    ///
    /// # Errors
    ///
    /// Returns an error if the wait or the status query fails, a status count does not fit, or an image's status is not success.
    pub fn check_host(mut self) -> Result<Vec<ProcessingStatus>> {
        self.wait_for_host()?;
        let statuses = self.processing_statuses()?;
        validate_processing_statuses(&statuses)?;
        Ok(statuses)
    }

    /// Waits for the CPU work, synchronizes `stream`, and returns the statuses if every image succeeded.
    ///
    /// Use it when the images used `stream` for GPU work and you need that work finished before reading their pixels.
    ///
    /// # Errors
    ///
    /// Returns an error if the wait, the synchronization, or the status query fails, a status count does not fit, or an image's status is not success.
    pub fn check_on(mut self, stream: &Stream) -> Result<Vec<ProcessingStatus>> {
        self.wait_for_host()?;
        stream.synchronize()?;
        let statuses = self.processing_statuses()?;
        validate_processing_statuses(&statuses)?;
        Ok(statuses)
    }
}

impl Drop for Future<'_> {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // SAFETY: `nvimgcodecFutureWaitForAll`, `nvimgcodecFutureDestroy` receive live handles owned by the wrappers.
            unsafe {
                if !self.host_completed
                    && try_ffi!(sys::nvimgcodecFutureWaitForAll(self.raw)).is_err()
                {
                    // Native destruction cannot release an uncompleted host submission.
                    process::abort();
                }
                if let Err(error) = try_ffi!(sys::nvimgcodecFutureDestroy(self.raw)) {
                    eprintln!(
                        "failed to free nvimagecodec future: {}",
                        singe_core::ErrorChain(&error)
                    );
                }
            }
        }
    }
}
