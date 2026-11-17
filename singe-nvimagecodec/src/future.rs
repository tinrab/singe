use std::marker::PhantomData;

use singe_core::{abort_after_cleanup_failure, report_cleanup_failure};
use singe_cuda::stream::Stream;
use singe_nvimagecodec_sys as sys;

use crate::{
    codec::RawBatch,
    error::{Error, Result},
    instance::InstanceRef,
    try_ffi,
    types::ProcessingStatus,
    utility::{to_u64, to_usize, validate_processing_statuses},
};

/// Submission from [`Decoder::decode_async`](crate::decode::Decoder::decode_async) or [`Encoder::encode_async`](crate::encode::Encoder::encode_async).
///
/// When the future is ready, nvImageCodec has finished its CPU work, but depending on the codec, GPU work may still be queued on the images' CUDA streams.
/// [`wait`](Self::wait) waits for both.
/// Dropping the future waits for the CPU work only.
#[derive(Debug)]
#[must_use = "call wait() to observe completion and statuses"]
pub struct Future<'a> {
    raw: sys::nvimgcodecFuture_t,
    _instance: InstanceRef,
    streams: Vec<Stream>,
    batch: RawBatch,
    host_completed: bool,
    _resources: PhantomData<&'a ()>,
}

impl Future<'_> {
    pub(crate) fn create(
        raw: sys::nvimgcodecFuture_t,
        instance: InstanceRef,
        streams: Vec<Stream>,
        batch: RawBatch,
    ) -> Result<Self> {
        if raw.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            raw,
            _instance: instance,
            streams,
            batch,
            host_completed: false,
            _resources: PhantomData,
        })
    }

    /// Returns the native handle, which this future still owns.
    #[must_use]
    pub fn as_raw(&self) -> sys::nvimgcodecFuture_t {
        self.raw
    }

    /// Waits until nvImageCodec has finished its CPU work for the submission.
    ///
    /// GPU work may still be queued on the images' CUDA streams afterwards.
    ///
    /// # Errors
    ///
    /// Returns an error if the wait fails.
    /// Failures of individual images are reported by [`processing_statuses`](Self::processing_statuses).
    pub fn wait_for_host(&mut self) -> Result<()> {
        // SAFETY: `self.raw` is the non-null future this value owns, checked in `create`, and is destroyed only in `Drop`.
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
    /// Returns an error if the status count does not fit nvImageCodec's integer types, nvImageCodec fails the query, or it reports a status count other than one per image.
    pub fn processing_statuses(&self) -> Result<Vec<ProcessingStatus>> {
        let status_count = self.batch.len();
        let mut count = to_u64(status_count, "processing_status_count")?;
        let mut statuses = vec![0; status_count];
        // SAFETY: `self.raw` is the live future this value owns, and `statuses` has room for the `count` entries passed in, one per batch item.
        unsafe {
            try_ffi!(sys::nvimgcodecFutureGetProcessingStatus(
                self.raw,
                statuses.as_mut_ptr(),
                &raw mut count,
            ))?;
        }

        let actual = to_usize(count, "processing_status_count")?;
        if actual != status_count {
            return Err(Error::LengthMismatch {
                name: "processing_status_count".into(),
                expected: status_count,
                actual,
            });
        }

        Ok(statuses.into_iter().map(ProcessingStatus::from).collect())
    }

    /// Waits for the CPU work and the images' CUDA streams, then checks that every image succeeded.
    ///
    /// # Errors
    ///
    /// Returns an error if a wait or the status query fails, a status count does not fit or is not one per image, or an image's status is not success.
    pub fn wait(mut self) -> Result<()> {
        self.wait_for_host()?;
        for stream in &self.streams {
            stream.synchronize()?;
        }

        validate_processing_statuses(&self.processing_statuses()?)
    }
}

impl Drop for Future<'_> {
    fn drop(&mut self) {
        if self.raw.is_null() {
            return;
        }

        if !self.host_completed
            // SAFETY: `self.raw` is the non-null future this value owns, not yet destroyed.
            && let Err(error) = unsafe { try_ffi!(sys::nvimgcodecFutureWaitForAll(self.raw)) }
        {
            // Native destruction cannot release an uncompleted host submission.
            abort_after_cleanup_failure("wait for", "nvimagecodec future", &error);
        }

        // SAFETY: `self.raw` is the future this value owns, its host work has finished, and `Drop` runs once, so it is destroyed once.
        if let Err(error) = unsafe { try_ffi!(sys::nvimgcodecFutureDestroy(self.raw)) } {
            report_cleanup_failure("destroy", "nvimagecodec future", &error);
        }
    }
}
