use std::{
    ffi,
    fmt::{self, Debug, Formatter},
    marker::PhantomData,
    mem::ManuallyDrop,
    process, ptr,
    sync::Arc,
    time::Duration,
};

use singe_cuda::{context::Context, error::Error as CudaError, view::DeviceRegion};
use singe_cufile_sys as sys;

use crate::{
    buffer::{IoInput, IoOutput},
    driver::Driver,
    error::{Error, Result},
    file::{FileHandle, checked_io_size, io_pointer, operation_result, prepare_buffer},
    try_ffi,
    types::{BatchStatus, Operation},
    utility::{to_i64, to_u32, to_u64},
};

/// An owned cuFile batch queue retaining its driver session.
#[derive(Debug)]
pub struct Batch {
    handle: sys::CUfileBatchHandle_t,
    capacity: u32,
    driver: Arc<Driver>,
}

// SAFETY: cuFile batch handles own submission state. The owner may move between
// threads, while status polling and submission stay tied to Rust borrows of the
// batch and request buffers.
unsafe impl Send for Batch {}

/// Transferred native ownership of a batch handle and the driver session it belongs to.
#[derive(Debug)]
pub struct RawBatch {
    /// The owned native batch handle.
    pub handle: sys::CUfileBatchHandle_t,
    /// Maximum number of requests accepted by the native batch.
    pub capacity: u32,
    /// The driver session that owns the batch handle.
    pub driver: Arc<Driver>,
}

impl Batch {
    /// Takes ownership of a native batch handle; dropping the batch destroys it.
    ///
    /// # Safety
    /// `raw.handle` must be a live batch handle created by `cuFileBatchIOSetUp` with
    /// `raw.capacity` entries in the session kept open by `raw.driver`, with no other owner.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NullHandle`] if `raw.handle` is null.
    pub unsafe fn from_raw(raw: RawBatch) -> Result<Self> {
        if raw.handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            handle: raw.handle,
            capacity: raw.capacity,
            driver: raw.driver,
        })
    }

    /// Returns the borrowed native batch handle.
    #[must_use]
    pub fn as_raw(&self) -> sys::CUfileBatchHandle_t {
        self.handle
    }

    /// Releases ownership without destroying the batch handle.
    #[must_use]
    pub fn into_raw(self) -> RawBatch {
        let this = ManuallyDrop::new(self);
        RawBatch {
            handle: this.handle,
            capacity: this.capacity,
            // SAFETY: `this` is never dropped, so the driver reference moves out exactly once.
            driver: unsafe { ptr::read(&raw const this.driver) },
        }
    }

    /// Creates a native batch queue for at most `capacity` requests.
    ///
    /// # Errors
    ///
    /// Returns an error if `capacity` is zero or exceeds the `u32` range, cuFile cannot create the batch, or the returned handle is null.
    pub fn create(driver: &Arc<Driver>, capacity: usize) -> Result<Self> {
        let capacity = to_u32(capacity, "capacity")?;
        if capacity == 0 {
            return Err(Error::RequestBatchTooBig);
        }
        let mut handle = ptr::null_mut();
        // SAFETY: `cuFileBatchIOSetUp` receives valid output locations.
        unsafe {
            try_ffi!(sys::cuFileBatchIOSetUp(&raw mut handle, capacity))?;
        }
        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            handle,
            capacity,
            driver: Arc::clone(driver),
        })
    }

    /// Returns the maximum number of requests per submission.
    #[must_use]
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Completes every borrowed request before returning its native completion events.
    ///
    /// # Errors
    ///
    /// Returns an error if the request count exceeds the batch capacity, CUDA-associated requests use different contexts, or readiness synchronization or context binding fails.
    /// Also returns an error if cuFile rejects the batch submission.
    /// Errors while polling for completion are propagated.
    /// Inspect each returned event with [`BatchEvent::bytes_transferred`] for request-specific failures.
    pub fn execute(&mut self, requests: Vec<BatchRequest<'_>>) -> Result<Vec<BatchEvent>> {
        // SAFETY: request builders establish representation and lifetime contracts.
        // The local submission cannot escape or be forgotten before wait drains it.
        unsafe { self.submit(requests) }?.wait()
    }

    /// Enqueues borrowed requests; completion is established by wait or native status polling.
    ///
    /// # Safety
    /// Retain every request buffer and prevent conflicting access until terminal native completion.
    /// This obligation includes forgotten submissions.
    ///
    /// # Errors
    ///
    /// Returns an error if the request count exceeds the batch capacity, CUDA-associated requests use different contexts, or readiness synchronization or context binding fails.
    /// Also returns an error if cuFile rejects the batch submission.
    pub unsafe fn submit<'batch, 'request>(
        &'batch mut self,
        requests: Vec<BatchRequest<'request>>,
    ) -> Result<BatchSubmission<'batch, 'request>> {
        let count = u32::try_from(requests.len()).map_err(|_| Error::RequestBatchTooBig)?;
        if count > self.capacity {
            return Err(Error::RequestBatchTooBig);
        }

        let mut context: Option<&Arc<Context>> = None;
        for request in &requests {
            if let Some(current) = &request.context {
                if let Some(context) = context
                    && context != current
                {
                    return Err(CudaError::MemoryContextMismatch.into());
                }
                context = Some(current);
            }
            if let Some(region) = request.region {
                region.synchronize()?;
            }
        }
        if let Some(context) = context {
            context.bind()?;
        }
        if requests.is_empty() {
            return Ok(BatchSubmission {
                batch: self,
                requests,
                _raw_requests: Vec::new(),
                completed: 0,
            });
        }
        let mut raw_requests: Vec<_> = requests.iter().map(|request| request.handle).collect();
        // cuFile documents unsuccessful API calls as having no application or filesystem side effects.
        // SAFETY: `cuFileBatchIOSubmit` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::cuFileBatchIOSubmit(
                self.handle,
                count,
                raw_requests.as_mut_ptr(),
                0
            ))?;
        }

        Ok(BatchSubmission {
            batch: self,
            requests,
            _raw_requests: raw_requests,
            completed: 0,
        })
    }

    fn status(
        &self,
        min_count: u32,
        max_count: u32,
        timeout: Option<Duration>,
    ) -> Result<Vec<BatchEvent>> {
        let mut count = max_count;
        let mut events = vec![sys::CUfileIOEvents_t::default(); count as usize];
        let mut timeout = timeout.map(duration_to_timespec);
        let timeout_ptr = timeout.as_mut().map_or(ptr::null_mut(), ptr::from_mut);

        // SAFETY: `cuFileBatchIOGetStatus` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::cuFileBatchIOGetStatus(
                self.handle,
                min_count,
                &raw mut count,
                events.as_mut_ptr(),
                timeout_ptr,
            ))?;
        }
        events.truncate(count as usize);
        Ok(events.into_iter().map(BatchEvent::from).collect())
    }

    fn cancel(&self) -> Result<()> {
        // SAFETY: `cuFileBatchIOCancel` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::cuFileBatchIOCancel(self.handle))?;
        }
        Ok(())
    }
}

/// A pending batch retaining its borrowed requests until all native events are drained.
///
/// Dropping an incomplete submission cancels and drains it before releasing the borrows.
#[must_use = "batch submissions must be completed with `wait` or cancelled with `cancel`"]
pub struct BatchSubmission<'batch, 'request> {
    batch: &'batch Batch,
    requests: Vec<BatchRequest<'request>>,
    _raw_requests: Vec<sys::CUfileIOParams_t>,
    completed: usize,
}

impl Debug for BatchSubmission<'_, '_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("BatchSubmission").finish_non_exhaustive()
    }
}

impl BatchSubmission<'_, '_> {
    /// Returns the number of completion events collected so far.
    #[must_use]
    pub fn completed_count(&self) -> usize {
        self.completed
    }

    /// Returns the number of submitted requests.
    #[must_use]
    pub fn request_count(&self) -> usize {
        self.requests.len()
    }

    /// Returns the number of requests whose completion events have not been collected.
    #[must_use]
    pub fn remaining_count(&self) -> usize {
        self.requests.len().saturating_sub(self.completed)
    }

    /// Returns whether all completion events have been collected.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.remaining_count() == 0
    }

    /// Collects up to `max_count` completion events, waiting for at least `min_count`.
    ///
    /// The maximum is limited to the number of remaining requests.
    /// `timeout` is passed to the native status poll; `None` passes a null timeout pointer.
    ///
    /// # Errors
    ///
    /// Returns an error if `min_count` exceeds the effective maximum for a nonempty poll or cuFile cannot retrieve completion events.
    ///
    /// # Panics
    ///
    /// Panics if the internal remaining request count exceeds the `u32` batch-capacity range.
    pub fn status(
        &mut self,
        min_count: u32,
        max_count: u32,
        timeout: Option<Duration>,
    ) -> Result<Vec<BatchEvent>> {
        let remaining = u32::try_from(self.remaining_count())
            .expect("remaining requests are bounded by the u32 batch capacity");
        let max_count = max_count.min(remaining);
        if max_count == 0 {
            return Ok(Vec::new());
        }

        if min_count > max_count {
            return Err(Error::out_of_range("min_count"));
        }
        let events = self.batch.status(min_count, max_count, timeout)?;
        self.completed += events.len();
        Ok(events)
    }

    /// Waits for every remaining request and returns its completion events.
    ///
    /// Events collected by earlier calls to [`Self::status`] are not returned again.
    ///
    /// # Errors
    ///
    /// Returns an error if polling cuFile completion events fails.
    /// Individual request failures remain in the returned [`BatchEvent`] values.
    ///
    /// # Panics
    ///
    /// Panics if the internal remaining request count exceeds the `u32` batch-capacity range.
    pub fn wait(mut self) -> Result<Vec<BatchEvent>> {
        let mut events = Vec::with_capacity(self.requests.len());
        while !self.is_complete() {
            events.extend(
                self.status(
                    1,
                    u32::try_from(self.remaining_count())
                        .expect("remaining requests are bounded by the u32 batch capacity"),
                    None,
                )?,
            );
        }
        Ok(events)
    }

    /// Requests cancellation and drains the remaining completion events.
    ///
    /// # Errors
    ///
    /// Returns an error if cuFile rejects cancellation or polling completion events fails.
    pub fn cancel(self) -> Result<Vec<BatchEvent>> {
        self.batch.cancel()?;
        self.wait()
    }
}

impl Drop for BatchSubmission<'_, '_> {
    fn drop(&mut self) {
        if self.is_complete() {
            return;
        }

        if let Err(error) = self.batch.cancel() {
            #[cfg(debug_assertions)]
            {
                let remaining = self.remaining_count();
                eprintln!(
                    "failed to cancel cufile batch submission with {remaining} pending requests: {error}"
                );
            }
        }

        while !self.is_complete() {
            match self.status(
                1,
                u32::try_from(self.remaining_count())
                    .expect("remaining requests are bounded by the u32 batch capacity"),
                None,
            ) {
                Ok(events) if !events.is_empty() => {}
                Ok(_) => {
                    #[cfg(debug_assertions)]
                    {
                        let remaining = self.remaining_count();
                        eprintln!(
                            "cufile batch status made no progress with {remaining} pending requests after cancellation"
                        );
                    }
                    process::abort();
                }
                Err(error) => {
                    #[cfg(debug_assertions)]
                    {
                        let remaining = self.remaining_count();
                        eprintln!(
                            "failed to drain cancelled cufile batch submission with {remaining} pending requests: {error}"
                        );
                    }
                    process::abort();
                }
            }
        }
    }
}

impl Drop for Batch {
    fn drop(&mut self) {
        // SAFETY: `cuFileBatchIODestroy` receives live handles owned by the wrappers.
        unsafe {
            if !self.handle.is_null() {
                sys::cuFileBatchIODestroy(self.handle);
            }
            self.handle = ptr::null_mut();
        }
    }
}

/// A file I/O request retaining its file registration and buffer borrow.
pub struct BatchRequest<'a> {
    handle: sys::CUfileIOParams_t,
    _borrows: PhantomData<&'a mut ()>,
    _file: FileHandle,
    context: Option<Arc<Context>>,
    region: Option<DeviceRegion<'a>>,
}

impl Debug for BatchRequest<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("BatchRequest").finish_non_exhaustive()
    }
}

impl<'a> BatchRequest<'a> {
    /// Builds a read request targeting an exclusively borrowed buffer.
    ///
    /// `size`, `file_offset`, and `buffer_offset` are in bytes; `cookie` is returned with the completion event.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer length cannot be computed, the requested byte range exceeds the buffer, or a size or offset exceeds its native integer range.
    /// Also returns an error if CUDA readiness, capture checks, or context binding fails.
    pub fn read<B>(
        file: &FileHandle,
        buffer: &'a mut B,
        size: usize,
        file_offset: u64,
        buffer_offset: u64,
        cookie: BatchCookie,
    ) -> Result<Self>
    where
        B: IoOutput + ?Sized,
    {
        prepare_buffer(buffer, None)?;
        let pointer = buffer.ptr_mut();
        let (pointer, native_offset) = io_pointer(buffer, pointer, size, buffer_offset)?;
        let region = buffer.device_region();
        let context = buffer.context().cloned();
        let mut request = Self::create(
            Operation::Read,
            file,
            pointer.cast(),
            if buffer.is_registered() {
                buffer.byte_len()?
            } else {
                size
            },
            size,
            file_offset,
            native_offset as u64,
            cookie,
        )?;
        request.region = region;
        request.context = context;
        Ok(request)
    }

    /// Builds a write request from initialized buffer bytes.
    ///
    /// `size`, `file_offset`, and `buffer_offset` are in bytes; `cookie` is returned with the completion event.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer length cannot be computed, the requested byte range exceeds the buffer, or a size or offset exceeds its native integer range.
    /// Also returns an error if CUDA readiness, capture checks, or context binding fails.
    pub fn write<B>(
        file: &FileHandle,
        buffer: &'a B,
        size: usize,
        file_offset: u64,
        buffer_offset: u64,
        cookie: BatchCookie,
    ) -> Result<Self>
    where
        B: IoInput + ?Sized,
    {
        prepare_buffer(buffer, None)?;
        let (pointer, native_offset) = io_pointer(buffer, buffer.ptr(), size, buffer_offset)?;
        let region = buffer.device_region();
        let context = buffer.context().cloned();
        let mut request = Self::create(
            Operation::Write,
            file,
            pointer.cast(),
            if buffer.is_registered() {
                buffer.byte_len()?
            } else {
                size
            },
            size,
            file_offset,
            native_offset as u64,
            cookie,
        )?;
        request.region = region;
        request.context = context;
        Ok(request)
    }

    fn create(
        operation: Operation,
        file: &FileHandle,
        buffer: *mut ffi::c_void,
        buffer_size: usize,
        size: usize,
        file_offset: u64,
        buffer_offset: u64,
        cookie: BatchCookie,
    ) -> Result<Self> {
        let size = checked_io_size(buffer_size, size, buffer_offset)?;
        Ok(Self {
            handle: sys::CUfileIOParams_t {
                mode: sys::cufileBatchMode::CUFILE_BATCH,
                u: sys::CUfileIOParams__bindgen_ty_1 {
                    batch: sys::CUfileIOParams__bindgen_ty_1__bindgen_ty_1 {
                        devPtr_base: buffer,
                        file_offset: to_i64(file_offset, "file_offset")?,
                        devPtr_offset: to_i64(buffer_offset, "buffer_offset")?,
                        size: to_u64(size, "size")?,
                    },
                },
                fh: file.as_raw(),
                opcode: operation.into(),
                cookie: cookie.into_raw().cast(),
            },
            _borrows: PhantomData,
            _file: file.clone(),
            context: None,
            region: None,
        })
    }
}

/// The status, caller cookie, and available result of one batch request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BatchEvent {
    /// The cookie supplied when the request was built.
    pub cookie: BatchCookie,
    /// The native request status reported by cuFile.
    pub status: BatchStatus,
    raw_result: Option<i64>,
}

impl BatchEvent {
    /// Decodes the successful byte count or the request failure.
    ///
    /// # Errors
    ///
    /// Returns a cuFile or operating-system error for a negative native result.
    /// Returns [`Error::BatchOperationFailed`] when no successful byte count is available for the event status, or an error if the result cannot be represented as `usize`.
    pub fn bytes_transferred(self) -> Result<usize> {
        match self.raw_result {
            Some(value) if self.status == BatchStatus::Complete || value < 0 => {
                operation_result(value)
            }
            Some(_) | None => Err(Error::BatchOperationFailed(self.status)),
        }
    }

    /// Returns the native signed result for complete or failed requests.
    ///
    /// Other statuses have no result.
    #[must_use]
    pub fn raw_result(self) -> Option<i64> {
        self.raw_result
    }
}

impl From<sys::CUfileIOEvents_t> for BatchEvent {
    fn from(value: sys::CUfileIOEvents_t) -> Self {
        Self {
            cookie: BatchCookie::from_raw(value.cookie.cast()),
            status: value.status.into(),
            raw_result: if matches!(
                value.status,
                sys::CUFILEStatus_enum::CUFILE_COMPLETE | sys::CUFILEStatus_enum::CUFILE_FAILED
            ) {
                Some(value.ret.cast_signed())
            } else {
                None
            },
        }
    }
}

/// An opaque integer-sized request identifier returned unchanged in completion events.
///
/// The native API carries it as a pointer value; this wrapper never dereferences it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BatchCookie {
    handle: *mut (),
}

impl BatchCookie {
    /// The null cookie, with integer value zero.
    pub const NONE: Self = Self {
        handle: ptr::null_mut(),
    };

    /// Encodes `value` as a request cookie.
    #[must_use]
    pub const fn new(value: usize) -> Self {
        Self {
            handle: value as *mut (),
        }
    }

    /// Returns the integer value encoded in this cookie.
    #[must_use]
    pub fn get(self) -> usize {
        self.handle as usize
    }

    pub(crate) const fn from_raw(raw: *mut ()) -> Self {
        Self { handle: raw }
    }

    pub(crate) const fn into_raw(self) -> *mut () {
        self.handle
    }
}

fn duration_to_timespec(duration: Duration) -> sys::timespec {
    sys::timespec {
        tv_sec: sys::__time_t::try_from(duration.as_secs()).unwrap_or(sys::__time_t::MAX),
        tv_nsec: sys::__syscall_slong_t::from(duration.subsec_nanos()),
    }
}

#[cfg(test)]
mod tests {
    use std::ptr;

    use crate::{
        batch::BatchEvent,
        error::{Error, Status},
        sys,
        types::BatchStatus,
    };

    #[test]
    fn test_batch_event_preserves_negative_results() {
        let raw_error = -(sys::CUfileOpError::CU_FILE_INVALID_VALUE as i64);
        let event = BatchEvent::from(sys::CUfileIOEvents_t {
            cookie: ptr::null_mut(),
            status: sys::CUFILEStatus_enum::CUFILE_FAILED,
            ret: raw_error as sys::size_t,
        });

        assert_eq!(event.status, BatchStatus::Failed);
        assert_eq!(event.raw_result(), Some(raw_error));
        assert!(matches!(
            event.bytes_transferred(),
            Err(Error::Cufile {
                code: Status::InvalidValue,
                ..
            })
        ));
    }
}
