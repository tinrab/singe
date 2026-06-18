use std::{ptr, time::Duration};

use singe_cufile_sys as sys;

use crate::{
    buffer::{IoBuffer, IoBufferMut},
    error::{Error, Result},
    file::{FileHandle, checked_io_size, operation_result},
    try_ffi,
    types::{BatchStatus, Operation},
    utility::{to_i64, to_u32, to_u64},
};

#[derive(Debug)]
pub struct Batch {
    handle: sys::CUfileBatchHandle_t,
    capacity: u32,
}

// cuFile batch handles own submission state. The owner may move between
// threads, while status polling and submission stay tied to Rust borrows of the
// batch and request buffers.
unsafe impl Send for Batch {}

impl Batch {
    pub fn create(capacity: usize) -> Result<Self> {
        let capacity = to_u32(capacity, "capacity")?;
        let mut handle = ptr::null_mut();
        unsafe {
            try_ffi!(sys::cuFileBatchIOSetUp(&raw mut handle, capacity))?;
        }
        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self { handle, capacity })
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    pub fn submit<'batch, 'request>(
        &'batch self,
        requests: Vec<BatchRequest<'request>>,
    ) -> Result<BatchSubmission<'batch, 'request>> {
        let count = u32::try_from(requests.len()).map_err(|_| Error::RequestBatchTooBig)?;
        if count > self.capacity {
            return Err(Error::RequestBatchTooBig);
        }

        let mut raw_requests: Vec<_> = requests.iter().map(|request| request.handle).collect();
        unsafe {
            try_ffi!(sys::cuFileBatchIOSubmit(
                self.handle,
                count,
                raw_requests.as_mut_ptr(),
                0,
            ))?;
        }

        Ok(BatchSubmission {
            batch: self,
            requests,
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
        let timeout_ptr = timeout
            .as_mut()
            .map_or(ptr::null_mut(), |timeout| timeout as *mut _);

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
        unsafe {
            try_ffi!(sys::cuFileBatchIOCancel(self.handle))?;
        }
        Ok(())
    }
}

#[must_use = "batch submissions must be completed with `wait` or cancelled with `cancel`"]
pub struct BatchSubmission<'batch, 'request> {
    batch: &'batch Batch,
    requests: Vec<BatchRequest<'request>>,
    completed: usize,
}

impl BatchSubmission<'_, '_> {
    pub fn completed_count(&self) -> usize {
        self.completed
    }

    pub fn request_count(&self) -> usize {
        self.requests.len()
    }

    pub fn remaining_count(&self) -> usize {
        self.requests.len().saturating_sub(self.completed)
    }

    pub fn is_complete(&self) -> bool {
        self.remaining_count() == 0
    }

    pub fn status(
        &mut self,
        min_count: u32,
        max_count: u32,
        timeout: Option<Duration>,
    ) -> Result<Vec<BatchEvent>> {
        let remaining = self.remaining_count() as u32;
        let max_count = max_count.min(remaining);
        if max_count == 0 {
            return Ok(Vec::new());
        }

        let events = self.batch.status(min_count, max_count, timeout)?;
        self.completed += events.len();
        Ok(events)
    }

    pub fn wait(mut self) -> Result<Vec<BatchEvent>> {
        let mut events = Vec::with_capacity(self.requests.len());
        while !self.is_complete() {
            events.extend(self.status(1, self.remaining_count() as u32, None)?);
        }
        Ok(events)
    }

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

        if let Err(err) = self.batch.cancel() {
            #[cfg(debug_assertions)]
            {
                let remaining = self.remaining_count();
                eprintln!(
                    "failed to cancel cufile batch submission with {remaining} pending requests: {err}"
                );
            }
        }

        while !self.is_complete() {
            match self.status(1, self.remaining_count() as u32, None) {
                Ok(events) if !events.is_empty() => {}
                Ok(_) => {
                    #[cfg(debug_assertions)]
                    {
                        let remaining = self.remaining_count();
                        eprintln!(
                            "cufile batch status made no progress with {remaining} pending requests after cancellation"
                        );
                    }
                    std::process::abort();
                }
                Err(err) => {
                    #[cfg(debug_assertions)]
                    {
                        let remaining = self.remaining_count();
                        eprintln!(
                            "failed to drain cancelled cufile batch submission with {remaining} pending requests: {err}"
                        );
                    }
                    std::process::abort();
                }
            }
        }
    }
}

impl Drop for Batch {
    fn drop(&mut self) {
        unsafe {
            if !self.handle.is_null() {
                sys::cuFileBatchIODestroy(self.handle);
            }
            self.handle = ptr::null_mut();
        }
    }
}

pub struct BatchRequest<'a> {
    handle: sys::CUfileIOParams_t,
    _borrows: std::marker::PhantomData<&'a mut ()>,
}

impl<'a> BatchRequest<'a> {
    pub fn read<B>(
        file: &'a FileHandle,
        buffer: &'a mut B,
        size: usize,
        file_offset: u64,
        buffer_offset: u64,
        cookie: BatchCookie,
    ) -> Result<Self>
    where
        B: IoBufferMut + ?Sized,
    {
        Self::create(
            Operation::Read,
            file,
            buffer.ptr_mut().as_mut() as _,
            buffer.byte_len()?,
            size,
            file_offset,
            buffer_offset,
            cookie,
        )
    }

    pub fn write<B>(
        file: &'a FileHandle,
        buffer: &'a B,
        size: usize,
        file_offset: u64,
        buffer_offset: u64,
        cookie: BatchCookie,
    ) -> Result<Self>
    where
        B: IoBuffer + ?Sized,
    {
        Self::create(
            Operation::Write,
            file,
            buffer.ptr().as_mut() as _,
            buffer.byte_len()?,
            size,
            file_offset,
            buffer_offset,
            cookie,
        )
    }

    fn create(
        operation: Operation,
        file: &'a FileHandle,
        buffer: *mut std::ffi::c_void,
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
                cookie: cookie.into_raw() as _,
            },
            _borrows: std::marker::PhantomData,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BatchEvent {
    pub cookie: BatchCookie,
    pub status: BatchStatus,
    raw_result: i64,
}

impl BatchEvent {
    pub fn bytes_transferred(self) -> Result<usize> {
        operation_result(self.raw_result)
    }

    pub fn raw_result(self) -> i64 {
        self.raw_result
    }
}

impl From<sys::CUfileIOEvents_t> for BatchEvent {
    fn from(value: sys::CUfileIOEvents_t) -> Self {
        Self {
            cookie: BatchCookie::from_raw(value.cookie as _),
            status: value.status.into(),
            raw_result: value.ret as i64,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BatchCookie {
    handle: *mut (),
}

impl BatchCookie {
    pub const NONE: Self = Self {
        handle: ptr::null_mut(),
    };

    pub const fn new(value: usize) -> Self {
        Self {
            handle: value as *mut (),
        }
    }

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
        tv_sec: duration.as_secs() as sys::__time_t,
        tv_nsec: duration.subsec_nanos() as sys::__syscall_slong_t,
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
    fn batch_event_preserves_negative_results() {
        let raw_error = -(sys::CUfileOpError::CU_FILE_INVALID_VALUE as i64);
        let event = BatchEvent::from(sys::CUfileIOEvents_t {
            cookie: ptr::null_mut(),
            status: sys::CUFILEStatus_enum::CUFILE_FAILED,
            ret: raw_error as sys::size_t,
        });

        assert_eq!(event.status, BatchStatus::Failed);
        assert_eq!(event.raw_result(), raw_error);
        assert!(matches!(
            event.bytes_transferred(),
            Err(Error::Cufile {
                code: Status::InvalidValue,
                ..
            })
        ));
    }
}
