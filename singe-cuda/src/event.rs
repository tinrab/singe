use std::{
    cmp::Ordering,
    fmt::{self, Display, Formatter},
    mem::ManuallyDrop,
    ptr,
    sync::Arc,
};

use bitflags::parser;
use singe_cuda_sys::{driver, runtime};

use crate::{
    context::Context,
    error::{Error, Result},
    stream::{BorrowedStream, Stream, StreamBinding},
    try_ffi,
};

bitflags::bitflags! {
    /// Flags for CUDA event creation ([`Context::create_event_with_flags`]).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct EventFlags: u32 {
        /// Default event flag.
        const DEFAULT = driver::CUevent_flags::CU_EVENT_DEFAULT as _;
        /// Event uses blocking synchronization.
        const BLOCKING_SYNC = driver::CUevent_flags::CU_EVENT_BLOCKING_SYNC as _;
        /// Event will not record timing data.
        const DISABLE_TIMING = driver::CUevent_flags::CU_EVENT_DISABLE_TIMING as _;
        /// Event is suitable for interprocess use.
        ///
        /// `CUevent_flags::CU_EVENT_DISABLE_TIMING` must be set.
        const INTERPROCESS = driver::CUevent_flags::CU_EVENT_INTERPROCESS as _;
    }
}

impl Display for EventFlags {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        parser::to_writer(self, f)
    }
}

bitflags::bitflags! {
    /// Flags controlling event recording and stream capture in [`Event::record`].
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct EventRecordFlags: u32 {
        /// Record normally, using internal event dependencies during stream capture.
        const DEFAULT = runtime::cudaEventRecordDefault;
        /// Create an external event-record node during stream capture.
        const EXTERNAL = runtime::cudaEventRecordExternal;
    }
}

impl Display for EventRecordFlags {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        parser::to_writer(self, f)
    }
}

/// A CUDA event and its retained context.
///
/// Recording captures preceding work on a stream; querying, waiting, and elapsed-time calculations refer to the event's most recent recording. Dropping the wrapper destroys its native event handle.
#[derive(Debug)]
pub struct Event {
    handle: runtime::cudaEvent_t,
    context: Arc<Context>,
}

impl PartialEq for Event {
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle && Arc::ptr_eq(&self.context, &other.context)
    }
}

impl Eq for Event {}

impl PartialOrd for Event {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Event {
    fn cmp(&self, other: &Self) -> Ordering {
        let lhs = (Arc::as_ptr(&self.context) as usize, self.handle as usize);
        let rhs = (Arc::as_ptr(&other.context) as usize, other.handle as usize);
        lhs.cmp(&rhs)
    }
}

impl Event {
    /// Wraps an existing CUDA event handle.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid CUDA event owned by `context`, and ownership of the
    /// handle is transferred to the returned [`Event`]. The handle must not be
    /// destroyed elsewhere after calling this function.
    ///
    /// # Errors
    ///
    /// Returns `Error::NullHandle` if `handle` is null. The caller must satisfy the remaining handle validity and ownership requirements.
    pub unsafe fn from_raw(handle: runtime::cudaEvent_t, context: Arc<Context>) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self { handle, context })
    }

    /// Records the work preceding this call on the supplied stream.
    ///
    /// Re-recording replaces the event's captured work. `flags` controls whether capture treats this as an external event record.
    ///
    /// # Errors
    ///
    /// Returns `Error::StreamContextMismatch` if the stream and event belong to different contexts. Also returns an error if the event context cannot be bound, CUDA rejects the recording or flags, or previous asynchronous work reports an error.
    pub fn record(&self, stream: &Stream, flags: EventRecordFlags) -> Result<()> {
        if stream.context() != self.context() {
            return Err(Error::StreamContextMismatch);
        }
        self.record_raw(stream.as_raw(), flags)
    }

    /// Records the work preceding this call on the supplied borrowed stream.
    ///
    /// Re-recording replaces the event's captured work. `flags` controls whether capture treats this as an external event record.
    ///
    /// # Errors
    ///
    /// Returns `Error::StreamContextMismatch` if the stream and event belong to different contexts. Also returns an error if the event context cannot be bound, CUDA rejects the recording or flags, or previous asynchronous work reports an error.
    pub fn record_borrowed(&self, stream: &BorrowedStream, flags: EventRecordFlags) -> Result<()> {
        if stream.context() != self.context() {
            return Err(Error::StreamContextMismatch);
        }
        self.record_raw(stream.as_raw(), flags)
    }

    /// Records the work preceding this call on the supplied stream binding.
    ///
    /// Re-recording replaces the event's captured work. `flags` controls whether capture treats this as an external event record.
    ///
    /// # Errors
    ///
    /// Returns `Error::StreamContextMismatch` if the stream and event belong to different contexts. Also returns an error if the event context cannot be bound, CUDA rejects the recording or flags, or previous asynchronous work reports an error.
    pub fn record_on(&self, stream: &StreamBinding, flags: EventRecordFlags) -> Result<()> {
        if stream.context() != self.context() {
            return Err(Error::StreamContextMismatch);
        }
        self.record_raw(stream.as_raw(), flags)
    }

    fn record_raw(&self, stream: runtime::cudaStream_t, flags: EventRecordFlags) -> Result<()> {
        self.context.bind()?;
        // SAFETY: `cudaEventRecordWithFlags` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(runtime::cudaEventRecordWithFlags(
                self.as_raw(),
                stream,
                flags.bits(),
            ))?;
        }
        Ok(())
    }

    /// Queries the status of all work currently captured by event.
    /// See [`sys::cudaEventRecord`](singe_cuda_sys::runtime::cudaEventRecord) for details on what is captured by an event.
    ///
    /// Returns `true` if all captured work has been completed, or `false` if any captured work is incomplete.
    ///
    /// For the purposes of Unified Memory, a return value of `true` is equivalent to having called [`Event::synchronize`].
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot query the event. CUDA may also report
    /// errors from previous asynchronous launches, internal runtime
    /// initialization errors such as [`crate::error::Status::NotInitialized`],
    /// [`crate::error::Status::CallRequiresNewerDriver`], or [`crate::error::Status::NoDevice`], and
    /// callback diagnostics such as [`crate::error::Status::NotPermitted`].
    pub fn query(&self) -> Result<bool> {
        self.context.bind()?;
        // SAFETY: `cudaEventQuery` receives live handles owned by the wrappers.
        let error = unsafe { runtime::cudaEventQuery(self.as_raw()) };
        match error {
            runtime::cudaError_t::CUDA_SUCCESS => Ok(true),
            runtime::cudaError_t::CUDA_ERROR_NOT_READY => Ok(false),
            _ => Err(error.into()),
        }
    }

    /// Waits until the completion of all work currently captured in event.
    /// See [`sys::cudaEventRecord`](singe_cuda_sys::runtime::cudaEventRecord) for details on what is captured by an event.
    ///
    /// Waiting for an event created with [`EventFlags::BLOCKING_SYNC`] causes the calling CPU thread to block until the event has been completed by the device.
    /// Without [`EventFlags::BLOCKING_SYNC`], the CPU thread will busy-wait until the event has been completed by the device.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot wait for the event. CUDA may also report
    /// errors from previous asynchronous launches, internal runtime
    /// initialization errors such as [`crate::error::Status::NotInitialized`],
    /// [`crate::error::Status::CallRequiresNewerDriver`], or [`crate::error::Status::NoDevice`], and
    /// callback diagnostics such as [`crate::error::Status::NotPermitted`].
    pub fn synchronize(&self) -> Result<()> {
        self.context.bind()?;
        // SAFETY: `cudaEventSynchronize` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(runtime::cudaEventSynchronize(self.as_raw()))?;
        }
        Ok(())
    }

    /// Computes the elapsed time between two events (in milliseconds with a resolution of around 0.5 microseconds).
    /// Note this call is not guaranteed to return the latest errors for pending work.
    /// Use it only for elapsed-time calculation; poll for completion on the events to be compared with [`Event::query`] instead.
    ///
    /// If either event was last recorded in a non-default stream, the resulting time may be greater than expected, even if both used the same stream handle.
    /// This happens because the [`sys::cudaEventRecord`](singe_cuda_sys::runtime::cudaEventRecord) operation takes place asynchronously and there is no guarantee that the measured latency is actually just between the two events.
    /// Any number of other different stream operations could execute in between the two measured events, thus altering the timing in a significant way.
    ///
    /// If [`sys::cudaEventRecord`](singe_cuda_sys::runtime::cudaEventRecord) has not been called on either event, then [`crate::error::Status::InvalidHandle`] is returned.
    /// If [`sys::cudaEventRecord`](singe_cuda_sys::runtime::cudaEventRecord) has been called on both events but one or both of them has not yet been completed (that is, [`Event::query`] would return [`crate::error::Status::NotReady`] on at least one of the events), [`crate::error::Status::NotReady`] is returned.
    /// If either event was created with [`EventFlags::DISABLE_TIMING`], this returns [`crate::error::Status::InvalidHandle`].
    ///
    /// # Errors
    ///
    /// Returns an error if the events belong to different contexts, either
    /// event has not been recorded, either event is incomplete, timing was
    /// disabled on either event, or CUDA rejects the elapsed-time query. CUDA
    /// may also report errors from previous asynchronous launches, internal
    /// runtime initialization errors such as [`crate::error::Status::NotInitialized`],
    /// [`crate::error::Status::CallRequiresNewerDriver`], or [`crate::error::Status::NoDevice`], and
    /// callback diagnostics such as [`crate::error::Status::NotPermitted`].
    pub fn elapsed_time_since(&self, start: &Self) -> Result<f32> {
        if self.context() != start.context() {
            return Err(runtime::cudaError_t::CUDA_ERROR_INVALID_CONTEXT.into());
        }

        self.context.bind()?;
        let mut milliseconds = 0.0f32;
        // SAFETY: `cudaEventElapsedTime` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(runtime::cudaEventElapsedTime(
                &raw mut milliseconds,
                start.as_raw(),
                self.as_raw(),
            ))?;
        }
        Ok(milliseconds)
    }

    /// Returns the context retained by this event.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        &self.context
    }

    /// Returns the borrowed native event handle without transferring ownership.
    #[must_use]
    pub const fn as_raw(&self) -> runtime::cudaEvent_t {
        self.handle
    }

    /// Consumes the event and returns the raw CUDA event handle without
    /// destroying it.
    ///
    /// The caller becomes responsible for eventually destroying the returned
    /// handle with CUDA.
    #[must_use]
    pub fn into_raw(self) -> runtime::cudaEvent_t {
        let event = ManuallyDrop::new(self);
        event.handle
    }
}

// SAFETY: CUDA events are synchronization handles. Recording/waiting uses CUDA's event
// semantics and does not mutate Rust-owned state through shared references.
unsafe impl Send for Event {}

unsafe impl Sync for Event {}

impl Drop for Event {
    fn drop(&mut self) {
        if let Err(error) = self.context.bind() {
            #[cfg(debug_assertions)]
            eprintln!("failed to bind context before destroying event: {error}");
        }
        // SAFETY: `cudaEventDestroy` receives live handles owned by the wrappers.
        unsafe {
            if let Err(error) = try_ffi!(runtime::cudaEventDestroy(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cuda event: {error}");
            }
        }
    }
}

impl Context {
    /// Creates a timing-enabled event with [`EventFlags::DEFAULT`] in this context.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`Self::create_event_with_flags`].
    pub fn create_event(self: &Arc<Self>) -> Result<Event> {
        self.create_event_with_flags(EventFlags::DEFAULT)
    }

    /// Creates an event object for the current device with the specified flags.
    /// Valid flags include:
    ///
    /// * [`EventFlags::DEFAULT`]: default event creation flag.
    /// * [`EventFlags::BLOCKING_SYNC`]: the event uses blocking synchronization.
    ///   A host thread that uses [`Event::synchronize`] to wait on an event created with this flag will block until the event actually completes.
    /// * [`EventFlags::DISABLE_TIMING`]: the created event does not record timing data.
    ///   Events created with this flag specified and [`EventFlags::BLOCKING_SYNC`] not specified will provide the best performance when used with [`Stream::wait_event`] and [`Event::query`].
    /// * [`EventFlags::INTERPROCESS`]: the created event may be used as an interprocess event by [`sys::cudaIpcGetEventHandle`](singe_cuda_sys::runtime::cudaIpcGetEventHandle).
    ///   [`EventFlags::INTERPROCESS`] must be specified along with [`EventFlags::DISABLE_TIMING`].
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound, the flag combination is
    /// invalid, CUDA cannot create the event, or CUDA returns a null event
    /// handle. CUDA may also report errors from previous asynchronous launches,
    /// internal runtime initialization errors such as
    /// [`crate::error::Status::NotInitialized`], [`crate::error::Status::CallRequiresNewerDriver`],
    /// or [`crate::error::Status::NoDevice`], and callback diagnostics such as
    /// [`crate::error::Status::NotPermitted`].
    pub fn create_event_with_flags(self: &Arc<Self>, flags: EventFlags) -> Result<Event> {
        self.bind()?;
        let mut handle = ptr::null_mut();
        // SAFETY: `cudaEventCreateWithFlags` receives valid output locations.
        unsafe {
            try_ffi!(runtime::cudaEventCreateWithFlags(
                &raw mut handle,
                flags.bits()
            ))?;
        }
        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        // SAFETY: CUDA just created this handle in this context, and nothing else owns it.
        unsafe { Event::from_raw(handle, Arc::clone(self)) }
    }
}
