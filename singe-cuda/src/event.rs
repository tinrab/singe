//! CUDA events, timing, and stream ordering.

use std::{
    cmp::Ordering,
    fmt::{self, Display, Formatter},
    mem::ManuallyDrop,
    ptr,
    sync::Arc,
};

use bitflags::parser;
use singe_core::report_cleanup_failure;
use singe_cuda_sys::{driver, runtime};

use crate::{
    context::Context,
    error::{Error, Result},
    stream::AsStream,
    try_ffi,
};

bitflags::bitflags! {
    /// Flags for CUDA event creation ([`Context::create_event`]).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct EventFlags: u32 {
        /// Default event flag.
        const DEFAULT = driver::CUevent_flags::CU_EVENT_DEFAULT.0;
        /// Event uses blocking synchronization.
        const BLOCKING_SYNC = driver::CUevent_flags::CU_EVENT_BLOCKING_SYNC.0;
        /// Event will not record timing data.
        const DISABLE_TIMING = driver::CUevent_flags::CU_EVENT_DISABLE_TIMING.0;
        /// Event is suitable for interprocess use.
        ///
        /// [`EventFlags::DISABLE_TIMING`] must also be set.
        const INTERPROCESS = driver::CUevent_flags::CU_EVENT_INTERPROCESS.0;
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

bitflags::bitflags! {
    /// Flags controlling how a stream waits on an event, used by [`Stream::wait_event`](crate::stream::Stream::wait_event).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct EventWaitFlags: u32 {
        /// Wait normally, using internal event dependencies during stream capture.
        const DEFAULT = driver::CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT.0;
        /// Create an external event-wait node during stream capture.
        const EXTERNAL = driver::CUevent_wait_flags::CU_EVENT_WAIT_EXTERNAL.0;
    }
}

impl Display for EventWaitFlags {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        parser::to_writer(self, f)
    }
}

/// A CUDA event and its retained context.
///
/// Recording captures preceding work on a stream; querying, waiting, and elapsed-time calculations refer to the event's most recent recording.
/// Dropping the wrapper destroys its native event handle.
#[derive(Debug)]
pub struct Event {
    handle: runtime::cudaEvent_t,
    context: Arc<Context>,
}

/// Raw ownership of a CUDA event and the context it belongs to.
///
/// Destroy `handle` with CUDA while `context` is alive, or re-adopt the value with [`Event::from_raw`].
#[derive(Debug)]
pub struct RawEvent {
    /// Native event handle.
    pub handle: runtime::cudaEvent_t,
    /// Context that owns the event.
    pub context: Arc<Context>,
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
        let lhs = (Arc::as_ptr(&self.context).addr(), self.handle.addr());
        let rhs = (Arc::as_ptr(&other.context).addr(), other.handle.addr());
        lhs.cmp(&rhs)
    }
}

impl Event {
    /// Takes ownership of a raw CUDA event.
    ///
    /// # Safety
    ///
    /// `raw.handle` must be a valid CUDA event created in `raw.context`, and ownership of the handle is transferred to the returned [`Event`].
    /// The handle must not be destroyed elsewhere after calling this function.
    ///
    /// # Errors
    ///
    /// Returns `Error::NullHandle` if `raw.handle` is null.
    pub unsafe fn from_raw(raw: RawEvent) -> Result<Self> {
        if raw.handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle: raw.handle,
            context: raw.context,
        })
    }

    /// Records the work preceding this call on `stream`.
    ///
    /// Re-recording replaces the event's captured work.
    /// `flags` controls whether capture treats this as an external event record.
    ///
    /// # Errors
    ///
    /// Returns `Error::StreamContextMismatch` if the stream and event belong to different contexts.
    /// Also returns an error if the event context cannot be bound, CUDA rejects the recording or flags, or previous asynchronous work reports an error.
    pub fn record(&self, stream: &(impl AsStream + ?Sized), flags: EventRecordFlags) -> Result<()> {
        if stream.context() != self.context() {
            return Err(Error::StreamContextMismatch);
        }

        self.context.bind()?;
        // SAFETY: the event and stream are live handles of the bound context.
        unsafe {
            try_ffi!(runtime::cudaEventRecordWithFlags(
                self.as_raw(),
                stream.as_raw(),
                flags.bits(),
            ))?;
        }
        Ok(())
    }

    /// Queries the status of all work currently captured by event.
    /// See [`Event::record`] for what an event captures.
    ///
    /// Returns `true` if all captured work has been completed, or `false` if any captured work is incomplete.
    ///
    /// For the purposes of Unified Memory, a return value of `true` is equivalent to having called [`Event::synchronize`].
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot query the event.
    /// CUDA may also report errors from previous asynchronous launches, internal runtime initialization errors such as [`crate::error::Status::NotInitialized`], [`crate::error::Status::CallRequiresNewerDriver`], or [`crate::error::Status::NoDevice`], and callback diagnostics such as [`crate::error::Status::NotPermitted`].
    pub fn query(&self) -> Result<bool> {
        self.context.bind()?;
        // SAFETY: `self` owns the live event, and its context is bound above.
        let error = unsafe { runtime::cudaEventQuery(self.as_raw()) };
        match error {
            runtime::cudaError_t::CUDA_SUCCESS => Ok(true),
            runtime::cudaError_t::CUDA_ERROR_NOT_READY => Ok(false),
            _ => Err(error.into()),
        }
    }

    /// Waits until the completion of all work currently captured in event.
    /// See [`Event::record`] for what an event captures.
    ///
    /// Waiting for an event created with [`EventFlags::BLOCKING_SYNC`] causes the calling CPU thread to block until the event has been completed by the device.
    /// Without [`EventFlags::BLOCKING_SYNC`], the CPU thread will busy-wait until the event has been completed by the device.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot wait for the event.
    /// CUDA may also report errors from previous asynchronous launches, internal runtime initialization errors such as [`crate::error::Status::NotInitialized`], [`crate::error::Status::CallRequiresNewerDriver`], or [`crate::error::Status::NoDevice`], and callback diagnostics such as [`crate::error::Status::NotPermitted`].
    pub fn synchronize(&self) -> Result<()> {
        self.context.bind()?;
        // SAFETY: `self` owns the live event, and its context is bound above.
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
    /// This happens because [`Event::record`] takes place asynchronously and there is no guarantee that the measured latency is actually just between the two events.
    /// Any number of other different stream operations could execute in between the two measured events, thus altering the timing in a significant way.
    ///
    /// If either event has not been recorded with [`Event::record`], [`crate::error::Status::InvalidHandle`] is returned.
    /// If both events have been recorded but one or both of them has not yet been completed (that is, [`Event::query`] returns `false` for at least one of them), [`crate::error::Status::NotReady`] is returned.
    /// If either event was created with [`EventFlags::DISABLE_TIMING`], this returns [`crate::error::Status::InvalidHandle`].
    ///
    /// # Errors
    ///
    /// Returns an error if the events belong to different contexts, either event has not been recorded, either event is incomplete, timing was disabled on either event, or CUDA rejects the elapsed-time query.
    /// CUDA may also report errors from previous asynchronous launches, internal runtime initialization errors such as [`crate::error::Status::NotInitialized`], [`crate::error::Status::CallRequiresNewerDriver`], or [`crate::error::Status::NoDevice`], and callback diagnostics such as [`crate::error::Status::NotPermitted`].
    pub fn elapsed_time_since(&self, start: &Self) -> Result<f32> {
        if self.context() != start.context() {
            return Err(Error::context_mismatch("start event"));
        }

        self.context.bind()?;
        let mut milliseconds = 0.0f32;
        // SAFETY: both events are live, owned by their wrappers, and of the context bound above, and `milliseconds` is a local output.
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

    /// Consumes the event and returns its raw handle and context without destroying the handle.
    ///
    /// The caller becomes responsible for eventually destroying the returned handle with CUDA.
    #[must_use]
    pub fn into_raw(self) -> RawEvent {
        let event = ManuallyDrop::new(self);
        RawEvent {
            handle: event.handle,
            // SAFETY: `event` is never dropped or used again, so its context is moved out exactly once.
            context: unsafe { ptr::read(&raw const event.context) },
        }
    }
}

// SAFETY: CUDA event handles may be recorded, queried, synchronized, and destroyed from any host thread once their context is bound, which every method and `Drop` do first.
unsafe impl Send for Event {}

// SAFETY: every `&self` method is a thread-safe driver call on the handle, and recording through `&self` only changes native state, never Rust-owned state.
unsafe impl Sync for Event {}

impl Drop for Event {
    fn drop(&mut self) {
        if let Err(error) = self.context.bind() {
            report_cleanup_failure("bind the cuda context to destroy", "cuda event", &error);
            return;
        }

        // SAFETY: `self` owns the live event, its context is bound above, and CUDA defers destruction until recorded work completes.
        if let Err(error) = unsafe { try_ffi!(runtime::cudaEventDestroy(self.handle)) } {
            report_cleanup_failure("destroy", "cuda event", &error);
        }
    }
}

impl Context {
    /// Creates an event in this context with the specified flags.
    /// Valid flags include:
    ///
    /// * [`EventFlags::DEFAULT`]: default event creation flag.
    /// * [`EventFlags::BLOCKING_SYNC`]: the event uses blocking synchronization.
    ///   A host thread that uses [`Event::synchronize`] to wait on an event created with this flag will block until the event actually completes.
    /// * [`EventFlags::DISABLE_TIMING`]: the created event does not record timing data.
    ///   Events created with this flag specified and [`EventFlags::BLOCKING_SYNC`] not specified will provide the best performance when used with [`Stream::wait_event`](crate::stream::Stream::wait_event) and [`Event::query`].
    /// * [`EventFlags::INTERPROCESS`]: the created event may be shared with another process through an [`IpcEventHandle`](crate::ipc::IpcEventHandle).
    ///   [`EventFlags::INTERPROCESS`] must be specified along with [`EventFlags::DISABLE_TIMING`].
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound, the flag combination is invalid, CUDA cannot create the event, or CUDA returns a null event handle.
    /// CUDA may also report errors from previous asynchronous launches, internal runtime initialization errors such as [`crate::error::Status::NotInitialized`], [`crate::error::Status::CallRequiresNewerDriver`], or [`crate::error::Status::NoDevice`], and callback diagnostics such as [`crate::error::Status::NotPermitted`].
    pub fn create_event(self: &Arc<Self>, flags: EventFlags) -> Result<Event> {
        self.bind()?;
        let mut handle = ptr::null_mut();
        // SAFETY: `handle` is a local output, and this context is bound above.
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
        unsafe {
            Event::from_raw(RawEvent {
                handle,
                context: Arc::clone(self),
            })
        }
    }
}
