use std::{
    cmp::Ordering,
    fmt::{self, Display, Formatter},
    mem::ManuallyDrop,
    ptr,
    sync::Arc,
};

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
        const DEFAULT = driver::CUevent_flags::CU_EVENT_DEFAULT as _;
        const BLOCKING_SYNC = driver::CUevent_flags::CU_EVENT_BLOCKING_SYNC as _;
        const DISABLE_TIMING = driver::CUevent_flags::CU_EVENT_DISABLE_TIMING as _;
        const INTERPROCESS = driver::CUevent_flags::CU_EVENT_INTERPROCESS as _;
    }
}

impl Display for EventFlags {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        bitflags::parser::to_writer(self, f)
    }
}

bitflags::bitflags! {
    /// Flags for `Event::record_raw`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct EventRecordFlags: u32 {
        const DEFAULT = runtime::cudaEventRecordDefault;
        const EXTERNAL = runtime::cudaEventRecordExternal;
    }
}

impl Display for EventRecordFlags {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        bitflags::parser::to_writer(self, f)
    }
}

#[derive(Debug)]
pub struct Event {
    handle: runtime::cudaEvent_t,
    ctx: Arc<Context>,
}

impl PartialEq for Event {
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle && Arc::ptr_eq(&self.ctx, &other.ctx)
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
        let lhs = (Arc::as_ptr(&self.ctx) as usize, self.handle as usize);
        let rhs = (Arc::as_ptr(&other.ctx) as usize, other.handle as usize);
        lhs.cmp(&rhs)
    }
}

impl Event {
    /// Wraps an existing CUDA event handle.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid CUDA event owned by `ctx`, and ownership of the
    /// handle is transferred to the returned [`Event`]. The handle must not be
    /// destroyed elsewhere after calling this function.
    pub unsafe fn from_raw(handle: runtime::cudaEvent_t, ctx: Arc<Context>) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self { handle, ctx })
    }

    pub fn record(&self, stream: &Stream, flags: EventRecordFlags) -> Result<()> {
        if stream.context() != self.context() {
            return Err(Error::StreamContextMismatch);
        }
        self.record_raw(stream.as_raw(), flags)
    }

    pub fn record_borrowed(&self, stream: &BorrowedStream, flags: EventRecordFlags) -> Result<()> {
        if stream.context() != self.context() {
            return Err(Error::StreamContextMismatch);
        }
        self.record_raw(stream.as_raw(), flags)
    }

    pub fn record_on(&self, stream: &StreamBinding, flags: EventRecordFlags) -> Result<()> {
        if stream.context() != self.context() {
            return Err(Error::StreamContextMismatch);
        }
        self.record_raw(stream.as_raw(), flags)
    }

    fn record_raw(&self, stream: runtime::cudaStream_t, flags: EventRecordFlags) -> Result<()> {
        self.ctx.bind()?;
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
        self.ctx.bind()?;
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
        self.ctx.bind()?;
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
    pub fn elapsed_time_since(&self, start: &Event) -> Result<f32> {
        if self.context() != start.context() {
            return Err(runtime::cudaError_t::CUDA_ERROR_INVALID_CONTEXT.into());
        }

        self.ctx.bind()?;
        let mut milliseconds = 0.0f32;
        unsafe {
            try_ffi!(runtime::cudaEventElapsedTime(
                &raw mut milliseconds,
                start.as_raw(),
                self.as_raw(),
            ))?;
        }
        Ok(milliseconds)
    }

    pub fn context(&self) -> &Context {
        &self.ctx
    }

    pub const fn as_raw(&self) -> runtime::cudaEvent_t {
        self.handle
    }

    /// Consumes the event and returns the raw CUDA event handle without
    /// destroying it.
    ///
    /// The caller becomes responsible for eventually destroying the returned
    /// handle with CUDA.
    pub fn into_raw(self) -> runtime::cudaEvent_t {
        let event = ManuallyDrop::new(self);
        event.handle
    }
}

// CUDA events are synchronization handles. Recording/waiting uses CUDA's event
// semantics and does not mutate Rust-owned state through shared references.
unsafe impl Send for Event {}
unsafe impl Sync for Event {}

impl Drop for Event {
    fn drop(&mut self) {
        if let Err(err) = self.ctx.bind() {
            #[cfg(debug_assertions)]
            eprintln!("failed to bind context before destroying event: {err}");
        }
        unsafe {
            if let Err(err) = try_ffi!(runtime::cudaEventDestroy(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy CUDA event: {err}");
            }
        }
    }
}

impl Context {
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
        unsafe {
            try_ffi!(runtime::cudaEventCreateWithFlags(
                &raw mut handle,
                flags.bits()
            ))?;
        }
        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        unsafe { Event::from_raw(handle, Arc::clone(self)) }
    }
}
