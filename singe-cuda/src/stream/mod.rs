//! CUDA streams, capture scopes, synchronization, and ordered operations.

mod callback;
mod capture;
mod context;
mod raw;
mod scope;

pub mod memory;

mod cleanup;

mod sealed {
    #[expect(
        unnameable_types,
        reason = "sealed supertrait that other crates must not name or implement"
    )]
    pub trait Sealed {}
}

use std::{
    collections::HashMap,
    fmt::{self, Display, Formatter},
    hash::{Hash, Hasher},
    mem, ptr,
    sync::{Arc, Mutex, MutexGuard, OnceLock, Weak},
};

use bitflags::parser;
use singe_core::{abort_after_cleanup_failure, report_cleanup_failure};
use singe_cuda_sys::runtime;

use crate::{
    context::Context,
    device::Device,
    error::{Error, Result, Status},
    event::{Event, EventFlags, EventRecordFlags, EventWaitFlags},
    memory::DeferredFree,
    try_ffi,
    utility::lock_for_cleanup,
    view::DeviceRegion,
};

use self::raw::ForeignOwner;
pub use self::{
    capture::{
        StreamCaptureDependencyUpdate, StreamCaptureInfo, StreamCaptureMode, StreamCaptureScope,
        StreamCaptureStatus, exchange_capture_mode,
    },
    raw::{RawStream, RawStreamOwnership},
    scope::StreamScope,
};

bitflags::bitflags! {
    /// Flags for CUDA stream creation ([`StreamOptions::flags`]).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct StreamFlags: u32 {
        /// Creates a stream that participates in implicit synchronization with the legacy default stream.
        const DEFAULT = runtime::cudaStreamDefault;
        /// Creates a stream without implicit synchronization with the legacy default stream.
        const NON_BLOCKING = runtime::cudaStreamNonBlocking;
    }
}

impl Display for StreamFlags {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        parser::to_writer(self, f)
    }
}

/// Options for [`Context::create_stream`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StreamOptions {
    /// Stream creation behavior, [`StreamFlags::DEFAULT`] by default.
    pub flags: StreamFlags,
    /// Scheduling priority, where lower numbers are higher priorities and `0` is the default priority.
    ///
    /// CUDA clamps a priority outside [`Device::stream_priority_range`] to the nearest end of that range.
    pub priority: i32,
}

impl Default for StreamOptions {
    fn default() -> Self {
        Self {
            flags: StreamFlags::DEFAULT,
            priority: 0,
        }
    }
}

/// A shared CUDA stream retaining its actual context.
///
/// Clones share native ownership and capture cleanup.
/// Owned handles are synchronized and destroyed when their last wrapper drops.
/// Externally owned handles instead retain an ownership token and are not destroyed by this wrapper.
#[derive(Debug, Clone)]
pub struct Stream {
    inner: Arc<StreamInner>,
}

#[derive(Debug)]
struct StreamInner {
    handle: runtime::cudaStream_t,
    context: Arc<Context>,
    capture_cleanup: Arc<Mutex<CaptureCleanup>>,
    foreign_owner: Option<ForeignOwner>,
}

impl PartialEq for Stream {
    fn eq(&self, other: &Self) -> bool {
        self.identity() == other.identity()
    }
}

impl Eq for Stream {}

impl Hash for Stream {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Hash::hash(&self.identity(), state);
    }
}

/// A CUDA stream handle without native destruction responsibility.
///
/// Values returned by [`Stream::to_borrowed`] retain the owning stream.
/// Raw borrowed streams rely on the lifetime contract of [`Self::from_raw`].
#[derive(Debug, Clone)]
pub struct BorrowedStream {
    handle: runtime::cudaStream_t,
    context: Arc<Context>,
    _owner: Option<Stream>,
}

/// An explicit CUDA stream selection with its retained context.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum StreamBinding {
    /// Selects the legacy default stream in the retained context.
    Default(Arc<Context>),
    /// Selects a borrowed stream and its associated context.
    Borrowed(BorrowedStream),
}

/// A CUDA stream selection that work can be ordered on.
///
/// Implemented by [`Stream`], [`BorrowedStream`], [`StreamBinding`], and [`StreamScope`].
/// The trait is sealed.
pub trait AsStream: sealed::Sealed {
    /// Returns the context the stream belongs to.
    fn context(&self) -> &Arc<Context>;

    /// Returns the native stream handle without transferring ownership.
    fn as_raw(&self) -> runtime::cudaStream_t;

    /// Blocks until the stream has completed all queued work.
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound, synchronization fails, or earlier asynchronous work reported an error.
    fn synchronize(&self) -> Result<()> {
        self.context().bind()?;
        // SAFETY: the handle is a live stream of the bound context.
        unsafe { try_ffi!(runtime::cudaStreamSynchronize(self.as_raw())) }
    }

    /// Waits for queued work on the stream, or prints the error and aborts the process if the wait fails.
    ///
    /// Cleanup code calls this before releasing borrowed storage that queued work may still access.
    fn synchronize_or_abort(&self) {
        if let Err(error) = self.synchronize() {
            abort_after_cleanup_failure("synchronize", "cuda stream", &error);
        }
    }

    /// Reports the stream's capture state without invalidating an active capture.
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound or CUDA cannot query the capture state.
    fn capture_status(&self) -> Result<StreamCaptureStatus> {
        self.context().bind()?;
        let mut status = runtime::cudaStreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
        // SAFETY: the handle is a live stream of the bound context, and `status` is a valid output location.
        unsafe {
            try_ffi!(runtime::cudaStreamIsCapturing(
                self.as_raw(),
                &raw mut status
            ))?;
        }
        Ok(status.try_into()?)
    }

    /// Rejects active or invalidated capture before an operation that cannot be recorded.
    ///
    /// # Errors
    ///
    /// Returns the stream-capture-unsupported status for active capture, the stream-capture-invalidated status for invalidated capture, or an error if the capture state cannot be queried.
    fn ensure_not_capturing(&self) -> Result<()> {
        match self.capture_status()? {
            StreamCaptureStatus::None => Ok(()),
            StreamCaptureStatus::Active => Err(Status::StreamCaptureUnsupported.into()),
            StreamCaptureStatus::Invalidated => Err(Status::StreamCaptureInvalidated.into()),
        }
    }

    /// Makes work submitted to the stream after this call wait for the work captured in `event`.
    ///
    /// `event` may belong to another context or device.
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound, CUDA rejects the wait or flags, or earlier asynchronous work reported an error.
    fn wait_event(&self, event: &Event, flags: EventWaitFlags) -> Result<()> {
        self.context().bind()?;
        // SAFETY: both handles are live and the stream's context is bound.
        unsafe {
            try_ffi!(runtime::cudaStreamWaitEvent(
                self.as_raw(),
                event.as_raw(),
                flags.bits(),
            ))
        }
    }

    /// Orders subsequent work on the stream after the readiness streams of these device regions.
    ///
    /// Each distinct producer stream contributes one event dependency.
    /// This does not establish initialization or retain allocations for later work.
    ///
    /// # Errors
    ///
    /// Returns `Error::MemoryContextMismatch` before submitting dependencies if any region belongs to another context, or an error if an event cannot be recorded or waited on.
    fn join_regions(&self, regions: &[DeviceRegion<'_>]) -> Result<()> {
        if regions
            .iter()
            .any(|region| region.context() != self.context())
        {
            return Err(Error::MemoryContextMismatch);
        }

        let identity = StreamIdentity::new(self.context(), self.as_raw());
        for (index, region) in regions.iter().enumerate() {
            if let Some(producer) = region.stream()
                && producer.identity() != identity
                && !regions[..index]
                    .iter()
                    .any(|earlier| earlier.stream() == Some(producer))
            {
                let event = producer
                    .context()
                    .create_event(EventFlags::DISABLE_TIMING)?;
                event.record(producer, EventRecordFlags::DEFAULT)?;
                self.wait_event(&event, EventWaitFlags::DEFAULT)?;
            }
        }
        Ok(())
    }
}

const LEGACY_STREAM: usize = 1;
const PER_THREAD_STREAM: usize = 2;

/// Native stream identity shared by independent wrappers and legacy stream aliases.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct StreamIdentity {
    context: usize,
    handle: usize,
}

impl StreamIdentity {
    pub(crate) fn new(context: &Context, handle: runtime::cudaStream_t) -> Self {
        let handle = handle.addr();
        Self {
            context: context.as_raw().addr(),
            handle: if handle == LEGACY_STREAM { 0 } else { handle },
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct CaptureCleanup {
    owners: Vec<Arc<dyn fmt::Debug + Send + Sync>>,
    frees: Vec<DeferredFree>,
    foreign: Vec<ForeignOwner>,
}

impl CaptureCleanup {
    fn for_stream(key: StreamIdentity) -> Result<Arc<Mutex<Self>>> {
        type Registry = HashMap<StreamIdentity, Weak<Mutex<CaptureCleanup>>>;

        static LIFECYCLES: OnceLock<Mutex<Registry>> = OnceLock::new();
        let mut lifecycles = LIFECYCLES
            .get_or_init(Mutex::default)
            .lock()
            .map_err(|_| Error::lock_poisoned("stream lifecycles"))?;
        lifecycles.retain(|_, lifecycle| lifecycle.strong_count() != 0);
        if let Some(lifecycle) = lifecycles.get(&key).and_then(Weak::upgrade) {
            return Ok(lifecycle);
        }

        let lifecycle = Arc::new(Mutex::new(Self::default()));
        lifecycles.insert(key, Arc::downgrade(&lifecycle));
        drop(lifecycles);
        Ok(lifecycle)
    }

    fn is_empty(&self) -> bool {
        self.owners.is_empty() && self.frees.is_empty() && self.foreign.is_empty()
    }

    fn flush_frees(&mut self, stream: runtime::cudaStream_t) -> Result<()> {
        while let Some(free) = self.frees.last() {
            free.enqueue(stream)?;
            self.frees.pop();
        }

        self.foreign.clear();
        Ok(())
    }
}

impl Stream {
    pub(crate) fn identity(&self) -> StreamIdentity {
        StreamIdentity::new(self.context(), self.as_raw())
    }

    /// Defers destruction of an unused resource until capture ends, if active.
    /// This permits eager APIs to reject capture without calling a capture-incompatible destructor while releasing their arguments.
    /// It does not retain resources for graph replay.
    /// End capture through this stream wrapper to release these owners.
    /// Avoid owners that retain this stream, which would create a reference cycle.
    ///
    /// # Errors
    ///
    /// Propagates capture-state query and context binding errors.
    /// If capture is inactive, `owner` is dropped before this call returns.
    pub fn defer_drop_until_capture_end<T>(&self, owner: Arc<T>) -> Result<()>
    where
        T: fmt::Debug + Send + Sync + 'static,
    {
        let mut cleanup = lock_for_cleanup(&self.inner.capture_cleanup);
        if self.capture_status()? != StreamCaptureStatus::None {
            cleanup.owners.push(owner);
        }

        drop(cleanup);
        Ok(())
    }

    fn capture_cleanup_lock(&self) -> Result<MutexGuard<'_, CaptureCleanup>> {
        self.inner
            .capture_cleanup
            .lock()
            .map_err(|_| Error::lock_poisoned("stream capture cleanup"))
    }

    pub(crate) fn allocation_guard(&self) -> Result<MutexGuard<'_, CaptureCleanup>> {
        let mut cleanup = self.capture_cleanup_lock()?;
        self.ensure_not_capturing()?;
        cleanup.flush_frees(self.as_raw())?;
        Ok(cleanup)
    }

    pub(crate) fn defer_free(&self, free: DeferredFree) -> Result<()> {
        let mut cleanup = lock_for_cleanup(&self.inner.capture_cleanup);
        cleanup.frees.push(free);
        if self.capture_status()? == StreamCaptureStatus::None {
            cleanup.flush_frees(self.as_raw())?;
        }

        drop(cleanup);
        Ok(())
    }

    /// Returns a borrowed stream wrapper retaining a clone of this stream's owner.
    #[must_use]
    pub fn to_borrowed(&self) -> BorrowedStream {
        BorrowedStream {
            handle: self.as_raw(),
            context: Arc::clone(&self.inner.context),
            _owner: Some(self.clone()),
        }
    }

    /// Returns a borrowed stream wrapper that does not retain this stream's owner.
    ///
    /// A native owner passed to [`Self::defer_drop_until_capture_end`] holds one, since retaining the stream would form a reference cycle.
    ///
    /// # Safety
    ///
    /// The returned wrapper must not be used after the last owner of this stream is dropped.
    #[must_use]
    pub unsafe fn to_unretained(&self) -> BorrowedStream {
        BorrowedStream {
            handle: self.as_raw(),
            context: Arc::clone(&self.inner.context),
            _owner: None,
        }
    }

    /// Blocks until stream has completed all operations.
    /// If [`ContextFlags::SCHEDULE_BLOCKING_SYNC`](crate::context::ContextFlags::SCHEDULE_BLOCKING_SYNC) was set for this device, the host thread will block until the stream is finished with all of its tasks.
    ///
    /// Uses standard `default stream` semantics.
    ///
    /// # Errors
    ///
    /// Returns an error if stream synchronization fails or if a previous asynchronous launch reported an error.
    /// CUDA may also return initialization-related errors such as [`crate::error::Status::NotInitialized`], [`crate::error::Status::CallRequiresNewerDriver`], or [`crate::error::Status::NoDevice`] if this call initializes internal runtime state.
    /// Callbacks must not call CUDA functions; see [`Stream::add_callback`].
    pub fn synchronize(&self) -> Result<()> {
        AsStream::synchronize(self)
    }

    /// Waits for queued work on this stream, or prints the error and aborts the process if the wait fails.
    ///
    /// Cleanup code calls this before releasing borrowed storage that queued work may still access.
    pub fn synchronize_or_abort(&self) {
        AsStream::synchronize_or_abort(self);
    }

    /// Synchronizes this stream and releases its native ownership.
    ///
    /// Owned handles are destroyed.
    /// Borrowed handles release their external owner token after synchronization instead of destroying the handle.
    /// This may block while waiting for stream work and callbacks.
    /// For owned handles, destruction is attempted even if synchronization fails, and the synchronization error takes precedence over a destruction error.
    ///
    /// # Errors
    ///
    /// Returns `Error::OwnershipShared` if another stream clone or borrowed owner still retains the handle.
    /// Also propagates capture-state rejection, deferred-free enqueue, context binding, synchronization, or owned-handle destruction errors.
    /// Synchronization errors take precedence over destruction errors.
    pub fn shutdown(self) -> Result<()> {
        let lifecycle = Arc::clone(&self.inner.capture_cleanup);
        let mut cleanup = lifecycle
            .lock()
            .map_err(|_| Error::lock_poisoned("stream capture cleanup"))?;
        self.ensure_not_capturing()?;
        cleanup.flush_frees(self.as_raw())?;
        let mut inner = match Arc::try_unwrap(self.inner) {
            Ok(inner) => inner,
            Err(inner) => {
                // Releasing this reference can destroy the last wrapper after a concurrent drop.
                drop(cleanup);
                drop(inner);
                return Err(Error::ownership_shared("stream"));
            }
        };
        let result = if inner.foreign_owner.is_some() {
            inner
                .context
                .bind()
                // SAFETY: `inner.handle` is the live foreign stream this last wrapper still holds, and its context is bound here.
                .and_then(|()| unsafe { try_ffi!(runtime::cudaStreamSynchronize(inner.handle)) })
        } else {
            Self::destroy_handle(inner.context.as_ref(), inner.handle)
        };
        inner.handle = ptr::null_mut();
        let owner = inner.foreign_owner.take();
        // An external token may itself own a Singe wrapper for this native stream.
        drop(cleanup);
        drop(owner);
        result
    }

    /// Returns `true` if all operations in stream have completed, or `false` if not.
    ///
    /// For the purposes of Unified Memory, a return value of `true` is equivalent to having called [`Stream::synchronize`].
    ///
    /// Uses standard `default stream` semantics.
    ///
    /// # Errors
    ///
    /// Returns an error if querying the stream fails or if a previous asynchronous launch reported an error.
    /// CUDA may also return initialization-related errors such as [`crate::error::Status::NotInitialized`], [`crate::error::Status::CallRequiresNewerDriver`], or [`crate::error::Status::NoDevice`] if this call initializes internal runtime state.
    /// Callbacks must not call CUDA functions; see [`Stream::add_callback`].
    pub fn query(&self) -> Result<bool> {
        self.inner.context.bind()?;
        // SAFETY: `self.as_raw()` is the live stream this wrapper retains, in the context bound above.
        let error = unsafe { runtime::cudaStreamQuery(self.as_raw()) };
        match error {
            runtime::cudaError_t::CUDA_SUCCESS => Ok(true),
            runtime::cudaError_t::CUDA_ERROR_NOT_READY => Ok(false),
            _ => Err(error.into()),
        }
    }

    /// Makes all future work submitted to stream wait for all work captured in event.
    /// See [`sys::cudaEventRecord`](singe_cuda_sys::runtime::cudaEventRecord) for details on what is captured by an event.
    /// `flags` controls how strictly the wait is enforced during capture.
    /// Synchronization is performed efficiently on the device when applicable.
    /// `event` may be from a different device than `stream`.
    ///
    /// # Errors
    ///
    /// Returns an error if the stream cannot wait on the event or if a previous asynchronous launch reported an error.
    pub fn wait_event(&self, event: &Event, flags: EventWaitFlags) -> Result<()> {
        AsStream::wait_event(self, event, flags)
    }

    /// Creates a nonblocking stream ordered after this stream's current work.
    ///
    /// The new stream uses this stream's context and priority.
    /// Later submissions to either stream can execute independently.
    /// Call [`Stream::join`] to order subsequent work after a branch completes.
    /// During capture, keep the branch alive until capture ends and join it back into the capturing stream before ending capture.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot create the stream or establish its event dependency.
    pub fn fork(&self) -> Result<Self> {
        let branch = self.context().create_stream(StreamOptions {
            flags: StreamFlags::NON_BLOCKING,
            priority: self.priority()?,
        })?;
        branch.join(self)?;
        Ok(branch)
    }

    /// Orders subsequent work on this stream after the other stream's current work.
    ///
    /// Records an event on `other` and enqueues a wait on this stream without blocking the host.
    /// The other stream may belong to a different CUDA context or device.
    /// Work submitted to `other` after this call is independent of the dependency.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot record the event or enqueue the wait.
    pub fn join(&self, other: &Self) -> Result<()> {
        if self == other {
            return Ok(());
        }

        let event = other.context().create_event(EventFlags::DISABLE_TIMING)?;
        event.record(other, EventRecordFlags::DEFAULT)?;
        self.wait_event(&event, EventWaitFlags::DEFAULT)
    }

    /// Orders subsequent work after the readiness streams of these device regions.
    ///
    /// All regions must belong to this stream's context.
    /// Each distinct producer stream contributes one event dependency.
    /// This does not establish initialization or retain allocations for later work.
    /// Borrowed consumers must keep their resources in a draining scope; raw queued consumers must uphold their resource and access contracts.
    ///
    /// # Errors
    ///
    /// Returns `Error::MemoryContextMismatch` before submitting dependencies if any region belongs to another context, or a CUDA error if an event cannot be recorded or waited on.
    pub fn join_regions(&self, regions: &[DeviceRegion<'_>]) -> Result<()> {
        AsStream::join_regions(self, regions)
    }

    /// Query the flags of a stream.
    /// Returns the stream flags.
    /// See [`StreamFlags`] for the valid flags.
    ///
    /// Uses standard `default stream` semantics.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot query the flags or if a previous asynchronous launch reported an error.
    /// CUDA may also return initialization-related errors such as [`crate::error::Status::NotInitialized`], [`crate::error::Status::CallRequiresNewerDriver`], or [`crate::error::Status::NoDevice`] if this call initializes internal runtime state.
    /// Callbacks must not call CUDA functions; see [`Stream::add_callback`].
    pub fn flags(&self) -> Result<StreamFlags> {
        self.inner.context.bind()?;
        let mut flags_raw = 0u32;
        // SAFETY: `self.as_raw()` is the live stream this wrapper retains, and `flags_raw` is a writable local.
        unsafe {
            try_ffi!(runtime::cudaStreamGetFlags(
                self.as_raw(),
                &raw mut flags_raw
            ))?;
        }
        Ok(StreamFlags::from_bits_retain(flags_raw))
    }

    /// Query the priority of a stream.
    /// Returns the stream priority.
    /// If the stream was created with a priority outside the meaningful numerical range returned by [`Device::stream_priority_range`], this returns the clamped priority.
    /// See [`StreamOptions::priority`] for details about priority clamping.
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound, CUDA cannot query the priority, a previous asynchronous launch reports an error, or CUDA reports runtime initialization diagnostics.
    pub fn priority(&self) -> Result<i32> {
        self.inner.context.bind()?;
        let mut priority = 0i32;
        // SAFETY: `self.as_raw()` is the live stream this wrapper retains, and `priority` is a writable local.
        unsafe {
            try_ffi!(runtime::cudaStreamGetPriority(
                self.as_raw(),
                &raw mut priority
            ))?;
        }
        Ok(priority)
    }

    /// Returns a stream identifier that remains unique for the life of the program.
    ///
    /// The stream handle may refer to any of the following:
    ///
    /// * a stream created via any of the CUDA runtime APIs such as [`sys::cudaStreamCreate`](singe_cuda_sys::runtime::cudaStreamCreate), [`Context::create_stream`], or their driver API equivalents such as [`sys::cuStreamCreate`](singe_cuda_sys::driver::cuStreamCreate) or [`sys::cuStreamCreateWithPriority`](singe_cuda_sys::driver::cuStreamCreateWithPriority).
    ///   Passing an invalid handle results in undefined behavior.
    /// * the special legacy default stream and per-thread default stream.
    ///   The driver API equivalents of these are also accepted.
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound, CUDA cannot query the stream identifier, a previous asynchronous launch reports an error, or CUDA reports runtime initialization diagnostics.
    pub fn id(&self) -> Result<u64> {
        self.inner.context.bind()?;
        let mut id = 0u64;
        // SAFETY: `self.as_raw()` is the live stream this wrapper retains, and `id` is a writable local.
        unsafe {
            try_ffi!(runtime::cudaStreamGetId(self.as_raw(), &raw mut id))?;
        }
        Ok(id)
    }

    /// Returns the device of the stream.
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound, CUDA cannot query the stream device, a previous asynchronous launch reports an error, or CUDA reports runtime initialization diagnostics.
    pub fn device(&self) -> Result<Device> {
        self.inner.context.bind()?;
        let mut device = 0i32;
        // SAFETY: `self.as_raw()` is the live stream this wrapper retains, and `device` is a writable local.
        unsafe {
            try_ffi!(runtime::cudaStreamGetDevice(self.as_raw(), &raw mut device))?;
        }
        Ok(Device::new(device))
    }

    /// Returns the context retained by this stream.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        &self.inner.context
    }

    /// Returns the borrowed native stream handle without transferring ownership.
    #[must_use]
    pub fn as_raw(&self) -> runtime::cudaStream_t {
        self.inner.handle
    }
}

impl BorrowedStream {
    /// Wraps an existing CUDA stream handle without taking ownership.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid CUDA stream associated with `context`, and it must remain valid for every operation using the returned borrowed stream.
    pub const unsafe fn from_raw(handle: runtime::cudaStream_t, context: Arc<Context>) -> Self {
        Self {
            handle,
            context,
            _owner: None,
        }
    }
}

impl StreamBinding {
    /// Returns whether this binding uses the [`Self::Default`] variant.
    #[must_use]
    pub fn is_default(&self) -> bool {
        matches!(self, Self::Default(..))
    }
}

impl sealed::Sealed for Stream {}

impl AsStream for Stream {
    fn context(&self) -> &Arc<Context> {
        &self.inner.context
    }

    fn as_raw(&self) -> runtime::cudaStream_t {
        self.inner.handle
    }
}

impl sealed::Sealed for BorrowedStream {}

impl AsStream for BorrowedStream {
    fn context(&self) -> &Arc<Context> {
        &self.context
    }

    fn as_raw(&self) -> runtime::cudaStream_t {
        self.handle
    }
}

impl sealed::Sealed for StreamBinding {}

impl AsStream for StreamBinding {
    fn context(&self) -> &Arc<Context> {
        match self {
            Self::Default(context) => context,
            Self::Borrowed(stream) => &stream.context,
        }
    }

    fn as_raw(&self) -> runtime::cudaStream_t {
        match self {
            Self::Default(_) => ptr::null_mut(),
            Self::Borrowed(stream) => stream.handle,
        }
    }
}

// SAFETY: a CUDA stream may be used and destroyed from any thread once its context is bound, and the other fields are `Send`.
unsafe impl Send for StreamInner {}

// SAFETY: `&self` methods only pass the immutable handle to thread-safe runtime calls, and the capture cleanup state sits behind a mutex.
unsafe impl Sync for StreamInner {}

impl Drop for StreamInner {
    fn drop(&mut self) {
        if self.handle.is_null() && self.foreign_owner.is_none() {
            return;
        }

        let mut cleanup = lock_for_cleanup(&self.capture_cleanup);
        if Arc::strong_count(&self.capture_cleanup) == 1 && !cleanup.is_empty() {
            // SAFETY: the last stream owner passes its live stream in the context bound here, `status` and `graph` are writable locals, and it destroys only the graph that ending its own capture returned.
            let ended = self.context.bind().and_then(|()| unsafe {
                let mut status = runtime::cudaStreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
                try_ffi!(runtime::cudaStreamIsCapturing(self.handle, &raw mut status))?;
                if status != runtime::cudaStreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE {
                    let mut graph = ptr::null_mut();
                    // An invalidated capture also ends here with an error; the status query below decides.
                    if let Err(error) =
                        try_ffi!(runtime::cudaStreamEndCapture(self.handle, &raw mut graph))
                    {
                        report_cleanup_failure("end", "cuda stream capture", &error);
                    }

                    if !graph.is_null()
                        && let Err(error) = try_ffi!(runtime::cudaGraphDestroy(graph))
                    {
                        report_cleanup_failure("destroy", "cuda graph", &error);
                    }

                    try_ffi!(runtime::cudaStreamIsCapturing(self.handle, &raw mut status))?;
                    if status != runtime::cudaStreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE {
                        return Err(Error::out_of_range("stream capture status"));
                    }
                }
                Ok(())
            });
            if let Err(error) = ended {
                // CUDA may still retain captured addresses. Preserve their owners.
                report_cleanup_failure(
                    "end the capture to release deferred owners of",
                    "cuda stream",
                    &error,
                );
                mem::forget(mem::take(&mut *cleanup));
            } else {
                if let Err(error) = cleanup.flush_frees(self.handle) {
                    report_cleanup_failure("enqueue", "deferred cuda frees", &error);
                    mem::forget(mem::take(&mut cleanup.frees));
                }

                cleanup.owners.clear();
            }
        }

        drop(cleanup);
        if self.foreign_owner.is_some() {
            // SAFETY: the foreign stream stays live while this last wrapper holds its owner token, its context is bound here, and `status` is a writable local.
            let result = self.context.bind().and_then(|()| unsafe {
                let mut status = runtime::cudaStreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
                try_ffi!(runtime::cudaStreamIsCapturing(self.handle, &raw mut status))?;
                if status == runtime::cudaStreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE {
                    try_ffi!(runtime::cudaStreamSynchronize(self.handle))?;
                }
                Ok(())
            });
            if let Err(error) = result {
                report_cleanup_failure("synchronize", "foreign cuda stream", &error);
            }
        } else {
            if let Err(error) = self.context.bind() {
                report_cleanup_failure("bind the cuda context to destroy", "cuda stream", &error);
                return;
            }

            if let Err(error) = Stream::destroy_handle(self.context.as_ref(), self.handle) {
                report_cleanup_failure("destroy", "cuda stream", &error);
            }
        }
    }
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use std::{
        panic,
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        },
        thread,
    };

    use super::*;
    use crate::{
        context::ContextFlags,
        event::EventRecordFlags,
        graph::{GraphDependency, GraphEdgeAttributes, GraphInstantiateFlags, GraphNode},
        memory::{DeviceMemory, MemoryCopyKind},
        testing,
    };

    #[test]
    fn test_it_works() -> Result<()> {
        let test_device = testing::bootstrap()?;
        let context = Arc::clone(test_device.context());
        let stream1 = context.create_stream(StreamOptions::default())?;
        let _stream2 = context.create_stream(StreamOptions {
            flags: StreamFlags::NON_BLOCKING,
            ..StreamOptions::default()
        })?;

        let stream1_called = Arc::new(AtomicBool::new(false));
        stream1.add_callback(Box::new({
            let stream1_called = Arc::clone(&stream1_called);
            move |_status| {
                stream1_called.store(true, Ordering::SeqCst);
            }
        }))?;

        let is_done = stream1.query()?;
        assert!(!is_done);

        stream1.synchronize()?;

        let is_done_after = stream1.query()?;
        assert!(is_done_after);

        assert!(stream1_called.load(Ordering::SeqCst));

        Ok(())
    }

    #[test]
    fn test_event_query_uses_event_context() -> Result<()> {
        let test_device = testing::bootstrap()?;
        let context = Arc::clone(test_device.context());
        let stream = context.create_stream(StreamOptions::default())?;
        let event = context.create_event(EventFlags::DEFAULT)?;

        event.record(&stream, EventRecordFlags::DEFAULT)?;
        stream.synchronize()?;

        assert!(event.query()?);
        Ok(())
    }

    #[test]
    fn test_shutdown_synchronizes_and_destroys_stream() -> Result<()> {
        let test_device = testing::bootstrap()?;
        let context = Arc::clone(test_device.context());
        let stream = context.create_stream(StreamOptions::default())?;

        let called = Arc::new(AtomicBool::new(false));
        stream.add_callback(Box::new({
            let called = Arc::clone(&called);
            move |_status| {
                called.store(true, Ordering::SeqCst);
            }
        }))?;

        stream.shutdown()?;
        assert!(called.load(Ordering::SeqCst));
        Ok(())
    }

    #[test]
    fn test_launch_host_func_runs_after_stream_work() -> Result<()> {
        let test_device = testing::bootstrap()?;
        let context = Arc::clone(test_device.context());
        let stream = context.create_stream(StreamOptions::default())?;

        let called = Arc::new(AtomicBool::new(false));
        stream.launch_host_func({
            let called = Arc::clone(&called);
            move || {
                called.store(true, Ordering::SeqCst);
            }
        })?;
        stream.synchronize()?;

        assert!(called.load(Ordering::SeqCst));
        Ok(())
    }

    #[test]
    fn test_scoped_capture_returns_context_associated_graph() -> Result<()> {
        let test_device = testing::bootstrap()?;
        let context = Arc::clone(test_device.context());
        let stream = context.create_stream(StreamOptions::default())?;

        let graph = stream.capture(StreamCaptureMode::Relaxed, |scope| {
            assert_eq!(scope.stream().context(), &context);
            Ok(())
        })?;

        assert_eq!(graph.context(), &context);
        Ok(())
    }

    #[test]
    fn test_capture_to_graph_rejects_graph_from_different_context() -> Result<()> {
        let test_device = testing::bootstrap()?;
        let context = Arc::clone(test_device.context());
        let other_context = Context::create_dedicated(context.device(), ContextFlags::empty())?;

        let stream = context.create_stream(StreamOptions::default())?;
        let graph = other_context.create_graph()?;

        assert!(matches!(
            // SAFETY: the test ends the capture before the graph is dropped.
            unsafe { stream.begin_capture_to_graph(&graph, &[], StreamCaptureMode::Relaxed) },
            Err(Error::ContextMismatch { .. })
        ));
        assert_eq!(stream.capture_status()?, StreamCaptureStatus::None);
        Ok(())
    }

    #[test]
    fn test_capture_to_graph_rejects_node_from_different_graph() -> Result<()> {
        let test_device = testing::bootstrap()?;
        let context = Arc::clone(test_device.context());

        let stream = context.create_stream(StreamOptions::default())?;
        let graph = context.create_graph()?;
        let mut other_graph = context.create_graph()?;
        let other_node = other_graph.add_empty_node(&[])?;

        assert!(matches!(
            // SAFETY: the test ends the capture before the graph is dropped.
            unsafe {
                stream.begin_capture_to_graph(
                    &graph,
                    &[GraphDependency {
                        node: other_node,
                        data: GraphEdgeAttributes::default(),
                    }],
                    StreamCaptureMode::Relaxed,
                )
            },
            Err(Error::GraphNodeMismatch)
        ));
        assert_eq!(stream.capture_status()?, StreamCaptureStatus::None);
        Ok(())
    }

    #[test]
    fn test_capture_to_graph_rejects_unassociated_raw_dependency_node() -> Result<()> {
        let test_device = testing::bootstrap()?;
        let context = Arc::clone(test_device.context());

        let stream = context.create_stream(StreamOptions::default())?;
        let graph = context.create_graph()?;
        // SAFETY: the fake handle is only compared, never passed to CUDA.
        let raw_node = unsafe { GraphNode::from_raw(0x1usize as _) };

        assert!(matches!(
            // SAFETY: the test ends the capture before the graph is dropped.
            unsafe {
                stream.begin_capture_to_graph(
                    &graph,
                    &[GraphDependency {
                        node: raw_node,
                        data: GraphEdgeAttributes::default(),
                    }],
                    StreamCaptureMode::Relaxed,
                )
            },
            Err(Error::GraphNodeMismatch)
        ));
        assert_eq!(stream.capture_status()?, StreamCaptureStatus::None);
        Ok(())
    }

    #[test]
    fn test_capture_dependency_update_rejects_node_from_different_context() -> Result<()> {
        let test_device = testing::bootstrap()?;
        let context = Arc::clone(test_device.context());
        let other_context = Context::create_dedicated(context.device(), ContextFlags::empty())?;

        let stream = context.create_stream(StreamOptions::default())?;
        let mut other_graph = other_context.create_graph()?;
        let other_node = other_graph.add_empty_node(&[])?;

        let result = stream.capture(StreamCaptureMode::Relaxed, |_scope| {
            stream.update_capture_dependencies(
                &[GraphDependency {
                    node: other_node,
                    data: GraphEdgeAttributes::default(),
                }],
                StreamCaptureDependencyUpdate::Add,
            )
        });

        assert!(matches!(result, Err(Error::ContextMismatch { .. })));
        assert_eq!(stream.capture_status()?, StreamCaptureStatus::None);
        Ok(())
    }

    #[test]
    fn test_capture_dependency_update_rejects_node_from_different_graph() -> Result<()> {
        let test_device = testing::bootstrap()?;
        let context = Arc::clone(test_device.context());

        let stream = context.create_stream(StreamOptions::default())?;
        let mut other_graph = context.create_graph()?;
        let other_node = other_graph.add_empty_node(&[])?;

        stream.begin_capture(StreamCaptureMode::Relaxed)?;
        assert!(matches!(
            stream.update_capture_dependencies(
                &[GraphDependency {
                    node: other_node,
                    data: GraphEdgeAttributes::default(),
                }],
                StreamCaptureDependencyUpdate::Add,
            ),
            Err(Error::GraphNodeMismatch)
        ));
        drop(stream.end_capture());
        assert_eq!(stream.capture_status()?, StreamCaptureStatus::None);
        Ok(())
    }

    #[test]
    fn test_capture_dependency_update_rejects_unassociated_raw_node() -> Result<()> {
        let test_device = testing::bootstrap()?;
        let context = Arc::clone(test_device.context());

        let stream = context.create_stream(StreamOptions::default())?;
        // SAFETY: the fake handle is only compared, never passed to CUDA.
        let raw_node = unsafe { GraphNode::from_raw(0x1usize as _) };

        stream.begin_capture(StreamCaptureMode::Relaxed)?;
        assert!(matches!(
            stream.update_capture_dependencies(
                &[GraphDependency {
                    node: raw_node,
                    data: GraphEdgeAttributes::default(),
                }],
                StreamCaptureDependencyUpdate::Add,
            ),
            Err(Error::GraphNodeMismatch)
        ));
        drop(stream.end_capture());
        assert_eq!(stream.capture_status()?, StreamCaptureStatus::None);
        Ok(())
    }

    #[test]
    fn test_captures_on_separate_streams_can_overlap() -> Result<()> {
        let test_device = testing::bootstrap()?;
        let context = Arc::clone(test_device.context());
        let stream_a = context.create_stream(StreamOptions::default())?;
        let stream_b = context.create_stream(StreamOptions::default())?;

        thread::scope(|scope| {
            let a = scope.spawn(|| {
                let graph = stream_a.capture(StreamCaptureMode::Relaxed, |_scope| Ok(()))?;
                assert_eq!(graph.context(), &context);
                Result::<()>::Ok(())
            });
            let b = scope.spawn(|| {
                let graph = stream_b.capture(StreamCaptureMode::Relaxed, |_scope| Ok(()))?;
                assert_eq!(graph.context(), &context);
                Result::<()>::Ok(())
            });

            a.join().expect("capture thread panicked")?;
            b.join().expect("capture thread panicked")?;
            Result::<()>::Ok(())
        })?;

        Ok(())
    }

    #[test]
    fn test_scoped_capture_error_leaves_stream_usable() -> Result<()> {
        let test_device = testing::bootstrap()?;
        let context = Arc::clone(test_device.context());
        let stream = context.create_stream(StreamOptions::default())?;

        let result = stream.capture(StreamCaptureMode::Relaxed, |_scope| {
            Err(Error::out_of_range("capture body"))
        });

        assert!(matches!(result, Err(Error::OutOfRange { .. })));
        stream.synchronize()?;
        Ok(())
    }

    #[test]
    fn test_scoped_capture_panic_leaves_stream_usable() -> Result<()> {
        let test_device = testing::bootstrap()?;
        let context = Arc::clone(test_device.context());
        let stream = context.create_stream(StreamOptions::default())?;

        let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            let _ = stream.capture(StreamCaptureMode::Relaxed, |_scope| -> Result<()> {
                panic!("capture body panic");
            });
        }));

        assert!(result.is_err());
        stream.synchronize()?;
        Ok(())
    }

    #[test]
    fn test_scoped_capture_records_memory_operations() -> Result<()> {
        let test_device = testing::bootstrap()?;
        let context = Arc::clone(test_device.context());
        let stream = context.create_stream(StreamOptions::default())?;

        let input = [1u8, 2, 3, 4];
        let source = DeviceMemory::from_slice(&context, &input)?;
        let copied = DeviceMemory::<u8>::zeroed(&context, input.len())?;
        let filled = DeviceMemory::<u8>::zeroed(&context, input.len())?;

        let graph = stream.capture(StreamCaptureMode::Relaxed, |scope| {
            // SAFETY: both buffers outlive the captured graph launches in this test.
            unsafe {
                DeviceMemory::copy_async(
                    copied.as_mut_ptr(),
                    source.as_ptr(),
                    input.len(),
                    MemoryCopyKind::DeviceToDevice,
                    scope.stream(),
                )?;
                DeviceMemory::<u8>::set_async(
                    filled.as_mut_ptr(),
                    0xab,
                    input.len(),
                    scope.stream(),
                )
            }
        })?;

        let executable = graph.instantiate(GraphInstantiateFlags::empty())?;
        // SAFETY: the buffers captured by the graph outlive the launch, which the test drains.
        unsafe {
            executable.launch_async(&stream)?;
        }
        stream.synchronize()?;

        assert_eq!(copied.copy_to_host_vec()?, input);
        assert_eq!(filled.copy_to_host_vec()?, [0xab; 4]);
        Ok(())
    }
}
