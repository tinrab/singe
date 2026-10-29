use std::sync::PoisonError;

pub mod memory;

use std::{
    any::Any,
    cell::RefCell,
    collections::HashMap,
    ffi,
    fmt::{self, Display, Formatter},
    hash::{Hash, Hasher},
    iter,
    marker::PhantomData,
    mem, panic, process, ptr,
    result::Result as StdResult,
    sync::{Arc, Mutex, MutexGuard, OnceLock, Weak},
};

use bitflags::parser;
use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{impl_enum_conversion, impl_enum_display};
use singe_cuda_sys::runtime;

use crate::{
    context::Context,
    device::Device,
    error::{Error, Result, Status},
    event::{Event, EventFlags, EventRecordFlags},
    graph::{
        ExecutableGraph, Graph, GraphDependency, GraphEdgeAttributes, GraphInstantiateFlags,
        GraphNode,
    },
    memory::DeferredFree,
    module::{CodeOwnerId, Module},
    submission::{Access, Execution, Resources},
    try_ffi,
    view::DeviceRegion,
};

bitflags::bitflags! {
    /// Flags for CUDA stream creation ([`Context::create_stream_with_flags`]).
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

/// State of a CUDA stream capture sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum StreamCaptureStatus {
    /// The stream is not capturing.
    None = runtime::cudaStreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE as _,
    /// The stream is recording work into a graph.
    Active = runtime::cudaStreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE as _,
    /// A capture rule was violated. End capture on the initiating stream before reusing it.
    Invalidated = runtime::cudaStreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_INVALIDATED as _,
}

impl_enum_conversion!(u32, runtime::cudaStreamCaptureStatus, StreamCaptureStatus);

/// Scope of CUDA's checks for potentially unsafe calls during stream capture.
///
/// Calls that necessarily conflict with capture remain prohibited in every mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum StreamCaptureMode {
    /// Checks the local capture sequence and concurrent captures begun in global mode on other host threads.
    Global = runtime::cudaStreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL as _,
    /// Checks the local capture sequence without considering captures on other host threads.
    ThreadLocal = runtime::cudaStreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL as _,
    /// Allows potentially unsafe CUDA calls while preserving restrictions on calls that necessarily conflict with capture.
    Relaxed = runtime::cudaStreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED as _,
}

impl_enum_conversion!(u32, runtime::cudaStreamCaptureMode, StreamCaptureMode);

/// Flags for [`Stream::update_capture_dependencies_with_dependencies`]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum StreamCaptureDependencyUpdate {
    /// Add new nodes to the dependency set.
    Add = runtime::cudaStreamUpdateCaptureDependenciesFlags::cudaStreamAddCaptureDependencies as _,
    /// Replace the dependency set with the new nodes.
    Set = runtime::cudaStreamUpdateCaptureDependenciesFlags::cudaStreamSetCaptureDependencies as _,
}

impl_enum_conversion!(
    u32,
    runtime::cudaStreamUpdateCaptureDependenciesFlags,
    StreamCaptureDependencyUpdate,
);

/// Capture status and the native ID of an active capture sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StreamCaptureInfo {
    /// Current capture status of the queried stream.
    pub status: StreamCaptureStatus,
    /// Native capture sequence ID. Meaningful only when `status` is [`StreamCaptureStatus::Active`].
    pub id: u64,
}

// Type alias for the trait object Box itself (inner box).
type RustStreamCallbackDyn = Box<dyn FnOnce(Result<()>) + Send + 'static>;

// Type alias for the pointer type stored in the outer box.
type BoxedCallbackPtr = *mut RustStreamCallbackDyn;

type RustHostFunctionDyn = Box<dyn FnOnce() + Send + 'static>;

type BoxedHostFunctionPtr = *mut RustHostFunctionDyn;

/// A shared CUDA stream retaining its actual context.
///
/// Clones share native ownership and capture cleanup. Owned handles are synchronized and destroyed when their last wrapper drops. Externally owned handles instead retain an ownership token and are not destroyed by this wrapper.
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
    // TODO: Store device ID? Could be useful for multi-GPU.
    // device_id: DeviceId,
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

/// A borrowed stream scope that drains queued work before borrowed storage can be released.
///
/// Constructed by [`Stream::sync_scope`]. Cleanup also drains during unwinding and aborts the process if completion cannot be established.
#[derive(Debug)]
pub struct StreamScope<'scope, 'env> {
    stream: &'scope Stream,
    _env: PhantomData<&'env mut &'env ()>,
}

struct ScopeDrain {
    stream: StreamBinding,
    drained: bool,
}

impl Drop for ScopeDrain {
    fn drop(&mut self) {
        if !self.drained && self.stream.synchronize().is_err() {
            // Returning or unwinding would release borrowed storage that CUDA
            // might still access. Unlike owned submissions, a scope cannot
            // retain or leak its caller's stack storage after it exits.
            process::abort();
        }
    }
}

/// An active stream capture and the kernel-code owners retained by recorded operations.
///
/// Constructed by [`Stream::capture`]. Recording does not execute producers or initialize their outputs.
#[derive(Debug)]
pub struct StreamCaptureScope<'scope> {
    stream: &'scope Stream,
    modules: RefCell<HashMap<CodeOwnerId, Module>>,
    _not_send: PhantomData<*const ()>,
}

/// Operation that may be recorded into a CUDA graph capture scope.
///
/// # Safety
///
/// Implementors must only enqueue CUDA work that is valid during stream
/// capture. Every pointer, handle, and side effect captured into the resulting
/// graph must have its replay safety contract represented by the operation's
/// type and constructor.
pub unsafe trait GraphRecordable {
    /// Value returned after recording the operation, without implying execution or completion.
    type Output;

    /// Records this operation in the supplied active capture.
    ///
    /// # Errors
    ///
    /// Returns operation-specific argument, context, or native capture errors. The implementation must preserve the captured operands' replay lifetimes even if some recording steps fail.
    fn record(self, scope: &StreamCaptureScope<'_>) -> Result<Self::Output>;
}

struct ActiveStreamCapture<'stream> {
    stream: &'stream Stream,
    finished: bool,
}

impl ActiveStreamCapture<'_> {
    fn finish(mut self) -> Result<Graph> {
        self.finished = true;
        self.stream.end_capture()
    }

    fn discard(mut self) {
        self.finished = true;
        drop(self.stream.end_capture());
    }
}

impl Drop for ActiveStreamCapture<'_> {
    fn drop(&mut self) {
        if !self.finished {
            drop(self.stream.end_capture());
        }
    }
}

/// A CUDA stream handle without native destruction responsibility.
///
/// Values returned by [`Stream::to_borrowed`] retain the owning stream. Raw borrowed streams rely on the lifetime contract of [`Self::from_raw`].
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

/// Release responsibility returned with a raw native stream.
pub enum RawStreamOwnership {
    /// The caller must eventually destroy the returned native handle.
    Owned,
    /// The external owner token must remain live through every use of the native handle.
    Borrowed(Arc<dyn Any + Send + Sync>),
}

impl fmt::Debug for RawStreamOwnership {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Owned => "Owned",
            Self::Borrowed(_) => "Borrowed",
        })
    }
}

/// Native stream parts retaining their actual context and any external owner.
/// Destroy owned handles explicitly; keep borrowed tokens alive through every use.
#[derive(Debug)]
pub struct RawStreamParts {
    /// Native stream handle whose release responsibility is described by `ownership`.
    pub handle: runtime::cudaStream_t,
    /// Actual CUDA context retained for native stream operations.
    pub context: Arc<Context>,
    /// Native destruction responsibility or external owner token.
    pub ownership: RawStreamOwnership,
}

struct ForeignOwner(Arc<dyn Any + Send + Sync>);

impl fmt::Debug for ForeignOwner {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("ForeignOwner")
            .field("references", &Arc::strong_count(&self.0))
            .finish_non_exhaustive()
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
        let handle = handle as usize;
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
    fn for_stream(key: StreamIdentity) -> Arc<Mutex<Self>> {
        type Registry = HashMap<StreamIdentity, Weak<Mutex<CaptureCleanup>>>;

        static LIFECYCLES: OnceLock<Mutex<Registry>> = OnceLock::new();
        let mut lifecycles = LIFECYCLES
            .get_or_init(Mutex::default)
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        lifecycles.retain(|_, lifecycle| lifecycle.strong_count() != 0);
        if let Some(lifecycle) = lifecycles.get(&key).and_then(Weak::upgrade) {
            return lifecycle;
        }
        let lifecycle = Arc::new(Mutex::new(Self::default()));
        lifecycles.insert(key, Arc::downgrade(&lifecycle));
        lifecycle
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

    /// Wraps an existing CUDA stream handle and takes ownership of it.
    ///
    /// Dropping the returned stream may block while synchronizing the stream
    /// before destruction. Use [`Stream::shutdown`] to surface synchronization
    /// or destruction errors explicitly.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid CUDA stream owned by `context`, and ownership of
    /// the handle is transferred to the returned [`Stream`]. The handle must
    /// not be destroyed elsewhere after calling this function.
    ///
    /// # Errors
    ///
    /// Returns `Error::NullHandle` if `handle` is null. The caller must satisfy the remaining handle validity and ownership requirements.
    pub unsafe fn from_raw(handle: runtime::cudaStream_t, context: Arc<Context>) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            inner: Arc::new(StreamInner {
                handle,
                capture_cleanup: CaptureCleanup::for_stream(StreamIdentity::new(&context, handle)),
                context,
                foreign_owner: None,
            }),
        })
    }

    /// Borrows a stream while retaining the library object that owns its native handle.
    /// Default legacy streams are accepted; per-thread default streams cannot be shared across threads.
    ///
    /// # Safety
    /// The handle must belong to `context` and remain valid while `owner` is retained.
    /// External work must obey this wrapper's access ordering and capture lifecycle.
    ///
    /// # Errors
    ///
    /// Returns `Error::ThreadLocalStreamUnsupported` for the per-thread default-stream sentinel. Also propagates context binding errors. Other handle validity and external ownership requirements remain the caller's responsibility.
    pub unsafe fn borrow_with_owner(
        handle: runtime::cudaStream_t,
        context: Arc<Context>,
        owner: Arc<dyn Any + Send + Sync>,
    ) -> Result<Self> {
        if handle as usize == PER_THREAD_STREAM {
            return Err(Error::ThreadLocalStreamUnsupported);
        }
        context.bind()?;
        Ok(Self {
            inner: Arc::new(StreamInner {
                handle,
                capture_cleanup: CaptureCleanup::for_stream(StreamIdentity::new(&context, handle)),
                context,
                foreign_owner: Some(ForeignOwner(owner)),
            }),
        })
    }

    #[cfg(feature = "interop_cudarc")]
    pub(crate) fn release_foreign_owner<T: Any + Send + Sync>(&self, owner: Arc<T>) -> Result<()> {
        let mut cleanup = self
            .inner
            .capture_cleanup
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        cleanup.foreign.push(ForeignOwner(owner));
        if self.capture_status()? == StreamCaptureStatus::None {
            cleanup.flush_frees(self.as_raw())?;
        }
        drop(cleanup);
        Ok(())
    }

    /// Defers destruction of an unused resource until capture ends, if active.
    /// This permits eager APIs to reject capture without calling a capture-incompatible
    /// destructor while releasing their arguments. It does not retain resources for
    /// graph replay. End capture through this stream wrapper to release these owners.
    /// Avoid owners that retain this stream, which would create a reference cycle.
    ///
    /// # Errors
    ///
    /// Propagates capture-state query and context binding errors. If capture is inactive, `owner` is dropped before this call returns.
    ///
    /// # Panics
    ///
    /// Panics if the shared capture-cleanup mutex is poisoned.
    pub fn defer_drop_until_capture_end<T>(&self, owner: Arc<T>) -> Result<()>
    where
        T: fmt::Debug + Send + Sync + 'static,
    {
        let mut cleanup = self
            .inner
            .capture_cleanup
            .lock()
            .expect("capture cleanup mutex poisoned");
        if self.capture_status()? != StreamCaptureStatus::None {
            cleanup.owners.push(owner);
        }
        drop(cleanup);
        Ok(())
    }

    pub(crate) fn allocation_guard(&self) -> Result<MutexGuard<'_, CaptureCleanup>> {
        let mut cleanup = self
            .inner
            .capture_cleanup
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        self.ensure_not_capturing()?;
        cleanup.flush_frees(self.as_raw())?;
        Ok(cleanup)
    }

    pub(crate) fn defer_free(&self, free: DeferredFree) -> Result<()> {
        let mut cleanup = self
            .inner
            .capture_cleanup
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
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

    /// Runs `f` with a stream scope and synchronizes this stream before returning.
    ///
    /// Use this for scoped asynchronous operations that borrow host or device
    /// memory until stream completion. For CUDA graph capture, use
    /// [`Stream::capture`] or [`Stream::capture_executable`].
    /// The completion wait also runs when `f` unwinds.
    /// If CUDA completion cannot be established, the process aborts before
    /// returning borrowed storage to the caller.
    ///
    /// # Errors
    ///
    /// Propagates capture-state query errors, rejects active or invalidated capture, and returns errors from `f` or stream synchronization. If both `f` and synchronization fail and cleanup can drain the stream, the closure error takes precedence.
    pub fn sync_scope<'env, F, R>(&self, f: F) -> Result<R>
    where
        F: for<'scope> FnOnce(&'scope StreamScope<'scope, 'env>) -> Result<R>,
    {
        let scope = StreamScope {
            stream: self,
            _env: PhantomData,
        };
        StreamBinding::Borrowed(self.to_borrowed()).sync_scope(|| f(&scope))
    }

    /// Blocks until stream has completed all operations.
    /// If [`ContextFlags::SCHEDULE_BLOCKING_SYNC`](crate::context::ContextFlags::SCHEDULE_BLOCKING_SYNC) was set for this device, the host thread will block until the stream is finished with all of its tasks.
    ///
    /// Uses standard `default stream` semantics.
    ///
    /// # Errors
    ///
    /// Returns an error if stream synchronization fails or if a previous asynchronous launch
    /// reported an error. CUDA may also return initialization-related errors such as
    /// [`crate::error::Status::NotInitialized`], [`crate::error::Status::CallRequiresNewerDriver`], or
    /// [`crate::error::Status::NoDevice`] if this call initializes internal runtime state. Callbacks must not
    /// call CUDA functions; see [`Stream::add_callback`].
    pub fn synchronize(&self) -> Result<()> {
        self.inner.context.bind()?;
        // SAFETY: `cudaStreamSynchronize` receives live handles owned by the wrappers.
        unsafe { try_ffi!(runtime::cudaStreamSynchronize(self.as_raw())) }
    }

    /// Synchronizes this stream and releases its native ownership.
    ///
    /// Owned handles are destroyed. Borrowed handles release their external
    /// owner token after synchronization instead of destroying the handle.
    /// This may block while waiting for stream work and callbacks. For owned
    /// handles, destruction is attempted even if synchronization fails, and the
    /// synchronization error takes precedence over a destruction error.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidValue` if another stream clone or borrowed owner still retains the handle. Also propagates capture-state rejection, deferred-free enqueue, context binding, synchronization, or owned-handle destruction errors. Synchronization errors take precedence over destruction errors.
    pub fn shutdown(self) -> Result<()> {
        let lifecycle = Arc::clone(&self.inner.capture_cleanup);
        let mut cleanup = lifecycle.lock().unwrap_or_else(PoisonError::into_inner);
        self.ensure_not_capturing()?;
        cleanup.flush_frees(self.as_raw())?;
        let mut inner = match Arc::try_unwrap(self.inner) {
            Ok(inner) => inner,
            Err(inner) => {
                // Releasing this reference can destroy the last wrapper after a concurrent drop.
                drop(cleanup);
                drop(inner);
                return Err(Error::InvalidValue);
            }
        };
        let result = if inner.foreign_owner.is_some() {
            inner
                .context
                .bind()
                // SAFETY: `cudaStreamSynchronize` receives only plain values.
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
    /// Returns an error if querying the stream fails or if a previous asynchronous launch reported
    /// an error. CUDA may also return initialization-related errors such as
    /// [`crate::error::Status::NotInitialized`], [`crate::error::Status::CallRequiresNewerDriver`], or
    /// [`crate::error::Status::NoDevice`] if this call initializes internal runtime state. Callbacks must not
    /// call CUDA functions; see [`Stream::add_callback`].
    pub fn query(&self) -> Result<bool> {
        self.inner.context.bind()?;
        // SAFETY: `cudaStreamQuery` receives live handles owned by the wrappers.
        let error = unsafe { runtime::cudaStreamQuery(self.as_raw()) };
        match error {
            runtime::cudaError_t::CUDA_SUCCESS => Ok(true),
            runtime::cudaError_t::CUDA_ERROR_NOT_READY => Ok(false),
            _ => Err(error.into()),
        }
    }

    /// Makes all future work submitted to stream wait for all work captured in event.
    /// See [`sys::cudaEventRecord`](singe_cuda_sys::runtime::cudaEventRecord) for details on what is captured by an event.
    /// Synchronization is performed efficiently on the device when applicable.
    /// `event` may be from a different device than `stream`.
    ///
    /// Uses standard `default stream` semantics.
    ///
    /// # Errors
    ///
    /// Returns an error if the stream cannot wait on the event or if a previous asynchronous launch
    /// reported an error. CUDA may also return initialization-related errors such as
    /// [`crate::error::Status::NotInitialized`], [`crate::error::Status::CallRequiresNewerDriver`], or
    /// [`crate::error::Status::NoDevice`] if this call initializes internal runtime state. Callbacks must not
    /// call CUDA functions; see [`Stream::add_callback`].
    pub fn wait_event(&self, event: &Event) -> Result<()> {
        self.wait_event_with_flags(event, 0)
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
        let branch = self
            .context()
            .create_stream_with_priority(StreamFlags::NON_BLOCKING, self.priority()?)?;
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
        let event = other
            .context()
            .create_event_with_flags(EventFlags::DISABLE_TIMING)?;
        event.record(other, EventRecordFlags::DEFAULT)?;
        self.wait_event(&event)
    }

    /// Orders subsequent work after the readiness streams of these device regions.
    ///
    /// All regions must belong to this stream's context.
    /// Each distinct producer stream contributes one event dependency.
    /// This does not establish initialization or retain allocations for later work.
    /// Borrowed consumers must keep their resources in a draining scope; raw queued
    /// consumers must uphold their resource and access contracts.
    ///
    /// # Errors
    ///
    /// Returns a context mismatch before submitting dependencies if any region is
    /// foreign, or a CUDA error if an event cannot be recorded or waited on.
    pub fn join_regions(&self, regions: &[DeviceRegion<'_>]) -> Result<()> {
        if regions
            .iter()
            .any(|region| region.context() != self.context())
        {
            return Err(Error::StreamContextMismatch);
        }
        for (index, region) in regions.iter().enumerate() {
            if let Some(producer) = region.stream()
                && producer != self
                && !regions[..index]
                    .iter()
                    .any(|earlier| earlier.stream() == Some(producer))
            {
                self.join(producer)?;
            }
        }
        Ok(())
    }

    /// Makes all future work submitted to stream wait for all work captured in event.
    /// See [`sys::cudaEventRecord`](singe_cuda_sys::runtime::cudaEventRecord) for details on what is captured by an event.
    /// `flags` controls how strictly the wait is enforced.
    /// Synchronization is performed efficiently on the device when applicable.
    /// `event` may be from a different device than `stream`.
    ///
    /// Uses standard `default stream` semantics.
    ///
    /// # Errors
    ///
    /// Returns an error if the stream cannot wait on the event or if a previous asynchronous launch
    /// reported an error. CUDA may also return initialization-related errors such as
    /// [`crate::error::Status::NotInitialized`], [`crate::error::Status::CallRequiresNewerDriver`], or
    /// [`crate::error::Status::NoDevice`] if this call initializes internal runtime state. Callbacks must not
    /// call CUDA functions; see [`Stream::add_callback`].
    pub fn wait_event_with_flags(&self, event: &Event, flags: u32) -> Result<()> {
        self.inner.context.bind()?;
        // SAFETY: `cudaStreamWaitEvent` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(runtime::cudaStreamWaitEvent(
                self.as_raw(),
                event.as_raw(),
                flags,
            ))
        }
    }

    /// Begin graph capture on stream.
    /// When a stream is in capture mode, operations pushed into the stream are captured
    /// into a graph instead of executed. [`Stream::end_capture`] returns the graph.
    /// Capture may not be initiated on the legacy default stream.
    /// Capture must be ended on the same stream in which it was initiated, and it may only be initiated if the stream is not already in capture mode.
    /// The capture mode may be queried via [`Stream::capture_status`].
    /// A unique id representing the capture sequence may be queried via [`Stream::capture_info`].
    ///
    /// If mode is not [`StreamCaptureMode::Relaxed`], [`Stream::end_capture`] must be called on this stream from the same thread.
    ///
    /// Kernels captured using this API must not use texture and surface references.
    /// Reading or writing through any texture or surface reference is undefined behavior.
    /// This restriction does not apply to texture and surface objects.
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound, capture cannot begin on
    /// this stream, the capture mode is invalid for the current thread state,
    /// or a previous asynchronous launch reports an error.
    pub fn begin_capture(&self, mode: StreamCaptureMode) -> Result<()> {
        let _lifecycle = self.allocation_guard()?;
        self.inner.context.bind()?;
        // SAFETY: `cudaStreamBeginCapture` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(runtime::cudaStreamBeginCapture(self.as_raw(), mode.into()))?;
        }
        Ok(())
    }

    /// Begins stream capture into an existing graph.
    ///
    /// # Safety
    ///
    /// This low-level API captures into `graph`'s existing CUDA handle. Calling
    /// [`Stream::end_capture`] after this may return that same raw handle; the
    /// caller must not wrap it as a second owned [`Graph`]. Prefer
    /// [`Stream::capture`] unless manually managing capture into an existing
    /// graph is required.
    ///
    /// # Errors
    ///
    /// Propagates graph-context, dependency ownership, capture lifecycle, context binding, and native capture errors from [`Self::begin_capture_to_graph_with_dependencies`].
    pub unsafe fn begin_capture_to_graph(
        &self,
        graph: &Graph,
        dependencies: &[GraphNode],
        mode: StreamCaptureMode,
    ) -> Result<()> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { self.begin_capture_to_graph_with_data(graph, dependencies, &[], mode) }
    }

    /// Begins stream capture into an existing graph with annotated dependency edges.
    ///
    /// # Safety
    ///
    /// This has the same ownership restrictions as
    /// [`Stream::begin_capture_to_graph`].
    ///
    /// # Errors
    ///
    /// Returns `Error::GraphDependencyMismatch` if nonempty edge data has a different length from `dependencies`. Also propagates graph-context, dependency ownership, capture lifecycle, context binding, and native capture errors.
    pub unsafe fn begin_capture_to_graph_with_data(
        &self,
        graph: &Graph,
        dependencies: &[GraphNode],
        edge_data: &[GraphEdgeAttributes],
        mode: StreamCaptureMode,
    ) -> Result<()> {
        if !edge_data.is_empty() && edge_data.len() != dependencies.len() {
            return Err(Error::GraphDependencyMismatch);
        }

        let dependencies: Vec<_> = dependencies
            .iter()
            .zip(
                edge_data
                    .iter()
                    .copied()
                    .chain(iter::repeat(GraphEdgeAttributes::default())),
            )
            .map(|(node, data)| GraphDependency {
                node: node.clone(),
                data,
            })
            .collect();

        // SAFETY: the caller upholds this function's safety contract.
        unsafe { self.begin_capture_to_graph_with_dependencies(graph, &dependencies, mode) }
    }

    /// Begin graph capture on stream.
    /// When a stream is in capture mode, operations pushed into the stream are captured
    /// into a graph instead of executed. [`Stream::end_capture`] returns the graph.
    ///
    /// Capture may not be initiated on the legacy default stream.
    /// Capture must be ended on the same stream in which it was initiated, and it may only be initiated if the stream is not already in capture mode.
    /// The capture mode may be queried via [`Stream::capture_status`].
    /// A unique id representing the capture sequence may be queried via [`Stream::capture_info`].
    ///
    /// If mode is not [`StreamCaptureMode::Relaxed`], [`Stream::end_capture`] must be called on this stream from the same thread.
    ///
    /// Kernels captured using this API must not use texture and surface references.
    /// Reading or writing through any texture or surface reference is undefined behavior.
    /// This restriction does not apply to texture and surface objects.
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound, capture cannot begin on
    /// this stream, the graph dependencies are invalid, the capture mode is
    /// invalid for the current thread state, or a previous asynchronous launch
    /// reports an error.
    /// # Safety
    ///
    /// This captures into `graph`'s existing CUDA handle. Calling
    /// [`Stream::end_capture`] after this may return that same raw handle; the
    /// caller must not wrap it as a second owned [`Graph`].
    pub unsafe fn begin_capture_to_graph_with_dependencies(
        &self,
        graph: &Graph,
        dependencies: &[GraphDependency],
        mode: StreamCaptureMode,
    ) -> Result<()> {
        let _lifecycle = self.allocation_guard()?;
        self.check_graph_context(graph)?;
        self.check_capture_dependency_contexts(dependencies)?;
        Self::check_capture_graph_dependencies(graph, dependencies)?;
        self.inner.context.bind()?;

        let dependencies_raw: Vec<_> = dependencies
            .iter()
            .map(|dependency| dependency.node.as_raw())
            .collect();
        let edge_data_raw: Vec<_> = dependencies
            .iter()
            .map(|dependency| dependency.data.into())
            .collect();
        // SAFETY: `cudaStreamBeginCaptureToGraph` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(runtime::cudaStreamBeginCaptureToGraph(
                self.as_raw(),
                graph.as_raw(),
                dependencies_raw.as_ptr(),
                if edge_data_raw.is_empty() {
                    ptr::null()
                } else {
                    edge_data_raw.as_ptr()
                },
                dependencies_raw.len() as _,
                mode.into(),
            ))?;
        }
        Ok(())
    }

    /// Ends capture on this stream, returning the captured graph.
    /// Capture must have been initiated on stream via a call to [`Stream::begin_capture`].
    /// If capture was invalidated due to a violation of the rules of stream capture, an error is returned.
    ///
    /// If the mode argument to [`Stream::begin_capture`] was not [`StreamCaptureMode::Relaxed`], this call must be from the same thread as [`Stream::begin_capture`].
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound, capture is not active
    /// on this stream, the capture has been invalidated, or a previous
    /// asynchronous launch reports an error.
    ///
    /// # Panics
    ///
    /// Panics if the shared capture-cleanup mutex is poisoned.
    pub fn end_capture(&self) -> Result<Graph> {
        self.inner.context.bind()?;
        let mut handle = ptr::null_mut();
        let mut cleanup = self
            .inner
            .capture_cleanup
            .lock()
            .expect("capture cleanup mutex poisoned");
        // SAFETY: `cudaStreamEndCapture` receives live handles owned by the wrappers and valid output locations.
        let result = unsafe {
            try_ffi!(runtime::cudaStreamEndCapture(
                self.as_raw(),
                &raw mut handle
            ))
        };
        // Invalidated captures also end the lifecycle; failed frees remain queued.
        let ended =
            result.is_ok() || matches!(self.capture_status(), Ok(StreamCaptureStatus::None));
        let (released, cleanup_result) = if ended {
            (
                mem::take(&mut cleanup.owners),
                cleanup.flush_frees(self.as_raw()),
            )
        } else {
            (Vec::new(), Ok(()))
        };
        drop(cleanup);
        drop(released);
        result?;
        // SAFETY: CUDA returned this graph for the context that the new value retains.
        let graph = unsafe { Graph::from_raw_in_context(handle, Arc::clone(&self.inner.context)) }?;
        cleanup_result?;
        Ok(graph)
    }

    /// Captures stream work recorded by `f` into a CUDA graph.
    ///
    /// This is the scoped form of [`Stream::begin_capture`] and
    /// [`Stream::end_capture`]. This method attempts to end capture before it
    /// returns or resumes a panic. If `f` returns an error, this method attempts
    /// to end capture to restore stream usability, destroys any graph returned
    /// by CUDA, and returns the closure error.
    ///
    /// The scope is `!Send`, so it cannot move to another thread while capture
    /// is active. It retains recorded kernel-code owners in the returned graph.
    ///
    /// # Errors
    ///
    /// Returns an error if capture cannot begin, if `f` returns an error, or if
    /// capture cannot be ended successfully.
    pub fn capture<F>(&self, mode: StreamCaptureMode, f: F) -> Result<Graph>
    where
        F: FnOnce(&StreamCaptureScope<'_>) -> Result<()>,
    {
        self.begin_capture(mode)?;
        let capture = ActiveStreamCapture {
            stream: self,
            finished: false,
        };

        let scope = StreamCaptureScope {
            stream: self,
            modules: RefCell::new(HashMap::new()),
            _not_send: PhantomData,
        };

        let capture_result = panic::catch_unwind(panic::AssertUnwindSafe(|| f(&scope)));
        match capture_result {
            Ok(Ok(())) => {
                let graph = capture.finish()?;
                for module in scope.modules.into_inner().into_values() {
                    graph.retain_module(&module);
                }
                Ok(graph)
            }
            Ok(Err(error)) => {
                capture.discard();
                Err(error)
            }
            Err(payload) => {
                drop(capture);
                panic::resume_unwind(payload);
            }
        }
    }

    /// Captures work recorded by `f` and instantiates the resulting graph with default flags.
    ///
    /// Recorded kernel-code owners are retained by the executable.
    ///
    /// # Errors
    ///
    /// Propagates capture setup, closure, capture finalization, and graph-instantiation errors.
    pub fn capture_executable<F>(&self, mode: StreamCaptureMode, f: F) -> Result<ExecutableGraph>
    where
        F: FnOnce(&StreamCaptureScope<'_>) -> Result<()>,
    {
        self.capture_executable_with_flags(mode, GraphInstantiateFlags::empty(), f)
    }

    /// Captures work recorded by `f` and instantiates the resulting graph with `flags`.
    ///
    /// Recorded kernel-code owners are retained by the executable.
    ///
    /// # Errors
    ///
    /// Propagates capture setup, closure, capture finalization, and graph-instantiation errors, including unsupported flags or invalid captured nodes.
    pub fn capture_executable_with_flags<F>(
        &self,
        mode: StreamCaptureMode,
        flags: GraphInstantiateFlags,
        f: F,
    ) -> Result<ExecutableGraph>
    where
        F: FnOnce(&StreamCaptureScope<'_>) -> Result<()>,
    {
        let graph = self.capture(mode, f)?;
        graph.instantiate_with_flags(flags)
    }

    /// Returns the capture status of this stream.
    /// After a successful call, the status is one of the following:
    ///
    /// * [`StreamCaptureStatus::None`]: The stream is not capturing.
    /// * [`StreamCaptureStatus::Active`]: The stream is capturing.
    /// * [`StreamCaptureStatus::Invalidated`]: The stream was capturing but an error has invalidated the capture sequence.
    ///   The capture sequence must be terminated with
    ///   [`Stream::end_capture`] on the stream where it was initiated to continue using the stream.
    ///
    /// If this is called on the legacy default stream while a blocking stream on the same device is capturing, it returns [`crate::error::Status::StreamCaptureImplicit`].
    /// The blocking stream capture is not invalidated.
    ///
    /// When a blocking stream is capturing, the legacy stream is in an unusable state until the blocking stream capture is terminated.
    /// The legacy stream is not supported for stream capture, but attempted use would have an implicit dependency on the capturing stream(s).
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound, CUDA cannot query the
    /// capture status, or a previous asynchronous launch reports an error.
    pub fn capture_status(&self) -> Result<StreamCaptureStatus> {
        self.inner.context.bind()?;
        let mut status = runtime::cudaStreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
        // SAFETY: `cudaStreamIsCapturing` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(runtime::cudaStreamIsCapturing(
                self.as_raw(),
                &raw mut status
            ))?;
        }
        Ok(status.into())
    }

    /// Query stream state related to stream capture.
    ///
    /// If called on the legacy default stream while a stream not created with [`StreamFlags::NON_BLOCKING`] is capturing, returns [`crate::error::Status::StreamCaptureImplicit`].
    ///
    /// Valid data (other than capture status) is returned only if both of the following are true:
    ///
    /// * the call succeeds
    /// * the returned capture status is [`StreamCaptureStatus::Active`]
    ///
    /// If there is non-zero edge data for one or more current stream dependencies and the query cannot return that data, the call returns [`crate::error::Status::LossyQuery`].
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound, CUDA cannot query the
    /// capture info, the query would lose non-zero edge data, or a previous
    /// asynchronous launch reports an error.
    pub fn capture_info(&self) -> Result<StreamCaptureInfo> {
        self.inner.context.bind()?;
        let mut status = runtime::cudaStreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
        let mut id = 0;
        // SAFETY: `cudaStreamGetCaptureInfo` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(runtime::cudaStreamGetCaptureInfo(
                self.as_raw(),
                &raw mut status,
                &raw mut id,
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
            ))?;
        }
        Ok(StreamCaptureInfo {
            status: status.into(),
            id,
        })
    }

    /// Adds dependency nodes for the next operation recorded on this capturing stream.
    ///
    /// Uses default edge attributes and [`StreamCaptureDependencyUpdate::Add`].
    ///
    /// # Errors
    ///
    /// Propagates dependency context or graph mismatches, capture-query, context binding, and native dependency-update errors.
    pub fn update_capture_dependencies(&self, dependencies: &[GraphNode]) -> Result<()> {
        self.update_capture_dependencies_with_mode(
            dependencies,
            &[],
            StreamCaptureDependencyUpdate::Add,
        )
    }

    /// Adds dependency nodes and their edge attributes for the next recorded operation.
    ///
    /// # Errors
    ///
    /// Returns `Error::GraphDependencyMismatch` if nonempty edge data has a different length from `dependencies`. Also propagates dependency context or graph mismatches, capture-query, context binding, and native dependency-update errors.
    pub fn update_capture_dependencies_with_data(
        &self,
        dependencies: &[GraphNode],
        edge_data: &[GraphEdgeAttributes],
    ) -> Result<()> {
        self.update_capture_dependencies_with_mode(
            dependencies,
            edge_data,
            StreamCaptureDependencyUpdate::Add,
        )
    }

    /// Adds or replaces the next recorded operation's dependency set using `mode`.
    ///
    /// Empty `edge_data` uses default attributes for every dependency.
    ///
    /// # Errors
    ///
    /// Returns `Error::GraphDependencyMismatch` if nonempty edge data has a different length from `dependencies`. Also propagates dependency context or graph mismatches, capture-query, context binding, and native dependency-update errors.
    pub fn update_capture_dependencies_with_mode(
        &self,
        dependencies: &[GraphNode],
        edge_data: &[GraphEdgeAttributes],
        mode: StreamCaptureDependencyUpdate,
    ) -> Result<()> {
        if !edge_data.is_empty() && edge_data.len() != dependencies.len() {
            return Err(Error::GraphDependencyMismatch);
        }

        let dependencies: Vec<_> = dependencies
            .iter()
            .zip(
                edge_data
                    .iter()
                    .copied()
                    .chain(iter::repeat(GraphEdgeAttributes::default())),
            )
            .map(|(node, data)| GraphDependency {
                node: node.clone(),
                data,
            })
            .collect();

        self.update_capture_dependencies_with_dependencies(&dependencies, mode)
    }

    /// Modifies the dependency set of a capturing stream.
    /// The dependency set is the set of nodes that the next captured node in the stream will depend on.
    ///
    /// Valid flags are [`StreamCaptureDependencyUpdate::Add`] and [`StreamCaptureDependencyUpdate::Set`].
    /// These control whether the supplied set is added to the existing set or replaces it.
    /// A flags value of 0 defaults to [`StreamCaptureDependencyUpdate::Add`].
    ///
    /// Nodes that are removed from the dependency set by this call do not result in [`crate::error::Status::StreamCaptureUnjoined`] if they are unreachable from the stream at [`Stream::end_capture`].
    ///
    /// Returns [`crate::error::Status::IllegalState`] if the stream is not capturing.
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound, the stream is not
    /// capturing, the supplied dependencies are invalid, or a previous
    /// asynchronous launch reports an error.
    pub fn update_capture_dependencies_with_dependencies(
        &self,
        dependencies: &[GraphDependency],
        mode: StreamCaptureDependencyUpdate,
    ) -> Result<()> {
        self.check_capture_dependency_contexts(dependencies)?;
        self.check_active_capture_graph_dependencies(dependencies)?;
        self.inner.context.bind()?;

        let mut dependencies_raw: Vec<_> = dependencies
            .iter()
            .map(|dependency| dependency.node.as_raw())
            .collect();
        let edge_data_raw: Vec<_> = dependencies
            .iter()
            .map(|dependency| dependency.data.into())
            .collect();
        // SAFETY: `cudaStreamUpdateCaptureDependencies` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(runtime::cudaStreamUpdateCaptureDependencies(
                self.as_raw(),
                dependencies_raw.as_mut_ptr(),
                if edge_data_raw.is_empty() {
                    ptr::null()
                } else {
                    edge_data_raw.as_ptr()
                },
                dependencies_raw.len() as _,
                mode.into(),
            ))?;
        }
        Ok(())
    }

    /// This callback API is slated for eventual deprecation and removal.
    /// If you do not require the callback to execute after a device error, consider using [`sys::cudaLaunchHostFunc`](singe_cuda_sys::runtime::cudaLaunchHostFunc).
    /// Additionally, this callback mechanism is not supported with [`Stream::begin_capture`] and [`Stream::end_capture`], unlike [`sys::cudaLaunchHostFunc`](singe_cuda_sys::runtime::cudaLaunchHostFunc).
    ///
    /// Adds a callback to be called on the host after all currently enqueued items in the stream have completed.
    /// For each [`Stream::add_callback`] call, a callback is executed exactly once.
    /// The callback blocks later work in the stream until it is finished.
    ///
    /// The callback may be passed a successful status or an error code.
    /// In the event of a device error, all subsequently executed callbacks receive an appropriate [`Status`].
    ///
    /// Callbacks must not call CUDA functions.
    /// Attempting to do so may result in [`crate::error::Status::NotPermitted`].
    /// Callbacks must not perform any synchronization that may depend on outstanding device work or other callbacks that are not mandated to run earlier.
    /// Callbacks without a mandated order (in independent streams) execute in undefined order and may be serialized.
    ///
    /// For the purposes of Unified Memory, callback execution makes a number of guarantees:
    ///
    /// * The callback stream is considered idle for the duration of the callback.
    ///   Thus, for example, a callback may always use memory
    ///   attached to the callback stream.
    /// * The start of execution of a callback has the same effect as synchronizing an event recorded in the same stream immediately
    ///   before the callback.
    ///   It thus synchronizes streams which have been "joined" before the callback.
    /// * Adding device work to any stream does not have the effect of making the stream active until all preceding callbacks have executed.
    ///   Thus, for example, a callback might use global attached memory even if work has been added to another stream, if it has been
    ///   properly ordered with an event.
    /// * Completion of a callback does not cause a stream to become active except as described above.
    ///   The callback stream will remain
    ///   idle if no device work follows the callback, and will remain idle across consecutive callbacks without device work in between.
    ///   Thus, for example, stream synchronization can be done by signaling from a callback at the end of the stream.
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound, CUDA rejects the
    /// callback registration, a previous asynchronous launch reports an error,
    /// or CUDA reports runtime initialization diagnostics such as
    /// [`crate::error::Status::NotInitialized`], [`crate::error::Status::CallRequiresNewerDriver`],
    /// or [`crate::error::Status::NoDevice`].
    pub fn add_callback<F>(&self, callback: F) -> Result<()>
    where
        F: FnOnce(Result<()>) + Send + 'static,
    {
        self.inner.context.bind()?;

        let boxed_dyn_callback: RustStreamCallbackDyn = Box::new(callback);
        let boxed_wrapper: Box<RustStreamCallbackDyn> = Box::new(boxed_dyn_callback);
        let user_data_ptr: BoxedCallbackPtr = Box::into_raw(boxed_wrapper);
        let final_user_data = user_data_ptr.cast();

        let flags = 0u32;

        // SAFETY: the boxed callback is leaked into CUDA and reclaimed exactly once by the trampoline.
        unsafe {
            let status = runtime::cudaStreamAddCallback(
                self.as_raw(),
                Some(stream_callback_trampoline),
                final_user_data, // Pass the thin pointer
                flags,
            );

            // If adding the callback fails, manually reconstruct and drop the *outer* Box to prevent leaking both boxes.
            if status != runtime::cudaError_t::CUDA_SUCCESS {
                // Reconstruct the outer box (Box<Box<dyn Trait>>)
                let _leaked_box = Box::from_raw(user_data_ptr);
                // Drop the reconstructed outer box.
                try_ffi!(status)?;
            }
        }

        Ok(())
    }

    /// Enqueues a host function to run after all currently enqueued work in this stream completes.
    ///
    /// Unlike [`Stream::add_callback`], CUDA does not call this function if the CUDA context is already in an error state.
    /// This API is supported during stream capture by CUDA, but the host function still must not call CUDA
    /// APIs or perform synchronization that depends on outstanding device work.
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound, CUDA rejects the host
    /// function registration, a previous asynchronous launch reports an error,
    /// or CUDA reports runtime initialization diagnostics.
    pub fn launch_host_func<F>(&self, function: F) -> Result<()>
    where
        F: FnOnce() + Send + 'static,
    {
        self.inner.context.bind()?;

        let boxed_dyn_function: RustHostFunctionDyn = Box::new(function);
        let boxed_wrapper: Box<RustHostFunctionDyn> = Box::new(boxed_dyn_function);
        let user_data_ptr: BoxedHostFunctionPtr = Box::into_raw(boxed_wrapper);
        let final_user_data = user_data_ptr.cast();

        // SAFETY: the boxed callback is leaked into CUDA and reclaimed exactly once by the trampoline.
        unsafe {
            let status = runtime::cudaLaunchHostFunc(
                self.as_raw(),
                Some(stream_host_function_trampoline),
                final_user_data,
            );

            if status != runtime::cudaError_t::CUDA_SUCCESS {
                let _leaked_box = Box::from_raw(user_data_ptr);
                try_ffi!(status)?;
            }
        }

        Ok(())
    }

    /// Query the flags of a stream.
    /// Returns the stream flags.
    /// See [`Context::create_stream_with_flags`] for a list of valid flags.
    ///
    /// Uses standard `default stream` semantics.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot query the flags or if a previous asynchronous launch
    /// reported an error. CUDA may also return initialization-related errors such as
    /// [`crate::error::Status::NotInitialized`], [`crate::error::Status::CallRequiresNewerDriver`], or
    /// [`crate::error::Status::NoDevice`] if this call initializes internal runtime state. Callbacks must not
    /// call CUDA functions; see [`Stream::add_callback`].
    pub fn flags(&self) -> Result<StreamFlags> {
        self.inner.context.bind()?;
        let mut flags_raw = 0u32;
        // SAFETY: `cudaStreamGetFlags` receives live handles owned by the wrappers and valid output locations.
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
    /// See [`Context::create_stream_with_priority`] for details about priority clamping.
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound, CUDA cannot query the
    /// priority, a previous asynchronous launch reports an error, or CUDA
    /// reports runtime initialization diagnostics.
    pub fn priority(&self) -> Result<i32> {
        self.inner.context.bind()?;
        let mut priority = 0i32;
        // SAFETY: `cudaStreamGetPriority` receives live handles owned by the wrappers and valid output locations.
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
    /// * a stream created via any of the CUDA runtime APIs such as [`sys::cudaStreamCreate`](singe_cuda_sys::runtime::cudaStreamCreate), [`Context::create_stream_with_flags`] and [`Context::create_stream_with_priority`], or their driver API equivalents such as [`sys::cuStreamCreate`](singe_cuda_sys::driver::cuStreamCreate) or [`sys::cuStreamCreateWithPriority`](singe_cuda_sys::driver::cuStreamCreateWithPriority).
    ///   Passing an invalid handle results in undefined behavior.
    /// * the special legacy default stream and per-thread default stream.
    ///   The driver API equivalents of these are also accepted.
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound, CUDA cannot query the
    /// stream identifier, a previous asynchronous launch reports an error, or
    /// CUDA reports runtime initialization diagnostics.
    pub fn id(&self) -> Result<u64> {
        self.inner.context.bind()?;
        let mut id = 0u64;
        // SAFETY: `cudaStreamGetId` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(runtime::cudaStreamGetId(self.as_raw(), &raw mut id))?;
        }
        Ok(id)
    }

    /// Returns the device of the stream.
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound, CUDA cannot query the
    /// stream device, a previous asynchronous launch reports an error, or CUDA
    /// reports runtime initialization diagnostics.
    pub fn device(&self) -> Result<Device> {
        self.inner.context.bind()?;
        let mut device = 0i32;
        // SAFETY: `cudaStreamGetDevice` receives live handles owned by the wrappers and valid output locations.
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

    /// Transfers the native handle, context and destruction responsibility to the caller.
    /// Destroy owned handles explicitly; retain borrowed ownership tokens through every use.
    /// Returns the unchanged stream if another clone or pending capture cleanup still retains it.
    ///
    /// # Errors
    ///
    /// Returns `Err(self)` if another stream clone retains the wrapper or shared capture cleanup still holds deferred owners, frees, or foreign tokens. The returned wrapper remains usable and retains its release responsibility.
    pub fn into_raw(self) -> StdResult<RawStreamParts, Self> {
        let mut inner = match Arc::try_unwrap(self.inner) {
            Ok(inner) => inner,
            Err(inner) => return Err(Self { inner }),
        };
        if !inner
            .capture_cleanup
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .is_empty()
        {
            return Err(Self {
                inner: Arc::new(inner),
            });
        }
        let handle = mem::replace(&mut inner.handle, ptr::null_mut());
        let ownership = match inner.foreign_owner.take() {
            Some(ForeignOwner(owner)) => RawStreamOwnership::Borrowed(owner),
            None => RawStreamOwnership::Owned,
        };
        Ok(RawStreamParts {
            handle,
            context: Arc::clone(&inner.context),
            ownership,
        })
    }

    fn destroy_handle(context: &Context, handle: runtime::cudaStream_t) -> Result<()> {
        let bind_result = context.bind();
        let sync_result =
            // SAFETY: `cudaStreamSynchronize` receives only plain values.
            bind_result.and_then(|()| unsafe { try_ffi!(runtime::cudaStreamSynchronize(handle)) });
        // SAFETY: `cudaStreamDestroy` receives only plain values.
        let destroy_result = unsafe { try_ffi!(runtime::cudaStreamDestroy(handle)) };

        match (sync_result, destroy_result) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(error), _) | (Ok(()), Err(error)) => Err(error),
        }
    }

    // pub fn is_null(&self) -> bool {
    //     self.inner.handle.is_null()
    // }

    // --- Methods related to Memory Management ---
    // Add methods like malloc_async, free_async, attach_mem_async if needed
    // These would likely take wrappers around device memory pointers.

    fn check_graph_context(&self, graph: &Graph) -> Result<()> {
        if matches!(graph.context(), Some(context) if context != &self.inner.context) {
            return Err(Error::GraphContextMismatch);
        }
        Ok(())
    }

    fn check_capture_dependency_contexts(&self, dependencies: &[GraphDependency]) -> Result<()> {
        for dependency in dependencies {
            if matches!(dependency.node.context(), Some(context) if context != &self.inner.context)
            {
                return Err(Error::GraphContextMismatch);
            }
        }
        Ok(())
    }

    fn check_capture_graph_dependencies(
        graph: &Graph,
        dependencies: &[GraphDependency],
    ) -> Result<()> {
        for dependency in dependencies {
            graph.check_node(&dependency.node)?;
        }
        Ok(())
    }

    fn check_active_capture_graph_dependencies(
        &self,
        dependencies: &[GraphDependency],
    ) -> Result<()> {
        if dependencies.is_empty() {
            return Ok(());
        }

        self.inner.context.bind()?;
        let mut status = runtime::cudaStreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
        let mut graph = ptr::null_mut();
        // SAFETY: `cudaStreamGetCaptureInfo` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(runtime::cudaStreamGetCaptureInfo(
                self.as_raw(),
                &raw mut status,
                ptr::null_mut(),
                &raw mut graph,
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
            ))?;
        }
        if StreamCaptureStatus::from(status) != StreamCaptureStatus::Active {
            return Ok(());
        }
        if graph.is_null() {
            return Err(Error::NullHandle);
        }

        for dependency in dependencies {
            if !matches!(dependency.node.graph_raw(), Some(node_graph) if node_graph == graph) {
                return Err(Error::GraphNodeMismatch);
            }
        }
        Ok(())
    }

    /// Rejects active or invalidated capture before an operation that cannot be recorded.
    ///
    /// Call this before synchronous work or ordinary allocation so a rejected
    /// operation does not first change the capturing stream's state.
    ///
    /// # Errors
    ///
    /// Returns the native stream-capture-unsupported status for active capture or the stream-capture-invalidated status for invalidated capture. Also propagates capture-state query and context binding errors.
    pub fn ensure_not_capturing(&self) -> Result<()> {
        match self.capture_status()? {
            StreamCaptureStatus::None => Ok(()),
            StreamCaptureStatus::Active => Err(Status::StreamCaptureUnsupported.into()),
            StreamCaptureStatus::Invalidated => Err(Status::StreamCaptureInvalidated.into()),
        }
    }
}

impl<'scope> StreamScope<'scope, '_> {
    pub(crate) unsafe fn enqueue<T>(
        &self,
        accesses: &[Access],
        enqueue: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        let mut resources = Resources::new();
        resources.accesses.extend_from_slice(accesses);
        let mut execution = Execution::create(self.stream)?;
        // Caller borrows stay alive until the scope drains, including error and unwind paths.
        // SAFETY: the caller upholds this function's safety contract.
        let output = unsafe { execution.enqueue(resources, |_| enqueue())? };
        execution.finish(())?;
        Ok(output)
    }

    /// Returns the stream borrowed by this synchronous scope.
    #[must_use]
    pub const fn stream(&self) -> &'scope Stream {
        self.stream
    }

    /// Waits for queued work on the scope's stream without ending the scope.
    ///
    /// # Errors
    ///
    /// Propagates context binding and native stream synchronization errors from [`Stream::synchronize`].
    pub fn synchronize(&self) -> Result<()> {
        self.stream.synchronize()
    }
}

impl<'scope> StreamCaptureScope<'scope> {
    pub(crate) fn retain_module(&self, module: &Module) {
        self.modules
            .borrow_mut()
            .entry(module.code_owner_id())
            .or_insert_with(|| module.clone());
    }

    /// Returns the stream currently being captured.
    pub const fn stream(&self) -> &'scope Stream {
        self.stream
    }

    /// Records a graph-safe operation into this active stream capture.
    ///
    /// Only operations implementing [`GraphRecordable`] can be submitted
    /// through this method. Allocation/free and other capture-unsafe CUDA calls
    /// should stay outside this trait unless their replay ownership and address
    /// stability are explicitly modeled.
    ///
    /// # Errors
    ///
    /// Propagates errors from the operation's [`GraphRecordable::record`] implementation.
    pub fn record<O>(&self, operation: O) -> Result<O::Output>
    where
        O: GraphRecordable,
    {
        operation.record(self)
    }
}

impl BorrowedStream {
    /// Wraps an existing CUDA stream handle without taking ownership.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid CUDA stream associated with `context`, and it must
    /// remain valid for every operation using the returned borrowed stream.
    pub const unsafe fn from_raw(handle: runtime::cudaStream_t, context: Arc<Context>) -> Self {
        Self {
            handle,
            context,
            _owner: None,
        }
    }

    /// Waits for preceding work on the selected native stream.
    ///
    /// # Errors
    ///
    /// Returns an error if the retained context cannot be bound, native synchronization fails, or CUDA reports prior asynchronous errors.
    pub fn synchronize(&self) -> Result<()> {
        self.context.bind()?;
        // SAFETY: `cudaStreamSynchronize` receives live handles owned by the wrappers.
        unsafe { try_ffi!(runtime::cudaStreamSynchronize(self.as_raw())) }
    }

    /// Returns the context retained for this native stream selection.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        &self.context
    }

    /// Returns the selected native stream handle without transferring ownership.
    #[must_use]
    pub const fn as_raw(&self) -> runtime::cudaStream_t {
        self.handle
    }
}

impl StreamBinding {
    /// Completes borrowed work on this stream before returning or unwinding.
    ///
    /// Capture is rejected before calling `f`. Default and externally borrowed
    /// streams use the same completion contract as owned streams. If completion
    /// cannot be established, the process aborts before releasing borrowed storage.
    ///
    /// # Errors
    ///
    /// Converts capture-query, context binding, capture rejection, and stream synchronization errors into `E`. Errors returned by `f` pass through after cleanup drains the stream and take precedence over a synchronization error if both occur.
    pub fn sync_scope<R, E>(&self, f: impl FnOnce() -> StdResult<R, E>) -> StdResult<R, E>
    where
        E: From<Error>,
    {
        match self.capture_status()? {
            StreamCaptureStatus::None => {}
            StreamCaptureStatus::Active => {
                return Err(Error::from(Status::StreamCaptureUnsupported).into());
            }
            StreamCaptureStatus::Invalidated => {
                return Err(Error::from(Status::StreamCaptureInvalidated).into());
            }
        }
        let mut drain = ScopeDrain {
            stream: self.clone(),
            drained: false,
        };
        let result = f();
        let completion = self.synchronize();
        drain.drained = completion.is_ok();
        drop(drain);
        match (result, completion) {
            (Ok(value), Ok(())) => Ok(value),
            (Err(error), _) => Err(error),
            (Ok(_), Err(error)) => Err(error.into()),
        }
    }

    /// Reports capture state without invalidating an active capture.
    ///
    /// # Errors
    ///
    /// Returns an error if the retained context cannot be bound or CUDA cannot query capture state, including implicit capture on the legacy stream or prior asynchronous failures.
    pub fn capture_status(&self) -> Result<StreamCaptureStatus> {
        self.context().bind()?;
        let mut status = runtime::cudaStreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
        // SAFETY: `cudaStreamIsCapturing` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(runtime::cudaStreamIsCapturing(
                self.as_raw(),
                &raw mut status
            ))?;
        }
        Ok(status.into())
    }

    /// Waits for this explicit stream or the context's default stream.
    ///
    /// # Errors
    ///
    /// Returns an error if the retained context cannot be bound, native synchronization fails, or CUDA reports prior asynchronous errors.
    pub fn synchronize(&self) -> Result<()> {
        self.context().bind()?;
        // SAFETY: `cudaStreamSynchronize` receives live handles owned by the wrappers.
        unsafe { try_ffi!(runtime::cudaStreamSynchronize(self.as_raw())) }
    }

    /// Returns the context retained for this native stream selection.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        match self {
            Self::Default(context) => context,
            Self::Borrowed(stream) => stream.context(),
        }
    }

    /// Returns whether this binding uses the [`Self::Default`] variant.
    #[must_use]
    pub fn is_default(&self) -> bool {
        matches!(self, Self::Default(..))
    }

    /// Returns the selected native stream handle without transferring ownership.
    #[must_use]
    pub fn as_raw(&self) -> runtime::cudaStream_t {
        match self {
            Self::Default(_) => ptr::null_mut(),
            Self::Borrowed(stream) => stream.as_raw(),
        }
    }
}

// SAFETY: CUDA streams are ordering handles.
// Operations take &self and are serialized by CUDA stream semantics rather than mutable Rust state.
unsafe impl Send for StreamInner {}

unsafe impl Sync for StreamInner {}

unsafe impl Send for Stream {}

unsafe impl Sync for Stream {}

impl Drop for StreamInner {
    fn drop(&mut self) {
        if self.handle.is_null() && self.foreign_owner.is_none() {
            return;
        }
        let mut cleanup = self
            .capture_cleanup
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if Arc::strong_count(&self.capture_cleanup) == 1 && !cleanup.is_empty() {
            let mut graph = ptr::null_mut();
            // SAFETY: `cudaStreamEndCapture`, `cudaGraphDestroy`, `cudaStreamIsCapturing` receive live handles owned by the wrappers and valid output locations.
            // SAFETY: the last stream owner ends any capture before freeing deferred storage.
            let ended = self.context.bind().and_then(|()| unsafe {
                let _ = runtime::cudaStreamEndCapture(self.handle, &raw mut graph);
                if !graph.is_null() {
                    let _ = runtime::cudaGraphDestroy(graph);
                }
                let mut status = runtime::cudaStreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
                try_ffi!(runtime::cudaStreamIsCapturing(self.handle, &raw mut status))?;
                if status != runtime::cudaStreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE {
                    return Err(Error::InvalidValue);
                }
                Ok(())
            });
            if ended.is_err() {
                // CUDA may still retain captured addresses. Preserve their owners.
                mem::forget(mem::take(&mut *cleanup));
            } else {
                if let Err(error) = cleanup.flush_frees(self.handle) {
                    #[cfg(debug_assertions)]
                    eprintln!("failed to enqueue deferred cuda frees: {error}");
                    mem::forget(mem::take(&mut cleanup.frees));
                }
                cleanup.owners.clear();
            }
        }
        drop(cleanup);
        let result = if self.foreign_owner.is_some() {
            // SAFETY: `cudaStreamIsCapturing`, `cudaStreamSynchronize` receive live handles owned by the wrappers and valid output locations.
            self.context.bind().and_then(|()| unsafe {
                let mut status = runtime::cudaStreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
                try_ffi!(runtime::cudaStreamIsCapturing(self.handle, &raw mut status))?;
                if status == runtime::cudaStreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE {
                    try_ffi!(runtime::cudaStreamSynchronize(self.handle))?;
                }
                Ok(())
            })
        } else {
            Stream::destroy_handle(self.context.as_ref(), self.handle)
        };
        if let Err(error) = result {
            #[cfg(debug_assertions)]
            eprintln!("failed to synchronize or destroy cuda stream: {error}");
        }
    }
}

// Trampoline function to bridge C FFI callback to Rust closure
extern "C" fn stream_callback_trampoline(
    _stream: runtime::cudaStream_t,
    status: runtime::cudaError_t,
    user_data: *mut ffi::c_void,
) {
    if user_data.is_null() {
        return;
    }

    let user_data_ptr = user_data as BoxedCallbackPtr;

    // SAFETY: CUDA calls this trampoline once with the pointer leaked when the callback was enqueued.
    let boxed_callback: Box<RustStreamCallbackDyn> = unsafe { Box::from_raw(user_data_ptr) };
    let callback: RustStreamCallbackDyn = *boxed_callback;

    let result = if status == runtime::cudaError_t::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(status.into())
    };

    callback(result);
}

extern "C" fn stream_host_function_trampoline(user_data: *mut ffi::c_void) {
    if user_data.is_null() {
        return;
    }

    let user_data_ptr = user_data as BoxedHostFunctionPtr;
    // SAFETY: CUDA calls this trampoline once with the pointer leaked when the callback was enqueued.
    let boxed_function: Box<RustHostFunctionDyn> = unsafe { Box::from_raw(user_data_ptr) };
    let function: RustHostFunctionDyn = *boxed_function;
    function();
}

impl Context {
    /// Creates a stream in this context with [`StreamFlags::DEFAULT`].
    ///
    /// # Errors
    ///
    /// Propagates context binding and native stream creation errors from [`Self::create_stream_with_flags`].
    pub fn create_stream(self: &Arc<Self>) -> Result<Stream> {
        self.create_stream_with_flags(StreamFlags::DEFAULT)
    }

    /// Creates an asynchronous stream in this context and makes this context current on the calling thread.
    /// The flags argument determines the behaviors of the stream.
    /// Valid values are provided by [`StreamFlags`]:
    ///
    /// * [`StreamFlags::DEFAULT`]: default stream creation behavior.
    /// * [`StreamFlags::NON_BLOCKING`]: allows the created stream to run concurrently with the legacy default stream without implicit synchronization.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot create the stream, if it does not return a valid stream
    /// handle, or if a previous asynchronous launch reported an error. CUDA may also return
    /// initialization-related errors such as [`crate::error::Status::NotInitialized`],
    /// [`crate::error::Status::CallRequiresNewerDriver`], or [`crate::error::Status::NoDevice`] if this call initializes
    /// internal runtime state. Callbacks must not call CUDA functions; see
    /// [`Stream::add_callback`].
    pub fn create_stream_with_flags(self: &Arc<Self>, flags: StreamFlags) -> Result<Stream> {
        self.bind()?;
        let mut handle = ptr::null_mut();
        // SAFETY: `cudaStreamCreateWithFlags` receives valid output locations.
        unsafe {
            try_ffi!(runtime::cudaStreamCreateWithFlags(
                &raw mut handle,
                flags.bits(),
            ))?;
        }
        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        // SAFETY: let mut device_id = 0;
        // unsafe { check(cudaStreamGetDevice(stream, &mut device_id))?; }
        unsafe { Stream::from_raw(handle, Arc::clone(self)) }
    }

    /// Creates a stream with the specified priority.
    /// The stream is created on this context.
    /// This affects the scheduling priority of work in the stream.
    /// Priorities provide a hint to preferentially run work with higher priority when possible, but do not preempt already-running work or provide any other functional guarantee on execution order.
    ///
    /// `priority` follows a convention where lower numbers represent higher priorities.
    /// `0` represents default priority.
    /// The range of meaningful numerical priorities can be queried using [`Device::stream_priority_range`].
    /// If the specified priority is outside the numerical range returned by [`Device::stream_priority_range`], it will automatically be clamped to the lowest or the highest number in the range.
    ///
    /// * Stream priorities are supported only on GPUs with compute capability 3.5 or higher.
    /// * In the current implementation, only compute kernels launched in priority streams are affected by the stream's priority.
    ///   Stream
    ///   priorities have no effect on host-to-device and device-to-host memory operations.
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound, CUDA cannot create the
    /// stream, CUDA returns a null stream handle, a previous asynchronous launch
    /// reports an error, or CUDA reports runtime initialization diagnostics.
    pub fn create_stream_with_priority(
        self: &Arc<Self>,
        flags: StreamFlags,
        priority: i32,
    ) -> Result<Stream> {
        self.bind()?;
        let mut handle = ptr::null_mut();
        // SAFETY: `cudaStreamCreateWithPriority` receives valid output locations.
        unsafe {
            try_ffi!(runtime::cudaStreamCreateWithPriority(
                &raw mut handle,
                flags.bits(),
                priority,
            ))?;
        }
        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        // SAFETY: CUDA just created this handle in this context, and nothing else owns it.
        unsafe { Stream::from_raw(handle, Arc::clone(self)) }
    }
}

/// Sets the calling thread's stream capture interaction mode, returning the previous mode for the thread.
/// To facilitate deterministic behavior across function or module boundaries, use this in a push-pop fashion.
///
/// During stream capture (see [`Stream::begin_capture`]), some actions, such as a call to [`DeviceMemory::alloc`](crate::memory::DeviceMemory::alloc), may be unsafe.
/// In the case of [`DeviceMemory::alloc`](crate::memory::DeviceMemory::alloc), the operation is not enqueued asynchronously to a stream, and is not observed by stream capture.
/// If the sequence of operations captured via [`Stream::begin_capture`] depended on the allocation being replayed whenever the graph is launched, the captured graph would be invalid.
///
/// Therefore, stream capture places restrictions on CUDA calls that can be made within or concurrently to a [`Stream::begin_capture`]-[`Stream::end_capture`] sequence.
/// Control this behavior with this function and flags to [`Stream::begin_capture`].
///
/// A thread's mode is one of the following:
///
/// * [`StreamCaptureMode::Global`]: default mode.
///   If the local thread has an ongoing capture sequence that was not initiated with [`StreamCaptureMode::Relaxed`] at [`Stream::begin_capture`], or if any other thread has a concurrent capture sequence initiated with [`StreamCaptureMode::Global`], this thread is prohibited from potentially unsafe CUDA calls.
/// * [`StreamCaptureMode::ThreadLocal`]: If the local thread has an ongoing capture sequence not initiated with [`StreamCaptureMode::Relaxed`], it is prohibited from potentially unsafe CUDA calls.
///   Concurrent capture sequences in other threads are ignored.
/// * [`StreamCaptureMode::Relaxed`]: The local thread is not prohibited from potentially unsafe CUDA calls.
///   The thread is still prohibited from CUDA calls
///   which necessarily conflict with stream capture, for example, attempting [`Event::query`] on an event that was last recorded inside a capture sequence.
///
/// # Errors
///
/// Returns an error if CUDA rejects the capture-mode exchange or if a previous asynchronous launch
/// reported an error.
pub fn exchange_capture_mode(mode: StreamCaptureMode) -> Result<StreamCaptureMode> {
    let mut mode_raw: runtime::cudaStreamCaptureMode = mode.into();
    // SAFETY: `cudaThreadExchangeStreamCaptureMode` receives valid output locations.
    unsafe {
        try_ffi!(runtime::cudaThreadExchangeStreamCaptureMode(
            &raw mut mode_raw
        ))?;
    }
    Ok(mode_raw.into())
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use std::{
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        },
        thread,
    };

    use super::*;
    use crate::{event::EventRecordFlags, memory::DeviceMemory, testing};

    #[test]
    fn test_it_works() -> Result<()> {
        let (_lock, context) = testing::bootstrap()?;
        let stream1 = context.create_stream()?;
        let _stream2 = context.create_stream_with_flags(StreamFlags::NON_BLOCKING)?;

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
        let (_lock, context) = testing::bootstrap()?;
        let stream = context.create_stream()?;
        let event = context.create_event()?;

        event.record(&stream, EventRecordFlags::DEFAULT)?;
        stream.synchronize()?;

        assert!(event.query()?);
        Ok(())
    }

    #[test]
    fn test_shutdown_synchronizes_and_destroys_stream() -> Result<()> {
        let (_lock, context) = testing::bootstrap()?;
        let stream = context.create_stream()?;

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
        let (_lock, context) = testing::bootstrap()?;
        let stream = context.create_stream()?;

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
        let (_lock, context) = testing::bootstrap()?;
        let stream = context.create_stream()?;

        let graph = stream.capture(StreamCaptureMode::Relaxed, |scope| {
            assert_eq!(scope.stream().context(), &context);
            Ok(())
        })?;

        assert_eq!(graph.context(), Some(&context));
        Ok(())
    }

    #[test]
    fn test_capture_to_graph_rejects_graph_from_different_context() -> Result<()> {
        let (_lock, context) = testing::bootstrap()?;
        let other_context = Context::create()?;

        let stream = context.create_stream()?;
        let graph = other_context.create_graph()?;

        assert!(matches!(
            // SAFETY: the test ends the capture before the graph is dropped.
            unsafe { stream.begin_capture_to_graph(&graph, &[], StreamCaptureMode::Relaxed) },
            Err(Error::GraphContextMismatch)
        ));
        assert_eq!(stream.capture_status()?, StreamCaptureStatus::None);
        Ok(())
    }

    #[test]
    fn test_capture_to_graph_rejects_node_from_different_graph() -> Result<()> {
        let (_lock, context) = testing::bootstrap()?;

        let stream = context.create_stream()?;
        let graph = context.create_graph()?;
        let mut other_graph = context.create_graph()?;
        let other_node = other_graph.add_empty_node(&[])?;

        assert!(matches!(
            // SAFETY: the test ends the capture before the graph is dropped.
            unsafe {
                stream.begin_capture_to_graph(&graph, &[other_node], StreamCaptureMode::Relaxed)
            },
            Err(Error::GraphNodeMismatch)
        ));
        assert_eq!(stream.capture_status()?, StreamCaptureStatus::None);
        Ok(())
    }

    #[test]
    fn test_capture_to_graph_rejects_unassociated_raw_dependency_node() -> Result<()> {
        let (_lock, context) = testing::bootstrap()?;

        let stream = context.create_stream()?;
        let graph = context.create_graph()?;
        // SAFETY: the fake handle is only compared, never passed to CUDA.
        let raw_node = unsafe { GraphNode::from_raw(0x1usize as _) };

        assert!(matches!(
            // SAFETY: the test ends the capture before the graph is dropped.
            unsafe {
                stream.begin_capture_to_graph(&graph, &[raw_node], StreamCaptureMode::Relaxed)
            },
            Err(Error::GraphNodeMismatch)
        ));
        assert_eq!(stream.capture_status()?, StreamCaptureStatus::None);
        Ok(())
    }

    #[test]
    fn test_capture_dependency_update_rejects_node_from_different_context() -> Result<()> {
        let (_lock, context) = testing::bootstrap()?;
        let other_context = Context::create()?;

        let stream = context.create_stream()?;
        let mut other_graph = other_context.create_graph()?;
        let other_node = other_graph.add_empty_node(&[])?;

        let result = stream.capture(StreamCaptureMode::Relaxed, |_scope| {
            stream.update_capture_dependencies(&[other_node])
        });

        assert!(matches!(result, Err(Error::GraphContextMismatch)));
        assert_eq!(stream.capture_status()?, StreamCaptureStatus::None);
        Ok(())
    }

    #[test]
    fn test_capture_dependency_update_rejects_node_from_different_graph() -> Result<()> {
        let (_lock, context) = testing::bootstrap()?;

        let stream = context.create_stream()?;
        let mut other_graph = context.create_graph()?;
        let other_node = other_graph.add_empty_node(&[])?;

        stream.begin_capture(StreamCaptureMode::Relaxed)?;
        assert!(matches!(
            stream.update_capture_dependencies(&[other_node]),
            Err(Error::GraphNodeMismatch)
        ));
        drop(stream.end_capture());
        assert_eq!(stream.capture_status()?, StreamCaptureStatus::None);
        Ok(())
    }

    #[test]
    fn test_capture_dependency_update_rejects_unassociated_raw_node() -> Result<()> {
        let (_lock, context) = testing::bootstrap()?;

        let stream = context.create_stream()?;
        // SAFETY: the fake handle is only compared, never passed to CUDA.
        let raw_node = unsafe { GraphNode::from_raw(0x1usize as _) };

        stream.begin_capture(StreamCaptureMode::Relaxed)?;
        assert!(matches!(
            stream.update_capture_dependencies(&[raw_node]),
            Err(Error::GraphNodeMismatch)
        ));
        drop(stream.end_capture());
        assert_eq!(stream.capture_status()?, StreamCaptureStatus::None);
        Ok(())
    }

    #[test]
    fn test_captures_on_separate_streams_can_overlap() -> Result<()> {
        let (_lock, context) = testing::bootstrap()?;
        let stream_a = context.create_stream()?;
        let stream_b = context.create_stream()?;

        thread::scope(|scope| {
            let a = scope.spawn(|| {
                let graph = stream_a.capture(StreamCaptureMode::Relaxed, |_scope| Ok(()))?;
                assert_eq!(graph.context(), Some(&context));
                Result::<()>::Ok(())
            });
            let b = scope.spawn(|| {
                let graph = stream_b.capture(StreamCaptureMode::Relaxed, |_scope| Ok(()))?;
                assert_eq!(graph.context(), Some(&context));
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
        let (_lock, context) = testing::bootstrap()?;
        let stream = context.create_stream()?;

        let result = stream.capture(StreamCaptureMode::Relaxed, |_scope| {
            Err(Error::InvalidValue)
        });

        assert!(matches!(result, Err(Error::InvalidValue)));
        stream.synchronize()?;
        Ok(())
    }

    #[test]
    fn test_scoped_capture_panic_leaves_stream_usable() -> Result<()> {
        let (_lock, context) = testing::bootstrap()?;
        let stream = context.create_stream()?;

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
        let (_lock, context) = testing::bootstrap()?;
        let stream = context.create_stream()?;

        let input = [1u8, 2, 3, 4];
        let source = DeviceMemory::from_slice(&context, &input)?;
        let mut copied = DeviceMemory::<u8>::zeroed(&context, input.len())?;
        let mut filled = DeviceMemory::<u8>::zeroed(&context, input.len())?;

        let graph = stream.capture(StreamCaptureMode::Relaxed, |scope| {
            // SAFETY: both buffers outlive the captured graph launches in this test.
            let copy = unsafe { copied.copy_from_device_operation(&source)? };
            scope.record(copy)?;

            // SAFETY: both buffers outlive the captured graph launches in this test.
            let memset = unsafe { filled.set_value_operation(0xab) };
            scope.record(memset)
        })?;

        let executable = graph.instantiate()?;
        // SAFETY: the buffers captured by the graph outlive the launch, which the test drains.
        unsafe {
            executable.launch(&stream)?;
        }
        stream.synchronize()?;

        assert_eq!(copied.copy_to_host_vec()?, input);
        assert_eq!(filled.copy_to_host_vec()?, [0xab; 4]);
        Ok(())
    }
}

impl_enum_display!(StreamCaptureStatus, {
    StreamCaptureStatus::None => "CU_STREAM_CAPTURE_STATUS_NONE",
    StreamCaptureStatus::Active => "CU_STREAM_CAPTURE_STATUS_ACTIVE",
    StreamCaptureStatus::Invalidated => "CU_STREAM_CAPTURE_STATUS_INVALIDATED",
});

impl_enum_display!(StreamCaptureMode, {
    StreamCaptureMode::Global => "CU_STREAM_CAPTURE_MODE_GLOBAL",
    StreamCaptureMode::ThreadLocal => "CU_STREAM_CAPTURE_MODE_THREAD_LOCAL",
    StreamCaptureMode::Relaxed => "CU_STREAM_CAPTURE_MODE_RELAXED",
});

impl_enum_display!(StreamCaptureDependencyUpdate, {
    StreamCaptureDependencyUpdate::Add => "cudaStreamAddCaptureDependencies",
    StreamCaptureDependencyUpdate::Set => "cudaStreamSetCaptureDependencies",
});
