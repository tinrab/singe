#[allow(unused_imports)]
use crate::error::Status;

use std::{iter, marker::PhantomData, mem::ManuallyDrop, panic, ptr, sync::Arc};

use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::impl_enum_conversion;
use singe_cuda_sys::runtime;

use crate::{
    context::Context,
    device::Device,
    error::{Error, Result},
    event::Event,
    graph::{
        ExecutableGraph, Graph, GraphDependency, GraphEdgeData, GraphInstantiateFlags, GraphNode,
    },
    try_ffi,
};

bitflags::bitflags! {
    /// Flags for CUDA stream creation ([`Context::create_stream_with_flags`]).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct StreamFlags: u32 {
        const DEFAULT = runtime::cudaStreamDefault;
        const NON_BLOCKING = runtime::cudaStreamNonBlocking;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum StreamCaptureStatus {
    None = runtime::cudaStreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE as _,
    Active = runtime::cudaStreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE as _,
    Invalidated = runtime::cudaStreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_INVALIDATED as _,
}

impl_enum_conversion!(u32, runtime::cudaStreamCaptureStatus, StreamCaptureStatus);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum StreamCaptureMode {
    Global = runtime::cudaStreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL as _,
    ThreadLocal = runtime::cudaStreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL as _,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StreamCaptureInfo {
    pub status: StreamCaptureStatus,
    pub id: u64,
}

// Type alias for the trait object Box itself (inner box).
type RustStreamCallbackDyn = Box<dyn FnOnce(Result<()>) + Send + 'static>;

// Type alias for the pointer type stored in the outer box.
type BoxedCallbackPtr = *mut RustStreamCallbackDyn;

type RustHostFunctionDyn = Box<dyn FnOnce() + Send + 'static>;
type BoxedHostFunctionPtr = *mut RustHostFunctionDyn;

#[derive(Debug, Clone)]
pub struct Stream {
    inner: Arc<StreamInner>,
}

#[derive(Debug)]
struct StreamInner {
    handle: runtime::cudaStream_t,
    ctx: Arc<Context>,
    // TODO: Store device ID? Could be useful for multi-GPU.
    // device_id: DeviceId,
}

impl PartialEq for Stream {
    fn eq(&self, other: &Self) -> bool {
        self.as_raw() == other.as_raw() && Arc::ptr_eq(&self.inner.ctx, &other.inner.ctx)
    }
}

impl Eq for Stream {}

#[derive(Debug)]
pub struct StreamScope<'scope, 'env> {
    stream: &'scope Stream,
    _env: PhantomData<&'env mut &'env ()>,
}

#[derive(Debug)]
pub struct StreamCaptureScope<'scope> {
    stream: &'scope Stream,
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
    type Output;

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

#[derive(Debug, Clone)]
pub struct BorrowedStream {
    handle: runtime::cudaStream_t,
    ctx: Arc<Context>,
}

#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum StreamBinding {
    Default(Arc<Context>),
    Borrowed(BorrowedStream),
}

impl Stream {
    /// Wraps an existing CUDA stream handle and takes ownership of it.
    ///
    /// Dropping the returned stream may block while synchronizing the stream
    /// before destruction. Use [`Stream::shutdown`] to surface synchronization
    /// or destruction errors explicitly.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid CUDA stream owned by `ctx`, and ownership of
    /// the handle is transferred to the returned [`Stream`]. The handle must
    /// not be destroyed elsewhere after calling this function.
    pub unsafe fn from_raw(handle: runtime::cudaStream_t, ctx: Arc<Context>) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            inner: Arc::new(StreamInner { handle, ctx }),
        })
    }

    pub fn to_borrowed(&self) -> BorrowedStream {
        unsafe { BorrowedStream::from_raw(self.as_raw(), Arc::clone(&self.inner.ctx)) }
    }

    /// Runs `f` with a stream scope and synchronizes this stream before returning.
    ///
    /// Use this for scoped asynchronous operations that borrow host or device
    /// memory until stream completion. For CUDA graph capture, use
    /// [`Stream::capture`] or [`Stream::capture_executable`].
    pub fn sync_scope<'env, F, R>(&self, f: F) -> Result<R>
    where
        F: for<'scope> FnOnce(&'scope StreamScope<'scope, 'env>) -> Result<R>,
    {
        let scope = StreamScope {
            stream: self,
            _env: PhantomData,
        };
        let result = f(&scope);
        let sync_result = self.synchronize();

        match (result, sync_result) {
            (Ok(value), Ok(())) => Ok(value),
            (Ok(_), Err(err)) => Err(err),
            (Err(err), Ok(())) | (Err(err), Err(_)) => Err(err),
        }
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
        self.inner.ctx.bind()?;
        unsafe { try_ffi!(runtime::cudaStreamSynchronize(self.as_raw())) }
    }

    /// Synchronizes this stream, destroys it, and returns any CUDA error.
    ///
    /// This is the explicit version of the cleanup normally performed by
    /// [`Drop`]. It may block while waiting for stream work and callbacks to
    /// complete. If synchronization fails, destruction is still attempted and
    /// the synchronization error is returned. If synchronization succeeds but
    /// destruction fails, the destruction error is returned.
    pub fn shutdown(self) -> Result<()> {
        let inner = Arc::try_unwrap(self.inner).map_err(|_| Error::InvalidValue)?;
        let inner = ManuallyDrop::new(inner);
        Self::destroy_handle(inner.ctx.as_ref(), inner.handle)
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
        self.inner.ctx.bind()?;
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
        self.inner.ctx.bind()?;
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
    pub unsafe fn begin_capture_to_graph(
        &self,
        graph: &Graph,
        dependencies: &[GraphNode],
        mode: StreamCaptureMode,
    ) -> Result<()> {
        unsafe { self.begin_capture_to_graph_with_data(graph, dependencies, &[], mode) }
    }

    /// Begins stream capture into an existing graph with annotated dependency edges.
    ///
    /// # Safety
    ///
    /// This has the same ownership restrictions as
    /// [`Stream::begin_capture_to_graph`].
    pub unsafe fn begin_capture_to_graph_with_data(
        &self,
        graph: &Graph,
        dependencies: &[GraphNode],
        edge_data: &[GraphEdgeData],
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
                    .chain(iter::repeat(GraphEdgeData::default())),
            )
            .map(|(node, data)| GraphDependency {
                node: node.clone(),
                data,
            })
            .collect();

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
        self.check_graph_context(graph)?;
        self.check_capture_dependency_contexts(dependencies)?;
        self.check_capture_graph_dependencies(graph, dependencies)?;
        self.inner.ctx.bind()?;

        let dependencies_raw: Vec<_> = dependencies
            .iter()
            .map(|dependency| dependency.node.as_raw())
            .collect();
        let edge_data_raw: Vec<_> = dependencies
            .iter()
            .map(|dependency| dependency.data.into())
            .collect();
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
    pub fn end_capture(&self) -> Result<Graph> {
        self.inner.ctx.bind()?;
        let mut handle = ptr::null_mut();
        unsafe {
            try_ffi!(runtime::cudaStreamEndCapture(
                self.as_raw(),
                &raw mut handle
            ))?;
            Graph::from_raw_in_context(handle, Arc::clone(&self.inner.ctx))
        }
    }

    /// Captures stream work recorded by `f` into a CUDA graph.
    ///
    /// This is the scoped form of [`Stream::begin_capture`] and
    /// [`Stream::end_capture`]. The capture is always ended before this method
    /// returns or resumes a panic. If `f` returns an error, this method attempts
    /// to end capture to restore stream usability, destroys any graph returned
    /// by CUDA, and returns the closure error.
    ///
    /// The scope is intentionally `!Send`, so it cannot be moved to another
    /// thread while capture is active. Future graph-safe recording helpers can
    /// be added to [`StreamCaptureScope`] without changing this API shape.
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
            _not_send: PhantomData,
        };

        let capture_result = panic::catch_unwind(panic::AssertUnwindSafe(|| f(&scope)));
        match capture_result {
            Ok(Ok(())) => capture.finish(),
            Ok(Err(err)) => {
                capture.discard();
                Err(err)
            }
            Err(payload) => {
                drop(capture);
                panic::resume_unwind(payload);
            }
        }
    }

    pub fn capture_executable<F>(&self, mode: StreamCaptureMode, f: F) -> Result<ExecutableGraph>
    where
        F: FnOnce(&StreamCaptureScope<'_>) -> Result<()>,
    {
        self.capture_executable_with_flags(mode, GraphInstantiateFlags::empty(), f)
    }

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
        self.inner.ctx.bind()?;
        let mut status = runtime::cudaStreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
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
        self.inner.ctx.bind()?;
        let mut status = runtime::cudaStreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
        let mut id = 0;
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

    pub fn update_capture_dependencies(&self, dependencies: &[GraphNode]) -> Result<()> {
        self.update_capture_dependencies_with_mode(
            dependencies,
            &[],
            StreamCaptureDependencyUpdate::Add,
        )
    }

    pub fn update_capture_dependencies_with_data(
        &self,
        dependencies: &[GraphNode],
        edge_data: &[GraphEdgeData],
    ) -> Result<()> {
        self.update_capture_dependencies_with_mode(
            dependencies,
            edge_data,
            StreamCaptureDependencyUpdate::Add,
        )
    }

    pub fn update_capture_dependencies_with_mode(
        &self,
        dependencies: &[GraphNode],
        edge_data: &[GraphEdgeData],
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
                    .chain(iter::repeat(GraphEdgeData::default())),
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
        self.inner.ctx.bind()?;

        let mut dependencies_raw: Vec<_> = dependencies
            .iter()
            .map(|dependency| dependency.node.as_raw())
            .collect();
        let edge_data_raw: Vec<_> = dependencies
            .iter()
            .map(|dependency| dependency.data.into())
            .collect();
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
        self.inner.ctx.bind()?;

        let boxed_dyn_callback: RustStreamCallbackDyn = Box::new(callback);
        let boxed_wrapper: Box<RustStreamCallbackDyn> = Box::new(boxed_dyn_callback);
        let user_data_ptr: BoxedCallbackPtr = Box::into_raw(boxed_wrapper);
        let final_user_data = user_data_ptr.cast();

        let flags = 0u32;

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
        self.inner.ctx.bind()?;

        let boxed_dyn_function: RustHostFunctionDyn = Box::new(function);
        let boxed_wrapper: Box<RustHostFunctionDyn> = Box::new(boxed_dyn_function);
        let user_data_ptr: BoxedHostFunctionPtr = Box::into_raw(boxed_wrapper);
        let final_user_data = user_data_ptr.cast();

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
        self.inner.ctx.bind()?;
        let mut flags_raw = 0u32;
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
        self.inner.ctx.bind()?;
        let mut priority = 0i32;
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
        self.inner.ctx.bind()?;
        let mut id = 0u64;
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
        self.inner.ctx.bind()?;
        let mut device = 0i32;
        unsafe {
            try_ffi!(runtime::cudaStreamGetDevice(self.as_raw(), &raw mut device))?;
        }
        Ok(Device::new(device))
    }

    pub fn context(&self) -> &Context {
        &self.inner.ctx
    }

    pub fn as_raw(&self) -> runtime::cudaStream_t {
        self.inner.handle
    }

    /// Consumes the stream and returns the raw CUDA stream handle without
    /// destroying it.
    ///
    /// The caller becomes responsible for eventually destroying the returned
    /// handle with CUDA.
    pub fn into_raw(self) -> runtime::cudaStream_t {
        let inner = Arc::try_unwrap(self.inner)
            .expect("cannot transfer raw stream handle while cloned stream handles exist");
        let inner = ManuallyDrop::new(inner);
        inner.handle
    }

    fn destroy_handle(ctx: &Context, handle: runtime::cudaStream_t) -> Result<()> {
        let bind_result = ctx.bind();
        let sync_result =
            bind_result.and_then(|()| unsafe { try_ffi!(runtime::cudaStreamSynchronize(handle)) });
        let destroy_result = unsafe { try_ffi!(runtime::cudaStreamDestroy(handle)) };

        match (sync_result, destroy_result) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(err), _) | (Ok(()), Err(err)) => Err(err),
        }
    }

    // pub fn is_null(&self) -> bool {
    //     self.inner.handle.is_null()
    // }

    // --- Methods related to Memory Management ---
    // Add methods like malloc_async, free_async, attach_mem_async if needed
    // These would likely take wrappers around device memory pointers.

    fn check_graph_context(&self, graph: &Graph) -> Result<()> {
        if matches!(graph.context(), Some(ctx) if ctx != self.inner.ctx.as_ref()) {
            return Err(Error::GraphContextMismatch);
        }
        Ok(())
    }

    fn check_capture_dependency_contexts(&self, dependencies: &[GraphDependency]) -> Result<()> {
        for dependency in dependencies {
            if matches!(dependency.node.context(), Some(ctx) if ctx != self.inner.ctx.as_ref()) {
                return Err(Error::GraphContextMismatch);
            }
        }
        Ok(())
    }

    fn check_capture_graph_dependencies(
        &self,
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

        self.inner.ctx.bind()?;
        let mut status = runtime::cudaStreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
        let mut graph = ptr::null_mut();
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

    pub(crate) fn ensure_not_capturing_for_future(&self) -> Result<()> {
        match self.capture_status()? {
            StreamCaptureStatus::None => Ok(()),
            StreamCaptureStatus::Active => Err(Status::StreamCaptureUnsupported.into()),
            StreamCaptureStatus::Invalidated => Err(Status::StreamCaptureInvalidated.into()),
        }
    }
}

impl<'scope, 'env> StreamScope<'scope, 'env> {
    pub const fn stream(&self) -> &'scope Stream {
        self.stream
    }

    pub fn synchronize(&self) -> Result<()> {
        self.stream.synchronize()
    }
}

impl<'scope> StreamCaptureScope<'scope> {
    pub const fn stream(&self) -> &'scope Stream {
        self.stream
    }

    /// Records a graph-safe operation into this active stream capture.
    ///
    /// Only operations implementing [`GraphRecordable`] can be submitted
    /// through this method. Allocation/free and other capture-unsafe CUDA calls
    /// should stay outside this trait unless their replay ownership and address
    /// stability are explicitly modeled.
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
    /// `handle` must be a valid CUDA stream associated with `ctx`, and it must
    /// remain valid for every operation using the returned borrowed stream.
    pub const unsafe fn from_raw(handle: runtime::cudaStream_t, ctx: Arc<Context>) -> Self {
        Self { handle, ctx }
    }

    pub fn synchronize(&self) -> Result<()> {
        self.ctx.bind()?;
        unsafe { try_ffi!(runtime::cudaStreamSynchronize(self.as_raw())) }
    }

    pub fn context(&self) -> &Context {
        &self.ctx
    }

    pub const fn as_raw(&self) -> runtime::cudaStream_t {
        self.handle
    }
}

impl StreamBinding {
    pub fn context(&self) -> &Context {
        match self {
            Self::Default(ctx) => ctx.as_ref(),
            Self::Borrowed(stream) => stream.context(),
        }
    }

    pub fn is_default(&self) -> bool {
        matches!(self, Self::Default(..))
    }

    pub fn as_raw(&self) -> runtime::cudaStream_t {
        match self {
            Self::Default(_) => ptr::null_mut(),
            Self::Borrowed(stream) => stream.as_raw(),
        }
    }
}

// CUDA streams are ordering handles.
// Operations take &self and are serialized by CUDA stream semantics rather than mutable Rust state.
unsafe impl Send for StreamInner {}
unsafe impl Sync for StreamInner {}
unsafe impl Send for Stream {}
unsafe impl Sync for Stream {}

impl Drop for StreamInner {
    fn drop(&mut self) {
        if let Err(err) = Stream::destroy_handle(self.ctx.as_ref(), self.handle) {
            #[cfg(debug_assertions)]
            eprintln!("failed to synchronize or destroy CUDA stream: {err}");
        }
    }
}

// Trampoline function to bridge C FFI callback to Rust closure
extern "C" fn stream_callback_trampoline(
    _stream: runtime::cudaStream_t,
    status: runtime::cudaError_t,
    user_data: *mut std::ffi::c_void,
) {
    if user_data.is_null() {
        return;
    }

    let user_data_ptr = user_data as BoxedCallbackPtr;

    let boxed_callback: Box<RustStreamCallbackDyn> = unsafe { Box::from_raw(user_data_ptr) };
    let callback: RustStreamCallbackDyn = *boxed_callback;

    let result = if status == runtime::cudaError_t::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(status.into())
    };

    callback(result);
}

extern "C" fn stream_host_function_trampoline(user_data: *mut std::ffi::c_void) {
    if user_data.is_null() {
        return;
    }

    let user_data_ptr = user_data as BoxedHostFunctionPtr;
    let boxed_function: Box<RustHostFunctionDyn> = unsafe { Box::from_raw(user_data_ptr) };
    let function: RustHostFunctionDyn = *boxed_function;
    function();
}

impl Context {
    pub fn create_stream(self: &Arc<Self>) -> Result<Stream> {
        self.create_stream_with_flags(StreamFlags::DEFAULT)
    }

    /// Creates a new asynchronous stream on the context that is current to the calling host thread.
    /// If no context is current to the calling host thread, then the primary context for a device is selected, made current to the calling thread, and initialized before creating a stream on it.
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
        unsafe {
            try_ffi!(runtime::cudaStreamCreateWithFlags(
                &raw mut handle,
                flags.bits(),
            ))?;
        }
        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        // let mut device_id = 0;
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
    fn it_works() -> Result<()> {
        let (_lock, ctx) = testing::bootstrap()?;
        let stream1 = ctx.create_stream()?;
        let _stream2 = ctx.create_stream_with_flags(StreamFlags::NON_BLOCKING)?;

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
    fn event_query_uses_event_context() -> Result<()> {
        let (_lock, ctx) = testing::bootstrap()?;
        let stream = ctx.create_stream()?;
        let event = ctx.create_event()?;

        event.record(&stream, EventRecordFlags::DEFAULT)?;
        stream.synchronize()?;

        assert!(event.query()?);
        Ok(())
    }

    #[test]
    fn shutdown_synchronizes_and_destroys_stream() -> Result<()> {
        let (_lock, ctx) = testing::bootstrap()?;
        let stream = ctx.create_stream()?;

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
    fn launch_host_func_runs_after_stream_work() -> Result<()> {
        let (_lock, ctx) = testing::bootstrap()?;
        let stream = ctx.create_stream()?;

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
    fn scoped_capture_returns_context_associated_graph() -> Result<()> {
        let (_lock, ctx) = testing::bootstrap()?;
        let stream = ctx.create_stream()?;

        let graph = stream.capture(StreamCaptureMode::Relaxed, |scope| {
            assert_eq!(scope.stream().context(), ctx.as_ref());
            Ok(())
        })?;

        assert_eq!(graph.context(), Some(ctx.as_ref()));
        Ok(())
    }

    #[test]
    fn capture_to_graph_rejects_graph_from_different_context() -> Result<()> {
        let (_lock, ctx) = testing::bootstrap()?;
        let other_ctx = Context::create()?;

        let stream = ctx.create_stream()?;
        let graph = other_ctx.create_graph()?;

        assert!(matches!(
            unsafe { stream.begin_capture_to_graph(&graph, &[], StreamCaptureMode::Relaxed) },
            Err(Error::GraphContextMismatch)
        ));
        assert_eq!(stream.capture_status()?, StreamCaptureStatus::None);
        Ok(())
    }

    #[test]
    fn capture_to_graph_rejects_node_from_different_graph() -> Result<()> {
        let (_lock, ctx) = testing::bootstrap()?;

        let stream = ctx.create_stream()?;
        let graph = ctx.create_graph()?;
        let mut other_graph = ctx.create_graph()?;
        let other_node = other_graph.add_empty_node(&[])?;

        assert!(matches!(
            unsafe {
                stream.begin_capture_to_graph(&graph, &[other_node], StreamCaptureMode::Relaxed)
            },
            Err(Error::GraphNodeMismatch)
        ));
        assert_eq!(stream.capture_status()?, StreamCaptureStatus::None);
        Ok(())
    }

    #[test]
    fn capture_to_graph_rejects_unassociated_raw_dependency_node() -> Result<()> {
        let (_lock, ctx) = testing::bootstrap()?;

        let stream = ctx.create_stream()?;
        let graph = ctx.create_graph()?;
        let raw_node = unsafe { GraphNode::from_raw(0x1usize as _) };

        assert!(matches!(
            unsafe {
                stream.begin_capture_to_graph(&graph, &[raw_node], StreamCaptureMode::Relaxed)
            },
            Err(Error::GraphNodeMismatch)
        ));
        assert_eq!(stream.capture_status()?, StreamCaptureStatus::None);
        Ok(())
    }

    #[test]
    fn capture_dependency_update_rejects_node_from_different_context() -> Result<()> {
        let (_lock, ctx) = testing::bootstrap()?;
        let other_ctx = Context::create()?;

        let stream = ctx.create_stream()?;
        let mut other_graph = other_ctx.create_graph()?;
        let other_node = other_graph.add_empty_node(&[])?;

        let result = stream.capture(StreamCaptureMode::Relaxed, |_scope| {
            stream.update_capture_dependencies(&[other_node])
        });

        assert!(matches!(result, Err(Error::GraphContextMismatch)));
        assert_eq!(stream.capture_status()?, StreamCaptureStatus::None);
        Ok(())
    }

    #[test]
    fn capture_dependency_update_rejects_node_from_different_graph() -> Result<()> {
        let (_lock, ctx) = testing::bootstrap()?;

        let stream = ctx.create_stream()?;
        let mut other_graph = ctx.create_graph()?;
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
    fn capture_dependency_update_rejects_unassociated_raw_node() -> Result<()> {
        let (_lock, ctx) = testing::bootstrap()?;

        let stream = ctx.create_stream()?;
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
    fn captures_on_separate_streams_can_overlap() -> Result<()> {
        let (_lock, ctx) = testing::bootstrap()?;
        let stream_a = ctx.create_stream()?;
        let stream_b = ctx.create_stream()?;

        thread::scope(|scope| {
            let a = scope.spawn(|| {
                let graph = stream_a.capture(StreamCaptureMode::Relaxed, |_scope| Ok(()))?;
                assert_eq!(graph.context(), Some(ctx.as_ref()));
                Result::<()>::Ok(())
            });
            let b = scope.spawn(|| {
                let graph = stream_b.capture(StreamCaptureMode::Relaxed, |_scope| Ok(()))?;
                assert_eq!(graph.context(), Some(ctx.as_ref()));
                Result::<()>::Ok(())
            });

            a.join().expect("capture thread panicked")?;
            b.join().expect("capture thread panicked")?;
            Result::<()>::Ok(())
        })?;

        Ok(())
    }

    #[test]
    fn scoped_capture_error_leaves_stream_usable() -> Result<()> {
        let (_lock, ctx) = testing::bootstrap()?;
        let stream = ctx.create_stream()?;

        let result = stream.capture(StreamCaptureMode::Relaxed, |_scope| {
            Err(Error::InvalidValue)
        });

        assert!(matches!(result, Err(Error::InvalidValue)));
        stream.synchronize()?;
        Ok(())
    }

    #[test]
    fn scoped_capture_panic_leaves_stream_usable() -> Result<()> {
        let (_lock, ctx) = testing::bootstrap()?;
        let stream = ctx.create_stream()?;

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
    fn scoped_capture_records_memory_operations() -> Result<()> {
        let (_lock, ctx) = testing::bootstrap()?;
        let stream = ctx.create_stream()?;

        let input = [1u8, 2, 3, 4];
        let source = DeviceMemory::from_slice(&input)?;
        let mut copied = DeviceMemory::<u8>::zeroes(input.len())?;
        let mut filled = DeviceMemory::<u8>::zeroes(input.len())?;

        let graph = stream.capture(StreamCaptureMode::Relaxed, |scope| {
            let copy = unsafe { copied.copy_from_device_operation(&source)? };
            scope.record(copy)?;

            let memset = unsafe { filled.set_value_operation(0xab) };
            scope.record(memset)
        })?;

        let executable = graph.instantiate()?;
        executable.launch(&stream)?;
        stream.synchronize()?;

        assert_eq!(copied.copy_to_host_vec()?, input);
        assert_eq!(filled.copy_to_host_vec()?, [0xab; 4]);
        Ok(())
    }
}
