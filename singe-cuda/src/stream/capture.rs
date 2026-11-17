use std::{
    cell::RefCell, collections::HashMap, hash::Hash, marker::PhantomData, mem, panic, ptr,
    sync::Arc,
};

use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{impl_enum_display, report_cleanup_failure};
use singe_cuda_sys::runtime;

use super::{AsStream, Stream};
use crate::{
    error::{Error, Result},
    graph::{ExecutableGraph, Graph, GraphDependency, GraphInstantiateFlags},
    module::{CodeOwnerId, Module},
    try_ffi,
};

singe_core::native_enum! {
    /// State of a CUDA stream capture sequence.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum StreamCaptureStatus: runtime::cudaStreamCaptureStatus {
        /// The stream is not capturing.
        None = CU_STREAM_CAPTURE_STATUS_NONE,
        /// The stream is recording work into a graph.
        Active = CU_STREAM_CAPTURE_STATUS_ACTIVE,
        /// A capture rule was violated.
        /// End capture on the initiating stream before reusing it.
        Invalidated = CU_STREAM_CAPTURE_STATUS_INVALIDATED,
    }
}

singe_core::native_enum! {
    /// Scope of CUDA's checks for potentially unsafe calls during stream capture.
    ///
    /// Calls that necessarily conflict with capture remain prohibited in every mode.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum StreamCaptureMode: runtime::cudaStreamCaptureMode {
        /// Checks the local capture sequence and concurrent captures begun in global mode on other host threads.
        Global = CU_STREAM_CAPTURE_MODE_GLOBAL,
        /// Checks the local capture sequence without considering captures on other host threads.
        ThreadLocal = CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
        /// Allows potentially unsafe CUDA calls while preserving restrictions on calls that necessarily conflict with capture.
        Relaxed = CU_STREAM_CAPTURE_MODE_RELAXED,
    }
}

singe_core::native_enum! {
    /// Flags for [`Stream::update_capture_dependencies`]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum StreamCaptureDependencyUpdate: runtime::cudaStreamUpdateCaptureDependenciesFlags {
        /// Add new nodes to the dependency set.
        Add = cudaStreamAddCaptureDependencies,
        /// Replace the dependency set with the new nodes.
        Set = cudaStreamSetCaptureDependencies,
    }
}

/// Capture status and the native ID of an active capture sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StreamCaptureInfo {
    /// Current capture status of the queried stream.
    pub status: StreamCaptureStatus,
    /// Native capture sequence ID.
    /// Meaningful only when `status` is [`StreamCaptureStatus::Active`].
    pub id: u64,
}

/// An active stream capture and the kernel-code owners retained by recorded operations.
///
/// Constructed by [`Stream::capture`].
/// Recording does not execute producers or initialize their outputs.
#[derive(Debug)]
pub struct StreamCaptureScope<'scope> {
    stream: &'scope Stream,
    modules: RefCell<HashMap<CodeOwnerId, Module>>,
    _not_send: PhantomData<*const ()>,
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
        if let Err(error) = self.stream.end_capture() {
            report_cleanup_failure("end", "cuda stream capture", &error);
        }
    }
}

impl Drop for ActiveStreamCapture<'_> {
    fn drop(&mut self) {
        if !self.finished
            && let Err(error) = self.stream.end_capture()
        {
            report_cleanup_failure("end", "cuda stream capture", &error);
        }
    }
}

impl Stream {
    /// Begin graph capture on stream.
    /// When a stream is in capture mode, operations pushed into the stream are captured into a graph instead of executed.
    /// [`Stream::end_capture`] returns the graph.
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
    /// Returns an error if the context cannot be bound, capture cannot begin on this stream, the capture mode is invalid for the current thread state, or a previous asynchronous launch reports an error.
    pub fn begin_capture(&self, mode: StreamCaptureMode) -> Result<()> {
        let _lifecycle = self.allocation_guard()?;
        self.inner.context.bind()?;
        // SAFETY: `self.as_raw()` is the live stream this wrapper retains, in the context bound above, and the lifecycle guard excludes concurrent deferred frees.
        unsafe {
            try_ffi!(runtime::cudaStreamBeginCapture(self.as_raw(), mode.into()))?;
        }
        Ok(())
    }

    /// Begin graph capture on stream.
    /// When a stream is in capture mode, operations pushed into the stream are captured into a graph instead of executed.
    /// [`Stream::end_capture`] returns the graph.
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
    /// # Safety
    ///
    /// This captures into `graph`'s existing CUDA handle.
    /// Calling [`Stream::end_capture`] after this may return that same raw handle; the caller must not wrap it as a second owned [`Graph`].
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound, capture cannot begin on this stream, the graph dependencies are invalid, the capture mode is invalid for the current thread state, or a previous asynchronous launch reports an error.
    pub unsafe fn begin_capture_to_graph(
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
        let count = dependencies_raw
            .len()
            .try_into()
            .map_err(|_| Error::out_of_range("capture dependency count"))?;
        // SAFETY: the stream and graph are live in the context bound above, both dependency arrays are locals of `count` entries or null edge data, and the nodes belong to `graph` as checked above.
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
                count,
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
    /// Returns an error if the context cannot be bound, capture is not active on this stream, the capture has been invalidated, or a previous asynchronous launch reports an error.
    pub fn end_capture(&self) -> Result<Graph> {
        self.inner.context.bind()?;
        let mut handle = ptr::null_mut();
        let mut cleanup = self.capture_cleanup_lock()?;
        // SAFETY: `self.as_raw()` is the live stream this wrapper retains, in the context bound above, and `handle` is a writable local.
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
    /// This is the scoped form of [`Stream::begin_capture`] and [`Stream::end_capture`].
    /// This method attempts to end capture before it returns or resumes a panic.
    /// If `f` returns an error, this method attempts to end capture to restore stream usability, destroys any graph returned by CUDA, and returns the closure error.
    ///
    /// The scope is `!Send`, so it cannot move to another thread while capture is active.
    /// It retains recorded kernel-code owners in the returned graph.
    ///
    /// # Errors
    ///
    /// Returns an error if capture cannot begin, if `f` returns an error, or if capture cannot be ended successfully.
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
                    graph.retain_module(&module)?;
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

    /// Captures work recorded by `f` and instantiates the resulting graph with `flags`.
    ///
    /// Recorded kernel-code owners are retained by the executable.
    ///
    /// # Errors
    ///
    /// Propagates capture setup, closure, capture finalization, and graph-instantiation errors, including unsupported flags or invalid captured nodes.
    pub fn capture_executable<F>(
        &self,
        mode: StreamCaptureMode,
        flags: GraphInstantiateFlags,
        f: F,
    ) -> Result<ExecutableGraph>
    where
        F: FnOnce(&StreamCaptureScope<'_>) -> Result<()>,
    {
        let graph = self.capture(mode, f)?;
        graph.instantiate(flags)
    }

    /// Returns the capture status of this stream.
    /// After a successful call, the status is one of the following:
    ///
    /// * [`StreamCaptureStatus::None`]: The stream is not capturing.
    /// * [`StreamCaptureStatus::Active`]: The stream is capturing.
    /// * [`StreamCaptureStatus::Invalidated`]: The stream was capturing but an error has invalidated the capture sequence.
    ///   The capture sequence must be terminated with [`Stream::end_capture`] on the stream where it was initiated to continue using the stream.
    ///
    /// If this is called on the legacy default stream while a blocking stream on the same device is capturing, it returns [`crate::error::Status::StreamCaptureImplicit`].
    /// The blocking stream capture is not invalidated.
    ///
    /// When a blocking stream is capturing, the legacy stream is in an unusable state until the blocking stream capture is terminated.
    /// The legacy stream is not supported for stream capture, but attempted use would have an implicit dependency on the capturing stream(s).
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound, CUDA cannot query the capture status, or a previous asynchronous launch reports an error.
    pub fn capture_status(&self) -> Result<StreamCaptureStatus> {
        AsStream::capture_status(self)
    }

    /// Query stream state related to stream capture.
    ///
    /// If called on the legacy default stream while a stream not created with [`StreamFlags::NON_BLOCKING`](super::StreamFlags::NON_BLOCKING) is capturing, returns [`crate::error::Status::StreamCaptureImplicit`].
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
    /// Returns an error if the context cannot be bound, CUDA cannot query the capture info, the query would lose non-zero edge data, or a previous asynchronous launch reports an error.
    pub fn capture_info(&self) -> Result<StreamCaptureInfo> {
        self.inner.context.bind()?;
        let mut status = runtime::cudaStreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
        let mut id = 0;
        // SAFETY: `self.as_raw()` is the live stream this wrapper retains, in the context bound above, `status` and `id` are writable locals, and the other outputs are null.
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
            status: status.try_into()?,
            id,
        })
    }

    /// Modifies the dependency set of a capturing stream.
    /// The dependency set is the set of nodes that the next captured node in the stream will depend on.
    ///
    /// `mode` is [`StreamCaptureDependencyUpdate::Add`] to add the supplied set to the existing set, or [`StreamCaptureDependencyUpdate::Set`] to replace it.
    ///
    /// Nodes that are removed from the dependency set by this call do not result in [`crate::error::Status::StreamCaptureUnjoined`] if they are unreachable from the stream at [`Stream::end_capture`].
    ///
    /// Returns [`crate::error::Status::IllegalState`] if the stream is not capturing.
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound, the stream is not capturing, the supplied dependencies are invalid, or a previous asynchronous launch reports an error.
    pub fn update_capture_dependencies(
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
        let count = dependencies_raw
            .len()
            .try_into()
            .map_err(|_| Error::out_of_range("capture dependency count"))?;
        // SAFETY: the stream is live in the context bound above, both dependency arrays are locals of `count` entries or null edge data, and the nodes belong to the active capture graph as checked above.
        unsafe {
            try_ffi!(runtime::cudaStreamUpdateCaptureDependencies(
                self.as_raw(),
                dependencies_raw.as_mut_ptr(),
                if edge_data_raw.is_empty() {
                    ptr::null()
                } else {
                    edge_data_raw.as_ptr()
                },
                count,
                mode.into(),
            ))?;
        }
        Ok(())
    }
}

impl Stream {
    fn check_graph_context(&self, graph: &Graph) -> Result<()> {
        if graph.context() != &self.inner.context {
            return Err(Error::context_mismatch("graph"));
        }
        Ok(())
    }

    fn check_capture_dependency_contexts(&self, dependencies: &[GraphDependency]) -> Result<()> {
        for dependency in dependencies {
            if matches!(dependency.node.context(), Some(context) if context != &self.inner.context)
            {
                return Err(Error::context_mismatch("graph dependency"));
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
        // SAFETY: `self.as_raw()` is the live stream this wrapper retains, in the context bound above, `status` and `graph` are writable locals, and the other outputs are null.
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

        if StreamCaptureStatus::try_from(status)? != StreamCaptureStatus::Active {
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
    /// Call this before synchronous work or ordinary allocation so a rejected operation does not first change the capturing stream's state.
    ///
    /// # Errors
    ///
    /// Returns the native stream-capture-unsupported status for active capture or the stream-capture-invalidated status for invalidated capture.
    /// Also propagates capture-state query and context binding errors.
    pub fn ensure_not_capturing(&self) -> Result<()> {
        AsStream::ensure_not_capturing(self)
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
///   The thread is still prohibited from CUDA calls which necessarily conflict with stream capture, for example, attempting [`Event::query`](crate::event::Event::query) on an event that was last recorded inside a capture sequence.
///
/// # Errors
///
/// Returns an error if CUDA rejects the capture-mode exchange or if a previous asynchronous launch reported an error.
pub fn exchange_capture_mode(mode: StreamCaptureMode) -> Result<StreamCaptureMode> {
    let mut mode_raw: runtime::cudaStreamCaptureMode = mode.into();
    // SAFETY: `mode_raw` is a local that CUDA reads and overwrites through an exclusive pointer.
    unsafe {
        try_ffi!(runtime::cudaThreadExchangeStreamCaptureMode(
            &raw mut mode_raw
        ))?;
    }
    Ok(mode_raw.try_into()?)
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
