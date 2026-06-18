#[allow(unused_imports)]
use crate::error::Status;

use std::{iter, marker::PhantomData, mem::ManuallyDrop, ptr, sync::Arc};

use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::impl_enum_conversion;
use singe_cuda_sys::runtime;

use crate::{
    context::Context,
    device::Device,
    error::{Error, Result},
    event::Event,
    graph::{Graph, GraphDependency, GraphEdgeData, GraphNode},
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

#[derive(Debug)]
pub struct Stream {
    handle: runtime::cudaStream_t,
    ctx: Arc<Context>,
    // TODO: Store device ID? Could be useful for multi-GPU.
    // device_id: DeviceId,
}

impl PartialEq for Stream {
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle && Arc::ptr_eq(&self.ctx, &other.ctx)
    }
}

impl Eq for Stream {}

#[derive(Debug)]
pub struct StreamScope<'scope, 'env> {
    stream: &'scope Stream,
    _env: PhantomData<&'env mut &'env ()>,
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
    /// # Safety
    ///
    /// `handle` must be a valid CUDA stream owned by `ctx`, and ownership of
    /// the handle is transferred to the returned [`Stream`]. The handle must
    /// not be destroyed elsewhere after calling this function.
    pub unsafe fn from_raw(handle: runtime::cudaStream_t, ctx: Arc<Context>) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self { handle, ctx })
    }

    pub fn to_borrowed(&self) -> BorrowedStream {
        BorrowedStream::from_raw(self.as_raw(), Arc::clone(&self.ctx))
    }

    pub fn scope<'env, F, R>(&self, f: F) -> Result<R>
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
        self.ctx.bind()?;
        unsafe { try_ffi!(runtime::cudaStreamSynchronize(self.as_raw())) }
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
        self.ctx.bind()?;
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
        self.ctx.bind()?;
        unsafe {
            try_ffi!(runtime::cudaStreamBeginCapture(self.as_raw(), mode.into()))?;
        }
        Ok(())
    }

    pub fn begin_capture_to_graph(
        &self,
        graph: &Graph,
        dependencies: &[GraphNode],
        mode: StreamCaptureMode,
    ) -> Result<()> {
        self.begin_capture_to_graph_with_data(graph, dependencies, &[], mode)
    }

    pub fn begin_capture_to_graph_with_data(
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
            .map(|(&node, data)| GraphDependency { node, data })
            .collect();

        self.begin_capture_to_graph_with_dependencies(graph, &dependencies, mode)
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
    pub fn begin_capture_to_graph_with_dependencies(
        &self,
        graph: &Graph,
        dependencies: &[GraphDependency],
        mode: StreamCaptureMode,
    ) -> Result<()> {
        self.ctx.bind()?;

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
        self.ctx.bind()?;
        let mut handle = ptr::null_mut();
        unsafe {
            try_ffi!(runtime::cudaStreamEndCapture(
                self.as_raw(),
                &raw mut handle
            ))?;
            Graph::from_raw(handle)
        }
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
        self.ctx.bind()?;
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
        self.ctx.bind()?;
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
            .map(|(&node, data)| GraphDependency { node, data })
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
        self.ctx.bind()?;

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
        self.ctx.bind()?;

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
        self.ctx.bind()?;
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
        self.ctx.bind()?;
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
        self.ctx.bind()?;
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
        self.ctx.bind()?;
        let mut device = 0i32;
        unsafe {
            try_ffi!(runtime::cudaStreamGetDevice(self.as_raw(), &raw mut device))?;
        }
        Ok(Device::new(device))
    }

    pub fn context(&self) -> &Context {
        &self.ctx
    }

    pub const fn as_raw(&self) -> runtime::cudaStream_t {
        self.handle
    }

    /// Consumes the stream and returns the raw CUDA stream handle without
    /// destroying it.
    ///
    /// The caller becomes responsible for eventually destroying the returned
    /// handle with CUDA.
    pub fn into_raw(self) -> runtime::cudaStream_t {
        let stream = ManuallyDrop::new(self);
        stream.handle
    }

    // pub fn is_null(&self) -> bool {
    //     self.handle.is_null()
    // }

    // TODO
    // --- Methods related to Stream Capture (Graphs) ---
    // Add methods like begin_capture, end_capture, is_capturing if needed

    // --- Methods related to Memory Management ---
    // Add methods like malloc_async, free_async, attach_mem_async if needed
    // These would likely take wrappers around device memory pointers.
}

impl<'scope, 'env> StreamScope<'scope, 'env> {
    pub const fn stream(&self) -> &'scope Stream {
        self.stream
    }

    pub fn synchronize(&self) -> Result<()> {
        self.stream.synchronize()
    }
}

impl BorrowedStream {
    pub const fn from_raw(handle: runtime::cudaStream_t, ctx: Arc<Context>) -> Self {
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

    pub const fn as_raw(&self) -> runtime::cudaStream_t {
        match self {
            Self::Default(_) => ptr::null_mut(),
            Self::Borrowed(stream) => stream.as_raw(),
        }
    }
}

// CUDA streams are ordering handles. Operations take &self and are serialized by
// CUDA stream semantics rather than mutable Rust state.
unsafe impl Send for Stream {}
unsafe impl Sync for Stream {}

impl Drop for Stream {
    fn drop(&mut self) {
        if let Err(err) = self.ctx.bind() {
            #[cfg(debug_assertions)]
            eprintln!("failed to bind context before destroying stream: {err}");
        }
        unsafe {
            // Synchronize before destroying to ensure callbacks complete.
            if let Err(err) = try_ffi!(runtime::cudaStreamSynchronize(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to synchronize stream before destroy: {err}");
            }

            if let Err(err) = try_ffi!(runtime::cudaStreamDestroy(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy CUDA stream: {err}");
            }
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
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };

    use super::*;
    use crate::testing;

    #[test]
    fn it_works() -> Result<()> {
        let _lock = testing::device_lock(0)?;
        let ctx = match Context::create() {
            Ok(ctx) => ctx,
            Err(error) if testing::is_stub_library(&error) => return Ok(()),
            Err(error) => return Err(error),
        };
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
}
