use std::sync::Arc;

use singe_cuda::graph::{ExecutableGraph, Graph, GraphNode};
use singe_cupti_sys as sys;

use crate::{
    error::Result,
    try_ffi,
    types::{
        ActivityAutoBoostState, ActivityThreadIdType, ContextId, DeviceId, ExecutableGraphId,
        GraphId, GraphNodeId,
    },
};

#[derive(Debug, Clone)]
pub struct Context {
    inner: Arc<singe_cuda::context::Context>,
}

impl Context {
    pub fn create() -> Result<Self> {
        Ok(Self {
            inner: singe_cuda::context::Context::create()?,
        })
    }

    #[cfg(feature = "testing")]
    pub(crate) fn from_cuda_context(inner: Arc<singe_cuda::context::Context>) -> Self {
        Self { inner }
    }

    pub fn bind(&self) -> Result<()> {
        self.inner.bind()?;
        Ok(())
    }

    /// Returns the process-unique CUPTI ID for this context.
    ///
    /// # Errors
    ///
    /// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
    /// - Returns [`crate::error::Status::InvalidContext`] if the CUDA context is invalid.
    /// - Returns [`crate::error::Status::InvalidParameter`] if CUPTI rejects the request.
    pub fn id(&self) -> Result<ContextId> {
        let mut context_id = 0;
        unsafe {
            try_ffi!(sys::cuptiGetContextId(self.as_raw(), &mut context_id))?;
        }
        Ok(ContextId::from(context_id))
    }

    /// Returns the CUDA device ID associated with this context.
    ///
    /// This can be called from CUPTI callbacks, where querying CUDA state directly may not be appropriate.
    ///
    /// # Errors
    ///
    /// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
    /// - Returns [`crate::error::Status::InvalidDevice`] if the CUDA device is invalid or CUPTI cannot resolve it.
    /// - Returns [`crate::error::Status::InvalidParameter`] if CUPTI rejects the request.
    pub fn device_id(&self) -> Result<DeviceId> {
        let mut device_id = 0;
        unsafe {
            try_ffi!(sys::cuptiGetDeviceId(self.as_raw(), &mut device_id))?;
        }
        Ok(DeviceId::from(device_id))
    }

    /// Returns the current auto-boost state for this context.
    ///
    /// Auto boost can make profiling results less stable. CUPTI may try to disable it during profiling, but that can fail because of permissions or `CUDA_AUTO_BOOST`.
    ///
    /// # Errors
    ///
    /// - Returns [`crate::error::Status::InvalidParameter`] if CUPTI rejects the context or the request.
    /// - Returns [`crate::error::Status::NotSupported`] if the device does not support auto boost.
    /// - Returns [`crate::error::Status::Unknown`] if CUPTI reports an internal error.
    pub fn auto_boost_state(&self) -> Result<ActivityAutoBoostState> {
        let mut state = sys::CUpti_ActivityAutoBoostState::default();
        unsafe {
            try_ffi!(sys::cuptiGetAutoBoostState(self.as_raw(), &mut state))?;
        }
        Ok(state.into())
    }

    pub(crate) fn as_raw(&self) -> sys::CUcontext {
        self.inner.as_raw() as sys::CUcontext
    }
}

/// Returns the CUPTI API version reported by the loaded runtime library.
///
/// # Errors
///
/// - Returns [`crate::error::Status::InvalidParameter`] if CUPTI rejects the request.
pub fn runtime_version() -> Result<u32> {
    let mut version = 0u32;
    unsafe {
        try_ffi!(sys::cuptiGetVersion(&mut version))?;
    }
    Ok(version)
}

/// Returns and clears the last CUPTI error recorded for the current host thread.
pub fn get_last_error() -> Result<()> {
    unsafe {
        try_ffi!(sys::cuptiGetLastError())?;
    }
    Ok(())
}

/// Returns the unique CUPTI ID for an executable CUDA graph.
///
/// # Errors
///
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
/// - Returns [`crate::error::Status::InvalidParameter`] if the executable graph handle is invalid.
pub fn get_executable_graph_id(executable_graph: &ExecutableGraph) -> Result<ExecutableGraphId> {
    let mut executable_graph_id = 0;
    unsafe {
        try_ffi!(sys::cuptiGetGraphExecId(
            executable_graph.as_raw() as sys::CUgraphExec,
            &mut executable_graph_id,
        ))?;
    }
    Ok(ExecutableGraphId::from(executable_graph_id))
}

/// Returns the unique CUPTI ID for a CUDA graph.
///
/// # Errors
///
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
/// - Returns [`crate::error::Status::InvalidParameter`] if the graph handle is invalid.
pub fn get_graph_id(graph: &Graph) -> Result<GraphId> {
    let mut graph_id = 0;
    unsafe {
        try_ffi!(sys::cuptiGetGraphId(
            graph.as_raw() as sys::CUgraph,
            &mut graph_id,
        ))?;
    }
    Ok(GraphId::from(graph_id))
}

/// Returns the unique CUPTI ID for a CUDA graph node.
///
/// # Errors
///
/// - Returns [`crate::error::Status::NotInitialized`] if CUPTI has not been initialized.
/// - Returns [`crate::error::Status::InvalidParameter`] if the graph node handle is invalid.
pub fn get_graph_node_id(node: GraphNode) -> Result<GraphNodeId> {
    let mut node_id = 0;
    unsafe {
        try_ffi!(sys::cuptiGetGraphNodeId(
            node.as_raw() as sys::CUgraphNode,
            &mut node_id,
        ))?;
    }
    Ok(GraphNodeId::from(node_id))
}

/// Returns a CUPTI timestamp in nanoseconds.
///
/// The timestamp uses the same timebase as CUPTI activity record timestamps.
///
/// # Errors
///
/// - Returns [`crate::error::Status::InvalidParameter`] if CUPTI rejects the timestamp request.
pub fn get_timestamp() -> Result<u64> {
    let mut timestamp = 0;
    unsafe {
        try_ffi!(sys::cuptiGetTimestamp(&mut timestamp))?;
    }
    Ok(timestamp)
}

/// Returns the thread ID type CUPTI uses for activity records.
///
/// This legacy process-wide API is not supported when multiple subscribers are allowed. Prefer [`crate::types::ActivityAttribute::ThreadIdType`] for subscriber-scoped activity collection.
///
/// # Errors
///
/// - Returns [`crate::error::Status::InvalidParameter`] if CUPTI rejects the thread ID type.
pub fn get_thread_id_type() -> Result<ActivityThreadIdType> {
    let mut thread_id_type = sys::CUpti_ActivityThreadIdType::CUPTI_ACTIVITY_THREAD_ID_TYPE_DEFAULT;
    unsafe {
        try_ffi!(sys::cuptiGetThreadIdType(&mut thread_id_type))?;
    }
    Ok(ActivityThreadIdType::from(thread_id_type))
}

/// Sets the thread ID type CUPTI uses for activity records.
///
/// Do not change this during a profiling session; existing and future activity records may otherwise use different thread ID schemes. This legacy process-wide API is not supported when multiple subscribers are allowed. Prefer [`crate::types::ActivityAttribute::ThreadIdType`] for subscriber-scoped activity collection.
///
/// # Errors
///
/// - Returns [`crate::error::Status::NotSupported`] if the requested thread ID type is not supported on the platform.
pub fn set_thread_id_type(thread_id_type: ActivityThreadIdType) -> Result<()> {
    unsafe {
        try_ffi!(sys::cuptiSetThreadIdType(thread_id_type.into()))?;
    }
    Ok(())
}

/// Detaches CUPTI from the current process.
///
/// CUPTI finalization is process-global. It is intentionally not tied to [`Context`] drop, because dropping one cloned CUDA context wrapper should not silently disable profiling for unrelated CUPTI users in the same process.
///
/// Prefer unsubscribing individual [`crate::callback::Subscriber`] values when ending a tracing session. CUPTI may return before finalizing when multiple subscribers are active.
pub fn finalize_process() -> Result<()> {
    unsafe {
        try_ffi!(sys::cuptiFinalize())?;
    }
    Ok(())
}
