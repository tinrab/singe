pub mod raw;

use std::{
    any::Any,
    ffi::CString,
    fmt::{self, Display, Formatter},
    hash::{Hash, Hasher},
    marker::PhantomData,
    mem::ManuallyDrop,
    ops::Deref,
    ptr,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{impl_enum_conversion, impl_enum_display};
use singe_cuda_sys::{driver, runtime};

use crate::{
    context::Context,
    dim::Dim3,
    error::{Error, Result},
    event::Event,
    graph::raw::{HostNodeParams, MemoryCopyFromSymbolNodeParams, MemoryCopyToSymbolNodeParams},
    memory::{DeviceMemory, MemoryAccessDescriptor, MemoryCopyKind, MemoryPoolProps},
    module::{KernelLaunchArgs, LaunchConfig},
    stream::Stream,
    try_ffi,
    types::{DeviceFunction, DevicePtr},
    view::{ByteBuffer, ByteBufferMut, DeviceRepr},
};
use raw::{MemoryCopy1DNodeParams, MemoryCopy3DNodeParams};

/// Identifiers for [`GraphKernelNodeAttribute`] values used by CUDA graph kernel nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum GraphKernelNodeAttributeId {
    /// Identifies [`GraphKernelNodeAttribute::Cooperative`].
    Cooperative = runtime::cudaLaunchAttributeID::cudaLaunchAttributeCooperative as _,
    /// Identifies [`GraphKernelNodeAttribute::ClusterDimension`].
    ClusterDimension = runtime::cudaLaunchAttributeID::cudaLaunchAttributeClusterDimension as _,
    /// Identifies [`GraphKernelNodeAttribute::Priority`].
    Priority = runtime::cudaLaunchAttributeID::cudaLaunchAttributePriority as _,
    /// Identifies [`GraphKernelNodeAttribute::PreferredSharedMemoryCarveout`].
    /// The value is a percentage in the range `0..=100` describing the preferred
    /// shared-memory carveout for the launch. This is a hint, and the driver
    /// may choose a different configuration if required.
    PreferredSharedMemoryCarveout =
        runtime::cudaLaunchAttributeID::cudaLaunchAttributePreferredSharedMemoryCarveout as _,
}

impl_enum_conversion!(
    u32,
    runtime::cudaLaunchAttributeID,
    GraphKernelNodeAttributeId
);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum GraphKernelNodeAttribute {
    Cooperative(bool),
    ClusterDimension(Dim3),
    Priority(i32),
    PreferredSharedMemoryCarveout(u32),
}

#[derive(Debug, Clone)]
pub struct MemoryAllocationNodeInfo {
    ptr: DevicePtr,
    pub byte_size: usize,
    graph_id: Option<GraphId>,
    _graph: Option<Arc<GraphInner>>,
    ctx: Option<Arc<Context>>,
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct GraphInstantiateFlags: u64 {
        const AUTO_FREE_ON_LAUNCH = driver::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH as _;
        const UPLOAD = driver::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD as _;
        const DEVICE_LAUNCH = driver::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH as _;
        const USE_NODE_PRIORITY = driver::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY as _;
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct GraphDebugDotFlags: u32 {
        const VERBOSE = driver::CUgraphDebugDot_flags::CU_GRAPH_DEBUG_DOT_FLAGS_VERBOSE as _;
        const RUNTIME_TYPES = driver::CUgraphDebugDot_flags::CU_GRAPH_DEBUG_DOT_FLAGS_RUNTIME_TYPES as _;
        const KERNEL_NODE_PARAMS = driver::CUgraphDebugDot_flags::CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_PARAMS as _;
        const MEMCPY_NODE_PARAMS = driver::CUgraphDebugDot_flags::CU_GRAPH_DEBUG_DOT_FLAGS_MEMCPY_NODE_PARAMS as _;
        const MEMSET_NODE_PARAMS = driver::CUgraphDebugDot_flags::CU_GRAPH_DEBUG_DOT_FLAGS_MEMSET_NODE_PARAMS as _;
        const HOST_NODE_PARAMS = driver::CUgraphDebugDot_flags::CU_GRAPH_DEBUG_DOT_FLAGS_HOST_NODE_PARAMS as _;
        const EVENT_NODE_PARAMS = driver::CUgraphDebugDot_flags::CU_GRAPH_DEBUG_DOT_FLAGS_EVENT_NODE_PARAMS as _;
        const EXTERNAL_SEMAPHORE_SIGNAL_NODE_PARAMS = driver::CUgraphDebugDot_flags::CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_SIGNAL_NODE_PARAMS as _;
        const EXTERNAL_SEMAPHORE_WAIT_NODE_PARAMS = driver::CUgraphDebugDot_flags::CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_WAIT_NODE_PARAMS as _;
        const KERNEL_NODE_ATTRIBUTES = driver::CUgraphDebugDot_flags::CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_ATTRIBUTES as _;
        const HANDLES = driver::CUgraphDebugDot_flags::CU_GRAPH_DEBUG_DOT_FLAGS_HANDLES as _;
        const MEMORY_ALLOC_NODE_PARAMS = driver::CUgraphDebugDot_flags::CU_GRAPH_DEBUG_DOT_FLAGS_MEM_ALLOC_NODE_PARAMS as _;
        const MEMORY_FREE_NODE_PARAMS = driver::CUgraphDebugDot_flags::CU_GRAPH_DEBUG_DOT_FLAGS_MEM_FREE_NODE_PARAMS as _;
        const BATCH_MEM_OP_NODE_PARAMS = driver::CUgraphDebugDot_flags::CU_GRAPH_DEBUG_DOT_FLAGS_BATCH_MEM_OP_NODE_PARAMS as _;
        const EXTRA_TOPOLOGY_INFO = driver::CUgraphDebugDot_flags::CU_GRAPH_DEBUG_DOT_FLAGS_EXTRA_TOPO_INFO as _;
        const CONDITIONAL_NODE_PARAMS = driver::CUgraphDebugDot_flags::CU_GRAPH_DEBUG_DOT_FLAGS_CONDITIONAL_NODE_PARAMS as _;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum GraphNodeType {
    Kernel = driver::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL as _,
    Memcpy = driver::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMCPY as _,
    Memset = driver::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMSET as _,
    Host = driver::CUgraphNodeType::CU_GRAPH_NODE_TYPE_HOST as _,
    Graph = driver::CUgraphNodeType::CU_GRAPH_NODE_TYPE_GRAPH as _,
    Empty = driver::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY as _,
    WaitEvent = driver::CUgraphNodeType::CU_GRAPH_NODE_TYPE_WAIT_EVENT as _,
    EventRecord = driver::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EVENT_RECORD as _,
    ExternalSemaphoresSignal = driver::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL as _,
    ExternalSemaphoresWait = driver::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT as _,
    MemoryAlloc = driver::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEM_ALLOC as _,
    MemoryFree = driver::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEM_FREE as _,
    BatchMemOp = driver::CUgraphNodeType::CU_GRAPH_NODE_TYPE_BATCH_MEM_OP as _,
    Conditional = driver::CUgraphNodeType::CU_GRAPH_NODE_TYPE_CONDITIONAL as _,
}

impl_enum_conversion!(u32, runtime::cudaGraphNodeType, GraphNodeType);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u8)]
#[non_exhaustive]
pub enum GraphDependencyType {
    Default = driver::CUgraphDependencyType::CU_GRAPH_DEPENDENCY_TYPE_DEFAULT as _,
    Programmatic = driver::CUgraphDependencyType::CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC as _,
}

impl From<driver::CUgraphDependencyType> for GraphDependencyType {
    fn from(value: driver::CUgraphDependencyType) -> Self {
        match value {
            driver::CUgraphDependencyType::CU_GRAPH_DEPENDENCY_TYPE_DEFAULT => Self::Default,
            driver::CUgraphDependencyType::CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC => {
                Self::Programmatic
            }
        }
    }
}

impl From<GraphDependencyType> for driver::CUgraphDependencyType {
    fn from(value: GraphDependencyType) -> Self {
        match value {
            GraphDependencyType::Default => Self::CU_GRAPH_DEPENDENCY_TYPE_DEFAULT,
            GraphDependencyType::Programmatic => Self::CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC,
        }
    }
}

impl_enum_display!(GraphNodeType, {
    Self::Kernel => "cudaGraphNodeTypeKernel",
    Self::Memcpy => "cudaGraphNodeTypeMemcpy",
    Self::Memset => "cudaGraphNodeTypeMemset",
    Self::Host => "cudaGraphNodeTypeHost",
    Self::Graph => "cudaGraphNodeTypeGraph",
    Self::Empty => "cudaGraphNodeTypeEmpty",
    Self::WaitEvent => "cudaGraphNodeTypeWaitEvent",
    Self::EventRecord => "cudaGraphNodeTypeEventRecord",
    Self::ExternalSemaphoresSignal => "cudaGraphNodeTypeExternalSemaphoresSignal",
    Self::ExternalSemaphoresWait => "cudaGraphNodeTypeExternalSemaphoresWait",
    Self::MemoryAlloc => "cudaGraphNodeTypeMemAlloc",
    Self::MemoryFree => "cudaGraphNodeTypeMemFree",
    Self::BatchMemOp => "cudaGraphNodeTypeBatchMemOp",
    Self::Conditional => "cudaGraphNodeTypeConditional",
});

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum GraphExecUpdateResult {
    Success = driver::CUgraphExecUpdateResult::CU_GRAPH_EXEC_UPDATE_SUCCESS as _,
    Error = driver::CUgraphExecUpdateResult::CU_GRAPH_EXEC_UPDATE_ERROR as _,
    ErrorTopologyChanged =
        driver::CUgraphExecUpdateResult::CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED as _,
    ErrorNodeTypeChanged =
        driver::CUgraphExecUpdateResult::CU_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED as _,
    ErrorFunctionChanged =
        driver::CUgraphExecUpdateResult::CU_GRAPH_EXEC_UPDATE_ERROR_FUNCTION_CHANGED as _,
    ErrorParametersChanged =
        driver::CUgraphExecUpdateResult::CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED as _,
    ErrorNotSupported =
        driver::CUgraphExecUpdateResult::CU_GRAPH_EXEC_UPDATE_ERROR_NOT_SUPPORTED as _,
    ErrorUnsupportedFunctionChange =
        driver::CUgraphExecUpdateResult::CU_GRAPH_EXEC_UPDATE_ERROR_UNSUPPORTED_FUNCTION_CHANGE
            as _,
    ErrorAttributesChanged =
        driver::CUgraphExecUpdateResult::CU_GRAPH_EXEC_UPDATE_ERROR_ATTRIBUTES_CHANGED as _,
}

impl_enum_conversion!(driver::CUgraphExecUpdateResult, GraphExecUpdateResult);

impl_enum_display!(GraphExecUpdateResult, {
    Self::Success => "CU_GRAPH_EXEC_UPDATE_SUCCESS",
    Self::Error => "CU_GRAPH_EXEC_UPDATE_ERROR",
    Self::ErrorTopologyChanged => "CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED",
    Self::ErrorNodeTypeChanged => "CU_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED",
    Self::ErrorFunctionChanged => "CU_GRAPH_EXEC_UPDATE_ERROR_FUNCTION_CHANGED",
    Self::ErrorParametersChanged => "CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED",
    Self::ErrorNotSupported => "CU_GRAPH_EXEC_UPDATE_ERROR_NOT_SUPPORTED",
    Self::ErrorUnsupportedFunctionChange => "CU_GRAPH_EXEC_UPDATE_ERROR_UNSUPPORTED_FUNCTION_CHANGE",
    Self::ErrorAttributesChanged => "CU_GRAPH_EXEC_UPDATE_ERROR_ATTRIBUTES_CHANGED",
});

#[derive(Debug, Clone)]
pub struct GraphNode {
    handle: runtime::cudaGraphNode_t,
    graph_id: Option<GraphId>,
    graph: Option<Arc<GraphInner>>,
    ctx: Option<Arc<Context>>,
}

impl PartialEq for GraphNode {
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle && self.graph_id == other.graph_id
    }
}
impl Eq for GraphNode {}

impl Hash for GraphNode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.handle.hash(state);
        self.graph_id.hash(state);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GraphEdgeData {
    pub from_port: u8,
    pub to_port: u8,
    pub dependency_type: GraphDependencyType,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GraphDependency {
    pub node: GraphNode,
    pub data: GraphEdgeData,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GraphEdge {
    pub from: GraphNode,
    pub to: GraphNode,
    pub data: GraphEdgeData,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct GraphTopologySummary {
    pub nodes: usize,
    pub root_nodes: usize,
    pub edges: usize,
    pub kernel_nodes: usize,
    pub memcpy_nodes: usize,
    pub memset_nodes: usize,
    pub host_nodes: usize,
    pub child_graph_nodes: usize,
    pub empty_nodes: usize,
    pub wait_event_nodes: usize,
    pub event_record_nodes: usize,
    pub external_semaphores_signal_nodes: usize,
    pub external_semaphores_wait_nodes: usize,
    pub memory_alloc_nodes: usize,
    pub memory_free_nodes: usize,
    pub batch_mem_op_nodes: usize,
    pub conditional_nodes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Position {
    pub x: usize,
    pub y: usize,
    pub z: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Extent {
    pub width: usize,
    pub height: usize,
    pub depth: usize,
}

#[derive(Debug, Clone)]
pub struct MemoryAllocationNodeParams<'a> {
    pub pool_props: MemoryPoolProps,
    pub access_descs: &'a [MemoryAccessDescriptor],
    pub byte_size: usize,
}

impl Default for GraphEdgeData {
    fn default() -> Self {
        Self {
            from_port: 0,
            to_port: 0,
            dependency_type: GraphDependencyType::Default,
        }
    }
}

impl From<runtime::cudaGraphEdgeData> for GraphEdgeData {
    fn from(value: runtime::cudaGraphEdgeData) -> Self {
        Self {
            from_port: value.from_port,
            to_port: value.to_port,
            dependency_type: GraphDependencyType::try_from(value.type_)
                .unwrap_or(GraphDependencyType::Default),
        }
    }
}

impl From<GraphEdgeData> for runtime::cudaGraphEdgeData {
    fn from(value: GraphEdgeData) -> Self {
        Self {
            from_port: value.from_port,
            to_port: value.to_port,
            type_: value.dependency_type.into(),
            reserved: [0; 5],
        }
    }
}

impl From<Position> for runtime::cudaPos {
    fn from(value: Position) -> Self {
        Self {
            x: value.x as _,
            y: value.y as _,
            z: value.z as _,
        }
    }
}

impl From<Extent> for runtime::cudaExtent {
    fn from(value: Extent) -> Self {
        Self {
            width: value.width as _,
            height: value.height as _,
            depth: value.depth as _,
        }
    }
}

impl GraphNode {
    /// Wraps an existing CUDA graph node handle.
    ///
    /// The returned node is not associated with any [`Graph`] identity, so
    /// graph and executable-graph methods cannot validate that it belongs to
    /// the target graph before calling CUDA.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid CUDA graph node handle. The caller must ensure
    /// the node remains valid for every operation using the returned token and
    /// that it belongs to the graph or executable graph passed to those
    /// operations.
    pub const unsafe fn from_raw(handle: runtime::cudaGraphNode_t) -> Self {
        Self {
            handle,
            graph_id: None,
            graph: None,
            ctx: None,
        }
    }

    fn from_raw_in_graph(
        handle: runtime::cudaGraphNode_t,
        graph_id: GraphId,
        graph: Arc<GraphInner>,
        ctx: Option<Arc<Context>>,
    ) -> Self {
        Self {
            handle,
            graph_id: Some(graph_id),
            graph: Some(graph),
            ctx,
        }
    }

    fn from_raw_like(handle: runtime::cudaGraphNode_t, node: &Self) -> Self {
        Self {
            handle,
            graph_id: node.graph_id,
            graph: node.graph.clone(),
            ctx: node.ctx.clone(),
        }
    }

    fn bind_context(&self) -> Result<()> {
        if let Some(ctx) = &self.ctx {
            ctx.bind()?;
        }
        Ok(())
    }

    /// Returns the node type.
    ///
    /// Graph objects are not threadsafe.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot query the node type or if a previous asynchronous launch
    /// reported an error. CUDA may also return initialization-related errors such as
    /// [`crate::error::Status::NotInitialized`], [`crate::error::Status::CallRequiresNewerDriver`], or
    /// [`crate::error::Status::NoDevice`] if this call initializes internal runtime state. Callbacks must not
    /// call CUDA functions; see [`Stream::add_callback`].
    pub fn node_type(&self) -> Result<GraphNodeType> {
        self.bind_context()?;
        let mut kind = runtime::cudaGraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL;
        unsafe {
            try_ffi!(runtime::cudaGraphNodeGetType(self.as_raw(), &raw mut kind))?;
        }
        Ok(kind.into())
    }

    /// Returns this node's dependencies.
    ///
    /// Graph objects are not threadsafe.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot query the dependencies, a previous
    /// asynchronous launch reports an error, or CUDA reports runtime
    /// initialization diagnostics.
    pub fn dependencies(&self) -> Result<Vec<GraphDependency>> {
        self.bind_context()?;
        unsafe {
            let mut count = 0;
            try_ffi!(runtime::cudaGraphNodeGetDependencies(
                self.as_raw(),
                ptr::null_mut(),
                ptr::null_mut(),
                &raw mut count,
            ))?;

            if count == 0 {
                return Ok(Vec::new());
            }

            let mut handles = Vec::with_capacity(count as usize);
            let mut edge_data = Vec::with_capacity(count as usize);
            try_ffi!(runtime::cudaGraphNodeGetDependencies(
                self.as_raw(),
                handles.as_mut_ptr(),
                edge_data.as_mut_ptr(),
                &raw mut count,
            ))?;
            handles.set_len(count as usize);
            edge_data.set_len(count as usize);

            Ok(handles
                .into_iter()
                .zip(edge_data)
                .map(|(handle, data)| GraphDependency {
                    node: Self::from_raw_like(handle, self),
                    data: data.into(),
                })
                .collect())
        }
    }

    /// Returns this node's dependent nodes.
    ///
    /// Graph objects are not threadsafe.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot query the dependent nodes, a previous
    /// asynchronous launch reports an error, or CUDA reports runtime
    /// initialization diagnostics.
    pub fn dependent_nodes(&self) -> Result<Vec<GraphDependency>> {
        self.bind_context()?;
        unsafe {
            let mut count = 0;
            try_ffi!(runtime::cudaGraphNodeGetDependentNodes(
                self.as_raw(),
                ptr::null_mut(),
                ptr::null_mut(),
                &raw mut count,
            ))?;

            if count == 0 {
                return Ok(Vec::new());
            }

            let mut handles = Vec::with_capacity(count as usize);
            let mut edge_data = Vec::with_capacity(count as usize);
            try_ffi!(runtime::cudaGraphNodeGetDependentNodes(
                self.as_raw(),
                handles.as_mut_ptr(),
                edge_data.as_mut_ptr(),
                &raw mut count,
            ))?;
            handles.set_len(count as usize);
            edge_data.set_len(count as usize);

            Ok(handles
                .into_iter()
                .zip(edge_data)
                .map(|(handle, data)| GraphDependency {
                    node: Self::from_raw_like(handle, self),
                    data: data.into(),
                })
                .collect())
        }
    }

    /// Returns the event of this event record node.
    ///
    /// Graph objects are not threadsafe.
    ///
    /// # Errors
    ///
    /// Returns an error if this is not an event-record node, CUDA cannot query
    /// the event, CUDA returns a null event handle, a previous asynchronous
    /// launch reports an error, or CUDA reports runtime initialization
    /// diagnostics.
    pub fn event_record_node_event(&self) -> Result<runtime::cudaEvent_t> {
        self.bind_context()?;
        let mut event = ptr::null_mut();
        unsafe {
            try_ffi!(runtime::cudaGraphEventRecordNodeGetEvent(
                self.as_raw(),
                &raw mut event,
            ))?;
        }
        if event.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(event)
    }

    /// Returns the event of this event wait node.
    ///
    /// Graph objects are not threadsafe.
    ///
    /// # Errors
    ///
    /// Returns an error if this is not an event-wait node, CUDA cannot query the
    /// event, CUDA returns a null event handle, a previous asynchronous launch
    /// reports an error, or CUDA reports runtime initialization diagnostics.
    pub fn event_wait_node_event(&self) -> Result<runtime::cudaEvent_t> {
        self.bind_context()?;
        let mut event = ptr::null_mut();
        unsafe {
            try_ffi!(runtime::cudaGraphEventWaitNodeGetEvent(
                self.as_raw(),
                &raw mut event,
            ))?;
        }
        if event.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(event)
    }

    /// Returns a borrowed handle to the embedded graph in a child graph node.
    /// This does not clone the graph.
    /// Changes to the returned graph are reflected in the node, and the child
    /// node retains ownership of the embedded graph handle.
    /// The returned [`BorrowedGraph`] is tied to this node borrow and does not
    /// destroy the embedded graph when dropped.
    ///
    /// Allocation and free nodes cannot be added to the returned graph.
    /// Attempting to do so returns an error.
    ///
    /// Graph objects are not threadsafe.
    ///
    /// # Errors
    ///
    /// Returns an error if this is not a child-graph node, CUDA cannot query the
    /// child graph, CUDA returns a null graph handle, a previous asynchronous
    /// launch reports an error, or CUDA reports runtime initialization
    /// diagnostics.
    pub fn child_graph(&self) -> Result<BorrowedGraph<'_>> {
        self.bind_context()?;
        let mut graph = ptr::null_mut();
        unsafe {
            try_ffi!(runtime::cudaGraphChildGraphNodeGetGraph(
                self.as_raw(),
                &raw mut graph,
            ))?;
        }
        if graph.is_null() {
            return Err(Error::NullHandle);
        }
        unsafe { BorrowedGraph::from_raw_in_context(graph, self.ctx.clone()) }
    }

    /// Returns the parameters of this memcpy node.
    ///
    /// Graph objects are not threadsafe.
    ///
    /// # Errors
    ///
    /// Returns an error if this is not a memcpy node, CUDA cannot query the
    /// parameters, a previous asynchronous launch reports an error, or CUDA
    /// reports runtime initialization diagnostics.
    pub fn memory_copy_node_params(&self) -> Result<runtime::cudaMemcpy3DParms> {
        self.bind_context()?;
        let mut params = runtime::cudaMemcpy3DParms::default();
        unsafe {
            try_ffi!(runtime::cudaGraphMemcpyNodeGetParams(
                self.as_raw(),
                &raw mut params,
            ))?;
        }
        Ok(params)
    }

    /// Returns the parameters of this memset node.
    ///
    /// Graph objects are not threadsafe.
    ///
    /// # Errors
    ///
    /// Returns an error if this is not a memset node, CUDA cannot query the
    /// parameters, a previous asynchronous launch reports an error, or CUDA
    /// reports runtime initialization diagnostics.
    pub fn memory_set_node_params(&self) -> Result<driver::CUDA_MEMSET_NODE_PARAMS> {
        self.bind_context()?;
        let mut params = driver::CUDA_MEMSET_NODE_PARAMS::default();
        unsafe {
            try_ffi!(runtime::cudaGraphMemsetNodeGetParams(
                self.as_raw(),
                &raw mut params,
            ))?;
        }
        Ok(params)
    }

    /// Returns the parameters of this host node.
    ///
    /// Graph objects are not threadsafe.
    ///
    /// # Errors
    ///
    /// Returns an error if this is not a host node, CUDA cannot query the
    /// parameters, a previous asynchronous launch reports an error, or CUDA
    /// reports runtime initialization diagnostics.
    pub fn host_node_params(&self) -> Result<driver::CUDA_HOST_NODE_PARAMS> {
        self.bind_context()?;
        let mut params = driver::CUDA_HOST_NODE_PARAMS::default();
        unsafe {
            try_ffi!(runtime::cudaGraphHostNodeGetParams(
                self.as_raw(),
                &raw mut params,
            ))?;
        }
        Ok(params)
    }

    /// Returns the parameters of a memory allocation node.
    /// The `poolProps` and `accessDescs` values in the returned parameters are owned by the node.
    /// This memory remains valid until the node is destroyed.
    /// The returned parameters must not be modified.
    ///
    /// Graph objects are not threadsafe.
    ///
    /// # Errors
    ///
    /// Returns an error if this is not a memory-allocation node, CUDA cannot
    /// query the parameters, a previous asynchronous launch reports an error,
    /// or CUDA reports runtime initialization diagnostics.
    pub fn memory_allocation_node_info(&self) -> Result<MemoryAllocationNodeInfo> {
        self.bind_context()?;
        let mut params = runtime::cudaMemAllocNodeParams::default();
        unsafe {
            try_ffi!(runtime::cudaGraphMemAllocNodeGetParams(
                self.as_raw(),
                &raw mut params,
            ))?;
        }
        Ok(MemoryAllocationNodeInfo::from_raw(
            unsafe { DevicePtr::new(params.dptr as _) },
            params.bytesize as usize,
            self.graph_id,
            self.graph.clone(),
            self.ctx.clone(),
        ))
    }

    /// Returns the address of this memory free node.
    ///
    /// Graph objects are not threadsafe.
    ///
    /// # Errors
    ///
    /// Returns an error if this is not a memory-free node, CUDA cannot query the
    /// pointer, a previous asynchronous launch reports an error, or CUDA reports
    /// runtime initialization diagnostics.
    ///
    /// # Safety
    ///
    /// The node must still be a valid memory-free node in a live graph, and the
    /// returned pointer must not be used after the graph frees it.
    pub unsafe fn memory_free_node_ptr(&self) -> Result<DevicePtr> {
        self.bind_context()?;
        let mut ptr = ptr::null_mut();
        unsafe {
            try_ffi!(runtime::cudaGraphMemFreeNodeGetParams(
                self.as_raw(),
                &raw mut ptr as *mut _,
            ))?;
        }
        Ok(unsafe { DevicePtr::new(ptr as _) })
    }

    /// Returns the requested kernel node attribute.
    ///
    /// # Errors
    ///
    /// Returns an error if this is not a kernel node, CUDA cannot query the
    /// attribute, or a previous asynchronous launch reports an error.
    pub fn kernel_node_attribute(
        self,
        id: GraphKernelNodeAttributeId,
    ) -> Result<GraphKernelNodeAttribute> {
        self.bind_context()?;
        let mut value = runtime::cudaLaunchAttributeValue::default();
        unsafe {
            try_ffi!(runtime::cudaGraphKernelNodeGetAttribute(
                self.as_raw(),
                id.into(),
                &raw mut value,
            ))?;

            Ok(match id {
                GraphKernelNodeAttributeId::Cooperative => {
                    GraphKernelNodeAttribute::Cooperative(*value.cooperative.as_ref() != 0)
                }
                GraphKernelNodeAttributeId::ClusterDimension => {
                    let dim = value.clusterDim.as_ref();
                    GraphKernelNodeAttribute::ClusterDimension(Dim3::new(dim.x, dim.y, dim.z))
                }
                GraphKernelNodeAttributeId::Priority => {
                    GraphKernelNodeAttribute::Priority(*value.priority.as_ref())
                }
                GraphKernelNodeAttributeId::PreferredSharedMemoryCarveout => {
                    GraphKernelNodeAttribute::PreferredSharedMemoryCarveout(
                        *value.sharedMemCarveout.as_ref(),
                    )
                }
            })
        }
    }

    /// Sets a kernel node attribute.
    ///
    /// # Errors
    ///
    /// Returns an error if this is not a kernel node, CUDA rejects the
    /// attribute update, or a previous asynchronous launch reports an error.
    pub fn set_kernel_node_attribute(&mut self, attribute: GraphKernelNodeAttribute) -> Result<()> {
        self.bind_context()?;
        let (id, value) = match attribute {
            GraphKernelNodeAttribute::Cooperative(value) => {
                let mut attr = runtime::cudaLaunchAttributeValue {
                    cooperative: runtime::__BindgenUnionField::new(),
                    ..runtime::cudaLaunchAttributeValue::default()
                };
                unsafe { *attr.cooperative.as_mut() = i32::from(value) };
                (GraphKernelNodeAttributeId::Cooperative, attr)
            }
            GraphKernelNodeAttribute::ClusterDimension(value) => {
                let mut attr = runtime::cudaLaunchAttributeValue {
                    clusterDim: runtime::__BindgenUnionField::new(),
                    ..runtime::cudaLaunchAttributeValue::default()
                };
                unsafe {
                    *attr.clusterDim.as_mut() = runtime::cudaLaunchAttributeValue__bindgen_ty_1 {
                        x: value.x,
                        y: value.y,
                        z: value.z,
                    };
                }
                (GraphKernelNodeAttributeId::ClusterDimension, attr)
            }
            GraphKernelNodeAttribute::Priority(value) => {
                let mut attr = runtime::cudaLaunchAttributeValue {
                    priority: runtime::__BindgenUnionField::new(),
                    ..runtime::cudaLaunchAttributeValue::default()
                };
                unsafe { *attr.priority.as_mut() = value as _ };
                (GraphKernelNodeAttributeId::Priority, attr)
            }
            GraphKernelNodeAttribute::PreferredSharedMemoryCarveout(value) => {
                let mut attr = runtime::cudaLaunchAttributeValue {
                    sharedMemCarveout: runtime::__BindgenUnionField::new(),
                    ..runtime::cudaLaunchAttributeValue::default()
                };
                unsafe { *attr.sharedMemCarveout.as_mut() = value };
                (
                    GraphKernelNodeAttributeId::PreferredSharedMemoryCarveout,
                    attr,
                )
            }
        };

        unsafe {
            try_ffi!(runtime::cudaGraphKernelNodeSetAttribute(
                self.as_raw(),
                id.into(),
                &raw const value,
            ))?;
        }
        Ok(())
    }

    /// Copies attributes from `src` to this node.
    /// Both nodes must have the same context.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the attribute copy or if a previous asynchronous launch
    /// reported an error.
    pub fn copy_kernel_node_attributes(self, other: Self) -> Result<()> {
        if let (Some(ctx), Some(other_ctx)) = (self.context(), other.context())
            && ctx != other_ctx
        {
            return Err(Error::GraphContextMismatch);
        }
        self.bind_context()?;
        other.bind_context()?;
        unsafe {
            try_ffi!(runtime::cudaGraphKernelNodeCopyAttributes(
                self.as_raw(),
                other.handle
            ))?;
        }
        Ok(())
    }

    pub const fn as_raw(&self) -> runtime::cudaGraphNode_t {
        self.handle
    }

    pub(crate) fn graph_raw(&self) -> Option<runtime::cudaGraph_t> {
        self.graph.as_ref().map(|graph| graph.handle)
    }

    pub fn context(&self) -> Option<&Context> {
        self.ctx.as_deref()
    }
}

impl MemoryAllocationNodeInfo {
    pub const fn ptr(&self) -> DevicePtr {
        self.ptr
    }

    pub fn context(&self) -> Option<&Context> {
        self.ctx.as_deref()
    }

    fn from_raw_in_graph(
        ptr: DevicePtr,
        byte_size: usize,
        graph_id: GraphId,
        graph: Arc<GraphInner>,
        ctx: Option<Arc<Context>>,
    ) -> Self {
        Self::from_raw(ptr, byte_size, Some(graph_id), Some(graph), ctx)
    }

    fn from_raw(
        ptr: DevicePtr,
        byte_size: usize,
        graph_id: Option<GraphId>,
        graph: Option<Arc<GraphInner>>,
        ctx: Option<Arc<Context>>,
    ) -> Self {
        Self {
            ptr,
            byte_size,
            graph_id,
            _graph: graph,
            ctx,
        }
    }
}

impl PartialEq for MemoryAllocationNodeInfo {
    fn eq(&self, other: &Self) -> bool {
        self.ptr == other.ptr
            && self.byte_size == other.byte_size
            && self.graph_id == other.graph_id
    }
}

impl Eq for MemoryAllocationNodeInfo {}

impl Hash for MemoryAllocationNodeInfo {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.ptr.hash(state);
        self.byte_size.hash(state);
        self.graph_id.hash(state);
    }
}

impl GraphTopologySummary {
    fn record_node_type(&mut self, node_type: GraphNodeType) {
        match node_type {
            GraphNodeType::Kernel => self.kernel_nodes += 1,
            GraphNodeType::Memcpy => self.memcpy_nodes += 1,
            GraphNodeType::Memset => self.memset_nodes += 1,
            GraphNodeType::Host => self.host_nodes += 1,
            GraphNodeType::Graph => self.child_graph_nodes += 1,
            GraphNodeType::Empty => self.empty_nodes += 1,
            GraphNodeType::WaitEvent => self.wait_event_nodes += 1,
            GraphNodeType::EventRecord => self.event_record_nodes += 1,
            GraphNodeType::ExternalSemaphoresSignal => {
                self.external_semaphores_signal_nodes += 1;
            }
            GraphNodeType::ExternalSemaphoresWait => {
                self.external_semaphores_wait_nodes += 1;
            }
            GraphNodeType::MemoryAlloc => self.memory_alloc_nodes += 1,
            GraphNodeType::MemoryFree => self.memory_free_nodes += 1,
            GraphNodeType::BatchMemOp => self.batch_mem_op_nodes += 1,
            GraphNodeType::Conditional => self.conditional_nodes += 1,
        }
    }
}

#[derive(Debug)]
pub struct Graph {
    inner: Arc<GraphInner>,
    id: GraphId,
    ctx: Option<Arc<Context>>,
    retained: Vec<RetainedAllocation>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GraphId(u64);

#[derive(Debug)]
pub struct RawGraph {
    inner: Arc<GraphInner>,
}

#[derive(Debug)]
struct GraphInner {
    handle: runtime::cudaGraph_t,
    owns_handle: bool,
}

// CUDA graph handles can be retained and destroyed from any host thread after
// binding the associated context. Mutating graph APIs require `&mut Graph`.
unsafe impl Send for GraphInner {}
unsafe impl Sync for GraphInner {}

#[derive(Clone)]
struct RetainedAllocation(Arc<dyn Any + Send + Sync>);

impl fmt::Debug for RetainedAllocation {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("RetainedAllocation")
            .field("strong_count", &Arc::strong_count(&self.0))
            .finish()
    }
}

#[derive(Debug)]
pub struct BorrowedGraph<'node> {
    graph: Graph,
    _node: PhantomData<&'node GraphNode>,
}

/// Device memory whose allocation is retained by CUDA graph objects.
///
/// `GraphBuffer` values are created through [`Graph::create_buffer`],
/// [`Graph::zeroes_buffer`], or [`Graph::buffer_from_slice`]. Graph and
/// executable-graph APIs that accept `GraphBuffer` retain the underlying
/// allocation so graph replay cannot outlive the device pointers baked into
/// CUDA graph nodes.
#[derive(Debug)]
pub struct GraphBuffer<T: DeviceRepr> {
    memory: Arc<DeviceMemory<T>>,
    ctx: Option<Arc<Context>>,
}

impl<T> GraphBuffer<T>
where
    T: DeviceRepr + Send + Sync,
{
    fn from_memory(memory: DeviceMemory<T>, ctx: Option<Arc<Context>>) -> Self {
        Self {
            memory: Arc::new(memory),
            ctx,
        }
    }

    fn retained(&self) -> RetainedAllocation {
        let memory: Arc<DeviceMemory<T>> = Arc::clone(&self.memory);
        RetainedAllocation(memory)
    }

    pub fn len(&self) -> usize {
        self.memory.len()
    }

    pub fn is_empty(&self) -> bool {
        self.memory.is_empty()
    }

    pub fn byte_len(&self) -> usize {
        self.memory.byte_len()
    }

    pub fn context(&self) -> Option<&Context> {
        self.ctx.as_deref()
    }

    pub fn as_ptr(&self) -> *const T {
        self.memory.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.memory.as_mut_ptr()
    }

    /// Copies a host slice into this graph-retained device buffer.
    ///
    /// This updates the stable allocation used by graph-buffer node APIs. The
    /// caller is still responsible for ordering this copy against graph launches
    /// that read or write the same allocation.
    ///
    /// # Errors
    ///
    /// Returns an error if `host_slice` does not have the same length as this
    /// buffer or if CUDA rejects the copy.
    pub fn copy_from_host(&mut self, host_slice: &[T]) -> Result<()> {
        if let Some(ctx) = &self.ctx {
            ctx.bind()?;
        }
        if host_slice.len() != self.len() {
            return Err(Error::InvalidMemoryAccess);
        }
        if self.is_empty() {
            return Ok(());
        }
        unsafe {
            DeviceMemory::<T>::copy(
                self.as_mut_ptr(),
                host_slice.as_ptr(),
                self.len(),
                MemoryCopyKind::HostToDevice,
            )
        }
    }

    /// Copies this graph-retained device buffer into a host slice.
    ///
    /// # Errors
    ///
    /// Returns an error if `host_slice` does not have the same length as this
    /// buffer or if CUDA rejects the copy.
    pub fn copy_to_host(&self, host_slice: &mut [T]) -> Result<()> {
        if let Some(ctx) = &self.ctx {
            ctx.bind()?;
        }
        if host_slice.len() != self.len() {
            return Err(Error::InvalidMemoryAccess);
        }
        if self.is_empty() {
            return Ok(());
        }
        unsafe {
            DeviceMemory::<T>::copy(
                host_slice.as_mut_ptr(),
                self.as_ptr(),
                self.len(),
                MemoryCopyKind::DeviceToHost,
            )
        }
    }

    /// Copies another graph-retained buffer into this buffer.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffers have different lengths or if CUDA
    /// rejects the copy.
    pub fn copy_from_buffer(&mut self, src: &Self) -> Result<()> {
        if let (Some(dst_ctx), Some(src_ctx)) = (&self.ctx, &src.ctx)
            && dst_ctx.as_ref() != src_ctx.as_ref()
        {
            return Err(Error::GraphContextMismatch);
        }
        if let Some(ctx) = &self.ctx {
            ctx.bind()?;
        }
        if src.len() != self.len() {
            return Err(Error::InvalidMemoryAccess);
        }
        if self.is_empty() {
            return Ok(());
        }
        unsafe {
            DeviceMemory::<T>::copy(
                self.as_mut_ptr(),
                src.as_ptr(),
                self.len(),
                MemoryCopyKind::DeviceToDevice,
            )
        }
    }

    pub fn copy_to_host_vec(&self) -> Result<Vec<T>> {
        if let Some(ctx) = &self.ctx {
            ctx.bind()?;
        }
        if self.is_empty() {
            return Ok(Vec::new());
        }

        let mut host = Vec::<T>::with_capacity(self.len());
        unsafe {
            DeviceMemory::<T>::copy(
                host.as_mut_ptr(),
                self.as_ptr(),
                self.len(),
                MemoryCopyKind::DeviceToHost,
            )?;
            host.set_len(self.len());
        }
        Ok(host)
    }
}

impl RawGraph {
    /// Wraps an existing CUDA graph handle and takes ownership of it.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid CUDA graph handle. Ownership of `handle` is
    /// transferred to the returned [`RawGraph`], and the handle must not be
    /// destroyed elsewhere after calling this function.
    pub unsafe fn from_raw(handle: runtime::cudaGraph_t) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            inner: Arc::new(GraphInner {
                handle,
                owns_handle: true,
            }),
        })
    }

    /// Creates an empty raw graph without a Singe context association.
    ///
    /// Prefer [`Context::create_graph`] for ordinary Singe code. Raw graphs do
    /// not model context association, so the caller must keep CUDA context,
    /// node, executable update, upload, and launch relationships coherent.
    ///
    /// # Safety
    ///
    /// The returned graph has no modeled CUDA context association. The caller
    /// must ensure every node, kernel, memory operand, child graph, executable
    /// update, upload, and launch is used with the correct CUDA context.
    pub unsafe fn create() -> Result<Self> {
        let mut handle = ptr::null_mut();
        unsafe {
            try_ffi!(runtime::cudaGraphCreate(&raw mut handle, 0))?;
        }
        unsafe { Self::from_raw(handle) }
    }

    pub fn as_raw(&self) -> runtime::cudaGraph_t {
        self.inner.handle
    }

    /// Consumes the graph and returns the raw CUDA graph handle without
    /// destroying it.
    ///
    /// The caller becomes responsible for eventually destroying the returned
    /// handle with CUDA.
    pub fn into_raw(self) -> runtime::cudaGraph_t {
        let inner = Arc::try_unwrap(self.inner)
            .unwrap_or_else(|_| panic!("cannot take raw graph handle while it is still shared"));
        let inner = ManuallyDrop::new(inner);
        inner.handle
    }
}

static NEXT_GRAPH_ID: AtomicU64 = AtomicU64::new(1);

impl GraphId {
    pub fn generate() -> Self {
        Self(NEXT_GRAPH_ID.fetch_add(1, Ordering::Relaxed))
    }

    pub fn as_u64(self) -> u64 {
        self.0
    }
}

impl Display for GraphId {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl Graph {
    fn bind_context(&self) -> Result<()> {
        if let Some(ctx) = &self.ctx {
            ctx.bind()?;
        }
        Ok(())
    }

    /// Wraps an existing CUDA graph handle associated with `ctx` and takes
    /// ownership of it.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid CUDA graph handle associated with `ctx`.
    /// Ownership of `handle` is transferred to the returned [`Graph`], and the
    /// handle must not be destroyed elsewhere after calling this function.
    pub unsafe fn from_raw_in_context(
        handle: runtime::cudaGraph_t,
        ctx: Arc<Context>,
    ) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            inner: Arc::new(GraphInner {
                handle,
                owns_handle: true,
            }),
            id: GraphId::generate(),
            ctx: Some(ctx),
            retained: Vec::new(),
        })
    }

    unsafe fn from_raw_borrowed_in_context(
        handle: runtime::cudaGraph_t,
        ctx: Option<Arc<Context>>,
    ) -> Self {
        Self {
            inner: Arc::new(GraphInner {
                handle,
                owns_handle: false,
            }),
            id: GraphId::generate(),
            ctx,
            retained: Vec::new(),
        }
    }

    pub(crate) fn create_in_context(ctx: Arc<Context>) -> Result<Self> {
        ctx.bind()?;
        let mut handle = ptr::null_mut();
        unsafe {
            try_ffi!(runtime::cudaGraphCreate(&raw mut handle, 0))?;
        }
        Ok(Self {
            inner: Arc::new(GraphInner {
                handle,
                owns_handle: true,
            }),
            id: GraphId::generate(),
            ctx: Some(ctx),
            retained: Vec::new(),
        })
    }

    fn retain_buffer<T>(&mut self, buffer: &GraphBuffer<T>)
    where
        T: DeviceRepr + Send + Sync,
    {
        self.retained.push(buffer.retained());
    }

    fn check_buffer_context<T>(&self, buffer: &GraphBuffer<T>) -> Result<()>
    where
        T: DeviceRepr + Send + Sync,
    {
        if let (Some(graph_ctx), Some(buffer_ctx)) = (&self.ctx, buffer.context())
            && graph_ctx.as_ref() != buffer_ctx
        {
            return Err(Error::GraphContextMismatch);
        }
        Ok(())
    }

    fn check_buffer_contexts<T>(&self, dst: &GraphBuffer<T>, src: &GraphBuffer<T>) -> Result<()>
    where
        T: DeviceRepr + Send + Sync,
    {
        self.check_buffer_context(dst)?;
        self.check_buffer_context(src)?;
        Ok(())
    }

    /// Allocates graph-retained device memory.
    ///
    /// The returned buffer can be used with graph-buffer node APIs. Any graph or
    /// executable graph that records the buffer retains the underlying device
    /// allocation for replay.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot allocate device memory, the requested
    /// byte count overflows, or CUDA reports runtime initialization diagnostics.
    pub fn create_buffer<T>(&mut self, length: usize) -> Result<GraphBuffer<T>>
    where
        T: DeviceRepr + Send + Sync,
    {
        self.bind_context()?;
        let buffer = GraphBuffer::from_memory(DeviceMemory::create(length)?, self.ctx.clone());
        self.retain_buffer(&buffer);
        Ok(buffer)
    }

    /// Allocates graph-retained device memory initialized to zero bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot allocate or initialize device memory, the
    /// requested byte count overflows, or CUDA reports runtime initialization
    /// diagnostics.
    pub fn zeroes_buffer<T>(&mut self, length: usize) -> Result<GraphBuffer<T>>
    where
        T: DeviceRepr + Send + Sync,
    {
        self.bind_context()?;
        let buffer = GraphBuffer::from_memory(DeviceMemory::zeroes(length)?, self.ctx.clone());
        self.retain_buffer(&buffer);
        Ok(buffer)
    }

    /// Allocates graph-retained device memory initialized from a host slice.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot allocate or copy device memory, the
    /// requested byte count overflows, or CUDA reports runtime initialization
    /// diagnostics.
    pub fn buffer_from_slice<T>(&mut self, values: &[T]) -> Result<GraphBuffer<T>>
    where
        T: DeviceRepr + Send + Sync,
    {
        self.bind_context()?;
        let buffer = GraphBuffer::from_memory(DeviceMemory::from_slice(values)?, self.ctx.clone());
        self.retain_buffer(&buffer);
        Ok(buffer)
    }

    pub fn instantiate(&self) -> Result<ExecutableGraph> {
        self.instantiate_with_flags(GraphInstantiateFlags::empty())
    }

    /// Instantiates graph as an executable graph.
    /// The graph is validated for any structural constraints or intra-node constraints which were not previously validated.
    /// If instantiation is successful, returns an instantiated executable graph.
    ///
    /// `flags` controls the behavior of instantiation and subsequent graph launches.
    /// Valid flags are:
    ///
    /// * [`GraphInstantiateFlags::AUTO_FREE_ON_LAUNCH`], which configures a graph containing memory allocation nodes to automatically free any unfreed memory allocations before
    ///   the graph is relaunched.
    ///
    /// * [`GraphInstantiateFlags::DEVICE_LAUNCH`], which configures the graph for launch from the device.
    ///   If this flag is passed, the executable graph handle returned can
    ///   be used to launch the graph from both the host and device.
    ///   This flag can only be used on platforms which support unified addressing.
    ///   This flag cannot be used in conjunction with [`GraphInstantiateFlags::AUTO_FREE_ON_LAUNCH`].
    ///
    /// * [`GraphInstantiateFlags::USE_NODE_PRIORITY`], which causes the graph to use the priorities from the per-node attributes rather than the priority of the launch stream
    ///   during execution.
    ///   Priorities are only available on kernel nodes and are copied from stream priority during stream capture.
    ///
    /// If the graph contains any allocation or free nodes, there can be at most one executable graph in existence for that graph at a time.
    /// An attempt to instantiate a second executable graph before dropping the first results in an error.
    /// The same also applies if the graph contains any device-updatable kernel nodes.
    ///
    /// If the graph contains kernels which call device-side [`ExecutableGraph::launch`] from multiple devices, this results in an error.
    ///
    /// Graphs instantiated for launch on the device have additional restrictions which do not apply to host graphs:
    ///
    /// * The graph's nodes must reside on a single device.
    /// * The graph can only contain kernel nodes, memcpy nodes, memset nodes, and child graph nodes.
    /// * The graph cannot be empty and must contain at least one kernel, memcpy, or memset node.
    ///   Operation-specific restrictions are
    ///   outlined below.
    /// * Kernel nodes:
    ///   + Use of CUDA Dynamic Parallelism is not permitted.
    ///   + Cooperative launches are permitted as long as MPS is not in use.
    /// * Memcpy nodes:
    ///   + Only copies involving device memory and/or pinned device-mapped host memory are permitted.
    ///   + Copies involving CUDA arrays are not permitted.
    ///   + Both operands must be accessible from the current device, and the current device must match the device of other nodes in the
    ///     graph.
    ///
    /// Graph objects are not threadsafe.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the graph operation, if a previous asynchronous launch
    /// reported an error, or if CUDA reports runtime initialization diagnostics. Callbacks must
    /// not call CUDA functions; see [`Stream::add_callback`].
    pub fn instantiate_with_flags(&self, flags: GraphInstantiateFlags) -> Result<ExecutableGraph> {
        self.bind_context()?;
        let mut handle = ptr::null_mut();
        unsafe {
            try_ffi!(runtime::cudaGraphInstantiateWithFlags(
                &raw mut handle,
                self.as_raw(),
                flags.bits(),
            ))?;
        }
        unsafe {
            ExecutableGraph::from_raw_with_graph(
                handle,
                self.ctx.clone(),
                Some(self.id),
                Some(Arc::clone(&self.inner)),
                self.retained.clone(),
            )
        }
    }

    /// Creates a copy of `original_graph`.
    /// All parameters are copied into the cloned graph.
    /// The original graph may be modified after this call without affecting the clone.
    ///
    /// Child graph nodes in the original graph are recursively copied into the clone.
    ///
    /// Cloning is not supported for graphs that contain memory allocation nodes, memory free nodes, or conditional nodes.
    ///
    /// Graph objects are not threadsafe.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the graph operation, if a previous asynchronous launch
    /// reported an error, or if CUDA reports runtime initialization diagnostics. Callbacks must
    /// not call CUDA functions; see [`Stream::add_callback`].
    pub fn try_clone(&self) -> Result<Self> {
        self.bind_context()?;
        let mut handle = ptr::null_mut();
        unsafe {
            try_ffi!(runtime::cudaGraphClone(&raw mut handle, self.as_raw()))?;
        }
        Ok(Self {
            inner: Arc::new(GraphInner {
                handle,
                owns_handle: true,
            }),
            id: GraphId::generate(),
            ctx: self.ctx.clone(),
            retained: self.retained.clone(),
        })
    }

    fn node_from_raw(&self, handle: runtime::cudaGraphNode_t) -> GraphNode {
        GraphNode::from_raw_in_graph(handle, self.id, Arc::clone(&self.inner), self.ctx.clone())
    }

    pub(crate) fn check_node(&self, node: &GraphNode) -> Result<()> {
        self.bind_context()?;
        if !matches!(node.graph_id, Some(id) if id == self.id) {
            return Err(Error::GraphNodeMismatch);
        }
        Ok(())
    }

    pub(crate) fn check_nodes(&self, nodes: &[GraphNode]) -> Result<()> {
        self.bind_context()?;
        for node in nodes {
            if !matches!(node.graph_id, Some(id) if id == self.id) {
                return Err(Error::GraphNodeMismatch);
            }
        }
        Ok(())
    }

    fn check_child_graph_context(&self, child_graph: &Graph) -> Result<()> {
        if let (Some(parent_ctx), Some(child_ctx)) = (&self.ctx, &child_graph.ctx)
            && parent_ctx.as_ref() != child_ctx.as_ref()
        {
            return Err(Error::GraphContextMismatch);
        }
        Ok(())
    }

    fn check_event_record_context(&self, event: &Event) -> Result<()> {
        if let Some(ctx) = &self.ctx
            && ctx.as_ref() != event.context()
        {
            return Err(Error::GraphContextMismatch);
        }
        Ok(())
    }

    pub fn add_dependency(&mut self, from: GraphNode, to: GraphNode) -> Result<()> {
        self.add_dependencies(&[from], &[to])
    }

    pub fn add_dependencies(&mut self, from: &[GraphNode], to: &[GraphNode]) -> Result<()> {
        self.add_dependencies_with_data(from, to, &[])
    }

    /// Elements in `from` and `to` at corresponding indices define each dependency to add.
    /// Each node in `from` and `to` must belong to this graph.
    ///
    /// If `from` and `to` are empty, the call returns without modifying the graph.
    /// Specifying an existing dependency returns an error.
    ///
    /// Graph objects are not threadsafe.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the graph operation, if a previous asynchronous launch
    /// reported an error, or if CUDA reports runtime initialization diagnostics. Callbacks must
    /// not call CUDA functions; see [`Stream::add_callback`].
    pub fn add_dependencies_with_data(
        &mut self,
        from: &[GraphNode],
        to: &[GraphNode],
        edge_data: &[GraphEdgeData],
    ) -> Result<()> {
        if from.len() != to.len() {
            return Err(Error::GraphDependencyMismatch);
        }
        if !edge_data.is_empty() && edge_data.len() != from.len() {
            return Err(Error::GraphDependencyMismatch);
        }
        if from.is_empty() {
            return Ok(());
        }
        self.check_nodes(from)?;
        self.check_nodes(to)?;

        let from_raw: Vec<_> = from.iter().map(GraphNode::as_raw).collect();
        let to_raw: Vec<_> = to.iter().map(GraphNode::as_raw).collect();
        let edge_data_raw: Vec<_> = edge_data.iter().copied().map(Into::into).collect();
        unsafe {
            try_ffi!(runtime::cudaGraphAddDependencies(
                self.as_raw(),
                from_raw.as_ptr(),
                to_raw.as_ptr(),
                if edge_data_raw.is_empty() {
                    ptr::null()
                } else {
                    edge_data_raw.as_ptr()
                },
                from_raw.len() as _,
            ))?;
        }
        Ok(())
    }

    pub fn remove_dependency(&mut self, from: GraphNode, to: GraphNode) -> Result<()> {
        self.remove_dependencies(&[from], &[to])
    }

    pub fn remove_dependencies(&mut self, from: &[GraphNode], to: &[GraphNode]) -> Result<()> {
        self.remove_dependencies_with_data(from, to, &[])
    }

    /// Elements in `from` and `to` at corresponding indices define each dependency to remove.
    /// Each node in `from` and `to` must belong to this graph.
    ///
    /// If `from` and `to` are empty, the call returns without modifying the graph.
    /// Specifying an edge that does not exist in the graph, with data matching `edge_data`, results in an error.
    /// Passing an empty `edge_data` slice is equivalent to passing default edge data for each edge.
    ///
    /// Graph objects are not threadsafe.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the graph operation, if a previous asynchronous launch
    /// reported an error, or if CUDA reports runtime initialization diagnostics. Callbacks must
    /// not call CUDA functions; see [`Stream::add_callback`].
    pub fn remove_dependencies_with_data(
        &mut self,
        from: &[GraphNode],
        to: &[GraphNode],
        edge_data: &[GraphEdgeData],
    ) -> Result<()> {
        if from.len() != to.len() {
            return Err(Error::GraphDependencyMismatch);
        }
        if !edge_data.is_empty() && edge_data.len() != from.len() {
            return Err(Error::GraphDependencyMismatch);
        }
        if from.is_empty() {
            return Ok(());
        }
        self.check_nodes(from)?;
        self.check_nodes(to)?;

        let from_raw: Vec<_> = from.iter().map(GraphNode::as_raw).collect();
        let to_raw: Vec<_> = to.iter().map(GraphNode::as_raw).collect();
        let edge_data_raw: Vec<_> = edge_data.iter().copied().map(Into::into).collect();
        unsafe {
            try_ffi!(runtime::cudaGraphRemoveDependencies(
                self.as_raw(),
                from_raw.as_ptr(),
                to_raw.as_ptr(),
                if edge_data_raw.is_empty() {
                    ptr::null()
                } else {
                    edge_data_raw.as_ptr()
                },
                from_raw.len() as _,
            ))?;
        }
        Ok(())
    }

    pub fn add_edges(&mut self, edges: &[GraphEdge]) -> Result<()> {
        if edges.is_empty() {
            return Ok(());
        }

        let from: Vec<_> = edges.iter().map(|edge| edge.from.clone()).collect();
        let to: Vec<_> = edges.iter().map(|edge| edge.to.clone()).collect();
        let data: Vec<_> = edges.iter().map(|edge| edge.data).collect();
        self.add_dependencies_with_data(&from, &to, &data)
    }

    pub fn remove_edges(&mut self, edges: &[GraphEdge]) -> Result<()> {
        if edges.is_empty() {
            return Ok(());
        }

        let from: Vec<_> = edges.iter().map(|edge| edge.from.clone()).collect();
        let to: Vec<_> = edges.iter().map(|edge| edge.to.clone()).collect();
        let data: Vec<_> = edges.iter().map(|edge| edge.data).collect();
        self.remove_dependencies_with_data(&from, &to, &data)
    }

    /// Creates a node that performs no operation and adds it to the graph with the given dependencies.
    /// The dependency list may be empty, in which case the node is placed at the
    /// graph root. It may not contain duplicate entries.
    ///
    /// An empty node performs no operation during execution, but can be used for transitive ordering.
    /// For example, a phased execution graph with 2 groups of n nodes with a barrier between them can be represented using an empty node and 2\*n dependency edges, rather than no empty node and n^2 dependency edges.
    ///
    /// Graph objects are not threadsafe.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the graph operation or reports runtime initialization
    /// diagnostics. Callbacks must not call CUDA functions; see [`Stream::add_callback`].
    pub fn add_empty_node(&mut self, dependencies: &[GraphNode]) -> Result<GraphNode> {
        self.check_nodes(dependencies)?;
        let mut handle = ptr::null_mut();
        let dependencies_raw: Vec<_> = dependencies.iter().map(GraphNode::as_raw).collect();
        unsafe {
            try_ffi!(runtime::cudaGraphAddEmptyNode(
                &raw mut handle,
                self.as_raw(),
                dependencies_raw.as_ptr(),
                dependencies_raw.len() as _,
            ))?;
            Ok(self.node_from_raw(handle))
        }
    }

    /// Creates an event record node and adds it to the graph with the given dependencies and event.
    /// The dependency list may be empty, in which case the node is placed at the
    /// graph root. It may not contain duplicate entries.
    ///
    /// Each graph launch records `event` to capture execution of the node's dependencies.
    ///
    /// These nodes may not be used in loops or conditionals.
    ///
    /// Graph objects are not threadsafe.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the graph operation, if a previous asynchronous launch
    /// reported an error, or if CUDA reports runtime initialization diagnostics. Callbacks must
    /// not call CUDA functions; see [`Stream::add_callback`].
    pub fn add_event_record_node(
        &mut self,
        dependencies: &[GraphNode],
        event: &Event,
    ) -> Result<GraphNode> {
        self.check_nodes(dependencies)?;
        self.check_event_record_context(event)?;
        let mut handle = ptr::null_mut();
        let dependencies_raw: Vec<_> = dependencies.iter().map(GraphNode::as_raw).collect();
        unsafe {
            try_ffi!(runtime::cudaGraphAddEventRecordNode(
                &raw mut handle,
                self.as_raw(),
                dependencies_raw.as_ptr(),
                dependencies_raw.len() as _,
                event.as_raw(),
            ))?;
            Ok(self.node_from_raw(handle))
        }
    }

    /// Creates an event wait node and adds it to the graph with the given dependencies and event.
    /// The dependency list may be empty, in which case the node is placed at the
    /// graph root. It may not contain duplicate entries.
    ///
    /// The graph node waits for all work captured in `event`.
    /// See [`sys::cuEventRecord`](singe_cuda_sys::driver::cuEventRecord) for details on what is captured by an event.
    /// Synchronization is performed efficiently on the device when applicable.
    /// `event` may come from a different context or device than the launch stream.
    ///
    /// These nodes may not be used in loops or conditionals.
    ///
    /// Graph objects are not threadsafe.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the graph operation, if a previous asynchronous launch
    /// reported an error, or if CUDA reports runtime initialization diagnostics. Callbacks must
    /// not call CUDA functions; see [`Stream::add_callback`].
    pub fn add_event_wait_node(
        &mut self,
        dependencies: &[GraphNode],
        event: &Event,
    ) -> Result<GraphNode> {
        self.check_nodes(dependencies)?;
        let mut handle = ptr::null_mut();
        let dependencies_raw: Vec<_> = dependencies.iter().map(GraphNode::as_raw).collect();
        unsafe {
            try_ffi!(runtime::cudaGraphAddEventWaitNode(
                &raw mut handle,
                self.as_raw(),
                dependencies_raw.as_ptr(),
                dependencies_raw.len() as _,
                event.as_raw(),
            ))?;
            Ok(self.node_from_raw(handle))
        }
    }

    /// Creates a CPU execution node and adds it to the graph with the given dependencies and host-node parameters.
    /// The dependency list may be empty, in which case the node is placed at the
    /// graph root. It may not contain duplicate entries.
    ///
    /// When the graph is launched, the node invokes the specified CPU function.
    /// Host nodes are not supported under MPS with pre-Volta GPUs.
    ///
    /// Graph objects are not threadsafe.
    ///
    /// # Safety
    ///
    /// CUDA stores the raw callback function and user-data pointer in the graph
    /// node for later replay. The caller must ensure `params` remains valid
    /// according to [`HostNodeParams::new`] for every graph instantiation and
    /// launch that can execute this node.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the graph operation, if a previous asynchronous launch
    /// reported an error, or if CUDA reports runtime initialization diagnostics. Callbacks must
    /// not call CUDA functions; see [`Stream::add_callback`].
    pub unsafe fn add_host_node(
        &mut self,
        dependencies: &[GraphNode],
        params: &HostNodeParams,
    ) -> Result<GraphNode> {
        self.check_nodes(dependencies)?;
        let mut handle = ptr::null_mut();
        let dependencies_raw: Vec<_> = dependencies.iter().map(GraphNode::as_raw).collect();
        let params = params.into();
        unsafe {
            try_ffi!(runtime::cudaGraphAddHostNode(
                &raw mut handle,
                self.as_raw(),
                dependencies_raw.as_ptr(),
                dependencies_raw.len() as _,
                &raw const params,
            ))?;
            Ok(self.node_from_raw(handle))
        }
    }

    /// Creates a kernel execution node and adds it to the graph with the given dependencies, launch configuration, and kernel parameters.
    /// The dependency list may be empty, in which case the node is placed at the
    /// graph root. It may not contain duplicate entries.
    ///
    /// When the graph is launched, the node invokes the kernel on the grid and blocks specified by [`LaunchConfig`].
    /// [`LaunchConfig::shared_memory_bytes`](crate::module::LaunchConfig::shared_memory_bytes) sets the amount of dynamic shared memory available to each thread block.
    /// Kernel parameters are passed with [`KernelParameters`](crate::module::KernelParameters) or tuples of shared or mutable references.
    ///
    /// Kernels launched using graphs must not use texture and surface references.
    /// Reading or writing through any texture or surface reference is undefined behavior.
    /// This restriction does not apply to texture and surface objects.
    ///
    /// Runtime kernel handles queried via [`sys::cudaLibraryGetKernel`](singe_cuda_sys::runtime::cudaLibraryGetKernel) or [`sys::cudaGetKernel`](singe_cuda_sys::runtime::cudaGetKernel) may be used.
    /// The symbol passed to [`sys::cudaGetKernel`](singe_cuda_sys::runtime::cudaGetKernel) must be registered with the same CUDA Runtime instance.
    /// Passing a symbol that belongs to a different runtime instance results in undefined behavior.
    ///
    /// Graph objects are not threadsafe.
    ///
    /// # Safety
    ///
    /// CUDA copies the kernel argument values during this call and stores those
    /// copied values in the graph node for later replay. If an argument value is
    /// itself a pointer, only the pointer address is copied. The caller must
    /// ensure every copied pointer value remains valid for every graph
    /// instantiation, update, and launch that can execute this node. Mutable
    /// pointer arguments must also remain exclusive for the work ordered by
    /// those launches.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the graph operation, if a previous asynchronous launch
    /// reported an error, or if CUDA reports runtime initialization diagnostics. Callbacks must
    /// not call CUDA functions; see [`Stream::add_callback`].
    pub unsafe fn add_kernel_node<'a, P>(
        &mut self,
        dependencies: &[GraphNode],
        function: DeviceFunction,
        config: &LaunchConfig,
        params: P,
    ) -> Result<GraphNode>
    where
        P: KernelLaunchArgs<'a>,
    {
        self.check_nodes(dependencies)?;
        let mut handle = ptr::null_mut();
        let dependencies_raw: Vec<_> = dependencies.iter().map(GraphNode::as_raw).collect();
        params.with_encoded_arguments(|mut arguments| unsafe {
            let params = runtime::cudaKernelNodeParams {
                func: function.as_raw().cast(),
                gridDim: config.grid_dim().into(),
                blockDim: config.block_dim().into(),
                sharedMemBytes: config.shared_memory_bytes_u32(),
                kernelParams: arguments.as_mut_ptr().cast(),
                extra: ptr::null_mut(),
            };
            try_ffi!(runtime::cudaGraphAddKernelNode(
                &raw mut handle,
                self.as_raw(),
                dependencies_raw.as_ptr(),
                dependencies_raw.len() as _,
                &raw const params,
            ))?;
            Ok(self.node_from_raw(handle))
        })
    }

    /// Creates a new 1D memcpy node and adds it to the graph with the given dependencies.
    /// The dependency list may be empty, in which case the node is placed at the root of the graph, and it may not contain duplicate entries.
    ///
    /// When the graph is launched, the node copies `count` bytes from `src` to `dst`.
    /// The transfer direction is described by [`MemoryCopyKind`].
    /// [`MemoryCopyKind::Default`] is recommended when unified virtual addressing is available, in which case the transfer direction is inferred from the pointer values.
    /// Launching a memcpy node with `dst` and `src` pointers that do not match the direction of the copy results in undefined behavior.
    ///
    /// Memcpy nodes have additional restrictions for managed memory if any device in the system does not support concurrent managed access.
    ///
    /// Graph objects are not threadsafe.
    ///
    /// # Safety
    ///
    /// CUDA stores the raw source and destination addresses in the graph node
    /// for later replay. The caller must ensure `params` remains valid
    /// according to [`MemoryCopy1DNodeParams::new`] for every graph instantiation
    /// and launch that can execute this node.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the graph operation, if a previous asynchronous launch
    /// reported an error, or if CUDA reports runtime initialization diagnostics. Callbacks must
    /// not call CUDA functions; see [`Stream::add_callback`].
    pub unsafe fn add_memory_copy_node_1d(
        &mut self,
        dependencies: &[GraphNode],
        params: &MemoryCopy1DNodeParams,
    ) -> Result<GraphNode> {
        self.check_nodes(dependencies)?;
        let mut handle = ptr::null_mut();
        let dependencies_raw: Vec<_> = dependencies.iter().map(GraphNode::as_raw).collect();
        unsafe {
            try_ffi!(runtime::cudaGraphAddMemcpyNode1D(
                &raw mut handle,
                self.as_raw(),
                dependencies_raw.as_ptr(),
                dependencies_raw.len() as _,
                params.dst().cast(),
                params.src().cast(),
                params.count() as _,
                params.kind().into(),
            ))?;
            Ok(self.node_from_raw(handle))
        }
    }

    /// Creates a device-to-device memcpy node from typed byte buffers.
    ///
    /// The node copies `src.byte_len()` bytes. `dst` must have at least that
    /// many bytes.
    ///
    /// # Safety
    ///
    /// CUDA stores the raw source and destination addresses in the graph node
    /// for later replay. The caller must ensure `dst` and `src` remain valid
    /// for every graph instantiation and launch that can execute this node.
    /// `dst` must not be accessed through another mutable path while graph
    /// launches using this node can write it.
    ///
    /// # Errors
    ///
    /// Returns an error if `dst` is smaller than `src`, if CUDA rejects the graph
    /// operation, if a previous asynchronous launch reported an error, or if CUDA
    /// reports runtime initialization diagnostics.
    pub unsafe fn add_memory_copy_node_1d_device_to_device<D, S>(
        &mut self,
        dependencies: &[GraphNode],
        dst: &mut D,
        src: &S,
    ) -> Result<GraphNode>
    where
        D: ByteBufferMut + ?Sized,
        S: ByteBuffer + ?Sized,
    {
        let count = src.byte_len();
        if dst.byte_len() < count {
            return Err(Error::InvalidMemoryAccess);
        }
        let params = unsafe {
            MemoryCopy1DNodeParams::new(
                dst.as_byte_mut_ptr().cast(),
                src.as_byte_ptr().cast(),
                count,
                MemoryCopyKind::DeviceToDevice,
            )
        };
        unsafe { self.add_memory_copy_node_1d(dependencies, &params) }
    }

    /// Creates a device-to-device memcpy node between graph-retained buffers.
    ///
    /// The node copies `src.byte_len()` bytes. `dst` must have at least that
    /// many bytes. The graph retains both allocations so the baked CUDA graph
    /// pointers remain live for future instantiation and replay.
    ///
    /// # Errors
    ///
    /// Returns an error if `dst` is smaller than `src`, if CUDA rejects the graph
    /// operation, if a previous asynchronous launch reported an error, or if CUDA
    /// reports runtime initialization diagnostics.
    pub fn add_buffer_memory_copy_node_1d_device_to_device<T>(
        &mut self,
        dependencies: &[GraphNode],
        dst: &mut GraphBuffer<T>,
        src: &GraphBuffer<T>,
    ) -> Result<GraphNode>
    where
        T: DeviceRepr + Send + Sync,
    {
        self.check_buffer_contexts(dst, src)?;
        let count = src.byte_len();
        if dst.byte_len() < count {
            return Err(Error::InvalidMemoryAccess);
        }
        let params = unsafe {
            MemoryCopy1DNodeParams::new(
                dst.as_mut_ptr().cast(),
                src.as_ptr().cast(),
                count,
                MemoryCopyKind::DeviceToDevice,
            )
        };
        let node = unsafe { self.add_memory_copy_node_1d(dependencies, &params)? };
        self.retain_buffer(dst);
        self.retain_buffer(src);
        Ok(node)
    }

    /// Creates a memcpy node and adds it to the graph with the given dependencies.
    /// The dependency list may be empty, in which case the node is placed at the
    /// graph root. It may not contain duplicate entries.
    ///
    /// When the graph is launched, the node performs the memcpy described by `params`.
    /// See [`sys::cudaMemcpy3D`](singe_cuda_sys::runtime::cudaMemcpy3D) for a description of the structure and its restrictions.
    ///
    /// Memcpy nodes have additional restrictions for managed memory if any device in the system does not support concurrent managed access.
    ///
    /// Graph objects are not threadsafe.
    ///
    /// # Safety
    ///
    /// CUDA stores the raw source and destination addresses in the graph node
    /// for later replay. The caller must ensure `params` remains valid
    /// according to [`MemoryCopy3DNodeParams`] for every graph instantiation and
    /// launch that can execute this node.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the graph operation, if a previous asynchronous launch
    /// reported an error, or if CUDA reports runtime initialization diagnostics. Callbacks must
    /// not call CUDA functions; see [`Stream::add_callback`].
    pub unsafe fn add_memory_copy_node(
        &mut self,
        dependencies: &[GraphNode],
        params: &MemoryCopy3DNodeParams,
    ) -> Result<GraphNode> {
        self.check_nodes(dependencies)?;
        let mut handle = ptr::null_mut();
        let dependencies_raw: Vec<_> = dependencies.iter().map(GraphNode::as_raw).collect();
        let params = params.into();
        unsafe {
            try_ffi!(runtime::cudaGraphAddMemcpyNode(
                &raw mut handle,
                self.as_raw(),
                dependencies_raw.as_ptr(),
                dependencies_raw.len() as _,
                &raw const params,
            ))?;
            Ok(self.node_from_raw(handle))
        }
    }

    /// # Safety
    ///
    /// CUDA stores the raw symbol and source pointer in the graph node for
    /// later replay. The caller must ensure `params` remains valid according to
    /// [`MemoryCopyToSymbolNodeParams::new`] for every graph instantiation and
    /// launch that can execute this node.
    pub unsafe fn add_memory_copy_node_to_symbol(
        &mut self,
        dependencies: &[GraphNode],
        params: &MemoryCopyToSymbolNodeParams,
    ) -> Result<GraphNode> {
        self.check_nodes(dependencies)?;
        let mut handle = ptr::null_mut();
        let dependencies_raw: Vec<_> = dependencies.iter().map(GraphNode::as_raw).collect();
        unsafe {
            try_ffi!(runtime::cudaGraphAddMemcpyNodeToSymbol(
                &raw mut handle,
                self.as_raw(),
                dependencies_raw.as_ptr(),
                dependencies_raw.len() as _,
                params.symbol().cast(),
                params.src().cast(),
                params.count() as _,
                params.offset() as _,
                params.kind().into(),
            ))?;
            Ok(self.node_from_raw(handle))
        }
    }

    /// # Safety
    ///
    /// CUDA stores the raw destination and symbol pointer in the graph node for
    /// later replay. The caller must ensure `params` remains valid according to
    /// [`MemoryCopyFromSymbolNodeParams::new`] for every graph instantiation and
    /// launch that can execute this node.
    pub unsafe fn add_memory_copy_node_from_symbol(
        &mut self,
        dependencies: &[GraphNode],
        params: &MemoryCopyFromSymbolNodeParams,
    ) -> Result<GraphNode> {
        self.check_nodes(dependencies)?;
        let mut handle = ptr::null_mut();
        let dependencies_raw: Vec<_> = dependencies.iter().map(GraphNode::as_raw).collect();
        unsafe {
            try_ffi!(runtime::cudaGraphAddMemcpyNodeFromSymbol(
                &raw mut handle,
                self.as_raw(),
                dependencies_raw.as_ptr(),
                dependencies_raw.len() as _,
                params.dst().cast(),
                params.symbol().cast(),
                params.count() as _,
                params.offset() as _,
                params.kind().into(),
            ))?;
            Ok(self.node_from_raw(handle))
        }
    }

    /// Creates a new memset node and adds it to the graph with the given dependencies.
    /// The dependency list may be empty, in which case the node is placed at the root of the graph, and it may not contain duplicate entries.
    ///
    /// The element size must be 1, 2, or 4 bytes.
    /// When the graph is launched, the node performs the memset described by `params`.
    ///
    /// Graph objects are not threadsafe.
    ///
    /// # Safety
    ///
    /// CUDA stores the destination address in the graph node for later replay.
    /// The caller must ensure `params` remains valid according to
    /// [`MemorySetNodeParams::new`] for every graph instantiation and launch that
    /// can execute this node.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the graph operation, if a previous asynchronous launch
    /// reported an error, or if CUDA reports runtime initialization diagnostics. Callbacks must
    /// not call CUDA functions; see [`Stream::add_callback`].
    pub unsafe fn add_memory_set_node(
        &mut self,
        dependencies: &[GraphNode],
        params: &MemorySetNodeParams,
    ) -> Result<GraphNode> {
        self.check_nodes(dependencies)?;
        let mut handle = ptr::null_mut();
        let dependencies_raw: Vec<_> = dependencies.iter().map(GraphNode::as_raw).collect();
        let params = params.into();
        unsafe {
            try_ffi!(runtime::cudaGraphAddMemsetNode(
                &raw mut handle,
                self.as_raw(),
                dependencies_raw.as_ptr(),
                dependencies_raw.len() as _,
                &raw const params,
            ))?;
            Ok(self.node_from_raw(handle))
        }
    }

    /// Creates a new node which executes an embedded graph, and adds it to the graph with the given dependencies.
    /// The dependency list may be empty, in which case the node is placed at the root of the graph, and it may not contain duplicate entries.
    ///
    /// If `child_graph` contains allocation nodes, free nodes, or conditional nodes, this call returns an error.
    ///
    /// The node executes an embedded child graph.
    /// The child graph is cloned in this call.
    ///
    /// Graph objects are not threadsafe.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the graph operation, if a previous asynchronous launch
    /// reported an error, or if CUDA reports runtime initialization diagnostics. Callbacks must
    /// not call CUDA functions; see [`Stream::add_callback`].
    pub fn add_child_graph_node(
        &mut self,
        dependencies: &[GraphNode],
        child_graph: &Self,
    ) -> Result<GraphNode> {
        self.check_nodes(dependencies)?;
        self.check_child_graph_context(child_graph)?;
        let mut handle = ptr::null_mut();
        let dependencies_raw: Vec<_> = dependencies.iter().map(GraphNode::as_raw).collect();
        unsafe {
            try_ffi!(runtime::cudaGraphAddChildGraphNode(
                &raw mut handle,
                self.as_raw(),
                dependencies_raw.as_ptr(),
                dependencies_raw.len() as _,
                child_graph.as_raw(),
            ))?;
            Ok(self.node_from_raw(handle))
        }
    }

    /// Creates a new memory free node for a graph allocation and adds it to the graph.
    /// The dependency list may be empty, in which case the node is placed at the root of the graph, and it may not contain duplicate entries.
    ///
    /// [`Graph::add_memory_free_node`] returns [`crate::error::Status::InvalidValue`] if the caller attempts to free:
    ///
    /// * an allocation twice in the same graph.
    /// * an address that was not returned by an allocation node.
    /// * an invalid address.
    ///
    /// The following restrictions apply to graphs which contain allocation and/or memory free nodes:
    ///
    /// * Nodes and edges of the graph cannot be deleted.
    /// * The graph can only be used in a child node if the ownership is moved to the parent.
    /// * Only one instantiation of the graph may exist at any point in time.
    /// * The graph cannot be cloned.
    ///
    /// Graph objects are not threadsafe.
    ///
    /// # Errors
    ///
    /// Returns [`Error::GraphNodeMismatch`] if `allocation` did not come from this
    /// graph. Returns an error if CUDA rejects the graph operation or if a
    /// previous asynchronous launch reported an error.
    pub fn add_memory_free_node(
        &mut self,
        dependencies: &[GraphNode],
        allocation: &MemoryAllocationNodeInfo,
    ) -> Result<GraphNode> {
        if allocation.graph_id != Some(self.id) {
            return Err(Error::GraphNodeMismatch);
        }
        unsafe { self.add_memory_free_node_raw(dependencies, allocation.ptr) }
    }

    /// Creates a new memory free node from a raw device address.
    ///
    /// # Safety
    ///
    /// CUDA stores the raw address in the graph. The caller must ensure `ptr`
    /// is a graph allocation that may be freed by this graph, is ordered after
    /// the allocation node, and is not freed more than once or by another graph
    /// in a way that violates CUDA graph allocation ownership rules.
    pub unsafe fn add_memory_free_node_raw(
        &mut self,
        dependencies: &[GraphNode],
        ptr: DevicePtr,
    ) -> Result<GraphNode> {
        self.check_nodes(dependencies)?;
        let mut handle = ptr::null_mut();
        let dependencies_raw: Vec<_> = dependencies.iter().map(GraphNode::as_raw).collect();
        unsafe {
            try_ffi!(runtime::cudaGraphAddMemFreeNode(
                &raw mut handle,
                self.as_raw(),
                dependencies_raw.as_ptr(),
                dependencies_raw.len() as _,
                ptr.as_ptr() as _,
            ))?;
            Ok(self.node_from_raw(handle))
        }
    }

    /// Creates a new allocation node and adds it to the graph with the given dependencies and allocation parameters.
    /// The dependency list may be empty, in which case the node is placed at the root of the graph, and it may not contain duplicate entries.
    ///
    /// When [`Graph::add_memory_allocation_node`] creates an allocation node, it returns the allocation metadata in [`MemoryAllocationNodeInfo`].
    /// The allocation's address remains fixed across instantiations and launches.
    ///
    /// If the allocation is freed in the same graph, by creating a free node using [`Graph::add_memory_free_node`], the allocation can be accessed by nodes ordered after the allocation node but before the free node.
    /// These allocations cannot be freed outside the owning graph, and they can only be freed once in the owning graph.
    ///
    /// If the allocation is not freed in the same graph, then it can be accessed not only by nodes in the graph which are ordered after the allocation node, but also by stream operations ordered after the graph's execution but before the allocation is freed.
    ///
    /// Allocations which are not freed in the same graph can be freed by:
    ///
    /// * passing the allocation to [`DeviceMemory::free_async`](crate::memory::DeviceMemory::free_async) or [`DeviceMemory::free`](crate::memory::DeviceMemory::free);
    /// * launching a graph with a free node for that allocation; or
    /// * specifying [`GraphInstantiateFlags::AUTO_FREE_ON_LAUNCH`] during instantiation, which makes each launch behave as though it called [`DeviceMemory::free_async`](crate::memory::DeviceMemory::free_async) for every unfreed allocation.
    ///
    /// It is not possible to free an allocation in both the owning graph and another graph.
    /// If the allocation is freed in the same graph, a free node cannot be added to another graph.
    /// If the allocation is freed in another graph, a free node can no longer be added to the owning graph.
    ///
    /// The following restrictions apply to graphs which contain allocation and/or memory free nodes:
    ///
    /// * Nodes and edges of the graph cannot be deleted.
    /// * The graph can only be used in a child node if the ownership is moved to the parent.
    /// * Only one instantiation of the graph may exist at any point in time.
    /// * The graph cannot be cloned.
    ///
    /// Graph objects are not threadsafe.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the graph operation or if a previous asynchronous
    /// launch reported an error.
    pub fn add_memory_allocation_node(
        &mut self,
        dependencies: &[GraphNode],
        params: &MemoryAllocationNodeParams<'_>,
    ) -> Result<(GraphNode, MemoryAllocationNodeInfo)> {
        self.check_nodes(dependencies)?;
        let mut handle = ptr::null_mut();
        let dependencies_raw: Vec<_> = dependencies.iter().map(GraphNode::as_raw).collect();
        let access_descs: Vec<_> = params
            .access_descs
            .iter()
            .copied()
            .map(Into::into)
            .collect();
        let mut params_raw = runtime::cudaMemAllocNodeParams {
            poolProps: params.pool_props.into(),
            accessDescs: access_descs.as_ptr(),
            accessDescCount: access_descs.len() as _,
            bytesize: params.byte_size as _,
            dptr: 0,
        };
        unsafe {
            try_ffi!(runtime::cudaGraphAddMemAllocNode(
                &raw mut handle,
                self.as_raw(),
                dependencies_raw.as_ptr(),
                dependencies_raw.len() as _,
                &raw mut params_raw,
            ))?;
            // TODO: verify dptr?
            let node = self.node_from_raw(handle);
            let allocation = MemoryAllocationNodeInfo::from_raw_in_graph(
                DevicePtr::new(params_raw.dptr as *mut ()),
                params.byte_size,
                self.id,
                Arc::clone(&self.inner),
                self.ctx.clone(),
            );
            Ok((node, allocation))
        }
    }

    /// Returns this graph's nodes.
    ///
    /// Graph objects are not threadsafe.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the graph operation, if a previous asynchronous launch
    /// reported an error, or if CUDA reports runtime initialization diagnostics. Callbacks must
    /// not call CUDA functions; see [`Stream::add_callback`].
    pub fn nodes(&self) -> Result<Vec<GraphNode>> {
        unsafe {
            let mut count = 0;
            try_ffi!(runtime::cudaGraphGetNodes(
                self.as_raw(),
                ptr::null_mut(),
                &raw mut count,
            ))?;

            if count == 0 {
                return Ok(Vec::new());
            }

            let mut handles = Vec::with_capacity(count as usize);
            try_ffi!(runtime::cudaGraphGetNodes(
                self.as_raw(),
                handles.as_mut_ptr(),
                &raw mut count,
            ))?;
            handles.set_len(count as usize);

            Ok(handles
                .into_iter()
                .map(|handle| self.node_from_raw(handle))
                .collect())
        }
    }

    /// Returns this graph's root nodes.
    ///
    /// Graph objects are not threadsafe.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the graph operation, if a previous asynchronous launch
    /// reported an error, or if CUDA reports runtime initialization diagnostics. Callbacks must
    /// not call CUDA functions; see [`Stream::add_callback`].
    pub fn root_nodes(&self) -> Result<Vec<GraphNode>> {
        unsafe {
            let mut count = 0;
            try_ffi!(runtime::cudaGraphGetRootNodes(
                self.as_raw(),
                ptr::null_mut(),
                &raw mut count,
            ))?;

            if count == 0 {
                return Ok(Vec::new());
            }

            let mut handles = Vec::with_capacity(count as usize);
            try_ffi!(runtime::cudaGraphGetRootNodes(
                self.as_raw(),
                handles.as_mut_ptr(),
                &raw mut count,
            ))?;
            handles.set_len(count as usize);

            Ok(handles
                .into_iter()
                .map(|handle| self.node_from_raw(handle))
                .collect())
        }
    }

    /// Returns this graph's dependency edges.
    ///
    /// Graph objects are not threadsafe.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the graph operation, if a previous asynchronous launch
    /// reported an error, or if CUDA reports runtime initialization diagnostics. Callbacks must
    /// not call CUDA functions; see [`Stream::add_callback`].
    pub fn edges(&self) -> Result<Vec<GraphEdge>> {
        unsafe {
            let mut count = 0;
            try_ffi!(runtime::cudaGraphGetEdges(
                self.as_raw(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                &raw mut count,
            ))?;

            if count == 0 {
                return Ok(Vec::new());
            }

            let len = count as usize;
            let mut from = Vec::with_capacity(len);
            let mut to = Vec::with_capacity(len);
            let mut edge_data = Vec::with_capacity(len);
            try_ffi!(runtime::cudaGraphGetEdges(
                self.as_raw(),
                from.as_mut_ptr(),
                to.as_mut_ptr(),
                edge_data.as_mut_ptr(),
                &raw mut count,
            ))?;
            let len = count as usize;
            from.set_len(len);
            to.set_len(len);
            edge_data.set_len(len);

            Ok(from
                .into_iter()
                .zip(to)
                .zip(edge_data)
                .map(|((from, to), data)| GraphEdge {
                    from: self.node_from_raw(from),
                    to: self.node_from_raw(to),
                    data: data.into(),
                })
                .collect())
        }
    }

    /// Returns a compact summary of this graph's native CUDA topology.
    ///
    /// The summary is computed from CUDA graph introspection APIs and counts
    /// node kinds, root nodes, and dependency edges in this graph. Child graph
    /// nodes are counted as child nodes here; callers that need recursive
    /// details can query the child graph returned by [`GraphNode::child_graph`].
    ///
    /// Graph objects are not threadsafe.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects a topology query, if a previous
    /// asynchronous launch reported an error, or if CUDA reports runtime
    /// initialization diagnostics.
    pub fn topology_summary(&self) -> Result<GraphTopologySummary> {
        let nodes = self.nodes()?;
        let mut summary = GraphTopologySummary {
            nodes: nodes.len(),
            root_nodes: self.root_nodes()?.len(),
            edges: self.edges()?.len(),
            ..GraphTopologySummary::default()
        };
        for node in nodes {
            summary.record_node_type(node.node_type()?);
        }
        Ok(summary)
    }

    /// Writes a DOT-formatted description of the graph to `path`.
    /// By default this includes the graph topology, node types, node ID, kernel names, and memcpy direction.
    /// `flags` can request more detailed information about each node type, such as parameter values, kernel attributes, node handles, and function handles.
    ///
    /// # Errors
    ///
    /// Returns an error if `path` contains an interior NUL byte or if CUDA
    /// Runtime cannot write the DOT file.
    pub fn write_dot(&self, path: &str, flags: GraphDebugDotFlags) -> Result<()> {
        let path = CString::new(path)?;
        unsafe {
            try_ffi!(runtime::cudaGraphDebugDotPrint(
                self.as_raw(),
                path.as_ptr(),
                flags.bits(),
            ))?;
        }
        Ok(())
    }

    pub fn as_raw(&self) -> runtime::cudaGraph_t {
        self.inner.handle
    }

    pub fn context(&self) -> Option<&Context> {
        self.ctx.as_deref()
    }

    /// Consumes the graph and returns the raw CUDA graph handle without
    /// destroying it.
    ///
    /// The caller becomes responsible for eventually destroying the returned
    /// handle with CUDA.
    pub fn into_raw(self) -> runtime::cudaGraph_t {
        let inner = Arc::try_unwrap(self.inner)
            .unwrap_or_else(|_| panic!("cannot take raw graph handle while it is still shared"));
        let inner = ManuallyDrop::new(inner);
        inner.handle
    }
}

impl Drop for GraphInner {
    fn drop(&mut self) {
        if !self.owns_handle {
            return;
        }
        unsafe {
            if let Err(err) = try_ffi!(runtime::cudaGraphDestroy(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cuda graph: {err}");
            }
        }
    }
}

impl<'graph> BorrowedGraph<'graph> {
    /// Wraps an existing CUDA graph handle without taking ownership.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid CUDA graph handle for the returned lifetime.
    /// The returned graph view will not destroy `handle` when dropped.
    pub unsafe fn from_raw(handle: runtime::cudaGraph_t) -> Result<Self> {
        unsafe { Self::from_raw_in_context(handle, None) }
    }

    /// Wraps an existing CUDA graph handle without taking ownership and keeps a
    /// modeled context association for safe graph operations through the
    /// borrowed view.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid CUDA graph handle for the returned lifetime,
    /// and it must be associated with `ctx` when `ctx` is present. The returned
    /// graph view will not destroy `handle` when dropped.
    pub unsafe fn from_raw_in_context(
        handle: runtime::cudaGraph_t,
        ctx: Option<Arc<Context>>,
    ) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            graph: unsafe { Graph::from_raw_borrowed_in_context(handle, ctx) },
            _node: PhantomData,
        })
    }

    pub const fn as_graph(&self) -> &Graph {
        &self.graph
    }

    pub fn as_raw(&self) -> runtime::cudaGraph_t {
        self.graph.as_raw()
    }
}

impl Deref for BorrowedGraph<'_> {
    type Target = Graph;

    fn deref(&self) -> &Self::Target {
        self.as_graph()
    }
}

#[derive(Debug)]
pub struct ExecutableGraph {
    handle: runtime::cudaGraphExec_t,
    ctx: Option<Arc<Context>>,
    source_graph_id: Option<GraphId>,
    _source_graph: Option<Arc<GraphInner>>,
    retained: Vec<RetainedAllocation>,
}

#[derive(Debug, Clone, Copy)]
pub struct ExecutableGraphLaunchOperation<'graph> {
    graph: &'graph ExecutableGraph,
}

#[derive(Debug)]
pub struct RawExecutableGraph {
    handle: runtime::cudaGraphExec_t,
}

impl RawExecutableGraph {
    /// Wraps an existing CUDA executable graph handle and takes ownership of it.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid CUDA executable graph handle.
    /// Ownership of `handle` is transferred to the returned [`RawExecutableGraph`], and the handle must not be destroyed elsewhere after calling this function.
    pub unsafe fn from_raw(handle: runtime::cudaGraphExec_t) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self { handle })
    }

    pub const fn as_raw(&self) -> runtime::cudaGraphExec_t {
        self.handle
    }

    /// Consumes the executable graph and returns the raw CUDA executable graph
    /// handle without destroying it.
    ///
    /// The caller becomes responsible for eventually destroying the returned
    /// handle with CUDA.
    pub fn into_raw(self) -> runtime::cudaGraphExec_t {
        let graph = ManuallyDrop::new(self);
        graph.as_raw()
    }
}

impl Drop for RawExecutableGraph {
    fn drop(&mut self) {
        unsafe {
            if let Err(err) = try_ffi!(runtime::cudaGraphExecDestroy(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cuda graph exec: {err}");
            }
        }
    }
}

impl ExecutableGraph {
    fn bind_context(&self) -> Result<()> {
        if let Some(ctx) = &self.ctx {
            ctx.bind()?;
        }
        Ok(())
    }

    unsafe fn from_raw_with_graph(
        handle: runtime::cudaGraphExec_t,
        ctx: Option<Arc<Context>>,
        source_graph_id: Option<GraphId>,
        source_graph: Option<Arc<GraphInner>>,
        retained: Vec<RetainedAllocation>,
    ) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle,
            ctx,
            source_graph_id,
            _source_graph: source_graph,
            retained,
        })
    }

    fn check_node(&self, node: &GraphNode) -> Result<()> {
        self.bind_context()?;
        if !matches!((self.source_graph_id, node.graph_id), (Some(source_id), Some(node_id)) if node_id == source_id)
        {
            return Err(Error::GraphNodeMismatch);
        }
        Ok(())
    }

    fn retain_buffer<T>(&mut self, buffer: &GraphBuffer<T>)
    where
        T: DeviceRepr + Send + Sync,
    {
        self.retained.push(buffer.retained());
    }

    /// Returns the flags that were passed to instantiation for the given executable graph.
    /// [`GraphInstantiateFlags::UPLOAD`] is not returned because it does not affect the resulting executable graph.
    ///
    /// Graph objects are not threadsafe.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the graph operation, if a previous asynchronous launch
    /// reported an error, or if CUDA reports runtime initialization diagnostics. Callbacks must
    /// not call CUDA functions; see [`Stream::add_callback`].
    pub fn flags(&self) -> Result<GraphInstantiateFlags> {
        self.bind_context()?;
        let mut flags = 0;
        unsafe {
            try_ffi!(runtime::cudaGraphExecGetFlags(
                self.as_raw(),
                &raw mut flags
            ))?;
        }
        Ok(GraphInstantiateFlags::from_bits_retain(flags))
    }

    /// Executes this executable graph in `stream`.
    /// Only one instance of this executable graph may be executing at a time.
    /// Each launch is ordered behind both any previous work in `stream` and any previous launches of this executable graph.
    /// To execute a graph concurrently, it must be instantiated multiple times into multiple executable graphs.
    ///
    /// If any allocations created by this executable graph remain unfreed from a previous launch and the graph was not instantiated with [`GraphInstantiateFlags::AUTO_FREE_ON_LAUNCH`], the launch fails with [`crate::error::Status::InvalidValue`].
    ///
    /// Graph objects are not threadsafe.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the graph operation, if a previous asynchronous launch
    /// reported an error, or if CUDA reports runtime initialization diagnostics. Callbacks must
    /// not call CUDA functions; see [`Stream::add_callback`].
    pub fn launch(&self, stream: &Stream) -> Result<()> {
        if let Some(ctx) = &self.ctx
            && stream.context() != ctx.as_ref()
        {
            return Err(Error::StreamContextMismatch);
        }
        self.bind_context()?;
        unsafe {
            try_ffi!(runtime::cudaGraphLaunch(self.as_raw(), stream.as_raw()))?;
        }
        Ok(())
    }

    /// Returns a reusable operation object that launches this executable graph.
    pub const fn launch_operation(&self) -> ExecutableGraphLaunchOperation<'_> {
        ExecutableGraphLaunchOperation { graph: self }
    }

    /// Uploads this executable graph to the device in `stream` without executing it.
    /// Uploads of the same executable graph are serialized.
    /// Each upload is ordered behind both any previous work in `stream` and any previous launches of this executable graph.
    /// Uses memory cached by `stream` to back the allocations owned by this executable graph.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the graph operation, if a previous asynchronous launch
    /// reported an error, or if CUDA reports runtime initialization diagnostics.
    pub fn upload(&self, stream: &Stream) -> Result<()> {
        if let Some(ctx) = &self.ctx
            && stream.context() != ctx.as_ref()
        {
            return Err(Error::StreamContextMismatch);
        }
        self.bind_context()?;
        unsafe {
            try_ffi!(runtime::cudaGraphUpload(self.as_raw(), stream.as_raw()))?;
        }
        Ok(())
    }

    /// Updates this executable graph with the node parameters in a topologically identical `graph`.
    ///
    /// Limitations:
    ///
    /// * Kernel nodes:
    ///   + The owning context of the kernel function cannot change.
    ///   + A node whose kernel function originally did not use CUDA dynamic parallelism cannot be updated to a kernel function that uses CDP.
    ///   + A node whose kernel function originally did not make device-side update calls cannot be updated to a kernel function that makes device-side
    ///     update calls.
    ///   + A cooperative node cannot be updated to a non-cooperative node, and vice-versa.
    ///   + If the graph was instantiated with [`GraphInstantiateFlags::USE_NODE_PRIORITY`], the priority attribute cannot change.
    ///     Equality
    ///     is checked on the originally requested priority values, before they are clamped to the device's supported range.
    ///   + If this executable graph was not instantiated for device launch, a node whose kernel function originally did not use device-side [`ExecutableGraph::launch`] cannot be updated to a kernel function that uses device-side [`ExecutableGraph::launch`] unless the node resides on the same device as nodes which contained such calls at instantiate-time.
    ///     If no such calls were
    ///     present at instantiation, these updates cannot be performed at all.
    ///   + Neither the source graph nor this executable graph may contain device-updatable kernel nodes.
    /// * Memset and memcpy nodes:
    ///   + The CUDA device(s) to which the operand(s) was allocated/mapped cannot change.
    ///   + The source/destination memory must be allocated from the same contexts as the original source/destination memory.
    ///   + For 2D memsets, only address and assigned value may be updated.
    ///   + For 1D memsets, updating dimensions is also allowed, but may fail if the resulting operation does not map onto the work resources
    ///     already allocated for the node.
    /// * Additional memcpy node restrictions:
    ///   + Changing either the source or destination memory type, such as [`MemoryType::Device`](crate::types::MemoryType::Device) or [`MemoryType::Array`](crate::types::MemoryType::Array), is not supported.
    /// * Conditional nodes:
    ///   + Changing node parameters is not supported.
    ///   + Changing parameters of nodes within the conditional body graph is subject to the rules above.
    ///   + Conditional handle flags and default values are updated as part of the graph update.
    ///
    /// CUDA may add further restrictions in future releases.
    /// [`ExecutableGraph::update`] sets the update result to [`GraphExecUpdateResult::ErrorTopologyChanged`] under the following conditions:
    ///
    /// * The count of nodes directly in the executable graph and the source graph differ.
    /// * The source graph has more exit nodes.
    /// * A node in the source graph has a different number of dependencies than the paired node from the executable graph.
    /// * A node in the source graph has a dependency that does not match the corresponding dependency of the paired node from the executable graph.
    ///   The dependencies are paired based on edge order and
    ///   a dependency does not match when the nodes are already paired based on other edges examined in the graph.
    ///
    /// [`ExecutableGraph::update`] sets the update result to:
    ///
    /// * [`GraphExecUpdateResult::Error`] if passed an invalid value.
    /// * [`GraphExecUpdateResult::ErrorTopologyChanged`] if the graph topology changed.
    /// * [`GraphExecUpdateResult::ErrorNodeTypeChanged`] if the type of a node changed.
    /// * [`GraphExecUpdateResult::ErrorFunctionChanged`] if the kernel function of a node changed (CUDA driver before 11.2).
    /// * [`GraphExecUpdateResult::ErrorUnsupportedFunctionChange`] if the kernel function changed in an unsupported way.
    /// * [`GraphExecUpdateResult::ErrorParametersChanged`] if any parameters to a node changed in a way that is not supported.
    /// * [`GraphExecUpdateResult::ErrorAttributesChanged`] if any attributes of a node changed in a way that is not supported.
    /// * [`GraphExecUpdateResult::ErrorNotSupported`] if something about a node is unsupported, like the node's type or configuration.
    ///
    /// If the update fails for a reason not listed above, the result is [`GraphExecUpdateResult::Error`].
    /// If the update succeeds, the result is [`GraphExecUpdateResult::Success`].
    ///
    /// [`ExecutableGraph::update`] succeeds when the update was performed successfully.
    /// It returns [`crate::error::Status::GraphExecUpdateFailure`] if the graph update was not performed because it included changes which violated constraints specific to instantiated graph update.
    ///
    /// Graph objects are not threadsafe.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the graph update, if the update violates instantiated graph
    /// update constraints, or if a previous asynchronous launch reported an error. CUDA may also
    /// return initialization-related errors such as [`crate::error::Status::NotInitialized`],
    /// [`crate::error::Status::CallRequiresNewerDriver`], or [`crate::error::Status::NoDevice`] if this call initializes
    /// internal runtime state. Callbacks must not call CUDA functions; see
    /// [`Stream::add_callback`].
    pub fn update(&mut self, graph: &Graph) -> Result<ExecutableGraphUpdate> {
        if let (Some(exec_ctx), Some(graph_ctx)) = (&self.ctx, &graph.ctx)
            && exec_ctx.as_ref() != graph_ctx.as_ref()
        {
            return Err(Error::GraphContextMismatch);
        }
        self.bind_context()?;
        let mut result_info = runtime::cudaGraphExecUpdateResultInfo::default();
        unsafe {
            try_ffi!(runtime::cudaGraphExecUpdate(
                self.as_raw(),
                graph.as_raw(),
                &raw mut result_info,
            ))?;
        }
        self.retained.extend(graph.retained.iter().cloned());
        Ok(ExecutableGraphUpdate::from_result_info(result_info, graph))
    }

    /// Sets the parameters of a kernel node in this executable graph.
    /// The node is identified by the corresponding `node` in the non-executable graph from which this executable graph was instantiated.
    ///
    /// `node` must not have been removed from the original graph.
    /// All node parameters may change, but the following restrictions apply to function updates:
    ///
    /// * The owning device of the kernel function cannot change.
    /// * A node whose kernel function originally did not use CUDA dynamic parallelism cannot be updated to a kernel function that uses CDP
    /// * A node whose kernel function originally did not make device-side update calls cannot be updated to a kernel function that makes device-side
    ///   update calls.
    /// * If this executable graph was not instantiated for device launch, a node whose kernel function originally did not use device-side [`ExecutableGraph::launch`] cannot be updated to a kernel function that uses device-side [`ExecutableGraph::launch`] unless the node resides on the same device as nodes which contained such calls at instantiate-time.
    ///   If no such calls were
    ///   present at instantiation, these updates cannot be performed at all.
    ///
    /// The modifications only affect future launches of this executable graph.
    /// Already enqueued or running launches of this executable graph are not affected by this call.
    /// The original `node` is also not modified by this call.
    ///
    /// If `node` is a device-updatable kernel node, the next upload or launch of this executable graph will overwrite any previous device-side updates.
    /// Additionally, applying host updates to a device-updatable kernel node while it is being updated from the device results in undefined behavior.
    /// This can also be used with a runtime kernel handle queried through [`sys::cudaLibraryGetKernel`](singe_cuda_sys::runtime::cudaLibraryGetKernel) or [`sys::cudaGetKernel`](singe_cuda_sys::runtime::cudaGetKernel) and then passed as a raw pointer.
    /// The symbol passed to [`sys::cudaGetKernel`](singe_cuda_sys::runtime::cudaGetKernel) must be registered with the same CUDA Runtime instance.
    /// Passing a symbol that belongs to a different runtime instance results in undefined behavior.
    /// The only type that can be reliably passed to a different runtime instance is the runtime kernel handle type itself.
    ///
    /// Graph objects are not threadsafe.
    ///
    /// # Safety
    ///
    /// CUDA copies the kernel argument values during this call and stores those
    /// copied values in the executable graph for future launches. If an
    /// argument value is itself a pointer, only the pointer address is copied.
    /// The caller must ensure every copied pointer value remains valid for
    /// every future launch that can execute this node. Mutable pointer
    /// arguments must also remain exclusive for the work ordered by those
    /// launches.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the graph operation, if a previous asynchronous launch
    /// reported an error, or if CUDA reports runtime initialization diagnostics. Callbacks must
    /// not call CUDA functions; see [`Stream::add_callback`].
    pub unsafe fn set_kernel_node_params<'a, P>(
        &mut self,
        node: GraphNode,
        function: DeviceFunction,
        config: &LaunchConfig,
        params: P,
    ) -> Result<()>
    where
        P: KernelLaunchArgs<'a>,
    {
        self.check_node(&node)?;
        params.with_encoded_arguments(|mut arguments| unsafe {
            let params = runtime::cudaKernelNodeParams {
                func: function.as_raw().cast(),
                gridDim: config.grid_dim().into(),
                blockDim: config.block_dim().into(),
                sharedMemBytes: config.shared_memory_bytes_u32(),
                kernelParams: arguments.as_mut_ptr().cast(),
                extra: ptr::null_mut(),
            };
            try_ffi!(runtime::cudaGraphExecKernelNodeSetParams(
                self.as_raw(),
                node.as_raw(),
                &raw const params,
            ))?;
            Ok(())
        })
    }

    /// Updates the work represented by `node` in this executable graph as though `node` had contained the given `params` at instantiation.
    /// `node` must remain in the graph which was used to instantiate this executable graph.
    /// Changed edges to and from `node` are ignored.
    ///
    /// The source and destination must be allocated from the same contexts as the original source and destination memory.
    /// The instantiation-time memory operands must be 1-dimensional.
    /// Zero-length operations are not supported.
    ///
    /// The modifications only affect future launches of this executable graph.
    /// Already enqueued or running launches of this executable graph are not affected by this call.
    /// The original `node` is also not modified by this call.
    ///
    /// Returns [`crate::error::Status::InvalidValue`] if the memory operands' mappings changed or the original memory operands are multidimensional.
    ///
    /// Graph objects are not threadsafe.
    ///
    /// # Safety
    ///
    /// CUDA stores the raw source and destination addresses in the executable
    /// graph for future launches. The caller must ensure `params` remains
    /// valid according to [`MemoryCopy1DNodeParams::new`] for every future launch
    /// that can execute this node.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the graph operation, if a previous asynchronous launch
    /// reported an error, or if CUDA reports runtime initialization diagnostics. Callbacks must
    /// not call CUDA functions; see [`Stream::add_callback`].
    pub unsafe fn set_memory_copy_node_1d_params(
        &mut self,
        node: GraphNode,
        params: &MemoryCopy1DNodeParams,
    ) -> Result<()> {
        self.check_node(&node)?;
        unsafe {
            try_ffi!(runtime::cudaGraphExecMemcpyNodeSetParams1D(
                self.as_raw(),
                node.as_raw(),
                params.dst().cast(),
                params.src().cast(),
                params.count() as _,
                params.kind().into(),
            ))?;
        }
        Ok(())
    }

    /// Updates a memcpy node to copy between typed device byte buffers.
    ///
    /// The node copies `src.byte_len()` bytes. `dst` must have at least that
    /// many bytes.
    ///
    /// # Safety
    ///
    /// CUDA stores the raw source and destination addresses in the executable
    /// graph for future launches. The caller must ensure `dst` and `src`
    /// remain valid for every future launch that can execute this node. `dst`
    /// must not be accessed through another mutable path while graph launches
    /// using this node can write it.
    ///
    /// # Errors
    ///
    /// Returns an error if `dst` is smaller than `src`, if CUDA rejects the graph
    /// operation, if a previous asynchronous launch reported an error, or if CUDA
    /// reports runtime initialization diagnostics.
    pub unsafe fn set_memory_copy_node_1d_device_to_device<D, S>(
        &mut self,
        node: GraphNode,
        dst: &mut D,
        src: &S,
    ) -> Result<()>
    where
        D: ByteBufferMut + ?Sized,
        S: ByteBuffer + ?Sized,
    {
        let count = src.byte_len();
        if dst.byte_len() < count {
            return Err(Error::InvalidMemoryAccess);
        }
        let params = unsafe {
            MemoryCopy1DNodeParams::new(
                dst.as_byte_mut_ptr().cast(),
                src.as_byte_ptr().cast(),
                count,
                MemoryCopyKind::DeviceToDevice,
            )
        };
        unsafe { self.set_memory_copy_node_1d_params(node, &params) }
    }

    /// Updates a memcpy node to copy between graph-retained buffers.
    ///
    /// The node copies `src.byte_len()` bytes. `dst` must have at least that
    /// many bytes. The executable graph retains both allocations so future
    /// launches cannot outlive the baked CUDA pointer values.
    ///
    /// # Errors
    ///
    /// Returns an error if `dst` is smaller than `src`, if `node` does not
    /// belong to the graph used to instantiate this executable graph, if CUDA
    /// rejects the graph update, if a previous asynchronous launch reported an
    /// error, or if CUDA reports runtime initialization diagnostics.
    pub fn set_buffer_memory_copy_node_1d_device_to_device<T>(
        &mut self,
        node: GraphNode,
        dst: &mut GraphBuffer<T>,
        src: &GraphBuffer<T>,
    ) -> Result<()>
    where
        T: DeviceRepr + Send + Sync,
    {
        if let (Some(exec_ctx), Some(dst_ctx)) = (&self.ctx, dst.context())
            && exec_ctx.as_ref() != dst_ctx
        {
            return Err(Error::GraphContextMismatch);
        }
        if let (Some(exec_ctx), Some(src_ctx)) = (&self.ctx, src.context())
            && exec_ctx.as_ref() != src_ctx
        {
            return Err(Error::GraphContextMismatch);
        }
        let count = src.byte_len();
        if dst.byte_len() < count {
            return Err(Error::InvalidMemoryAccess);
        }
        let params = unsafe {
            MemoryCopy1DNodeParams::new(
                dst.as_mut_ptr().cast(),
                src.as_ptr().cast(),
                count,
                MemoryCopyKind::DeviceToDevice,
            )
        };
        unsafe { self.set_memory_copy_node_1d_params(node, &params)? };
        self.retain_buffer(dst);
        self.retain_buffer(src);
        Ok(())
    }

    /// Updates the work represented by `node` in this executable graph as though `node` had contained the given `params` at instantiation.
    /// `node` must remain in the graph which was used to instantiate this executable graph.
    /// Changed edges to and from `node` are ignored.
    ///
    /// The source and destination memory in `params` must be allocated from the same contexts as the original source and destination memory.
    /// Both the instantiation-time memory operands and the memory operands in `params` must be 1-dimensional.
    /// Zero-length operations are not supported.
    ///
    /// The modifications only affect future launches of this executable graph.
    /// Already enqueued or running launches of this executable graph are not affected by this call.
    /// The original `node` is also not modified by this call.
    ///
    /// Returns [`crate::error::Status::InvalidValue`] if the memory operands' mappings changed or either the original or new memory operands are multidimensional.
    ///
    /// Graph objects are not threadsafe.
    ///
    /// # Safety
    ///
    /// CUDA stores the raw source and destination addresses in the executable
    /// graph for future launches. The caller must ensure `params` remains
    /// valid according to [`MemoryCopy3DNodeParams`] for every future launch that
    /// can execute this node.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the graph operation, if a previous asynchronous launch
    /// reported an error, or if CUDA reports runtime initialization diagnostics. Callbacks must
    /// not call CUDA functions; see [`Stream::add_callback`].
    pub unsafe fn set_memory_copy_node_params(
        &mut self,
        node: GraphNode,
        params: &MemoryCopy3DNodeParams,
    ) -> Result<()> {
        self.check_node(&node)?;
        let params = params.into();
        unsafe {
            try_ffi!(runtime::cudaGraphExecMemcpyNodeSetParams(
                self.as_raw(),
                node.as_raw(),
                &raw const params,
            ))?;
        }
        Ok(())
    }

    /// # Safety
    ///
    /// CUDA stores the raw symbol and source pointer in the executable graph
    /// for future launches. The caller must ensure `params` remains valid
    /// according to [`MemoryCopyToSymbolNodeParams::new`] for every future launch
    /// that can execute this node.
    pub unsafe fn set_memory_copy_node_to_symbol_params(
        &mut self,
        node: GraphNode,
        params: &MemoryCopyToSymbolNodeParams,
    ) -> Result<()> {
        self.check_node(&node)?;
        unsafe {
            try_ffi!(runtime::cudaGraphExecMemcpyNodeSetParamsToSymbol(
                self.as_raw(),
                node.as_raw(),
                params.symbol().cast(),
                params.src().cast(),
                params.count() as _,
                params.offset() as _,
                params.kind().into(),
            ))?;
        }
        Ok(())
    }

    /// # Safety
    ///
    /// CUDA stores the raw destination and symbol pointer in the executable
    /// graph for future launches. The caller must ensure `params` remains
    /// valid according to [`MemoryCopyFromSymbolNodeParams::new`] for every future
    /// launch that can execute this node.
    pub unsafe fn set_memory_copy_node_from_symbol_params(
        &mut self,
        node: GraphNode,
        params: &MemoryCopyFromSymbolNodeParams,
    ) -> Result<()> {
        self.check_node(&node)?;
        unsafe {
            try_ffi!(runtime::cudaGraphExecMemcpyNodeSetParamsFromSymbol(
                self.as_raw(),
                node.as_raw(),
                params.dst().cast(),
                params.symbol().cast(),
                params.count() as _,
                params.offset() as _,
                params.kind().into(),
            ))?;
        }
        Ok(())
    }

    /// Updates the work represented by `node` in this executable graph as though `node` had contained the given `params` at instantiation.
    /// `node` must remain in the graph which was used to instantiate this executable graph.
    /// Changed edges to and from `node` are ignored.
    ///
    /// Zero-sized operations are not supported.
    ///
    /// The new destination pointer in `params` must be to the same kind of allocation as the original destination pointer and have the same context association and device mapping as the original destination pointer.
    ///
    /// Both the value and pointer address may be updated.
    /// Changing other aspects of the memset (width, height, element size or pitch) may cause the update to be rejected.
    /// Specifically, for 2D memsets, all dimension changes are rejected.
    /// For 1D memsets, changes in height are explicitly rejected and other changes are opportunistically allowed if the resulting work maps onto the work resources already allocated for the node.
    ///
    /// The modifications only affect future launches of this executable graph.
    /// Already enqueued or running launches of this executable graph are not affected by this call.
    /// The original `node` is also not modified by this call.
    ///
    /// Graph objects are not threadsafe.
    ///
    /// # Safety
    ///
    /// CUDA stores the raw destination address in the executable graph for
    /// future launches. The caller must ensure `params` remains valid according
    /// to [`MemorySetNodeParams::new`] for every future launch that can execute
    /// this node.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the graph operation, if a previous asynchronous launch
    /// reported an error, or if CUDA reports runtime initialization diagnostics. Callbacks must
    /// not call CUDA functions; see [`Stream::add_callback`].
    pub unsafe fn set_memory_set_node_params(
        &mut self,
        node: GraphNode,
        params: &MemorySetNodeParams,
    ) -> Result<()> {
        self.check_node(&node)?;
        let params = params.into();
        unsafe {
            try_ffi!(runtime::cudaGraphExecMemsetNodeSetParams(
                self.as_raw(),
                node.as_raw(),
                &raw const params,
            ))?;
        }
        Ok(())
    }

    /// Updates the work represented by `node` in this executable graph as though `node` had contained the given `params` at instantiation.
    /// `node` must remain in the graph which was used to instantiate this executable graph.
    /// Changed edges to and from `node` are ignored.
    ///
    /// The modifications only affect future launches of this executable graph.
    /// Already enqueued or running launches of this executable graph are not affected by this call.
    /// The original `node` is also not modified by this call.
    ///
    /// Graph objects are not threadsafe.
    ///
    /// # Safety
    ///
    /// CUDA stores the raw callback function and user-data pointer in the
    /// executable graph for future launches. The caller must ensure `params`
    /// remains valid according to [`HostNodeParams::new`] for every future
    /// launch that can execute this node.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the graph operation, if a previous asynchronous launch
    /// reported an error, or if CUDA reports runtime initialization diagnostics. Callbacks must
    /// not call CUDA functions; see [`Stream::add_callback`].
    pub unsafe fn set_host_node_params(
        &mut self,
        node: GraphNode,
        params: &HostNodeParams,
    ) -> Result<()> {
        self.check_node(&node)?;
        let params = params.into();
        unsafe {
            try_ffi!(runtime::cudaGraphExecHostNodeSetParams(
                self.as_raw(),
                node.as_raw(),
                &raw const params,
            ))?;
        }
        Ok(())
    }

    /// Sets the event of an event record node in this executable graph.
    /// The node is identified by the corresponding `node` in the non-executable graph from which this executable graph was instantiated.
    ///
    /// The modifications only affect future launches of this executable graph.
    /// Already enqueued or running launches of this executable graph are not affected by this call.
    /// The original `node` is also not modified by this call.
    ///
    /// Graph objects are not threadsafe.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the graph operation, if a previous asynchronous launch
    /// reported an error, or if CUDA reports runtime initialization diagnostics. Callbacks must
    /// not call CUDA functions; see [`Stream::add_callback`].
    pub fn set_event_record_node_event(&mut self, node: GraphNode, event: &Event) -> Result<()> {
        self.check_node(&node)?;
        if let Some(ctx) = &self.ctx
            && ctx.as_ref() != event.context()
        {
            return Err(Error::GraphContextMismatch);
        }
        unsafe {
            try_ffi!(runtime::cudaGraphExecEventRecordNodeSetEvent(
                self.as_raw(),
                node.as_raw(),
                event.as_raw(),
            ))?;
        }
        Ok(())
    }

    /// Updates the work represented by `node` in this executable graph as though the nodes contained in `node`'s graph had the parameters contained in `child_graph`'s nodes at instantiation.
    /// `node` must remain in the graph which was used to instantiate this executable graph.
    /// Changed edges to and from `node` are ignored.
    ///
    /// The modifications only affect future launches of this executable graph.
    /// Already enqueued or running launches of this executable graph are not affected by this call.
    /// The original `node` is also not modified by this call.
    ///
    /// The topology of `child_graph`, as well as the node insertion order, must match that of the graph contained in `node`.
    /// See [`ExecutableGraph::update`] for a list of restrictions on what can be updated in an instantiated graph.
    /// The update is recursive, so child graph nodes contained within the top-level child graph are also updated.
    ///
    /// Graph objects are not threadsafe.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the graph operation, if a previous asynchronous launch
    /// reported an error, or if CUDA reports runtime initialization diagnostics. Callbacks must
    /// not call CUDA functions; see [`Stream::add_callback`].
    pub fn set_child_graph_node(&mut self, node: GraphNode, child_graph: &Graph) -> Result<()> {
        self.check_node(&node)?;
        if let (Some(exec_ctx), Some(child_ctx)) = (&self.ctx, &child_graph.ctx)
            && exec_ctx.as_ref() != child_ctx.as_ref()
        {
            return Err(Error::GraphContextMismatch);
        }
        unsafe {
            try_ffi!(runtime::cudaGraphExecChildGraphNodeSetParams(
                self.as_raw(),
                node.as_raw(),
                child_graph.as_raw(),
            ))?;
        }
        Ok(())
    }

    /// Sets the event of an event wait node in this executable graph.
    /// The node is identified by the corresponding `node` in the non-executable graph from which this executable graph was instantiated.
    ///
    /// The modifications only affect future launches of this executable graph.
    /// Already enqueued or running launches of this executable graph are not affected by this call.
    /// The original `node` is also not modified by this call.
    ///
    /// Graph objects are not threadsafe.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the graph operation, if a previous asynchronous launch
    /// reported an error, or if CUDA reports runtime initialization diagnostics. Callbacks must
    /// not call CUDA functions; see [`Stream::add_callback`].
    pub fn set_event_wait_node_event(&mut self, node: GraphNode, event: &Event) -> Result<()> {
        self.check_node(&node)?;
        unsafe {
            try_ffi!(runtime::cudaGraphExecEventWaitNodeSetEvent(
                self.as_raw(),
                node.as_raw(),
                event.as_raw(),
            ))?;
        }
        Ok(())
    }

    /// Sets `node` to be either enabled or disabled.
    /// Disabled nodes are functionally equivalent to empty nodes until they are reenabled.
    /// Existing node parameters are not affected by disabling/enabling the node.
    ///
    /// The node is identified by the corresponding `node` in the non-executable graph from which this executable graph was instantiated.
    ///
    /// `node` must not have been removed from the original graph.
    ///
    /// The modifications only affect future launches of this executable graph.
    /// Already enqueued or running launches of this executable graph are not affected by this call.
    /// The original `node` is also not modified by this call.
    ///
    /// Currently only kernel, memset and memcpy nodes are supported.
    ///
    /// Graph objects are not threadsafe.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the graph operation, if a previous asynchronous launch
    /// reported an error, or if CUDA reports runtime initialization diagnostics. Callbacks must
    /// not call CUDA functions; see [`Stream::add_callback`].
    fn set_node_enabled(&mut self, node: GraphNode, enabled: bool) -> Result<()> {
        self.check_node(&node)?;
        unsafe {
            try_ffi!(runtime::cudaGraphNodeSetEnabled(
                self.as_raw(),
                node.as_raw(),
                u32::from(enabled),
            ))?;
        }
        Ok(())
    }

    pub fn enable_node(&mut self, node: GraphNode) -> Result<()> {
        self.set_node_enabled(node, true)
    }

    pub fn disable_node(&mut self, node: GraphNode) -> Result<()> {
        self.set_node_enabled(node, false)
    }

    /// Returns whether `node` is enabled.
    ///
    /// The node is identified by the corresponding `node` in the non-executable graph from which this executable graph was instantiated.
    ///
    /// `node` must not have been removed from the original graph.
    ///
    /// Currently only kernel, memset and memcpy nodes are supported.
    ///
    /// Graph objects are not threadsafe.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the graph operation, if a previous asynchronous launch
    /// reported an error, or if CUDA reports runtime initialization diagnostics. Callbacks must
    /// not call CUDA functions; see [`Stream::add_callback`].
    pub fn is_node_enabled(&self, node: GraphNode) -> Result<bool> {
        self.check_node(&node)?;
        let mut enabled = 0;
        unsafe {
            try_ffi!(runtime::cudaGraphNodeGetEnabled(
                self.as_raw(),
                node.as_raw(),
                &raw mut enabled,
            ))?;
        }
        Ok(enabled != 0)
    }

    pub const fn as_raw(&self) -> runtime::cudaGraphExec_t {
        self.handle
    }

    pub fn context(&self) -> Option<&Context> {
        self.ctx.as_deref()
    }

    /// Consumes the executable graph and returns the raw CUDA executable graph
    /// handle without destroying it.
    ///
    /// The caller becomes responsible for eventually destroying the returned
    /// handle with CUDA.
    pub fn into_raw(self) -> runtime::cudaGraphExec_t {
        let graph = ManuallyDrop::new(self);
        graph.as_raw()
    }
}

impl Drop for ExecutableGraph {
    fn drop(&mut self) {
        unsafe {
            if let Err(err) = try_ffi!(runtime::cudaGraphExecDestroy(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cuda graph exec: {err}");
            }
        }
    }
}

impl ExecutableGraphLaunchOperation<'_> {
    /// Enqueues this graph launch in `stream`.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the graph operation, if `stream` belongs to a different context, if a previous asynchronous launch reported an error, or if CUDA reports runtime initialization diagnostics.
    pub fn enqueue(self, stream: &Stream) -> Result<()> {
        self.graph.launch(stream)
    }

    pub const fn graph(&self) -> &ExecutableGraph {
        self.graph
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutableGraphUpdate {
    pub result: GraphExecUpdateResult,
    pub error_node: Option<GraphNode>,
    pub error_from_node: Option<GraphNode>,
}

impl ExecutableGraphUpdate {
    fn from_result_info(value: runtime::cudaGraphExecUpdateResultInfo, graph: &Graph) -> Self {
        Self {
            result: value.result.into(),
            error_node: if value.errorNode.is_null() {
                None
            } else {
                Some(graph.node_from_raw(value.errorNode))
            },
            error_from_node: if value.errorFromNode.is_null() {
                None
            } else {
                Some(graph.node_from_raw(value.errorFromNode))
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MemorySetNodeParams {
    dst: DevicePtr,
    pitch: usize,
    value: u32,
    element_size: u32,
    width: usize,
    height: usize,
}

impl MemorySetNodeParams {
    /// Creates raw memset node parameters.
    ///
    /// # Safety
    ///
    /// `dst` must be valid for writes of `element_size * width` bytes when the
    /// graph executes. If `height` or `pitch` are changed after construction,
    /// the caller must account for those values as required by CUDA.
    pub const unsafe fn new(dst: DevicePtr, element_size: u32, width: usize) -> Self {
        Self {
            dst,
            pitch: 0,
            value: 0,
            element_size,
            width,
            height: 1,
        }
    }

    pub const fn with_pitch(mut self, pitch: usize) -> Self {
        self.pitch = pitch;
        self
    }

    pub const fn with_value(mut self, value: u32) -> Self {
        self.value = value;
        self
    }

    pub const fn with_height(mut self, height: usize) -> Self {
        self.height = height;
        self
    }

    pub const fn dst(self) -> DevicePtr {
        self.dst
    }

    pub const fn pitch(self) -> usize {
        self.pitch
    }

    pub const fn value(self) -> u32 {
        self.value
    }

    pub const fn element_size(self) -> u32 {
        self.element_size
    }

    pub const fn width(self) -> usize {
        self.width
    }

    pub const fn height(self) -> usize {
        self.height
    }
}

impl From<&MemorySetNodeParams> for driver::CUDA_MEMSET_NODE_PARAMS {
    fn from(value: &MemorySetNodeParams) -> Self {
        Self {
            dst: value.dst().as_ptr() as _,
            pitch: value.pitch() as _,
            value: value.value(),
            elementSize: value.element_size(),
            width: value.width() as _,
            height: value.height() as _,
        }
    }
}
