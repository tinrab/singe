pub mod contextual;
mod host_data;
pub mod memory;
pub mod owned;

/// Raw CUDA graph node parameters and pointer-based operations.
pub mod raw;

mod retained_resource;

mod buffer;
mod definition;
mod executable;
mod node;

use std::{
    any::Any,
    cell::Cell,
    collections::HashMap,
    fmt::{self, Display, Formatter},
    hash::{Hash, Hasher},
    marker::PhantomData,
    mem::{ManuallyDrop, MaybeUninit},
    ops::Deref,
    ptr,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

use bitflags::parser;
use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{impl_enum_conversion, impl_enum_display};
use singe_cuda_sys::{driver, runtime};

pub use self::{
    host_data::GraphHostBuffer,
    retained_resource::{
        GraphRetainedResource, collect_graph_resources, has_deferred_graph_resources,
    },
};
use crate::{
    context::Context,
    dim::Dim3,
    error::{Error, Result},
    memory::{DeviceMemory, MemoryAccessDescriptor, MemoryPoolProps},
    module::{CodeOwnerId, Module},
    try_ffi,
    types::DevicePtr,
    view::{
        DeviceInput, DeviceOutput, DeviceRegion, DeviceRegionProvider, DeviceRepr, DeviceSlice,
        DeviceSliceMut,
    },
};

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
    /// The value is a percentage in the range `0..=100` describing the preferred shared-memory carveout for the launch.
    /// This is a hint, and the driver may choose a different configuration if required.
    PreferredSharedMemoryCarveout =
        runtime::cudaLaunchAttributeID::cudaLaunchAttributePreferredSharedMemoryCarveout as _,
}

impl_enum_conversion!(
    u32,
    runtime::cudaLaunchAttributeID,
    GraphKernelNodeAttributeId
);

/// Typed launch attributes stored on a CUDA graph kernel node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum GraphKernelNodeAttribute {
    /// Enables or disables a cooperative kernel launch.
    Cooperative(
        /// Whether the kernel uses cooperative launch semantics.
        bool,
    ),
    /// Sets the thread-block cluster dimensions.
    ClusterDimension(
        /// Cluster dimensions in thread blocks.
        Dim3,
    ),
    /// Sets the scheduling priority hint for this kernel node.
    Priority(
        /// Requested node priority.
        i32,
    ),
    /// Sets the preferred percentage of shared memory in the shared-memory/L1 partition.
    PreferredSharedMemoryCarveout(
        /// Preferred shared-memory percentage.
        u32,
    ),
}

/// Address and size returned for a graph memory-allocation node.
///
/// The metadata may retain the graph and its context.
/// Access to the allocation must follow the ordering of its allocation and free nodes.
#[derive(Debug, Clone)]
pub struct MemoryAllocationNodeInfo {
    ptr: DevicePtr,
    /// Allocation size in bytes.
    pub byte_size: usize,
    graph_id: Option<GraphId>,
    _graph: Option<Arc<GraphInner>>,
    context: Option<Arc<Context>>,
}

bitflags::bitflags! {
    /// Flags for instantiating a graph
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct GraphInstantiateFlags: u64 {
        /// Automatically free memory allocated in a graph before relaunching.
        const AUTO_FREE_ON_LAUNCH = driver::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH as _;
        /// Automatically upload the graph after instantiation.
        ///
        /// Only supported by `cuGraphInstantiateWithParams`.
        /// The upload will be performed using the stream provided in `instantiateParams`.
        const UPLOAD = driver::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD as _;
        /// Instantiate the graph to be launchable from the device.
        ///
        /// This flag can only be used on platforms which support unified addressing.
        /// This flag cannot be used in conjunction with `CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH`.
        const DEVICE_LAUNCH = driver::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH as _;
        /// Run the graph using the per-node priority attributes rather than the priority of the stream it is launched into.
        const USE_NODE_PRIORITY = driver::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY as _;
    }
}

impl Display for GraphInstantiateFlags {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        parser::to_writer(self, f)
    }
}

bitflags::bitflags! {
    /// The additional write options for ::cuGraphDebugDotPrint
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct GraphDebugDotFlags: u32 {
        /// Output all debug data as if every debug flag is enabled.
        const VERBOSE = driver::CUgraphDebugDot_flags::CU_GRAPH_DEBUG_DOT_FLAGS_VERBOSE as _;
        /// Use CUDA Runtime structures for output.
        const RUNTIME_TYPES = driver::CUgraphDebugDot_flags::CU_GRAPH_DEBUG_DOT_FLAGS_RUNTIME_TYPES as _;
        /// Adds CUDA_KERNEL_NODE_PARAMS values to output.
        const KERNEL_NODE_PARAMS = driver::CUgraphDebugDot_flags::CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_PARAMS as _;
        /// Adds CUDA_MEMCPY3D values to output.
        const MEMCPY_NODE_PARAMS = driver::CUgraphDebugDot_flags::CU_GRAPH_DEBUG_DOT_FLAGS_MEMCPY_NODE_PARAMS as _;
        /// Adds CUDA_MEMSET_NODE_PARAMS values to output.
        const MEMSET_NODE_PARAMS = driver::CUgraphDebugDot_flags::CU_GRAPH_DEBUG_DOT_FLAGS_MEMSET_NODE_PARAMS as _;
        /// Adds CUDA_HOST_NODE_PARAMS values to output.
        const HOST_NODE_PARAMS = driver::CUgraphDebugDot_flags::CU_GRAPH_DEBUG_DOT_FLAGS_HOST_NODE_PARAMS as _;
        /// Adds CUevent handle from record and wait nodes to output.
        const EVENT_NODE_PARAMS = driver::CUgraphDebugDot_flags::CU_GRAPH_DEBUG_DOT_FLAGS_EVENT_NODE_PARAMS as _;
        /// Adds CUDA_EXT_SEM_SIGNAL_NODE_PARAMS values to output.
        const EXTERNAL_SEMAPHORE_SIGNAL_NODE_PARAMS = driver::CUgraphDebugDot_flags::CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_SIGNAL_NODE_PARAMS as _;
        /// Adds CUDA_EXT_SEM_WAIT_NODE_PARAMS values to output.
        const EXTERNAL_SEMAPHORE_WAIT_NODE_PARAMS = driver::CUgraphDebugDot_flags::CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_WAIT_NODE_PARAMS as _;
        /// Adds CUkernelNodeAttrValue values to output.
        const KERNEL_NODE_ATTRIBUTES = driver::CUgraphDebugDot_flags::CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_ATTRIBUTES as _;
        /// Adds node handles and every kernel function handle to output.
        const HANDLES = driver::CUgraphDebugDot_flags::CU_GRAPH_DEBUG_DOT_FLAGS_HANDLES as _;
        /// Adds memory alloc node parameters to output.
        const MEMORY_ALLOC_NODE_PARAMS = driver::CUgraphDebugDot_flags::CU_GRAPH_DEBUG_DOT_FLAGS_MEM_ALLOC_NODE_PARAMS as _;
        /// Adds memory free node parameters to output.
        const MEMORY_FREE_NODE_PARAMS = driver::CUgraphDebugDot_flags::CU_GRAPH_DEBUG_DOT_FLAGS_MEM_FREE_NODE_PARAMS as _;
        /// Adds batch mem op node parameters to output.
        const BATCH_MEM_OP_NODE_PARAMS = driver::CUgraphDebugDot_flags::CU_GRAPH_DEBUG_DOT_FLAGS_BATCH_MEM_OP_NODE_PARAMS as _;
        /// Adds edge numbering information.
        const EXTRA_TOPOLOGY_INFO = driver::CUgraphDebugDot_flags::CU_GRAPH_DEBUG_DOT_FLAGS_EXTRA_TOPO_INFO as _;
        /// Adds conditional node parameters to output.
        const CONDITIONAL_NODE_PARAMS = driver::CUgraphDebugDot_flags::CU_GRAPH_DEBUG_DOT_FLAGS_CONDITIONAL_NODE_PARAMS as _;
    }
}

impl Display for GraphDebugDotFlags {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        parser::to_writer(self, f)
    }
}

/// Graph node types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum GraphNodeType {
    /// GPU kernel node.
    Kernel = driver::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL as _,
    /// Memcpy node.
    Memcpy = driver::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMCPY as _,
    /// Memset node.
    Memset = driver::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMSET as _,
    /// Host (executable) node.
    Host = driver::CUgraphNodeType::CU_GRAPH_NODE_TYPE_HOST as _,
    /// Node which executes an embedded graph.
    Graph = driver::CUgraphNodeType::CU_GRAPH_NODE_TYPE_GRAPH as _,
    /// Empty (no-op) node.
    Empty = driver::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY as _,
    /// External event wait node.
    WaitEvent = driver::CUgraphNodeType::CU_GRAPH_NODE_TYPE_WAIT_EVENT as _,
    /// External event record node.
    EventRecord = driver::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EVENT_RECORD as _,
    /// External semaphore signal node.
    ExternalSemaphoresSignal = driver::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL as _,
    /// External semaphore wait node.
    ExternalSemaphoresWait = driver::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT as _,
    /// Memory Allocation Node.
    MemoryAlloc = driver::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEM_ALLOC as _,
    /// Memory Free Node.
    MemoryFree = driver::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEM_FREE as _,
    /// Batch `MemOp` Node See cuStreamBatchMemOp and [CUstreamBatchMemOpType](https://docs.nvidia.com/cuda/archive/13.4.2/cuda-driver-api/cuda_driver_api/group__CUDA__TYPES.html#group__cuda__types_1gab257b534afdb704b6ebdb99c16a5b292) for what these nodes can do.
    BatchMemOp = driver::CUgraphNodeType::CU_GRAPH_NODE_TYPE_BATCH_MEM_OP as _,
    /// Conditional Node.
    Conditional = driver::CUgraphNodeType::CU_GRAPH_NODE_TYPE_CONDITIONAL as _,
}

impl_enum_conversion!(u32, runtime::cudaGraphNodeType, GraphNodeType);

/// Dependency types applied to CUDA graph edges.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u8)]
#[non_exhaustive]
pub enum GraphDependencyType {
    /// This is an ordinary dependency.
    Default = driver::CUgraphDependencyType::CU_GRAPH_DEPENDENCY_TYPE_DEFAULT as _,
    /// This dependency type allows the downstream node to use `cudaGridDependencySynchronize()`.
    ///
    /// It may only be used between kernel nodes, and must be used with either the `CU_GRAPH_KERNEL_NODE_PORT_PROGRAMMATIC` or `CU_GRAPH_KERNEL_NODE_PORT_LAUNCH_ORDER` outgoing port.
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

/// CUDA Graph Update error types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum GraphExecUpdateResult {
    /// The update succeeded.
    Success = driver::CUgraphExecUpdateResult::CU_GRAPH_EXEC_UPDATE_SUCCESS as _,
    /// The update failed for an unexpected reason which is described in the return value of the function.
    Error = driver::CUgraphExecUpdateResult::CU_GRAPH_EXEC_UPDATE_ERROR as _,
    /// The update failed because the topology changed.
    ErrorTopologyChanged =
        driver::CUgraphExecUpdateResult::CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED as _,
    /// The update failed because a node type changed.
    ErrorNodeTypeChanged =
        driver::CUgraphExecUpdateResult::CU_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED as _,
    /// The update failed because the function of a kernel node changed (CUDA driver &lt; 11.2).
    ErrorFunctionChanged =
        driver::CUgraphExecUpdateResult::CU_GRAPH_EXEC_UPDATE_ERROR_FUNCTION_CHANGED as _,
    /// The update failed because the parameters changed in a way that is not supported.
    ErrorParametersChanged =
        driver::CUgraphExecUpdateResult::CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED as _,
    /// The update failed because something about the node is not supported.
    ErrorNotSupported =
        driver::CUgraphExecUpdateResult::CU_GRAPH_EXEC_UPDATE_ERROR_NOT_SUPPORTED as _,
    /// The update failed because the function of a kernel node changed in an unsupported way.
    ErrorUnsupportedFunctionChange =
        driver::CUgraphExecUpdateResult::CU_GRAPH_EXEC_UPDATE_ERROR_UNSUPPORTED_FUNCTION_CHANGE
            as _,
    /// The update failed because the node attributes changed in a way that is not supported.
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

/// CUDA graph node handle with optional retained graph and context metadata.
///
/// Nodes returned by a [`Graph`] retain their source graph and carry its identity so methods can reject nodes from another graph.
#[derive(Debug, Clone)]
pub struct GraphNode {
    handle: runtime::cudaGraphNode_t,
    graph_id: Option<GraphId>,
    graph: Option<Arc<GraphInner>>,
    context: Option<Arc<Context>>,
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

/// Ports and dependency semantics attached to a directed CUDA graph edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GraphEdgeAttributes {
    /// Output port of the source node.
    pub from_port: u8,
    /// Input port of the destination node.
    pub to_port: u8,
    /// Dependency semantics associated with the edge.
    pub dependency_type: GraphDependencyType,
}

/// Adjacent graph node and the attributes of its connecting edge.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GraphDependency {
    /// Node connected by this dependency.
    pub node: GraphNode,
    /// Port and dependency attributes of the connecting edge.
    pub data: GraphEdgeAttributes,
}

/// Directed edge between graph nodes, including its dependency attributes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GraphEdge {
    /// Source node of the directed dependency.
    pub from: GraphNode,
    /// Destination node that depends on the source.
    pub to: GraphNode,
    /// Port and dependency attributes of the edge.
    pub data: GraphEdgeAttributes,
}

/// Snapshot of graph node and edge counts, grouped by node kind.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct GraphTopologySummary {
    /// Total number of nodes.
    pub nodes: usize,
    /// Number of nodes with no incoming dependencies.
    pub root_nodes: usize,
    /// Total number of directed dependency edges.
    pub edges: usize,
    /// Number of kernel nodes.
    pub kernel_nodes: usize,
    /// Number of memory-copy nodes.
    pub memcpy_nodes: usize,
    /// Number of memory-fill nodes.
    pub memset_nodes: usize,
    /// Number of host callback nodes.
    pub host_nodes: usize,
    /// Number of child-graph nodes.
    pub child_graph_nodes: usize,
    /// Number of empty dependency nodes.
    pub empty_nodes: usize,
    /// Number of event-wait nodes.
    pub wait_event_nodes: usize,
    /// Number of event-record nodes.
    pub event_record_nodes: usize,
    /// Number of external-semaphore signal nodes.
    pub external_semaphores_signal_nodes: usize,
    /// Number of external-semaphore wait nodes.
    pub external_semaphores_wait_nodes: usize,
    /// Number of memory-allocation nodes.
    pub memory_alloc_nodes: usize,
    /// Number of memory-free nodes.
    pub memory_free_nodes: usize,
    /// Number of batched memory-operation nodes.
    pub batch_mem_op_nodes: usize,
    /// Number of conditional nodes.
    pub conditional_nodes: usize,
}

/// Three-dimensional origin passed to CUDA memory operations.
///
/// For copies, x is measured in bytes for linear memory and elements for arrays; y and z select rows and slices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Position {
    /// X coordinate in bytes for linear memory or elements for arrays.
    pub x: usize,
    /// Row index.
    pub y: usize,
    /// Slice index.
    pub z: usize,
}

/// Three-dimensional size passed to CUDA memory operations.
///
/// Width is measured in bytes for linear memory and elements for arrays; height and depth count rows and slices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Extent {
    /// Width in bytes for linear memory or elements for arrays.
    pub width: usize,
    /// Number of rows.
    pub height: usize,
    /// Number of slices.
    pub depth: usize,
}

/// Pool, access permissions, and byte count for a graph allocation node.
#[derive(Debug, Clone)]
pub struct MemoryAllocationNodeParams<'a> {
    /// Properties of the pool backing the graph allocation.
    pub pool_props: MemoryPoolProps,
    /// Locations and permissions that may access the allocation.
    pub access_descs: &'a [MemoryAccessDescriptor],
    /// Number of bytes allocated when the node executes.
    pub byte_size: usize,
}

impl Default for GraphEdgeAttributes {
    fn default() -> Self {
        Self {
            from_port: 0,
            to_port: 0,
            dependency_type: GraphDependencyType::Default,
        }
    }
}

impl From<runtime::cudaGraphEdgeData> for GraphEdgeAttributes {
    fn from(value: runtime::cudaGraphEdgeData) -> Self {
        Self {
            from_port: value.from_port,
            to_port: value.to_port,
            dependency_type: GraphDependencyType::try_from(value.type_)
                .unwrap_or(GraphDependencyType::Default),
        }
    }
}

impl From<GraphEdgeAttributes> for runtime::cudaGraphEdgeData {
    fn from(value: GraphEdgeAttributes) -> Self {
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

impl MemoryAllocationNodeInfo {
    /// Returns the address reserved for the graph allocation node.
    #[must_use]
    pub const fn ptr(&self) -> DevicePtr {
        self.ptr
    }

    /// Returns the retained context, if the allocation metadata includes one.
    #[must_use]
    pub fn context(&self) -> Option<&Arc<Context>> {
        self.context.as_ref()
    }

    fn from_raw_in_graph(
        ptr: DevicePtr,
        byte_size: usize,
        graph_id: GraphId,
        graph: Arc<GraphInner>,
        context: Option<Arc<Context>>,
    ) -> Self {
        Self::from_raw(ptr, byte_size, Some(graph_id), Some(graph), context)
    }

    fn from_raw(
        ptr: DevicePtr,
        byte_size: usize,
        graph_id: Option<GraphId>,
        graph: Option<Arc<GraphInner>>,
        context: Option<Arc<Context>>,
    ) -> Self {
        Self {
            ptr,
            byte_size,
            graph_id,
            _graph: graph,
            context,
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

/// A graph definition with exclusive host access to native graph operations.
///
/// Move it between threads or share it through external synchronization.
/// Native graph objects cannot be queried or modified concurrently.
#[derive(Debug)]
pub struct Graph {
    inner: Arc<GraphInner>,
    id: GraphId,
    context: Option<Arc<Context>>,
    _not_sync: PhantomData<Cell<()>>,
}

/// Process-local identity used to associate nodes with their source graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GraphId(u64);

/// Owns a CUDA graph handle without requiring an associated Singe context.
///
/// The caller must establish the CUDA context required for raw native operations.
#[derive(Debug)]
pub struct RawGraph {
    inner: Arc<GraphInner>,
}

/// Raw graph ownership, including context and every graph-retained resource.
///
/// Keep this value alive until the native handle is destroyed or re-adopt the complete value with `Graph::from_raw_parts`.
/// Bind its context before native use and complete graph work before releasing these owners.
#[derive(Debug)]
pub struct RawGraphParts {
    /// Native graph handle whose ownership accompanies the retained metadata.
    pub handle: runtime::cudaGraph_t,
    context: Option<Arc<Context>>,
    retained: RetainedResources,
    id: GraphId,
}

impl RawGraphParts {
    /// Returns the context retained alongside the raw graph, when present.
    #[must_use]
    pub fn context(&self) -> Option<&Arc<Context>> {
        self.context.as_ref()
    }
}

/// Raw executable ownership retaining its context, source graph, and resource dependencies.
///
/// Keep this value alive until native destruction or re-adopt it with `ExecutableGraph::from_raw_parts`.
/// Bind its context before native use and complete all launches before releasing these owners.
#[derive(Debug)]
pub struct RawExecutableGraphParts {
    /// Native executable handle whose ownership accompanies the retained metadata.
    pub handle: runtime::cudaGraphExec_t,
    context: Option<Arc<Context>>,
    retained: RetainedResources,
    source_graph: Option<Arc<GraphInner>>,
    source_graph_id: Option<GraphId>,
}

impl RawExecutableGraphParts {
    /// Returns the context retained alongside the raw executable, when present.
    #[must_use]
    pub fn context(&self) -> Option<&Arc<Context>> {
        self.context.as_ref()
    }
}

// SAFETY: these values expose only native handles and retained metadata.
// Native use remains unsafe, and all retained resources support thread-bound destruction.
unsafe impl Send for RawGraphParts {}

unsafe impl Sync for RawGraphParts {}

unsafe impl Send for RawExecutableGraphParts {}

unsafe impl Sync for RawExecutableGraphParts {}

#[derive(Debug)]
pub(crate) struct GraphInner {
    handle: runtime::cudaGraph_t,
    owns_handle: bool,
    context: Option<Arc<Context>>,
    retained: Mutex<RetainedResources>,
    parent: Option<Arc<Self>>,
}

// SAFETY: CUDA graph handles can be retained and destroyed from any host thread after
// binding the associated context. Mutating graph APIs require `&mut Graph`.
unsafe impl Send for GraphInner {}

unsafe impl Sync for GraphInner {}

#[derive(Clone)]
pub(crate) struct RetainedAllocation(Arc<dyn Any + Send + Sync>);

impl fmt::Debug for RetainedAllocation {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("RetainedAllocation")
            .field("strong_count", &Arc::strong_count(&self.0))
            .finish()
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct RetainedResources {
    allocations: HashMap<usize, RetainedAllocation>,
    modules: HashMap<CodeOwnerId, Module>,
}

impl RetainedResources {
    fn retain_allocation(&mut self, allocation: RetainedAllocation) {
        let identity = Arc::as_ptr(&allocation.0).cast::<()>().addr();
        self.allocations.entry(identity).or_insert(allocation);
    }

    fn retain_module(&mut self, module: &Module) {
        self.modules
            .entry(module.code_owner_id())
            .or_insert_with(|| module.clone());
    }

    fn extend(&mut self, other: Self) {
        self.allocations.extend(other.allocations);
        self.modules.extend(other.modules);
    }
}

/// Borrowed graph view tied to the graph node that exposes it.
///
/// Dropping the view does not destroy the borrowed native graph.
#[derive(Debug)]
pub struct BorrowedGraph<'node> {
    graph: Graph,
    _node: PhantomData<&'node GraphNode>,
}

/// Device memory whose allocation is retained by CUDA graph objects.
///
/// `GraphBuffer` values are created through [`Graph::create_buffer_uninit`], [`Graph::zeroed_buffer`], or [`Graph::buffer_from_slice`].
/// Graph and executable-graph APIs that accept `GraphBuffer` retain the underlying allocation so graph replay cannot outlive the device pointers baked into CUDA graph nodes.
/// Each resource snapshot retains a given allocation owner once, including after graph updates and child-graph merges.
#[derive(Debug)]
pub struct GraphBuffer<T: DeviceRepr> {
    memory: DeviceMemory<T>,
}

// SAFETY: this typed handle preserves the shared allocation's identity, bounds, and actual context.
unsafe impl<T: DeviceRepr> DeviceRegionProvider for GraphBuffer<T> {
    fn region(&self) -> DeviceRegion<'_> {
        self.memory.region()
    }
}

unsafe impl<T: DeviceRepr> DeviceInput<T> for GraphBuffer<T> {
    fn as_device_ptr(&self) -> *const T {
        self.memory.as_ptr()
    }

    fn len(&self) -> usize {
        self.memory.len()
    }
}

// SAFETY: the typed owner or view preserves initialized values for its borrow.
unsafe impl<T: DeviceRepr> DeviceSlice<T> for GraphBuffer<T> {}

unsafe impl<T: DeviceRepr> DeviceSliceMut<T> for GraphBuffer<T> {}

unsafe impl<T: DeviceRepr> DeviceOutput<T> for GraphBuffer<T> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.memory.as_mut_ptr()
    }
}

// SAFETY: output metadata describes the same live typed range without initialized reads.
unsafe impl<T: DeviceRepr> DeviceInput<T> for GraphBuffer<MaybeUninit<T>> {
    fn len(&self) -> usize {
        self.memory.len()
    }
}

unsafe impl<T: DeviceRepr> DeviceOutput<T> for GraphBuffer<MaybeUninit<T>> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.memory.as_mut_ptr().cast()
    }
}

impl RawGraph {
    /// Wraps an existing CUDA graph handle and takes ownership of it.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid CUDA graph handle.
    /// Ownership of `handle` is transferred to the returned [`RawGraph`], and the handle must not be destroyed elsewhere after calling this function.
    ///
    /// # Errors
    ///
    /// Returns an error if the native handle is null.
    pub unsafe fn from_raw(handle: runtime::cudaGraph_t) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            inner: Arc::new(GraphInner {
                handle,
                owns_handle: true,
                context: None,
                retained: Mutex::new(RetainedResources::default()),
                parent: None,
            }),
        })
    }

    /// Creates an empty raw graph without a Singe context association.
    ///
    /// Prefer [`Context::create_graph`] for ordinary Singe code.
    /// Raw graphs do not model context association, so the caller must keep CUDA context, node, executable update, upload, and launch relationships coherent.
    ///
    /// # Safety
    ///
    /// The returned graph has no modeled CUDA context association.
    /// The caller must ensure every node, kernel, memory operand, child graph, executable update, upload, and launch is used with the correct CUDA context.
    ///
    /// # Errors
    ///
    /// Propagates CUDA graph creation errors and rejects a null returned graph handle.
    pub unsafe fn create() -> Result<Self> {
        let mut handle = ptr::null_mut();
        // SAFETY: `cudaGraphCreate` receives valid output locations.
        unsafe {
            try_ffi!(runtime::cudaGraphCreate(&raw mut handle, 0))?;
        }
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { Self::from_raw(handle) }
    }

    /// Returns the owned native graph handle without transferring ownership.
    #[must_use]
    pub fn as_raw(&self) -> runtime::cudaGraph_t {
        self.inner.handle
    }

    /// Consumes the graph and returns the raw CUDA graph handle without destroying it.
    ///
    /// The caller becomes responsible for eventually destroying the returned handle with CUDA.
    ///
    /// # Panics
    ///
    /// Panics if another owner still retains the graph handle.
    #[must_use]
    pub fn into_raw(self) -> runtime::cudaGraph_t {
        let inner = Arc::try_unwrap(self.inner)
            .unwrap_or_else(|_| panic!("cannot take raw graph handle while it is still shared"));
        let inner = ManuallyDrop::new(inner);
        inner.handle
    }
}

static NEXT_GRAPH_ID: AtomicU64 = AtomicU64::new(1);

impl GraphId {
    /// Generates an identity for a graph in this process.
    pub fn generate() -> Self {
        Self(NEXT_GRAPH_ID.fetch_add(1, Ordering::Relaxed))
    }

    /// Returns the integer representation of this graph identity.
    #[must_use]
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

impl Display for GraphId {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl Drop for GraphInner {
    fn drop(&mut self) {
        if !self.owns_handle {
            return;
        }

        if let Some(context) = &self.context
            && let Err(error) = context.bind()
        {
            eprintln!(
                "failed to bind cuda graph context for destruction: {}",
                singe_core::ErrorChain(&error)
            );
            return;
        }
        // SAFETY: `cudaGraphDestroy` receives live handles owned by the wrappers.
        unsafe {
            if let Err(error) = try_ffi!(runtime::cudaGraphDestroy(self.handle)) {
                eprintln!(
                    "failed to destroy cuda graph: {}",
                    singe_core::ErrorChain(&error)
                );
            }
        }
    }
}

impl BorrowedGraph<'_> {
    /// Wraps an existing CUDA graph handle without taking ownership.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid CUDA graph handle for the returned lifetime.
    /// The returned graph view will not destroy `handle` when dropped.
    ///
    /// # Errors
    ///
    /// Returns an error if the native handle is null.
    pub unsafe fn from_raw(handle: runtime::cudaGraph_t) -> Result<Self> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { Self::from_raw_in_context(handle, None) }
    }

    /// Wraps an existing CUDA graph handle without taking ownership and keeps a modeled context association for safe graph operations through the borrowed view.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid CUDA graph handle for the returned lifetime, and it must be associated with `context` when `context` is present.
    /// The returned graph view will not destroy `handle` when dropped.
    ///
    /// # Errors
    ///
    /// Returns an error if the native handle is null.
    pub unsafe fn from_raw_in_context(
        handle: runtime::cudaGraph_t,
        context: Option<Arc<Context>>,
    ) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            // SAFETY: the caller upholds this function's safety contract.
            graph: unsafe { Graph::from_raw_borrowed_in_context(handle, context) },
            _node: PhantomData,
        })
    }

    /// Borrows the graph API while preserving this view’s native lifetime.
    #[must_use]
    pub const fn as_graph(&self) -> &Graph {
        &self.graph
    }

    /// Returns the borrowed native graph handle without transferring ownership.
    #[must_use]
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

/// Owns an instantiated CUDA graph and retains its execution resources.
///
/// When available, source graph identity and context metadata allow node updates and launches to validate their inputs.
#[derive(Debug)]
pub struct ExecutableGraph {
    handle: runtime::cudaGraphExec_t,
    context: Option<Arc<Context>>,
    source_graph_id: Option<GraphId>,
    source_graph: Option<Arc<GraphInner>>,
    retained: RetainedResources,
}

/// Borrowed executable-graph launch descriptor for stream capture.
#[derive(Debug, Clone, Copy)]
pub struct ExecutableGraphLaunchOperation<'graph> {
    graph: &'graph ExecutableGraph,
}

/// Owns a native executable-graph handle without retained context or resource metadata.
#[derive(Debug)]
pub struct RawExecutableGraph {
    handle: runtime::cudaGraphExec_t,
}

impl Drop for RawExecutableGraph {
    fn drop(&mut self) {
        // SAFETY: `cudaGraphExecDestroy` receives live handles owned by the wrappers.
        unsafe {
            if let Err(error) = try_ffi!(runtime::cudaGraphExecDestroy(self.handle)) {
                eprintln!(
                    "failed to destroy cuda graph exec: {}",
                    singe_core::ErrorChain(&error)
                );
            }
        }
    }
}

// SAFETY: native executable ownership may move between host threads.
// Mutation requires exclusive access; shared owned replay serializes native calls with a mutex.
unsafe impl Send for ExecutableGraph {}

impl Drop for ExecutableGraph {
    fn drop(&mut self) {
        if let Err(error) = self.bind_context() {
            eprintln!(
                "failed to bind cuda executable graph context for destruction: {}",
                singe_core::ErrorChain(&error)
            );
            return;
        }
        // SAFETY: `cudaGraphExecDestroy` receives live handles owned by the wrappers.
        unsafe {
            if let Err(error) = try_ffi!(runtime::cudaGraphExecDestroy(self.handle)) {
                eprintln!(
                    "failed to destroy cuda graph exec: {}",
                    singe_core::ErrorChain(&error)
                );
            }
        }
    }
}

/// CUDA graph update result and nodes associated with an update failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutableGraphUpdate {
    /// CUDA’s classification of the update outcome.
    pub result: GraphExecUpdateResult,
    /// Node associated with an update failure, when CUDA reports one.
    pub error_node: Option<GraphNode>,
    /// Source node of an edge associated with an update failure, when CUDA reports one.
    pub error_from_node: Option<GraphNode>,
}

/// Raw destination, geometry, and fill value for a graph memset node.
///
/// These parameters do not retain the destination allocation.
/// Its owner must outlive every graph replay that uses the stored address.
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
    /// `dst` must be valid for writes of `element_size * width` bytes when the graph executes.
    /// If `height` or `pitch` are changed after construction, the caller must account for those values as required by CUDA.
    #[must_use]
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

    /// Sets the row stride in bytes without validating the destination range.
    #[must_use]
    pub const fn with_pitch(mut self, pitch: usize) -> Self {
        self.pitch = pitch;
        self
    }

    /// Sets the fill value; CUDA uses the low bytes selected by the element size.
    #[must_use]
    pub const fn with_value(mut self, value: u32) -> Self {
        self.value = value;
        self
    }

    /// Sets the number of rows without validating the destination range.
    #[must_use]
    pub const fn with_height(mut self, height: usize) -> Self {
        self.height = height;
        self
    }

    /// Returns the raw destination address.
    #[must_use]
    pub const fn dst(self) -> DevicePtr {
        self.dst
    }

    /// Returns the row stride in bytes.
    #[must_use]
    pub const fn pitch(self) -> usize {
        self.pitch
    }

    /// Returns the fill value before truncation to the element size.
    #[must_use]
    pub const fn value(self) -> u32 {
        self.value
    }

    /// Returns the fill element size in bytes.
    #[must_use]
    pub const fn element_size(self) -> u32 {
        self.element_size
    }

    /// Returns the number of fill elements per row.
    #[must_use]
    pub const fn width(self) -> usize {
        self.width
    }

    /// Returns the number of rows.
    #[must_use]
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

impl_enum_display!(GraphKernelNodeAttributeId, {
    GraphKernelNodeAttributeId::Cooperative => "cudaLaunchAttributeCooperative",
    GraphKernelNodeAttributeId::ClusterDimension => "cudaLaunchAttributeClusterDimension",
    GraphKernelNodeAttributeId::Priority => "cudaLaunchAttributePriority",
    GraphKernelNodeAttributeId::PreferredSharedMemoryCarveout => "cudaLaunchAttributePreferredSharedMemoryCarveout",
});

impl_enum_display!(GraphDependencyType, {
    GraphDependencyType::Default => "CU_GRAPH_DEPENDENCY_TYPE_DEFAULT",
    GraphDependencyType::Programmatic => "CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC",
});
