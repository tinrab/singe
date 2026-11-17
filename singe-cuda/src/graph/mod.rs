//! CUDA graph construction, instantiation, update, and execution.

pub mod contextual;
mod host_data;
pub mod memory;
pub mod owned;
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
    mem::MaybeUninit,
    ops::Deref,
    ptr,
    result::Result as StdResult,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

use bitflags::parser;
use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{impl_enum_display, report_cleanup_failure};
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
    event::Event,
    memory::{DeviceMemory, MemoryAccessDescriptor, MemoryPoolProps},
    module::{CodeOwnerId, Module},
    try_ffi,
    types::DevicePtr,
    view::{
        DeviceOutput, DeviceRange, DeviceRegion, DeviceRegionProvider, DeviceRepr, DeviceSlice,
        DeviceSliceMut,
    },
};

singe_core::native_enum! {
    /// Identifiers for [`GraphKernelNodeAttribute`] values used by CUDA graph kernel nodes.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum GraphKernelNodeAttributeId: runtime::cudaLaunchAttributeID {
        /// Identifies [`GraphKernelNodeAttribute::Cooperative`].
        Cooperative = cudaLaunchAttributeCooperative,
        /// Identifies [`GraphKernelNodeAttribute::ClusterDimension`].
        ClusterDimension = cudaLaunchAttributeClusterDimension,
        /// Identifies [`GraphKernelNodeAttribute::Priority`].
        Priority = cudaLaunchAttributePriority,
        /// Identifies [`GraphKernelNodeAttribute::PreferredSharedMemoryCarveout`].
        /// The value is a percentage in the range `0..=100` describing the preferred shared-memory carveout for the launch.
        /// This is a hint, and the driver may choose a different configuration if required.
        PreferredSharedMemoryCarveout = cudaLaunchAttributePreferredSharedMemoryCarveout,
    }
}

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
        const AUTO_FREE_ON_LAUNCH = driver::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH.0 as u64;
        /// Automatically upload the graph after instantiation.
        ///
        /// Only supported when instantiating with parameters, which name the stream the upload uses.
        const UPLOAD = driver::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD.0 as u64;
        /// Instantiate the graph to be launchable from the device.
        ///
        /// This flag can only be used on platforms which support unified addressing.
        /// This flag cannot be used in conjunction with `CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH`.
        const DEVICE_LAUNCH = driver::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH.0 as u64;
        /// Run the graph using the per-node priority attributes rather than the priority of the stream it is launched into.
        const USE_NODE_PRIORITY = driver::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY.0 as u64;
    }
}

impl Display for GraphInstantiateFlags {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        parser::to_writer(self, f)
    }
}

bitflags::bitflags! {
    /// Options that select what [`Graph::write_dot`] includes in its DOT output.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct GraphDebugDotFlags: u32 {
        /// Output all debug data as if every debug flag is enabled.
        const VERBOSE = driver::CUgraphDebugDot_flags::CU_GRAPH_DEBUG_DOT_FLAGS_VERBOSE.0;
        /// Use CUDA Runtime structures for output.
        const RUNTIME_TYPES = driver::CUgraphDebugDot_flags::CU_GRAPH_DEBUG_DOT_FLAGS_RUNTIME_TYPES.0;
        /// Adds CUDA_KERNEL_NODE_PARAMS values to output.
        const KERNEL_NODE_PARAMS = driver::CUgraphDebugDot_flags::CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_PARAMS.0;
        /// Adds CUDA_MEMCPY3D values to output.
        const MEMCPY_NODE_PARAMS = driver::CUgraphDebugDot_flags::CU_GRAPH_DEBUG_DOT_FLAGS_MEMCPY_NODE_PARAMS.0;
        /// Adds CUDA_MEMSET_NODE_PARAMS values to output.
        const MEMSET_NODE_PARAMS = driver::CUgraphDebugDot_flags::CU_GRAPH_DEBUG_DOT_FLAGS_MEMSET_NODE_PARAMS.0;
        /// Adds CUDA_HOST_NODE_PARAMS values to output.
        const HOST_NODE_PARAMS = driver::CUgraphDebugDot_flags::CU_GRAPH_DEBUG_DOT_FLAGS_HOST_NODE_PARAMS.0;
        /// Adds CUevent handle from record and wait nodes to output.
        const EVENT_NODE_PARAMS = driver::CUgraphDebugDot_flags::CU_GRAPH_DEBUG_DOT_FLAGS_EVENT_NODE_PARAMS.0;
        /// Adds CUDA_EXT_SEM_SIGNAL_NODE_PARAMS values to output.
        const EXTERNAL_SEMAPHORE_SIGNAL_NODE_PARAMS = driver::CUgraphDebugDot_flags::CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_SIGNAL_NODE_PARAMS.0;
        /// Adds CUDA_EXT_SEM_WAIT_NODE_PARAMS values to output.
        const EXTERNAL_SEMAPHORE_WAIT_NODE_PARAMS = driver::CUgraphDebugDot_flags::CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_WAIT_NODE_PARAMS.0;
        /// Adds CUkernelNodeAttrValue values to output.
        const KERNEL_NODE_ATTRIBUTES = driver::CUgraphDebugDot_flags::CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_ATTRIBUTES.0;
        /// Adds node handles and every kernel function handle to output.
        const HANDLES = driver::CUgraphDebugDot_flags::CU_GRAPH_DEBUG_DOT_FLAGS_HANDLES.0;
        /// Adds memory alloc node parameters to output.
        const MEMORY_ALLOC_NODE_PARAMS = driver::CUgraphDebugDot_flags::CU_GRAPH_DEBUG_DOT_FLAGS_MEM_ALLOC_NODE_PARAMS.0;
        /// Adds memory free node parameters to output.
        const MEMORY_FREE_NODE_PARAMS = driver::CUgraphDebugDot_flags::CU_GRAPH_DEBUG_DOT_FLAGS_MEM_FREE_NODE_PARAMS.0;
        /// Adds batch mem op node parameters to output.
        const BATCH_MEM_OP_NODE_PARAMS = driver::CUgraphDebugDot_flags::CU_GRAPH_DEBUG_DOT_FLAGS_BATCH_MEM_OP_NODE_PARAMS.0;
        /// Adds edge numbering information.
        const EXTRA_TOPOLOGY_INFO = driver::CUgraphDebugDot_flags::CU_GRAPH_DEBUG_DOT_FLAGS_EXTRA_TOPO_INFO.0;
        /// Adds conditional node parameters to output.
        const CONDITIONAL_NODE_PARAMS = driver::CUgraphDebugDot_flags::CU_GRAPH_DEBUG_DOT_FLAGS_CONDITIONAL_NODE_PARAMS.0;
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
    Kernel = driver::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL.0,
    /// Memcpy node.
    Memcpy = driver::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMCPY.0,
    /// Memset node.
    Memset = driver::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMSET.0,
    /// Host (executable) node.
    Host = driver::CUgraphNodeType::CU_GRAPH_NODE_TYPE_HOST.0,
    /// Node which executes an embedded graph.
    Graph = driver::CUgraphNodeType::CU_GRAPH_NODE_TYPE_GRAPH.0,
    /// Empty (no-op) node.
    Empty = driver::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY.0,
    /// External event wait node.
    WaitEvent = driver::CUgraphNodeType::CU_GRAPH_NODE_TYPE_WAIT_EVENT.0,
    /// External event record node.
    EventRecord = driver::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EVENT_RECORD.0,
    /// External semaphore signal node.
    ExternalSemaphoresSignal = driver::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL.0,
    /// External semaphore wait node.
    ExternalSemaphoresWait = driver::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT.0,
    /// Memory Allocation Node.
    MemoryAlloc = driver::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEM_ALLOC.0,
    /// Memory Free Node.
    MemoryFree = driver::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEM_FREE.0,
    /// Batch memory-operation node; see [`Stream::memory_operations`](crate::stream::Stream::memory_operations) for what these nodes can do.
    BatchMemOp = driver::CUgraphNodeType::CU_GRAPH_NODE_TYPE_BATCH_MEM_OP.0,
    /// Conditional Node.
    Conditional = driver::CUgraphNodeType::CU_GRAPH_NODE_TYPE_CONDITIONAL.0,
}

impl TryFrom<runtime::cudaGraphNodeType> for GraphNodeType {
    type Error = Error;

    fn try_from(value: runtime::cudaGraphNodeType) -> Result<Self> {
        Self::try_from(value.0)
            .map_err(|_| singe_core::UnknownEnumValue::new("GraphNodeType", value.0).into())
    }
}

impl From<GraphNodeType> for runtime::cudaGraphNodeType {
    fn from(value: GraphNodeType) -> Self {
        match value {
            GraphNodeType::Kernel => Self::CU_GRAPH_NODE_TYPE_KERNEL,
            GraphNodeType::Memcpy => Self::CU_GRAPH_NODE_TYPE_MEMCPY,
            GraphNodeType::Memset => Self::CU_GRAPH_NODE_TYPE_MEMSET,
            GraphNodeType::Host => Self::CU_GRAPH_NODE_TYPE_HOST,
            GraphNodeType::Graph => Self::CU_GRAPH_NODE_TYPE_GRAPH,
            GraphNodeType::Empty => Self::CU_GRAPH_NODE_TYPE_EMPTY,
            GraphNodeType::WaitEvent => Self::CU_GRAPH_NODE_TYPE_WAIT_EVENT,
            GraphNodeType::EventRecord => Self::CU_GRAPH_NODE_TYPE_EVENT_RECORD,
            GraphNodeType::ExternalSemaphoresSignal => Self::CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL,
            GraphNodeType::ExternalSemaphoresWait => Self::CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT,
            GraphNodeType::MemoryAlloc => Self::CU_GRAPH_NODE_TYPE_MEM_ALLOC,
            GraphNodeType::MemoryFree => Self::CU_GRAPH_NODE_TYPE_MEM_FREE,
            GraphNodeType::BatchMemOp => Self::CU_GRAPH_NODE_TYPE_BATCH_MEM_OP,
            GraphNodeType::Conditional => Self::CU_GRAPH_NODE_TYPE_CONDITIONAL,
        }
    }
}

singe_core::native_enum! {
    /// Dependency types applied to CUDA graph edges.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u8)]
    #[non_exhaustive]
    pub enum GraphDependencyType: driver::CUgraphDependencyType {
        /// This is an ordinary dependency.
        Default = CU_GRAPH_DEPENDENCY_TYPE_DEFAULT,
        /// This dependency type allows the downstream kernel to synchronize on the upstream grid from device code.
        ///
        /// It may only be used between kernel nodes, with the programmatic or launch-order outgoing port.
        Programmatic = CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC,
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

singe_core::native_enum! {
    /// CUDA Graph Update error types
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum GraphExecUpdateResult: driver::CUgraphExecUpdateResult {
        /// The update succeeded.
        Success = CU_GRAPH_EXEC_UPDATE_SUCCESS,
        /// The update failed for an unexpected reason which is described in the return value of the function.
        Error = CU_GRAPH_EXEC_UPDATE_ERROR,
        /// The update failed because the topology changed.
        ErrorTopologyChanged = CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED,
        /// The update failed because a node type changed.
        ErrorNodeTypeChanged = CU_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED,
        /// The update failed because the function of a kernel node changed (CUDA driver &lt; 11.2).
        ErrorFunctionChanged = CU_GRAPH_EXEC_UPDATE_ERROR_FUNCTION_CHANGED,
        /// The update failed because the parameters changed in a way that is not supported.
        ErrorParametersChanged = CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED,
        /// The update failed because something about the node is not supported.
        ErrorNotSupported = CU_GRAPH_EXEC_UPDATE_ERROR_NOT_SUPPORTED,
        /// The update failed because the function of a kernel node changed in an unsupported way.
        ErrorUnsupportedFunctionChange = CU_GRAPH_EXEC_UPDATE_ERROR_UNSUPPORTED_FUNCTION_CHANGE,
        /// The update failed because the node attributes changed in a way that is not supported.
        ErrorAttributesChanged = CU_GRAPH_EXEC_UPDATE_ERROR_ATTRIBUTES_CHANGED,
    }
}

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

/// Identifies a node of a CUDA graph.
///
/// Nodes returned by a [`Graph`] keep their source graph alive and carry its identity, so graph methods can reject nodes from another graph.
/// Queries and changes of a node go through its [`Graph`].
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
    context: Arc<Context>,
    _not_sync: PhantomData<Cell<()>>,
}

/// Process-local identity used to associate nodes with their source graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GraphId(u64);

/// Owns a CUDA graph handle that is not bound to a Singe context.
///
/// The caller must establish the CUDA context required for raw native operations.
#[derive(Debug)]
pub struct UnboundGraph {
    inner: Arc<GraphInner>,
}

/// Raw ownership of a graph, its context, and every graph-retained resource.
///
/// Keep this value alive until the native handle is destroyed, or re-adopt it with [`Graph::from_raw`].
/// Bind its context before native use and complete graph work before releasing these owners.
#[derive(Debug)]
pub struct RawGraph {
    /// Native graph handle.
    pub handle: runtime::cudaGraph_t,
    /// Context the graph belongs to.
    pub context: Arc<Context>,
    /// Allocations, modules, and events the graph's nodes depend on.
    pub retained: RetainedResources,
    /// Identity that the graph's nodes record.
    pub id: GraphId,
}

/// Raw ownership of an executable graph, its context, source graph, and resource dependencies.
///
/// Keep this value alive until native destruction, or re-adopt it with [`ExecutableGraph::from_raw`].
/// Bind its context before native use and complete all launches before releasing these owners.
#[derive(Debug)]
pub struct RawExecutableGraph {
    /// Native executable graph handle.
    pub handle: runtime::cudaGraphExec_t,
    /// Context the executable graph belongs to.
    pub context: Arc<Context>,
    /// Allocations, modules, and events the executable graph's nodes depend on.
    pub retained: RetainedResources,
    /// Graph the executable was instantiated from, if it retains one.
    pub source_graph: Option<UnboundGraph>,
    /// Identity of the source graph, which node updates check.
    pub source_graph_id: Option<GraphId>,
}

// SAFETY: the graph handle is a plain value whose every native use is `unsafe` and binds `context`, and the retained owners are `Send` (`Arc<dyn Any + Send + Sync>` and modules).
unsafe impl Send for RawGraph {}

// SAFETY: a shared `RawGraph` only exposes the handle value and `Sync` owners; it makes no native call, so concurrent use is the caller's documented obligation.
unsafe impl Sync for RawGraph {}

// SAFETY: the executable handle is a plain value whose every native use is `unsafe` and binds `context`, and the retained owners and source graph are `Send`.
unsafe impl Send for RawExecutableGraph {}

// SAFETY: a shared `RawExecutableGraph` only exposes the handle value and `Sync` owners; it makes no native call, so concurrent launches are the caller's obligation.
unsafe impl Sync for RawExecutableGraph {}

#[derive(Debug)]
pub(crate) struct GraphInner {
    handle: runtime::cudaGraph_t,
    owns_handle: bool,
    context: Option<Arc<Context>>,
    retained: Mutex<RetainedResources>,
    parent: Option<Arc<Self>>,
}

// SAFETY: CUDA graph handles may be destroyed from any host thread once the owning context is bound, which `Drop` does first.
unsafe impl Send for GraphInner {}

// SAFETY: through `&GraphInner` only the handle value and the `Mutex`-guarded retained set are reachable; native graph calls go through `Graph`, which is `!Sync` and takes `&mut self` to mutate.
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

/// Owners that a graph retains for its nodes: device allocations, modules, and events.
///
/// Keep it alive until every graph and executable graph whose nodes refer to these owners is destroyed.
#[derive(Debug, Clone, Default)]
pub struct RetainedResources {
    allocations: HashMap<usize, RetainedAllocation>,
    modules: HashMap<CodeOwnerId, Module>,
    events: HashMap<usize, Arc<Event>>,
}

impl RetainedResources {
    fn retain_event(&mut self, event: &Arc<Event>) {
        self.events
            .entry(Arc::as_ptr(event).addr())
            .or_insert_with(|| Arc::clone(event));
    }

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
        self.events.extend(other.events);
    }
}

/// Borrowed graph view tied to the graph whose child graph node embeds it.
///
/// Dropping the view does not destroy the borrowed native graph.
#[derive(Debug)]
pub struct BorrowedGraph<'graph> {
    graph: Graph,
    _graph: PhantomData<&'graph Graph>,
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

// SAFETY: the pointer and length come from the owned `DeviceMemory<T>`, which `region` also describes.
unsafe impl<T: DeviceRepr> DeviceRange<T> for GraphBuffer<T> {
    fn as_device_ptr(&self) -> *const T {
        self.memory.as_ptr()
    }

    fn len(&self) -> usize {
        self.memory.len()
    }
}

// SAFETY: a `GraphBuffer<T>` is built zeroed, copied from host values, or by `assume_init`, whose caller guarantees initialization, so its values are initialized for the borrow.
unsafe impl<T: DeviceRepr> DeviceSlice<T> for GraphBuffer<T> {}

// SAFETY: `&mut GraphBuffer<T>` is the only path to mutable access of the owned allocation.
unsafe impl<T: DeviceRepr> DeviceSliceMut<T> for GraphBuffer<T> {}

// SAFETY: the mutable pointer covers the same `len()` elements of the owned allocation, and `&mut self` makes the access exclusive.
unsafe impl<T: DeviceRepr> DeviceOutput<T> for GraphBuffer<T> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.memory.as_mut_ptr()
    }
}

// SAFETY: the length is the owned allocation's element count, and `MaybeUninit<T>` has the layout of `T`, so the range describes the same bytes as `region`.
unsafe impl<T: DeviceRepr> DeviceRange<T> for GraphBuffer<MaybeUninit<T>> {
    fn len(&self) -> usize {
        self.memory.len()
    }
}

// SAFETY: the cast pointer addresses the same owned elements with `T`'s layout, and `&mut self` makes the access exclusive.
unsafe impl<T: DeviceRepr> DeviceOutput<T> for GraphBuffer<MaybeUninit<T>> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.memory.as_mut_ptr().cast()
    }
}

impl UnboundGraph {
    /// Wraps an existing CUDA graph handle and takes ownership of it.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid CUDA graph handle.
    /// Ownership of `handle` is transferred to the returned [`UnboundGraph`], and the handle must not be destroyed elsewhere after calling this function.
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

    /// Creates an empty graph without a Singe context association.
    ///
    /// Prefer [`Context::create_graph`] for ordinary Singe code.
    /// Unbound graphs do not model context association, so the caller must keep CUDA context, node, executable update, upload, and launch relationships coherent.
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
        // SAFETY: `handle` is a local that `cudaGraphCreate` writes through an exclusive pointer.
        unsafe {
            try_ffi!(runtime::cudaGraphCreate(&raw mut handle, 0))?;
        }
        // SAFETY: CUDA just created `handle`, nothing else owns it, and `from_raw` rejects null.
        unsafe { Self::from_raw(handle) }
    }

    /// Returns the owned native graph handle without transferring ownership.
    #[must_use]
    pub fn as_raw(&self) -> runtime::cudaGraph_t {
        self.inner.handle
    }

    /// Consumes the graph and returns its native handle without destroying it.
    ///
    /// The caller becomes responsible for eventually destroying the returned handle with CUDA.
    ///
    /// # Errors
    ///
    /// Returns the unchanged graph if a node or another owner still retains its native graph storage.
    pub fn into_raw(self) -> StdResult<runtime::cudaGraph_t, Self> {
        match Arc::try_unwrap(self.inner) {
            Ok(mut inner) => {
                inner.owns_handle = false;
                Ok(inner.handle)
            }
            Err(inner) => Err(Self { inner }),
        }
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
            report_cleanup_failure("bind the cuda context to destroy", "cuda graph", &error);
            return;
        }
        // SAFETY: `owns_handle` says this value owns the live graph, every node and executable holding it has dropped its `Arc`, and its context is bound above.
        unsafe {
            if let Err(error) = try_ffi!(runtime::cudaGraphDestroy(self.handle)) {
                report_cleanup_failure("destroy", "cuda graph", &error);
            }
        }
    }
}

impl BorrowedGraph<'_> {
    /// Wraps an existing CUDA graph handle of `context` without taking ownership.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid CUDA graph handle of `context` for the returned lifetime.
    /// The returned graph view will not destroy `handle` when dropped.
    ///
    /// # Errors
    ///
    /// Returns an error if the native handle is null.
    pub unsafe fn from_raw(handle: runtime::cudaGraph_t, context: Arc<Context>) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            // SAFETY: the caller guarantees `handle` is a live graph of `context` for the view's lifetime, which is what the borrowed constructor requires.
            graph: unsafe { Graph::from_raw_borrowed_in_context(handle, context) },
            _graph: PhantomData,
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
    context: Arc<Context>,
    source_graph_id: Option<GraphId>,
    source_graph: Option<Arc<GraphInner>>,
    retained: RetainedResources,
}

/// Owns a native executable-graph handle that is not bound to a Singe context or retained resources.
#[derive(Debug)]
pub struct UnboundExecutableGraph {
    handle: runtime::cudaGraphExec_t,
}

impl Drop for UnboundExecutableGraph {
    fn drop(&mut self) {
        // SAFETY: this value owns the live executable handle, and its constructor's caller keeps the right context current for native use.
        unsafe {
            if let Err(error) = try_ffi!(runtime::cudaGraphExecDestroy(self.handle)) {
                report_cleanup_failure("destroy", "cuda executable graph", &error);
            }
        }
    }
}

// SAFETY: an executable graph may be launched and destroyed from any host thread once its context is bound, which every method and `Drop` do; it is not `Sync`, so native calls on one executable never run concurrently.
unsafe impl Send for ExecutableGraph {}

impl Drop for ExecutableGraph {
    fn drop(&mut self) {
        if let Err(error) = self.bind_context() {
            report_cleanup_failure(
                "bind the cuda context to destroy",
                "cuda executable graph",
                &error,
            );
            return;
        }
        // SAFETY: this value owns the live executable handle, its context is bound above, and the retained resources drop only after destruction.
        unsafe {
            if let Err(error) = try_ffi!(runtime::cudaGraphExecDestroy(self.handle)) {
                report_cleanup_failure("destroy", "cuda executable graph", &error);
            }
        }
    }
}

/// Failure of [`ExecutableGraph::update`].
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum GraphUpdateError {
    /// CUDA did not perform the update because it violates the constraints of executable graph updates.
    #[error("cuda rejected the executable graph update ({result})")]
    Rejected {
        /// CUDA's classification of the failure.
        result: GraphExecUpdateResult,
        /// Node of the source graph that caused the failure, when CUDA reports one.
        error_node: Option<GraphNode>,
        /// Source node of the edge that caused the failure, when CUDA reports one.
        error_from_node: Option<GraphNode>,
    },
    /// A context mismatch or another CUDA failure.
    #[error(transparent)]
    Cuda(#[from] Error),
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
            dst: value.dst().addr() as _,
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
