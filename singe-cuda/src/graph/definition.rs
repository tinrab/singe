use std::{
    ffi::CString,
    marker::PhantomData,
    mem::{MaybeUninit, take},
    ptr,
    result::Result as StdResult,
    sync::{Arc, Mutex, MutexGuard, PoisonError},
};

use singe_cuda_sys::{driver, runtime};

use super::{
    ExecutableGraph, Graph, GraphBuffer, GraphDebugDotFlags, GraphEdge, GraphEdgeAttributes,
    GraphId, GraphInner, GraphInstantiateFlags, GraphNode, GraphTopologySummary,
    MemoryAllocationNodeInfo, MemoryAllocationNodeParams, MemorySetNodeParams, RawGraphParts,
    RetainedResources,
    raw::{MemoryCopy1DNodeParams, MemoryCopy3DNodeParams},
};
use crate::{
    context::Context,
    error::{Error, Result},
    event::Event,
    graph::raw::{HostNodeParams, MemoryCopyFromSymbolNodeParams, MemoryCopyToSymbolNodeParams},
    memory::{DeviceMemory, MemoryCopyKind},
    module::{KernelLaunchArgs, LaunchConfig, Module},
    try_ffi,
    types::{DeviceFunction, DevicePtr},
    utility::to_usize,
    view::{ByteBuffer, ByteBufferMut, DeviceOutput, DeviceRepr, ZeroableDeviceRepr},
};

impl Graph {
    pub(crate) fn bind_context(&self) -> Result<()> {
        if let Some(context) = &self.context {
            context.bind()?;
        }
        Ok(())
    }

    /// Wraps an existing CUDA graph handle associated with `context` and takes ownership of it.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid CUDA graph handle associated with `context`.
    /// Ownership of `handle` is transferred to the returned [`Graph`], and the handle must not be destroyed elsewhere after calling this function.
    ///
    /// # Errors
    ///
    /// Returns an error if the native handle is null.
    pub unsafe fn from_raw_in_context(
        handle: runtime::cudaGraph_t,
        context: Arc<Context>,
    ) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            inner: Arc::new(GraphInner {
                handle,
                owns_handle: true,
                context: Some(Arc::clone(&context)),
                retained: Mutex::new(RetainedResources::default()),
                parent: None,
            }),
            id: GraphId::generate(),
            context: Some(context),
            _not_sync: PhantomData,
        })
    }

    pub(crate) unsafe fn from_raw_borrowed_in_context(
        handle: runtime::cudaGraph_t,
        context: Option<Arc<Context>>,
    ) -> Self {
        Self {
            inner: Arc::new(GraphInner {
                handle,
                owns_handle: false,
                context: context.clone(),
                retained: Mutex::new(RetainedResources::default()),
                parent: None,
            }),
            id: GraphId::generate(),
            context,
            _not_sync: PhantomData,
        }
    }

    pub(crate) fn create_in_context(context: Arc<Context>) -> Result<Self> {
        context.bind()?;
        let mut handle = ptr::null_mut();
        // SAFETY: `cudaGraphCreate` receives valid output locations.
        unsafe {
            try_ffi!(runtime::cudaGraphCreate(&raw mut handle, 0))?;
        }
        Ok(Self {
            inner: Arc::new(GraphInner {
                handle,
                owns_handle: true,
                context: Some(Arc::clone(&context)),
                retained: Mutex::new(RetainedResources::default()),
                parent: None,
            }),
            id: GraphId::generate(),
            context: Some(context),
            _not_sync: PhantomData,
        })
    }

    pub(crate) fn retained_resources(&self) -> Result<RetainedResources> {
        Ok(self.retained_lock()?.clone())
    }

    fn retained_lock(&self) -> Result<MutexGuard<'_, RetainedResources>> {
        self.inner
            .retained
            .lock()
            .map_err(|_| Error::ResourcePoisoned {
                name: "graph resources",
            })
    }

    pub(crate) fn retain_module(&self, module: &Module) -> Result<()> {
        self.retained_lock()?.retain_module(module);
        Ok(())
    }

    fn retain_buffer<T>(&self, buffer: &GraphBuffer<T>) -> Result<()>
    where
        T: DeviceRepr,
    {
        self.retained_lock()?.retain_allocation(buffer.retained());
        Ok(())
    }

    fn check_buffer_context<T>(&self, buffer: &GraphBuffer<T>) -> Result<()>
    where
        T: DeviceRepr,
    {
        if let Some(graph_context) = &self.context
            && graph_context != buffer.context()
        {
            return Err(Error::GraphContextMismatch);
        }
        Ok(())
    }

    fn check_buffer_contexts<T, D>(&self, dst: &GraphBuffer<D>, src: &GraphBuffer<T>) -> Result<()>
    where
        T: DeviceRepr,
        D: DeviceRepr,
    {
        self.check_buffer_context(dst)?;
        self.check_buffer_context(src)?;
        Ok(())
    }

    /// Allocates graph-retained device memory.
    ///
    /// The returned buffer can be used with graph-buffer node APIs.
    /// Any graph or executable graph that records the buffer retains the underlying device allocation for replay.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot allocate device memory, the requested byte count overflows, or CUDA reports runtime initialization diagnostics.
    pub fn create_buffer_uninit<T>(&mut self, length: usize) -> Result<GraphBuffer<MaybeUninit<T>>>
    where
        T: DeviceRepr,
    {
        self.bind_context()?;
        let context = self.context.as_ref().ok_or(Error::GraphContextMismatch)?;
        let buffer = GraphBuffer::from_memory(DeviceMemory::<T>::create_uninit(context, length)?);
        self.retain_buffer(&buffer)?;
        Ok(buffer)
    }

    /// Allocates graph storage that raw graph kernels initialize before reading.
    ///
    /// # Safety
    ///
    /// Every element must be initialized to a valid `T` before any graph or host consumer reads it.
    ///
    /// # Errors
    ///
    /// Returns an error if the graph has no associated context, the byte count overflows, a nonempty zero-sized allocation is requested, context binding fails, or CUDA cannot allocate the buffer.
    pub unsafe fn create_buffer_unchecked<T>(&mut self, length: usize) -> Result<GraphBuffer<T>>
    where
        T: DeviceRepr,
    {
        let context = self.context.as_ref().ok_or(Error::GraphContextMismatch)?;
        let buffer =
            // SAFETY: the caller upholds this function's safety contract.
            GraphBuffer::from_memory(unsafe { DeviceMemory::create_unchecked(context, length)? });
        self.retain_buffer(&buffer)?;
        Ok(buffer)
    }

    /// Allocates graph-retained device memory initialized to zero bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot allocate or initialize device memory, the requested byte count overflows, or CUDA reports runtime initialization diagnostics.
    pub fn zeroed_buffer<T>(&mut self, length: usize) -> Result<GraphBuffer<T>>
    where
        T: ZeroableDeviceRepr,
    {
        self.bind_context()?;
        let context = self.context.as_ref().ok_or(Error::GraphContextMismatch)?;
        let buffer = GraphBuffer::from_memory(DeviceMemory::zeroed(context, length)?);
        self.retain_buffer(&buffer)?;
        Ok(buffer)
    }

    /// Allocates graph-retained device memory initialized from a host slice.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot allocate or copy device memory, the requested byte count overflows, or CUDA reports runtime initialization diagnostics.
    pub fn buffer_from_slice<T>(&mut self, values: &[T]) -> Result<GraphBuffer<T>>
    where
        T: DeviceRepr,
    {
        self.bind_context()?;
        let context = self.context.as_ref().ok_or(Error::GraphContextMismatch)?;
        let buffer = GraphBuffer::from_memory(DeviceMemory::from_slice(context, values)?);
        self.retain_buffer(&buffer)?;
        Ok(buffer)
    }

    /// Validates and instantiates this graph with the default instantiation flags.
    ///
    /// # Errors
    ///
    /// Propagates context-binding and CUDA instantiation errors, including invalid graph structure, unsupported node parameters, or native restrictions on simultaneous executable instances.
    /// See [`Graph::instantiate_with_flags`].
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
    /// * [`GraphInstantiateFlags::AUTO_FREE_ON_LAUNCH`], which configures a graph containing memory allocation nodes to automatically free any unfreed memory allocations before the graph is relaunched.
    ///
    /// * [`GraphInstantiateFlags::DEVICE_LAUNCH`], which configures the graph for launch from the device.
    ///   If this flag is passed, the executable graph handle returned can be used to launch the graph from both the host and device.
    ///   This flag can only be used on platforms which support unified addressing.
    ///   This flag cannot be used in conjunction with [`GraphInstantiateFlags::AUTO_FREE_ON_LAUNCH`].
    ///
    /// * [`GraphInstantiateFlags::USE_NODE_PRIORITY`], which causes the graph to use the priorities from the per-node attributes rather than the priority of the launch stream during execution.
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
    ///   Operation-specific restrictions are outlined below.
    /// * Kernel nodes:
    ///   + Use of CUDA Dynamic Parallelism is not permitted.
    ///   + Cooperative launches are permitted as long as MPS is not in use.
    /// * Memcpy nodes:
    ///   + Only copies involving device memory and/or pinned device-mapped host memory are permitted.
    ///   + Copies involving CUDA arrays are not permitted.
    ///   + Both operands must be accessible from the current device, and the current device must match the device of other nodes in the graph.
    ///
    /// Graph objects are not threadsafe.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the graph operation, if a previous asynchronous launch reported an error, or if CUDA reports runtime initialization diagnostics.
    /// Callbacks must not call CUDA functions; see [`Stream::add_callback`].
    pub fn instantiate_with_flags(&self, flags: GraphInstantiateFlags) -> Result<ExecutableGraph> {
        self.bind_context()?;
        let mut handle = ptr::null_mut();
        // SAFETY: `cudaGraphInstantiateWithFlags` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(runtime::cudaGraphInstantiateWithFlags(
                &raw mut handle,
                self.as_raw(),
                flags.bits(),
            ))?;
        }
        // SAFETY: CUDA instantiated this executable from the retained graph, and nothing else owns it.
        unsafe {
            ExecutableGraph::from_raw_with_graph(
                handle,
                self.context.clone(),
                Some(self.id),
                Some(Arc::clone(&self.inner)),
                self.retained_resources()?,
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
    /// Returns an error if CUDA rejects the graph operation, if a previous asynchronous launch reported an error, or if CUDA reports runtime initialization diagnostics.
    /// Callbacks must not call CUDA functions; see [`Stream::add_callback`].
    pub fn try_clone(&self) -> Result<Self> {
        self.bind_context()?;
        let mut handle = ptr::null_mut();
        // SAFETY: `cudaGraphClone` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(runtime::cudaGraphClone(&raw mut handle, self.as_raw()))?;
        }
        Ok(Self {
            inner: Arc::new(GraphInner {
                handle,
                owns_handle: true,
                context: self.context.clone(),
                retained: Mutex::new(self.retained_resources()?),
                parent: None,
            }),
            id: GraphId::generate(),
            context: self.context.clone(),
            _not_sync: PhantomData,
        })
    }

    pub(crate) fn node_from_raw(&self, handle: runtime::cudaGraphNode_t) -> GraphNode {
        GraphNode::from_raw_in_graph(
            handle,
            self.id,
            Arc::clone(&self.inner),
            self.context.clone(),
        )
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

    fn check_child_graph_context(&self, child_graph: &Self) -> Result<()> {
        if let (Some(parent_context), Some(child_context)) = (&self.context, &child_graph.context)
            && parent_context.as_ref() != child_context.as_ref()
        {
            return Err(Error::GraphContextMismatch);
        }
        Ok(())
    }

    fn check_event_record_context(&self, event: &Event) -> Result<()> {
        if let Some(context) = &self.context
            && context != event.context()
        {
            return Err(Error::GraphContextMismatch);
        }
        Ok(())
    }

    /// Adds a default dependency from `from` to `to`.
    ///
    /// # Errors
    ///
    /// Returns an error if any node belongs to another graph or context binding fails.
    /// Also propagates CUDA dependency-update errors.
    pub fn add_dependency(&mut self, from: GraphNode, to: GraphNode) -> Result<()> {
        self.add_dependencies(&[from], &[to])
    }

    /// Adds default dependencies between corresponding entries in `from` and `to`.
    ///
    /// # Errors
    ///
    /// Returns an error if any node belongs to another graph or context binding fails.
    /// Also propagates CUDA dependency-update errors.
    /// Source and destination slices must have equal lengths.
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
    /// Returns an error if CUDA rejects the graph operation, if a previous asynchronous launch reported an error, or if CUDA reports runtime initialization diagnostics.
    /// Callbacks must not call CUDA functions; see [`Stream::add_callback`].
    pub fn add_dependencies_with_data(
        &mut self,
        from: &[GraphNode],
        to: &[GraphNode],
        edge_data: &[GraphEdgeAttributes],
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
        // SAFETY: `cudaGraphAddDependencies` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
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

    /// Removes the default dependency from `from` to `to`.
    ///
    /// # Errors
    ///
    /// Returns an error if any node belongs to another graph or context binding fails.
    /// Also propagates CUDA dependency-update errors.
    pub fn remove_dependency(&mut self, from: GraphNode, to: GraphNode) -> Result<()> {
        self.remove_dependencies(&[from], &[to])
    }

    /// Removes default dependencies between corresponding entries in `from` and `to`.
    ///
    /// # Errors
    ///
    /// Returns an error if any node belongs to another graph or context binding fails.
    /// Also propagates CUDA dependency-update errors.
    /// Source and destination slices must have equal lengths.
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
    /// Returns an error if CUDA rejects the graph operation, if a previous asynchronous launch reported an error, or if CUDA reports runtime initialization diagnostics.
    /// Callbacks must not call CUDA functions; see [`Stream::add_callback`].
    pub fn remove_dependencies_with_data(
        &mut self,
        from: &[GraphNode],
        to: &[GraphNode],
        edge_data: &[GraphEdgeAttributes],
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
        // SAFETY: `cudaGraphRemoveDependencies` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
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

    /// Adds directed edges with explicit port and dependency attributes.
    ///
    /// # Errors
    ///
    /// Returns an error if any node belongs to another graph or context binding fails.
    /// Also propagates CUDA dependency-update errors.
    pub fn add_edges(&mut self, edges: &[GraphEdge]) -> Result<()> {
        if edges.is_empty() {
            return Ok(());
        }

        let from: Vec<_> = edges.iter().map(|edge| edge.from.clone()).collect();
        let to: Vec<_> = edges.iter().map(|edge| edge.to.clone()).collect();
        let data: Vec<_> = edges.iter().map(|edge| edge.data).collect();
        self.add_dependencies_with_data(&from, &to, &data)
    }

    /// Removes directed edges matching the supplied nodes and attributes.
    ///
    /// # Errors
    ///
    /// Returns an error if any node belongs to another graph or context binding fails.
    /// Also propagates CUDA dependency-update errors.
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
    /// The dependency list may be empty, in which case the node is placed at the graph root.
    /// It may not contain duplicate entries.
    ///
    /// An empty node performs no operation during execution, but can be used for transitive ordering.
    /// For example, a phased execution graph with 2 groups of n nodes with a barrier between them can be represented using an empty node and 2\*n dependency edges, rather than no empty node and n^2 dependency edges.
    ///
    /// Graph objects are not threadsafe.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the graph operation or reports runtime initialization diagnostics.
    /// Callbacks must not call CUDA functions; see [`Stream::add_callback`].
    pub fn add_empty_node(&mut self, dependencies: &[GraphNode]) -> Result<GraphNode> {
        self.check_nodes(dependencies)?;
        let mut handle = ptr::null_mut();
        let dependencies_raw: Vec<_> = dependencies.iter().map(GraphNode::as_raw).collect();
        // SAFETY: the graph and dependency handles are live, and the dependency array length matches.
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
    /// The dependency list may be empty, in which case the node is placed at the graph root.
    /// It may not contain duplicate entries.
    ///
    /// Each graph launch records `event` to capture execution of the node's dependencies.
    ///
    /// These nodes may not be used in loops or conditionals.
    ///
    /// Graph objects are not threadsafe.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the graph operation, if a previous asynchronous launch reported an error, or if CUDA reports runtime initialization diagnostics.
    /// Callbacks must not call CUDA functions; see [`Stream::add_callback`].
    pub fn add_event_record_node(
        &mut self,
        dependencies: &[GraphNode],
        event: &Event,
    ) -> Result<GraphNode> {
        self.check_nodes(dependencies)?;
        self.check_event_record_context(event)?;
        let mut handle = ptr::null_mut();
        let dependencies_raw: Vec<_> = dependencies.iter().map(GraphNode::as_raw).collect();
        // SAFETY: the graph and dependency handles are live, and the dependency array length matches.
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
    /// The dependency list may be empty, in which case the node is placed at the graph root.
    /// It may not contain duplicate entries.
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
    /// Returns an error if CUDA rejects the graph operation, if a previous asynchronous launch reported an error, or if CUDA reports runtime initialization diagnostics.
    /// Callbacks must not call CUDA functions; see [`Stream::add_callback`].
    pub fn add_event_wait_node(
        &mut self,
        dependencies: &[GraphNode],
        event: &Event,
    ) -> Result<GraphNode> {
        self.check_nodes(dependencies)?;
        let mut handle = ptr::null_mut();
        let dependencies_raw: Vec<_> = dependencies.iter().map(GraphNode::as_raw).collect();
        // SAFETY: the graph and dependency handles are live, and the dependency array length matches.
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
    /// The dependency list may be empty, in which case the node is placed at the graph root.
    /// It may not contain duplicate entries.
    ///
    /// When the graph is launched, the node invokes the specified CPU function.
    /// Host nodes are not supported under MPS with pre-Volta GPUs.
    ///
    /// Graph objects are not threadsafe.
    ///
    /// # Safety
    ///
    /// CUDA stores the raw callback function and user-data pointer in the graph node for later replay.
    /// The caller must ensure `params` remains valid according to [`HostNodeParams::new`] for every graph instantiation and launch that can execute this node.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the graph operation, if a previous asynchronous launch reported an error, or if CUDA reports runtime initialization diagnostics.
    /// Callbacks must not call CUDA functions; see [`Stream::add_callback`].
    pub unsafe fn add_host_node(
        &mut self,
        dependencies: &[GraphNode],
        params: &HostNodeParams,
    ) -> Result<GraphNode> {
        self.check_nodes(dependencies)?;
        let mut handle = ptr::null_mut();
        let dependencies_raw: Vec<_> = dependencies.iter().map(GraphNode::as_raw).collect();
        let params = params.into();
        // SAFETY: the caller upholds this function's safety contract.
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
    /// The dependency list may be empty, in which case the node is placed at the graph root.
    /// It may not contain duplicate entries.
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
    /// Arguments must match `function`'s ABI and satisfy its value and access requirements.
    /// The function must belong to the node's execution context.
    /// Keep its code owner alive until every queued graph execution completes.
    /// CUDA copies the kernel argument values during this call and stores those copied values in the graph node for later replay.
    /// If an argument value is itself a pointer, only the pointer address is copied.
    /// The caller must ensure every copied pointer value remains valid for every graph instantiation, update, and launch that can execute this node.
    /// Mutable pointer arguments must also remain exclusive for the work ordered by those launches.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the graph operation, if a previous asynchronous launch reported an error, or if CUDA reports runtime initialization diagnostics.
    /// Callbacks must not call CUDA functions; see [`Stream::add_callback`].
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
        // SAFETY: the caller upholds this function's safety contract.
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
            let node = self.node_from_raw(handle);
            for attribute in config.attributes() {
                if let Err(error) = node.set_launch_attribute(*attribute) {
                    let _ = driver::cuGraphDestroyNode(handle);
                    return Err(error);
                }
            }
            Ok(node)
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
    /// CUDA stores the raw source and destination addresses in the graph node for later replay.
    /// The caller must ensure `params` remains valid according to [`MemoryCopy1DNodeParams::new`] for every graph instantiation and launch that can execute this node.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the graph operation, if a previous asynchronous launch reported an error, or if CUDA reports runtime initialization diagnostics.
    /// Callbacks must not call CUDA functions; see [`Stream::add_callback`].
    pub unsafe fn add_memory_copy_node_1d(
        &mut self,
        dependencies: &[GraphNode],
        params: &MemoryCopy1DNodeParams,
    ) -> Result<GraphNode> {
        self.check_nodes(dependencies)?;
        let mut handle = ptr::null_mut();
        let dependencies_raw: Vec<_> = dependencies.iter().map(GraphNode::as_raw).collect();
        // SAFETY: the caller upholds this function's safety contract.
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
    /// The node copies `src.byte_len()` bytes.
    /// `dst` must have at least that many bytes.
    ///
    /// # Safety
    ///
    /// CUDA stores the raw source and destination addresses in the graph node for later replay.
    /// The caller must ensure `dst` and `src` remain valid for every graph instantiation and launch that can execute this node.
    /// `dst` must not be accessed through another mutable path while graph launches using this node can write it.
    ///
    /// # Errors
    ///
    /// Returns an error if `dst` is smaller than `src`, if CUDA rejects the graph operation, if a previous asynchronous launch reported an error, or if CUDA reports runtime initialization diagnostics.
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

        // SAFETY: the caller upholds this function's safety contract.
        let params = unsafe {
            MemoryCopy1DNodeParams::new(
                dst.as_byte_mut_ptr().cast(),
                src.as_byte_ptr().cast(),
                count,
                MemoryCopyKind::DeviceToDevice,
            )
        };
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { self.add_memory_copy_node_1d(dependencies, &params) }
    }

    /// Creates a device-to-device memcpy node between graph-retained buffers.
    ///
    /// The node copies `src.byte_len()` bytes.
    /// `dst` must have at least that many bytes.
    /// The graph retains both allocations so the baked CUDA graph pointers remain live for future instantiation and replay.
    ///
    /// # Errors
    ///
    /// Returns an error if `dst` is smaller than `src`, if CUDA rejects the graph operation, if a previous asynchronous launch reported an error, or if CUDA reports runtime initialization diagnostics.
    ///
    /// # Safety
    /// Keep source elements valid for every launch, and order their producers before replay.
    /// Keep destination accesses exclusive through completion, including failed submissions.
    /// Allocation retention preserves addresses; initialization and access ordering remain caller obligations.
    /// Serialize host updates with other operations on this graph executable.
    pub unsafe fn add_buffer_memory_copy_node_1d_device_to_device<T, D>(
        &mut self,
        dependencies: &[GraphNode],
        dst: &mut GraphBuffer<D>,
        src: &GraphBuffer<T>,
    ) -> Result<GraphNode>
    where
        T: DeviceRepr,
        D: DeviceRepr,
        GraphBuffer<D>: DeviceOutput<T>,
    {
        self.check_buffer_contexts(dst, src)?;
        let count = src.byte_len();
        if dst.byte_len() < count {
            return Err(Error::InvalidMemoryAccess);
        }

        // SAFETY: the caller upholds this function's safety contract.
        let params = unsafe {
            MemoryCopy1DNodeParams::new(
                dst.as_mut_ptr().cast(),
                src.as_ptr().cast(),
                count,
                MemoryCopyKind::DeviceToDevice,
            )
        };
        // SAFETY: the caller upholds this function's safety contract.
        let node = unsafe { self.add_memory_copy_node_1d(dependencies, &params)? };
        self.retain_buffer(dst)?;
        self.retain_buffer(src)?;
        Ok(node)
    }

    /// Creates a memcpy node and adds it to the graph with the given dependencies.
    /// The dependency list may be empty, in which case the node is placed at the graph root.
    /// It may not contain duplicate entries.
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
    /// CUDA stores the raw source and destination addresses in the graph node for later replay.
    /// The caller must ensure `params` remains valid according to [`MemoryCopy3DNodeParams`] for every graph instantiation and launch that can execute this node.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the graph operation, if a previous asynchronous launch reported an error, or if CUDA reports runtime initialization diagnostics.
    /// Callbacks must not call CUDA functions; see [`Stream::add_callback`].
    pub unsafe fn add_memory_copy_node(
        &mut self,
        dependencies: &[GraphNode],
        params: &MemoryCopy3DNodeParams,
    ) -> Result<GraphNode> {
        self.check_nodes(dependencies)?;
        let mut handle = ptr::null_mut();
        let dependencies_raw: Vec<_> = dependencies.iter().map(GraphNode::as_raw).collect();
        let params = params.into();
        // SAFETY: the caller upholds this function's safety contract.
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
    /// CUDA stores the raw symbol and source pointer in the graph node for later replay.
    /// The caller must ensure `params` remains valid according to [`MemoryCopyToSymbolNodeParams::new`] for every graph instantiation and launch that can execute this node.
    ///
    /// # Errors
    ///
    /// Returns an error if a dependency belongs to another graph, context binding fails, or CUDA rejects the symbol, pointers, copy range, or node dependencies.
    pub unsafe fn add_memory_copy_node_to_symbol(
        &mut self,
        dependencies: &[GraphNode],
        params: &MemoryCopyToSymbolNodeParams,
    ) -> Result<GraphNode> {
        self.check_nodes(dependencies)?;
        let mut handle = ptr::null_mut();
        let dependencies_raw: Vec<_> = dependencies.iter().map(GraphNode::as_raw).collect();
        // SAFETY: the caller upholds this function's safety contract.
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
    /// CUDA stores the raw destination and symbol pointer in the graph node for later replay.
    /// The caller must ensure `params` remains valid according to [`MemoryCopyFromSymbolNodeParams::new`] for every graph instantiation and launch that can execute this node.
    ///
    /// # Errors
    ///
    /// Returns an error if a dependency belongs to another graph, context binding fails, or CUDA rejects the symbol, pointers, copy range, or node dependencies.
    pub unsafe fn add_memory_copy_node_from_symbol(
        &mut self,
        dependencies: &[GraphNode],
        params: &MemoryCopyFromSymbolNodeParams,
    ) -> Result<GraphNode> {
        self.check_nodes(dependencies)?;
        let mut handle = ptr::null_mut();
        let dependencies_raw: Vec<_> = dependencies.iter().map(GraphNode::as_raw).collect();
        // SAFETY: the caller upholds this function's safety contract.
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
    /// The caller must ensure `params` remains valid according to [`MemorySetNodeParams::new`] for every graph instantiation and launch that can execute this node.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the graph operation, if a previous asynchronous launch reported an error, or if CUDA reports runtime initialization diagnostics.
    /// Callbacks must not call CUDA functions; see [`Stream::add_callback`].
    pub unsafe fn add_memory_set_node(
        &mut self,
        dependencies: &[GraphNode],
        params: &MemorySetNodeParams,
    ) -> Result<GraphNode> {
        self.check_nodes(dependencies)?;
        let mut handle = ptr::null_mut();
        let dependencies_raw: Vec<_> = dependencies.iter().map(GraphNode::as_raw).collect();
        let params = params.into();
        // SAFETY: the caller upholds this function's safety contract.
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
    /// Returns an error if CUDA rejects the graph operation, if a previous asynchronous launch reported an error, or if CUDA reports runtime initialization diagnostics.
    /// Callbacks must not call CUDA functions; see [`Stream::add_callback`].
    pub fn add_child_graph_node(
        &mut self,
        dependencies: &[GraphNode],
        child_graph: &Self,
    ) -> Result<GraphNode> {
        self.check_nodes(dependencies)?;
        self.check_child_graph_context(child_graph)?;
        let mut handle = ptr::null_mut();
        let dependencies_raw: Vec<_> = dependencies.iter().map(GraphNode::as_raw).collect();
        // SAFETY: the graph and dependency handles are live, and the dependency array length matches.
        unsafe {
            try_ffi!(runtime::cudaGraphAddChildGraphNode(
                &raw mut handle,
                self.as_raw(),
                dependencies_raw.as_ptr(),
                dependencies_raw.len() as _,
                child_graph.as_raw(),
            ))?;
            self.retained_lock()?
                .extend(child_graph.retained_resources()?);
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
    /// Returns [`Error::GraphNodeMismatch`] if `allocation` did not come from this graph.
    /// Returns an error if CUDA rejects the graph operation or if a previous asynchronous launch reported an error.
    pub fn add_memory_free_node(
        &mut self,
        dependencies: &[GraphNode],
        allocation: &MemoryAllocationNodeInfo,
    ) -> Result<GraphNode> {
        if allocation.graph_id != Some(self.id) {
            return Err(Error::GraphNodeMismatch);
        }
        // SAFETY: `allocation` is a graph allocation consumed here, so it is freed exactly once.
        unsafe { self.add_memory_free_node_raw(dependencies, allocation.ptr) }
    }

    /// Creates a new memory free node from a raw device address.
    ///
    /// # Safety
    ///
    /// CUDA stores the raw address in the graph.
    /// The caller must ensure `ptr` is a graph allocation that may be freed by this graph, is ordered after the allocation node, and is not freed more than once or by another graph in a way that violates CUDA graph allocation ownership rules.
    ///
    /// # Errors
    ///
    /// Returns an error if a dependency belongs to another graph, context binding fails, or CUDA rejects the free node or its allocation pointer.
    pub unsafe fn add_memory_free_node_raw(
        &mut self,
        dependencies: &[GraphNode],
        ptr: DevicePtr,
    ) -> Result<GraphNode> {
        self.check_nodes(dependencies)?;
        let mut handle = ptr::null_mut();
        let dependencies_raw: Vec<_> = dependencies.iter().map(GraphNode::as_raw).collect();
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            try_ffi!(runtime::cudaGraphAddMemFreeNode(
                &raw mut handle,
                self.as_raw(),
                dependencies_raw.as_ptr(),
                dependencies_raw.len() as _,
                ptr.as_ptr().cast(),
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
    /// Returns an error if CUDA rejects the graph operation or if a previous asynchronous launch reported an error.
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
        // SAFETY: the graph and dependency handles are live, and the dependency array length matches.
        unsafe {
            try_ffi!(runtime::cudaGraphAddMemAllocNode(
                &raw mut handle,
                self.as_raw(),
                dependencies_raw.as_ptr(),
                dependencies_raw.len() as _,
                &raw mut params_raw,
            ))?;
            if params_raw.dptr == 0 {
                return Err(Error::NullHandle);
            }

            let node = self.node_from_raw(handle);
            let allocation = MemoryAllocationNodeInfo::from_raw_in_graph(
                DevicePtr::new(params_raw.dptr as *mut ()),
                params.byte_size,
                self.id,
                Arc::clone(&self.inner),
                self.context.clone(),
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
    /// Returns an error if CUDA rejects the graph operation, if a previous asynchronous launch reported an error, or if CUDA reports runtime initialization diagnostics.
    /// Callbacks must not call CUDA functions; see [`Stream::add_callback`].
    pub fn nodes(&self) -> Result<Vec<GraphNode>> {
        // SAFETY: `cudaGraphGetNodes` first reports the count, then fills arrays sized to it.
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

            let mut handles = Vec::with_capacity(to_usize(count, "graph node count")?);
            try_ffi!(runtime::cudaGraphGetNodes(
                self.as_raw(),
                handles.as_mut_ptr(),
                &raw mut count,
            ))?;
            handles.set_len(to_usize(count, "graph node count")?);

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
    /// Returns an error if CUDA rejects the graph operation, if a previous asynchronous launch reported an error, or if CUDA reports runtime initialization diagnostics.
    /// Callbacks must not call CUDA functions; see [`Stream::add_callback`].
    pub fn root_nodes(&self) -> Result<Vec<GraphNode>> {
        // SAFETY: `cudaGraphGetRootNodes` first reports the count, then fills arrays sized to it.
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

            let mut handles = Vec::with_capacity(to_usize(count, "graph node count")?);
            try_ffi!(runtime::cudaGraphGetRootNodes(
                self.as_raw(),
                handles.as_mut_ptr(),
                &raw mut count,
            ))?;
            handles.set_len(to_usize(count, "graph node count")?);

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
    /// Returns an error if CUDA rejects the graph operation, if a previous asynchronous launch reported an error, or if CUDA reports runtime initialization diagnostics.
    /// Callbacks must not call CUDA functions; see [`Stream::add_callback`].
    pub fn edges(&self) -> Result<Vec<GraphEdge>> {
        // SAFETY: `cudaGraphGetEdges` first reports the count, then fills arrays sized to it.
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

            let len = to_usize(count, "graph node count")?;
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
            let len = to_usize(count, "graph node count")?;
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
    /// The summary is computed from CUDA graph introspection APIs and counts node kinds, root nodes, and dependency edges in this graph.
    /// Child graph nodes are counted as child nodes here; callers that need recursive details can query the child graph returned by [`GraphNode::child_graph`].
    ///
    /// Graph objects are not threadsafe.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects a topology query, if a previous asynchronous launch reported an error, or if CUDA reports runtime initialization diagnostics.
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
    /// Returns an error if `path` contains an interior NUL byte or if CUDA Runtime cannot write the DOT file.
    pub fn write_dot(&self, path: &str, flags: GraphDebugDotFlags) -> Result<()> {
        let path = CString::new(path)?;
        // SAFETY: `cudaGraphDebugDotPrint` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(runtime::cudaGraphDebugDotPrint(
                self.as_raw(),
                path.as_ptr(),
                flags.bits(),
            ))?;
        }
        Ok(())
    }

    /// Returns the native graph handle without transferring ownership.
    #[must_use]
    pub fn as_raw(&self) -> runtime::cudaGraph_t {
        self.inner.handle
    }

    /// Returns the retained graph context, when present.
    #[must_use]
    pub fn context(&self) -> Option<&Arc<Context>> {
        self.context.as_ref()
    }

    /// Transfers the native graph, context, and retained resources without destruction.
    ///
    /// Returns the unchanged graph if a node or executable still shares its native owner.
    /// Keep the returned parts alive through completed native work and handle destruction, or re-adopt them.
    ///
    /// # Errors
    ///
    /// Returns the original graph if a node, executable, child view, or another owner still retains its native graph storage.
    pub fn into_raw(self) -> StdResult<RawGraphParts, Self> {
        let Self {
            inner,
            id,
            context,
            _not_sync: _,
        } = self;
        match Arc::try_unwrap(inner) {
            Ok(mut inner) => {
                inner.owns_handle = false;
                Ok(RawGraphParts {
                    handle: inner.handle,
                    context,
                    // The caller takes over keeping these alive through native work, whatever a panic left here.
                    retained: take(
                        inner
                            .retained
                            .get_mut()
                            .unwrap_or_else(PoisonError::into_inner),
                    ),
                    id,
                })
            }
            Err(inner) => Err(Self {
                inner,
                id,
                context,
                _not_sync: PhantomData,
            }),
        }
    }

    /// Re-adopts a native graph together with its original context and resource owners.
    ///
    /// # Safety
    /// The native handle must still be live and unchanged, with no other destroying owner or unretained dependencies.
    ///
    /// # Errors
    ///
    /// Returns an error if the native handle is null.
    pub unsafe fn from_raw_parts(parts: RawGraphParts) -> Result<Self> {
        if parts.handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            inner: Arc::new(GraphInner {
                handle: parts.handle,
                owns_handle: true,
                context: parts.context.clone(),
                retained: Mutex::new(parts.retained),
                parent: None,
            }),
            id: parts.id,
            context: parts.context,
            _not_sync: PhantomData,
        })
    }
}
