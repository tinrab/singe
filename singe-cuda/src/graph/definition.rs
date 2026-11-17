use std::{
    ffi::CString,
    marker::PhantomData,
    mem::{MaybeUninit, take},
    ptr,
    result::Result as StdResult,
    sync::{Arc, Mutex, MutexGuard, PoisonError},
};

use singe_core::report_cleanup_failure;
use singe_cuda_sys::{driver, runtime};

use crate::{
    context::Context,
    error::{Error, Result},
    event::Event,
    graph::{
        ExecutableGraph, Graph, GraphBuffer, GraphDebugDotFlags, GraphEdge, GraphEdgeAttributes,
        GraphId, GraphInner, GraphInstantiateFlags, GraphNode, GraphTopologySummary,
        MemoryAllocationNodeInfo, MemoryAllocationNodeParams, MemorySetNodeParams, RawGraph,
        RetainedResources,
        raw::{
            HostNodeParams, MemoryCopy1DNodeParams, MemoryCopy3DNodeParams,
            MemoryCopyFromSymbolNodeParams, MemoryCopyToSymbolNodeParams,
        },
    },
    memory::{DeviceMemory, MemoryCopyKind},
    module::{KernelLaunchArgs, LaunchConfig, Module},
    try_ffi,
    types::{DeviceFunction, DevicePtr},
    utility::to_usize,
    view::{DeviceOutput, DeviceRepr, DeviceSlice, DeviceSliceMut, ZeroableDeviceRepr},
};

impl Graph {
    pub(crate) fn bind_context(&self) -> Result<()> {
        self.context.bind()
    }

    // The caller transfers ownership of a live graph handle of `context`.
    pub(crate) unsafe fn from_raw_in_context(
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
            context,
            _not_sync: PhantomData,
        })
    }

    pub(crate) unsafe fn from_raw_borrowed_in_context(
        handle: runtime::cudaGraph_t,
        context: Arc<Context>,
    ) -> Self {
        Self {
            inner: Arc::new(GraphInner {
                handle,
                owns_handle: false,
                context: Some(Arc::clone(&context)),
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
        // SAFETY: `handle` is a local that `cudaGraphCreate` writes through an exclusive pointer, and `context` is bound above.
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
            context,
            _not_sync: PhantomData,
        })
    }

    pub(crate) fn retained_resources(&self) -> Result<RetainedResources> {
        Ok(self.retained_lock()?.clone())
    }

    pub(crate) fn retained_lock(&self) -> Result<MutexGuard<'_, RetainedResources>> {
        self.inner
            .retained
            .lock()
            .map_err(|_| Error::lock_poisoned("graph resources"))
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
        if &self.context != buffer.context() {
            return Err(Error::MemoryContextMismatch);
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
        let context = &self.context;
        let buffer = GraphBuffer::from_memory(DeviceMemory::<T>::create_uninit(context, length)?);
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
        let context = &self.context;
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
        let context = &self.context;
        let buffer = GraphBuffer::from_memory(DeviceMemory::from_slice(context, values)?);
        self.retain_buffer(&buffer)?;
        Ok(buffer)
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
    /// If the graph contains kernels which call device-side [`ExecutableGraph::launch_async`] from multiple devices, this results in an error.
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
    /// Returns an error if the context cannot be bound or CUDA rejects instantiation, including for invalid graph structure, unsupported node parameters or flags, or native restrictions on simultaneous executable instances.
    /// CUDA may also report errors from previous asynchronous launches or runtime initialization diagnostics.
    pub fn instantiate(&self, flags: GraphInstantiateFlags) -> Result<ExecutableGraph> {
        self.bind_context()?;
        let mut handle = ptr::null_mut();
        // SAFETY: `self.inner` owns the live graph, `Graph` is not `Sync` so no other thread uses it, `handle` is a local output, and the context is bound above.
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
                Arc::clone(&self.context),
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
    /// Callbacks must not call CUDA functions; see [`Stream::add_callback`](crate::stream::Stream::add_callback).
    pub fn try_clone(&self) -> Result<Self> {
        self.bind_context()?;
        let mut handle = ptr::null_mut();
        // SAFETY: `self.inner` owns the live graph, `handle` is a local output, and the context is bound above.
        unsafe {
            try_ffi!(runtime::cudaGraphClone(&raw mut handle, self.as_raw()))?;
        }
        Ok(Self {
            inner: Arc::new(GraphInner {
                handle,
                owns_handle: true,
                context: Some(Arc::clone(&self.context)),
                retained: Mutex::new(self.retained_resources()?),
                parent: None,
            }),
            id: GraphId::generate(),
            context: Arc::clone(&self.context),
            _not_sync: PhantomData,
        })
    }

    pub(crate) fn node_from_raw(&self, handle: runtime::cudaGraphNode_t) -> GraphNode {
        GraphNode::from_raw_in_graph(
            handle,
            self.id,
            Arc::clone(&self.inner),
            Some(Arc::clone(&self.context)),
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
        if self.context.as_ref() != child_graph.context.as_ref() {
            return Err(Error::context_mismatch("child graph"));
        }
        Ok(())
    }

    fn check_event_record_context(&self, event: &Event) -> Result<()> {
        if &self.context != event.context() {
            return Err(Error::context_mismatch("event"));
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
        self.add_dependency_edges(from, to, &[])
    }

    // Empty `edge_data` uses default attributes for every edge; empty `from` and `to` leave the graph unchanged.
    fn add_dependency_edges(
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
        // SAFETY: the nodes were checked to belong to this exclusively borrowed graph, and the `from`, `to`, and edge arrays are locals of equal, validated length that outlive the call.
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
        self.remove_dependency_edges(from, to, &[])
    }

    // Empty `edge_data` matches default attributes for every edge; empty `from` and `to` leave the graph unchanged.
    fn remove_dependency_edges(
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
        // SAFETY: the nodes were checked to belong to this exclusively borrowed graph, and the `from`, `to`, and edge arrays are locals of equal, validated length that outlive the call.
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
        self.add_dependency_edges(&from, &to, &data)
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
        self.remove_dependency_edges(&from, &to, &data)
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
    /// Callbacks must not call CUDA functions; see [`Stream::add_callback`](crate::stream::Stream::add_callback).
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
    /// The graph and executables instantiated from it retain `event`.
    ///
    /// These nodes may not be used in loops or conditionals.
    ///
    /// Graph objects are not threadsafe.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the graph operation, if a previous asynchronous launch reported an error, or if CUDA reports runtime initialization diagnostics.
    /// Callbacks must not call CUDA functions; see [`Stream::add_callback`](crate::stream::Stream::add_callback).
    pub fn add_event_record_node(
        &mut self,
        dependencies: &[GraphNode],
        event: &Arc<Event>,
    ) -> Result<GraphNode> {
        self.check_nodes(dependencies)?;
        self.check_event_record_context(event)?;
        // Retain first, so a node CUDA created never references an unretained event.
        self.retained_lock()?.retain_event(event);
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
    /// See [`Event::record`] for what an event captures.
    /// Synchronization is performed efficiently on the device when applicable.
    /// `event` may come from a different context or device than the launch stream.
    /// The graph and executables instantiated from it retain `event`.
    ///
    /// These nodes may not be used in loops or conditionals.
    ///
    /// Graph objects are not threadsafe.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the graph operation, if a previous asynchronous launch reported an error, or if CUDA reports runtime initialization diagnostics.
    /// Callbacks must not call CUDA functions; see [`Stream::add_callback`](crate::stream::Stream::add_callback).
    pub fn add_event_wait_node(
        &mut self,
        dependencies: &[GraphNode],
        event: &Arc<Event>,
    ) -> Result<GraphNode> {
        self.check_nodes(dependencies)?;
        // Retain first, so a node CUDA created never references an unretained event.
        self.retained_lock()?.retain_event(event);
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
    /// Callbacks must not call CUDA functions; see [`Stream::add_callback`](crate::stream::Stream::add_callback).
    pub unsafe fn add_host_node(
        &mut self,
        dependencies: &[GraphNode],
        params: &HostNodeParams,
    ) -> Result<GraphNode> {
        self.check_nodes(dependencies)?;
        let mut handle = ptr::null_mut();
        let dependencies_raw: Vec<_> = dependencies.iter().map(GraphNode::as_raw).collect();
        let params = params.into();
        // SAFETY: `handle` is a local output, `check_nodes` confirmed the dependencies belong to this exclusively borrowed graph, and the dependency array is passed with its length; the caller keeps the callback and user data valid for every launch.
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
    /// A runtime kernel handle passed as a raw function must come from the same CUDA Runtime instance.
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
    /// Callbacks must not call CUDA functions; see [`Stream::add_callback`](crate::stream::Stream::add_callback).
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
        // SAFETY: `handle` is a local output, `check_nodes` confirmed the dependencies belong to this exclusively borrowed graph, and the dependency array is passed with its length; CUDA copies `arguments` during the call, and the caller guarantees they match `function` and point to storage valid for every launch.
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
                if let Err(error) = self.set_launch_attribute(&node, *attribute) {
                    if let Err(destroy_error) = try_ffi!(driver::cuGraphDestroyNode(handle)) {
                        report_cleanup_failure("destroy", "cuda graph node", &destroy_error);
                    }

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
    /// Callbacks must not call CUDA functions; see [`Stream::add_callback`](crate::stream::Stream::add_callback).
    pub unsafe fn add_memory_copy_node_1d(
        &mut self,
        dependencies: &[GraphNode],
        params: &MemoryCopy1DNodeParams,
    ) -> Result<GraphNode> {
        self.check_nodes(dependencies)?;
        let mut handle = ptr::null_mut();
        let dependencies_raw: Vec<_> = dependencies.iter().map(GraphNode::as_raw).collect();
        // SAFETY: `handle` is a local output, `check_nodes` confirmed the dependencies belong to this exclusively borrowed graph, and the dependency array is passed with its length; the caller keeps the copy's addresses valid for every launch.
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

    /// Creates a device-to-device memcpy node between byte slices.
    ///
    /// The node copies `src.len()` bytes.
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
        D: DeviceSliceMut<u8> + ?Sized,
        S: DeviceSlice<u8> + ?Sized,
    {
        let count = src.len();
        if dst.len() < count {
            return Err(Error::buffer_too_small("destination", count, dst.len()));
        }

        // SAFETY: `dst` holds at least `count` bytes (checked above) and `src` exactly `count`; the caller keeps both valid for every launch.
        let params = unsafe {
            MemoryCopy1DNodeParams::new(
                dst.as_mut_ptr().cast(),
                src.as_device_ptr().cast(),
                count,
                MemoryCopyKind::DeviceToDevice,
            )
        };
        // SAFETY: the caller keeps `dst` and `src`, the addresses in `params`, valid and exclusive for every launch.
        unsafe { self.add_memory_copy_node_1d(dependencies, &params) }
    }

    /// Creates a device-to-device memcpy node between graph-retained buffers.
    ///
    /// The node copies `src.byte_len()` bytes.
    /// `dst` must have at least that many bytes.
    /// The graph retains both allocations so the baked CUDA graph pointers remain live for future instantiation and replay.
    ///
    /// # Safety
    ///
    /// Keep source elements valid for every launch, and order their producers before replay.
    /// Keep destination accesses exclusive through completion, including failed submissions.
    /// Allocation retention preserves addresses; initialization and access ordering remain caller obligations.
    ///
    /// # Errors
    ///
    /// Returns an error if `dst` is smaller than `src`, if CUDA rejects the graph operation, if a previous asynchronous launch reported an error, or if CUDA reports runtime initialization diagnostics.
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
            return Err(Error::buffer_too_small(
                "destination",
                count,
                dst.byte_len(),
            ));
        }

        // SAFETY: `dst` holds at least `count` bytes (checked above) and `src` exactly `count`, and both allocations are retained by this graph below.
        let params = unsafe {
            MemoryCopy1DNodeParams::new(
                dst.as_mut_ptr().cast(),
                src.as_ptr().cast(),
                count,
                MemoryCopyKind::DeviceToDevice,
            )
        };
        // SAFETY: the graph retains both allocations below, so the addresses stay valid; the caller orders producers and keeps destination access exclusive.
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
    /// [`MemoryCopy3DNodeParams`] describes the copy and its restrictions.
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
    /// Callbacks must not call CUDA functions; see [`Stream::add_callback`](crate::stream::Stream::add_callback).
    pub unsafe fn add_memory_copy_node(
        &mut self,
        dependencies: &[GraphNode],
        params: &MemoryCopy3DNodeParams,
    ) -> Result<GraphNode> {
        self.check_nodes(dependencies)?;
        let mut handle = ptr::null_mut();
        let dependencies_raw: Vec<_> = dependencies.iter().map(GraphNode::as_raw).collect();
        let params = params.into();
        // SAFETY: `handle` is a local output, `check_nodes` confirmed the dependencies belong to this exclusively borrowed graph, and the dependency array is passed with its length; the caller keeps the 3D copy's addresses valid for every launch.
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

    /// Adds a node that copies host or device memory into a module symbol, as described by `params`.
    ///
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
        // SAFETY: `handle` is a local output, `check_nodes` confirmed the dependencies belong to this exclusively borrowed graph, and the dependency array is passed with its length; the caller keeps the symbol and source pointer valid for every launch.
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

    /// Adds a node that copies a module symbol into host or device memory, as described by `params`.
    ///
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
        // SAFETY: `handle` is a local output, `check_nodes` confirmed the dependencies belong to this exclusively borrowed graph, and the dependency array is passed with its length; the caller keeps the symbol and destination pointer valid for every launch.
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
    /// Callbacks must not call CUDA functions; see [`Stream::add_callback`](crate::stream::Stream::add_callback).
    pub unsafe fn add_memory_set_node(
        &mut self,
        dependencies: &[GraphNode],
        params: &MemorySetNodeParams,
    ) -> Result<GraphNode> {
        self.check_nodes(dependencies)?;
        let mut handle = ptr::null_mut();
        let dependencies_raw: Vec<_> = dependencies.iter().map(GraphNode::as_raw).collect();
        let params = params.into();
        // SAFETY: `handle` is a local output, `check_nodes` confirmed the dependencies belong to this exclusively borrowed graph, and the dependency array is passed with its length; the caller keeps the destination valid for every launch.
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
    /// Callbacks must not call CUDA functions; see [`Stream::add_callback`](crate::stream::Stream::add_callback).
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
        // SAFETY: the graph id check above proves `allocation` came from an allocation node of this graph, and CUDA rejects a second free of it in the same graph.
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
        // SAFETY: `handle` is a local output, `check_nodes` confirmed the dependencies belong to this exclusively borrowed graph, and the dependency array is passed with its length; the caller guarantees `ptr` is this graph's allocation, ordered after its allocation node and freed once.
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
                DevicePtr::new(ptr::without_provenance_mut(to_usize(
                    params_raw.dptr,
                    "memory allocation address",
                )?)),
                params.byte_size,
                self.id,
                Arc::clone(&self.inner),
                Some(Arc::clone(&self.context)),
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
    /// Callbacks must not call CUDA functions; see [`Stream::add_callback`](crate::stream::Stream::add_callback).
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
    /// Callbacks must not call CUDA functions; see [`Stream::add_callback`](crate::stream::Stream::add_callback).
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
    /// Callbacks must not call CUDA functions; see [`Stream::add_callback`](crate::stream::Stream::add_callback).
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
    /// Child graph nodes are counted as child nodes here; callers that need recursive details can query the child graph returned by [`Graph::child_graph`].
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
            summary.record_node_type(self.node_type(&node)?);
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
        // SAFETY: `self.inner` owns the live graph, and `path` is a NUL-terminated local that outlives the call.
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

    /// Returns the context this graph belongs to.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        &self.context
    }

    /// Transfers the native graph, context, and retained resources without destruction.
    ///
    /// Returns the unchanged graph if a node or executable still shares its native owner.
    /// Keep the returned parts alive through completed native work and handle destruction, or re-adopt them.
    ///
    /// # Errors
    ///
    /// Returns the original graph if a node, executable, child view, or another owner still retains its native graph storage.
    pub fn into_raw(self) -> StdResult<RawGraph, Self> {
        let Self {
            inner,
            id,
            context,
            _not_sync: _,
        } = self;
        match Arc::try_unwrap(inner) {
            Ok(mut inner) => {
                inner.owns_handle = false;
                Ok(RawGraph {
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

    /// Adopts a native graph together with its original context and resource owners.
    ///
    /// # Safety
    ///
    /// The native handle must still be live and unchanged, with no other destroying owner or unretained dependencies.
    ///
    /// # Errors
    ///
    /// Returns an error if the native handle is null.
    pub unsafe fn from_raw(raw: RawGraph) -> Result<Self> {
        if raw.handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            inner: Arc::new(GraphInner {
                handle: raw.handle,
                owns_handle: true,
                context: Some(Arc::clone(&raw.context)),
                retained: Mutex::new(raw.retained),
                parent: None,
            }),
            id: raw.id,
            context: raw.context,
            _not_sync: PhantomData,
        })
    }
}
