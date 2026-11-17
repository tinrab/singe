use std::{ffi::c_void, ptr, sync::Arc};

use singe_cuda_sys::{driver, runtime};

use crate::{
    context::Context,
    dim::Dim3,
    error::{Error, Result},
    graph::{
        BorrowedGraph, Graph, GraphDependency, GraphId, GraphInner, GraphKernelNodeAttribute,
        GraphKernelNodeAttributeId, GraphNode, GraphNodeType, MemoryAllocationNodeInfo,
        MemorySetNodeParams, raw::MemoryCopy1DNodeParams,
    },
    module::LaunchAttribute,
    try_ffi,
    types::{ClusterSchedulingPolicy, DevicePtr, SharedMemoryMode},
    utility::to_usize,
};

// Every query and change of a node goes through its `Graph`, which is not `Sync`, so native graph access follows Rust's borrow of the graph.
// A `GraphNode` only identifies a node and keeps its graph alive so that its handle stays valid.
impl GraphNode {
    /// Wraps an existing CUDA graph node handle.
    ///
    /// The returned node has no [`Graph`] identity, so graph, executable-graph, and stream-capture methods reject it with `Error::GraphNodeMismatch`.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid CUDA graph node handle.
    /// The caller must ensure the node remains valid for every operation using the returned token and that it belongs to the graph or executable graph passed to those operations.
    pub const unsafe fn from_raw(handle: runtime::cudaGraphNode_t) -> Self {
        Self {
            handle,
            graph_id: None,
            graph: None,
            context: None,
        }
    }

    pub(crate) fn from_raw_in_graph(
        handle: runtime::cudaGraphNode_t,
        graph_id: GraphId,
        graph: Arc<GraphInner>,
        context: Option<Arc<Context>>,
    ) -> Self {
        Self {
            handle,
            graph_id: Some(graph_id),
            graph: Some(graph),
            context,
        }
    }

    /// Returns the native node handle without transferring graph ownership.
    #[must_use]
    pub const fn as_raw(&self) -> runtime::cudaGraphNode_t {
        self.handle
    }

    pub(crate) fn graph_raw(&self) -> Option<runtime::cudaGraph_t> {
        self.graph.as_ref().map(|graph| graph.handle)
    }

    /// Returns the retained context, if this node has context metadata.
    #[must_use]
    pub fn context(&self) -> Option<&Arc<Context>> {
        self.context.as_ref()
    }
}

impl Graph {
    /// Returns the type of `node`.
    ///
    /// # Errors
    ///
    /// Returns `Error::GraphNodeMismatch` if `node` belongs to another graph, and `Error::UnknownEnumValue` if CUDA reports a node type this crate does not know.
    /// Also returns an error if CUDA cannot query the node type or a previous asynchronous launch reported an error.
    pub fn node_type(&self, node: &GraphNode) -> Result<GraphNodeType> {
        self.check_node(node)?;
        let mut kind = runtime::cudaGraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL;
        // SAFETY: `node` belongs to this live graph, and `kind` is a valid output location.
        unsafe {
            try_ffi!(runtime::cudaGraphNodeGetType(node.as_raw(), &raw mut kind))?;
        }
        GraphNodeType::try_from(kind)
    }

    /// Returns the dependencies of `node`.
    ///
    /// # Errors
    ///
    /// Returns `Error::GraphNodeMismatch` if `node` belongs to another graph.
    /// Also returns an error if CUDA cannot query the dependencies or a previous asynchronous launch reports an error.
    pub fn node_dependencies(&self, node: &GraphNode) -> Result<Vec<GraphDependency>> {
        self.check_node(node)?;
        // SAFETY: `node` belongs to this live graph, and the native call first reports the count, then fills arrays sized to it.
        unsafe {
            let mut count = 0;
            try_ffi!(runtime::cudaGraphNodeGetDependencies(
                node.as_raw(),
                ptr::null_mut(),
                ptr::null_mut(),
                &raw mut count,
            ))?;

            if count == 0 {
                return Ok(Vec::new());
            }

            let mut handles = Vec::with_capacity(to_usize(count, "graph node count")?);
            let mut edge_data = Vec::with_capacity(to_usize(count, "graph node count")?);
            try_ffi!(runtime::cudaGraphNodeGetDependencies(
                node.as_raw(),
                handles.as_mut_ptr(),
                edge_data.as_mut_ptr(),
                &raw mut count,
            ))?;
            handles.set_len(to_usize(count, "graph node count")?);
            edge_data.set_len(to_usize(count, "graph node count")?);

            Ok(handles
                .into_iter()
                .zip(edge_data)
                .map(|(handle, data)| GraphDependency {
                    node: self.node_from_raw(handle),
                    data: data.into(),
                })
                .collect())
        }
    }

    /// Returns the nodes that depend on `node`.
    ///
    /// # Errors
    ///
    /// Returns `Error::GraphNodeMismatch` if `node` belongs to another graph.
    /// Also returns an error if CUDA cannot query the dependent nodes or a previous asynchronous launch reports an error.
    pub fn node_dependents(&self, node: &GraphNode) -> Result<Vec<GraphDependency>> {
        self.check_node(node)?;
        // SAFETY: `node` belongs to this live graph, and the native call first reports the count, then fills arrays sized to it.
        unsafe {
            let mut count = 0;
            try_ffi!(runtime::cudaGraphNodeGetDependentNodes(
                node.as_raw(),
                ptr::null_mut(),
                ptr::null_mut(),
                &raw mut count,
            ))?;

            if count == 0 {
                return Ok(Vec::new());
            }

            let mut handles = Vec::with_capacity(to_usize(count, "graph node count")?);
            let mut edge_data = Vec::with_capacity(to_usize(count, "graph node count")?);
            try_ffi!(runtime::cudaGraphNodeGetDependentNodes(
                node.as_raw(),
                handles.as_mut_ptr(),
                edge_data.as_mut_ptr(),
                &raw mut count,
            ))?;
            handles.set_len(to_usize(count, "graph node count")?);
            edge_data.set_len(to_usize(count, "graph node count")?);

            Ok(handles
                .into_iter()
                .zip(edge_data)
                .map(|(handle, data)| GraphDependency {
                    node: self.node_from_raw(handle),
                    data: data.into(),
                })
                .collect())
        }
    }

    /// Borrows the graph embedded in the child graph node `node` without cloning it.
    ///
    /// The returned [`BorrowedGraph`] borrows this graph, so this graph cannot change while it lives, and it does not destroy the embedded graph when dropped.
    /// Allocation and free nodes cannot be added to the returned graph.
    ///
    /// # Errors
    ///
    /// Returns `Error::GraphNodeMismatch` if `node` belongs to another graph.
    /// Also returns an error if `node` is not a child graph node, CUDA cannot query the child graph, CUDA returns a null graph handle, or a previous asynchronous launch reports an error.
    ///
    /// # Panics
    ///
    /// Panics if the newly constructed borrowed graph unexpectedly has multiple owners.
    pub fn child_graph(&self, node: &GraphNode) -> Result<BorrowedGraph<'_>> {
        self.check_node(node)?;
        let mut graph = ptr::null_mut();
        // SAFETY: `node` belongs to this live graph, and `graph` is a valid output location.
        unsafe {
            try_ffi!(runtime::cudaGraphChildGraphNodeGetGraph(
                node.as_raw(),
                &raw mut graph,
            ))?;
        }

        if graph.is_null() {
            return Err(Error::NullHandle);
        }

        // SAFETY: CUDA returned this graph, owned by a node of this graph in this graph's context, and the view borrows this graph.
        let mut borrowed = unsafe { BorrowedGraph::from_raw(graph, Arc::clone(&self.context)) }?;
        let inner =
            Arc::get_mut(&mut borrowed.graph.inner).expect("new borrowed graph has one owner");
        inner.parent = Some(Arc::clone(&self.inner));
        *inner
            .retained
            .get_mut()
            .map_err(|_| Error::lock_poisoned("graph resources"))? = self.retained_resources()?;
        Ok(borrowed)
    }

    /// Returns the address and size of the memory allocation node `node`.
    ///
    /// # Errors
    ///
    /// Returns `Error::GraphNodeMismatch` if `node` belongs to another graph.
    /// Also returns an error if `node` is not a memory allocation node, CUDA cannot query the parameters, or a previous asynchronous launch reports an error.
    pub fn memory_allocation_node_info(
        &self,
        node: &GraphNode,
    ) -> Result<MemoryAllocationNodeInfo> {
        self.check_node(node)?;
        let mut params = runtime::cudaMemAllocNodeParams::default();
        // SAFETY: `node` belongs to this live graph, and `params` is a valid output location.
        unsafe {
            try_ffi!(runtime::cudaGraphMemAllocNodeGetParams(
                node.as_raw(),
                &raw mut params,
            ))?;
        }
        Ok(MemoryAllocationNodeInfo::from_raw_in_graph(
            // SAFETY: the address is the graph allocation CUDA reported and is only stored; dereferencing it requires further unsafe code.
            unsafe {
                DevicePtr::new(ptr::without_provenance_mut(to_usize(
                    params.dptr,
                    "memory allocation address",
                )?))
            },
            to_usize(params.bytesize, "memory allocation byte count")?,
            self.id,
            Arc::clone(&self.inner),
            Some(Arc::clone(&self.context)),
        ))
    }

    /// Returns the address that the memory free node `node` frees.
    ///
    /// # Safety
    ///
    /// The returned pointer must not be used after the graph frees it.
    ///
    /// # Errors
    ///
    /// Returns `Error::GraphNodeMismatch` if `node` belongs to another graph.
    /// Also returns an error if `node` is not a memory free node, CUDA cannot query the pointer, or a previous asynchronous launch reports an error.
    pub unsafe fn memory_free_node_ptr(&self, node: &GraphNode) -> Result<DevicePtr> {
        self.check_node(node)?;
        let mut ptr: *mut c_void = ptr::null_mut();
        // SAFETY: `node` belongs to this live graph, and `ptr` is a valid output location.
        unsafe {
            try_ffi!(runtime::cudaGraphMemFreeNodeGetParams(
                node.as_raw(),
                (&raw mut ptr).cast(),
            ))?;
        }
        // SAFETY: the address is the one CUDA's free node frees, and the caller must not use it after the graph frees it.
        Ok(unsafe { DevicePtr::new(ptr.cast()) })
    }

    /// Returns the attribute `id` of the kernel node `node`.
    ///
    /// # Errors
    ///
    /// Returns `Error::GraphNodeMismatch` if `node` belongs to another graph.
    /// Also returns an error if `node` is not a kernel node, CUDA cannot query the attribute, or a previous asynchronous launch reports an error.
    pub fn kernel_node_attribute(
        &self,
        node: &GraphNode,
        id: GraphKernelNodeAttributeId,
    ) -> Result<GraphKernelNodeAttribute> {
        self.check_node(node)?;
        let mut value = runtime::cudaLaunchAttributeValue::default();
        // SAFETY: `node` belongs to this live graph, and the attribute value is a writable local union sized for every attribute.
        unsafe {
            try_ffi!(runtime::cudaGraphKernelNodeGetAttribute(
                node.as_raw(),
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

    /// Returns the shared memory configuration of the kernel node `node`.
    ///
    /// # Errors
    ///
    /// Returns `Error::GraphNodeMismatch` if `node` belongs to another graph.
    /// Also propagates CUDA kernel-node attribute errors, including an incompatible node type or unsupported attribute.
    pub fn kernel_node_shared_memory_mode(&self, node: &GraphNode) -> Result<SharedMemoryMode> {
        self.check_node(node)?;
        let mut value = driver::CUlaunchAttributeValue::default();
        // SAFETY: `node` belongs to this live graph, and `value` is a valid output location.
        unsafe {
            try_ffi!(driver::cuGraphKernelNodeGetAttribute(
                node.as_raw(),
                driver::CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_SHARED_MEMORY_MODE,
                &raw mut value
            ))?;
            Ok(value.sharedMemoryMode.try_into()?)
        }
    }

    /// Returns the cluster scheduling policy preference of the kernel node `node`.
    ///
    /// # Errors
    ///
    /// Returns `Error::GraphNodeMismatch` if `node` belongs to another graph.
    /// Also propagates CUDA kernel-node attribute errors, including an incompatible node type or unsupported attribute.
    pub fn kernel_node_cluster_scheduling_policy(
        &self,
        node: &GraphNode,
    ) -> Result<ClusterSchedulingPolicy> {
        self.check_node(node)?;
        let mut value = driver::CUlaunchAttributeValue::default();
        // SAFETY: `node` belongs to this live graph, and `value` is a valid output location.
        unsafe {
            try_ffi!(driver::cuGraphKernelNodeGetAttribute(
                node.as_raw(),
                driver::CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE,
                &raw mut value
            ))?;
            Ok(value.clusterSchedulingPolicyPreference.try_into()?)
        }
    }

    /// Changes the one-dimensional copy node `node` in the graph definition.
    /// Existing executable graphs keep their previous parameters until explicitly updated.
    ///
    /// # Safety
    ///
    /// Source and destination must uphold [`MemoryCopy1DNodeParams::new`] for every affected replay.
    /// Retain old sources for existing executables.
    ///
    /// # Errors
    ///
    /// Returns `Error::GraphNodeMismatch` if `node` belongs to another graph.
    /// Also propagates context-binding and CUDA node-parameter update errors, including a node that is not a memory-copy node or invalid copy parameters.
    pub unsafe fn set_memory_copy_node_1d_params(
        &mut self,
        node: &GraphNode,
        params: &MemoryCopy1DNodeParams,
    ) -> Result<()> {
        self.check_node(node)?;
        // SAFETY: `node` belongs to this exclusively borrowed graph, and the caller vouches for the copy operands.
        unsafe {
            try_ffi!(runtime::cudaGraphMemcpyNodeSetParams1D(
                node.as_raw(),
                params.dst().cast(),
                params.src().cast(),
                params.count() as _,
                params.kind().into(),
            ))
        }
    }

    /// Replaces the graph embedded in the child graph node `node` with an independent snapshot of `child_graph`.
    /// Existing executable graphs retain their previous parameters until explicitly updated.
    /// The node identity and its incoming and outgoing dependencies are preserved.
    ///
    /// # Errors
    ///
    /// Returns `Error::GraphNodeMismatch` if `node` belongs to another graph, `Error::ContextMismatch` if `child_graph` has another context, or an error if CUDA rejects the replacement.
    /// CUDA validates the replacement before changing the node.
    pub fn set_child_graph(&mut self, node: &GraphNode, child_graph: &Self) -> Result<()> {
        self.check_node(node)?;
        if self.context.as_ref() != child_graph.context().as_ref() {
            return Err(Error::context_mismatch("child graph"));
        }

        let child_resources = child_graph.retained_resources()?;
        // Retain both snapshots before publishing native pointers.
        // A rejected replacement can retain extra owners, but cannot release resources used by an existing graph.
        self.retained_lock()?.extend(child_resources);
        let mut params = runtime::cudaGraphNodeParams {
            type_: GraphNodeType::Graph.into(),
            ..Default::default()
        };
        // SAFETY: the graph tag selects this zero-initialized union member.
        // Write its fields separately so native-required padding and reserved bytes stay zero.
        let graph_params = unsafe { &mut params.__bindgen_anon_1.graph };
        graph_params.graph = child_graph.as_raw();
        graph_params.ownership =
            driver::CUgraphChildGraphNodeOwnership::CU_GRAPH_CHILD_GRAPH_OWNERSHIP_CLONE;
        // SAFETY: `node` belongs to this exclusively borrowed graph, the zero-initialized parameters select a cloned child graph, both graphs stay live through the call, and retained resources cover the copied nodes.
        unsafe {
            try_ffi!(runtime::cudaGraphNodeSetParams(
                node.as_raw(),
                &raw mut params
            ))
        }
    }

    /// Changes the memset node `node` in the graph definition.
    /// Existing executables keep their previous parameters until explicitly updated.
    ///
    /// # Safety
    ///
    /// The destination must uphold [`MemorySetNodeParams::new`] for every affected replay.
    /// Retain previous destinations through existing executable launches.
    ///
    /// # Errors
    ///
    /// Returns `Error::GraphNodeMismatch` if `node` belongs to another graph, or an error if the context cannot bind or CUDA rejects the node parameters.
    pub unsafe fn set_memory_set_node_params(
        &mut self,
        node: &GraphNode,
        params: &MemorySetNodeParams,
    ) -> Result<()> {
        self.check_node(node)?;
        let params = params.into();
        // SAFETY: `node` belongs to this exclusively borrowed graph, `params` lives through the call, and the caller vouches for the destination.
        unsafe {
            try_ffi!(runtime::cudaGraphMemsetNodeSetParams(
                node.as_raw(),
                &raw const params,
            ))
        }
    }

    /// Sets an attribute of the kernel node `node`.
    ///
    /// # Errors
    ///
    /// Returns `Error::GraphNodeMismatch` if `node` belongs to another graph.
    /// Also returns an error if `node` is not a kernel node, CUDA rejects the attribute update, or a previous asynchronous launch reports an error.
    pub fn set_kernel_node_attribute(
        &mut self,
        node: &GraphNode,
        attribute: GraphKernelNodeAttribute,
    ) -> Result<()> {
        self.check_node(node)?;
        let (id, value) = match attribute {
            GraphKernelNodeAttribute::Cooperative(value) => {
                let mut attr = runtime::cudaLaunchAttributeValue {
                    cooperative: runtime::__BindgenUnionField::new(),
                    ..runtime::cudaLaunchAttributeValue::default()
                };
                // SAFETY: the attribute id set above selects this union field.
                unsafe { *attr.cooperative.as_mut() = i32::from(value) };
                (GraphKernelNodeAttributeId::Cooperative, attr)
            }
            GraphKernelNodeAttribute::ClusterDimension(value) => {
                let mut attr = runtime::cudaLaunchAttributeValue {
                    clusterDim: runtime::__BindgenUnionField::new(),
                    ..runtime::cudaLaunchAttributeValue::default()
                };
                // SAFETY: the attribute id set above selects this union field.
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
                // SAFETY: the attribute id set above selects this union field.
                unsafe { *attr.priority.as_mut() = value };
                (GraphKernelNodeAttributeId::Priority, attr)
            }
            GraphKernelNodeAttribute::PreferredSharedMemoryCarveout(value) => {
                let mut attr = runtime::cudaLaunchAttributeValue {
                    sharedMemCarveout: runtime::__BindgenUnionField::new(),
                    ..runtime::cudaLaunchAttributeValue::default()
                };
                // SAFETY: the attribute id set above selects this union field.
                unsafe { *attr.sharedMemCarveout.as_mut() = value };
                (
                    GraphKernelNodeAttributeId::PreferredSharedMemoryCarveout,
                    attr,
                )
            }
        };

        // SAFETY: `node` belongs to this exclusively borrowed graph, and `value` lives through the call.
        unsafe {
            try_ffi!(runtime::cudaGraphKernelNodeSetAttribute(
                node.as_raw(),
                id.into(),
                &raw const value,
            ))
        }
    }

    /// Copies the kernel attributes of `src` to `dst`, both kernel nodes of this graph.
    ///
    /// # Errors
    ///
    /// Returns `Error::GraphNodeMismatch` if either node belongs to another graph.
    /// Also returns an error if CUDA rejects the attribute copy or a previous asynchronous launch reported an error.
    pub fn copy_kernel_node_attributes(&mut self, dst: &GraphNode, src: &GraphNode) -> Result<()> {
        self.check_node(dst)?;
        self.check_node(src)?;
        // SAFETY: both nodes belong to this exclusively borrowed graph.
        unsafe {
            try_ffi!(runtime::cudaGraphKernelNodeCopyAttributes(
                dst.as_raw(),
                src.as_raw()
            ))
        }
    }

    /// Sets an attribute of the kernel node `node` using the extended launch attribute representation.
    ///
    /// # Errors
    ///
    /// Returns `Error::GraphNodeMismatch` if `node` belongs to another graph.
    /// Also propagates context-binding and CUDA kernel-node attribute errors, including an incompatible node type or unsupported attribute.
    pub fn set_launch_attribute(
        &mut self,
        node: &GraphNode,
        attribute: LaunchAttribute,
    ) -> Result<()> {
        self.check_node(node)?;
        let attribute = attribute.raw();
        // SAFETY: `node` belongs to this exclusively borrowed graph, and the attribute value lives through the call.
        unsafe {
            try_ffi!(driver::cuGraphKernelNodeSetAttribute(
                node.as_raw(),
                attribute.id,
                &raw const attribute.value
            ))
        }
    }
}
