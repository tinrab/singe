use std::{ffi::c_void, ptr, sync::Arc};

use singe_cuda_sys::{driver, runtime};

use super::{
    BorrowedGraph, Graph, GraphDependency, GraphId, GraphInner, GraphKernelNodeAttribute,
    GraphKernelNodeAttributeId, GraphNode, GraphNodeType, MemoryAllocationNodeInfo,
    MemorySetNodeParams, raw::MemoryCopy1DNodeParams,
};
use crate::{
    context::Context,
    dim::Dim3,
    error::{Error, Result},
    module::LaunchAttribute,
    try_ffi,
    types::{ClusterSchedulingPolicy, DevicePtr, SharedMemoryMode},
    utility::to_usize,
};

impl GraphNode {
    /// Wraps an existing CUDA graph node handle.
    ///
    /// The returned node is not associated with any [`Graph`] identity, so graph and executable-graph methods cannot validate that it belongs to the target graph before calling CUDA.
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

    fn from_raw_like(handle: runtime::cudaGraphNode_t, node: &Self) -> Self {
        Self {
            handle,
            graph_id: node.graph_id,
            graph: node.graph.clone(),
            context: node.context.clone(),
        }
    }

    pub(crate) fn bind_context(&self) -> Result<()> {
        if let Some(context) = &self.context {
            context.bind()?;
        }
        Ok(())
    }

    /// Returns the node type.
    ///
    /// Graph objects are not threadsafe.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot query the node type or if a previous asynchronous launch reported an error.
    /// CUDA may also return initialization-related errors such as [`crate::error::Status::NotInitialized`], [`crate::error::Status::CallRequiresNewerDriver`], or [`crate::error::Status::NoDevice`] if this call initializes internal runtime state.
    /// Callbacks must not call CUDA functions; see [`Stream::add_callback`].
    pub fn node_type(&self) -> Result<GraphNodeType> {
        self.bind_context()?;
        let mut kind = runtime::cudaGraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL;
        // SAFETY: `cudaGraphNodeGetType` receives live handles owned by the wrappers and valid output locations.
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
    /// Returns an error if CUDA cannot query the dependencies, a previous asynchronous launch reports an error, or CUDA reports runtime initialization diagnostics.
    pub fn dependencies(&self) -> Result<Vec<GraphDependency>> {
        self.bind_context()?;
        // SAFETY: `cudaGraphNodeGetDependencies` first reports the count, then fills arrays sized to it.
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

            let mut handles = Vec::with_capacity(to_usize(count, "graph node count")?);
            let mut edge_data = Vec::with_capacity(to_usize(count, "graph node count")?);
            try_ffi!(runtime::cudaGraphNodeGetDependencies(
                self.as_raw(),
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
    /// Returns an error if CUDA cannot query the dependent nodes, a previous asynchronous launch reports an error, or CUDA reports runtime initialization diagnostics.
    pub fn dependent_nodes(&self) -> Result<Vec<GraphDependency>> {
        self.bind_context()?;
        // SAFETY: `cudaGraphNodeGetDependentNodes` first reports the count, then fills arrays sized to it.
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

            let mut handles = Vec::with_capacity(to_usize(count, "graph node count")?);
            let mut edge_data = Vec::with_capacity(to_usize(count, "graph node count")?);
            try_ffi!(runtime::cudaGraphNodeGetDependentNodes(
                self.as_raw(),
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
    /// Returns an error if this is not an event-record node, CUDA cannot query the event, CUDA returns a null event handle, a previous asynchronous launch reports an error, or CUDA reports runtime initialization diagnostics.
    pub fn event_record_node_event(&self) -> Result<runtime::cudaEvent_t> {
        self.bind_context()?;
        let mut event = ptr::null_mut();
        // SAFETY: `cudaGraphEventRecordNodeGetEvent` receives live handles owned by the wrappers and valid output locations.
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
    /// Returns an error if this is not an event-wait node, CUDA cannot query the event, CUDA returns a null event handle, a previous asynchronous launch reports an error, or CUDA reports runtime initialization diagnostics.
    pub fn event_wait_node_event(&self) -> Result<runtime::cudaEvent_t> {
        self.bind_context()?;
        let mut event = ptr::null_mut();
        // SAFETY: `cudaGraphEventWaitNodeGetEvent` receives live handles owned by the wrappers and valid output locations.
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
    /// Changes to the returned graph are reflected in the node, and the child node retains ownership of the embedded graph handle.
    /// The returned [`BorrowedGraph`] is tied to this node borrow and does not destroy the embedded graph when dropped.
    ///
    /// Allocation and free nodes cannot be added to the returned graph.
    /// Attempting to do so returns an error.
    ///
    /// Graph objects are not threadsafe.
    ///
    /// # Errors
    ///
    /// Returns an error if this is not a child-graph node, CUDA cannot query the child graph, CUDA returns a null graph handle, a previous asynchronous launch reports an error, or CUDA reports runtime initialization diagnostics.
    ///
    /// # Panics
    ///
    /// Panics if the newly constructed borrowed graph unexpectedly has multiple owners.
    pub fn child_graph(&self) -> Result<BorrowedGraph<'_>> {
        self.bind_context()?;
        let mut graph = ptr::null_mut();
        // SAFETY: `cudaGraphChildGraphNodeGetGraph` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(runtime::cudaGraphChildGraphNodeGetGraph(
                self.as_raw(),
                &raw mut graph,
            ))?;
        }

        if graph.is_null() {
            return Err(Error::NullHandle);
        }

        // SAFETY: CUDA returned this graph for the context that the new value retains.
        let mut borrowed =
            unsafe { BorrowedGraph::from_raw_in_context(graph, self.context.clone()) }?;
        let inner =
            Arc::get_mut(&mut borrowed.graph.inner).expect("new borrowed graph has one owner");
        inner.parent.clone_from(&self.graph);
        if let Some(parent) = &self.graph {
            *inner
                .retained
                .get_mut()
                .map_err(|_| Error::ResourcePoisoned {
                    name: "graph resources",
                })? = parent
                .retained
                .lock()
                .map_err(|_| Error::ResourcePoisoned {
                    name: "graph resources",
                })?
                .clone();
        }
        Ok(borrowed)
    }

    /// Returns the parameters of this memcpy node.
    ///
    /// Graph objects are not threadsafe.
    ///
    /// # Errors
    ///
    /// Returns an error if this is not a memcpy node, CUDA cannot query the parameters, a previous asynchronous launch reports an error, or CUDA reports runtime initialization diagnostics.
    pub fn memory_copy_node_params(&self) -> Result<runtime::cudaMemcpy3DParms> {
        self.bind_context()?;
        let mut params = runtime::cudaMemcpy3DParms::default();
        // SAFETY: `cudaGraphMemcpyNodeGetParams` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(runtime::cudaGraphMemcpyNodeGetParams(
                self.as_raw(),
                &raw mut params,
            ))?;
        }
        Ok(params)
    }

    /// Changes a one-dimensional copy node in the graph definition.
    /// Existing executable graphs keep their previous parameters until explicitly updated.
    ///
    /// # Safety
    /// Source and destination must uphold [`MemoryCopy1DNodeParams::new`] for every affected replay.
    /// Order mutations against other graph-definition operations and retain old sources for existing executables.
    ///
    /// # Errors
    ///
    /// Propagates context-binding and CUDA node-parameter update errors, including a node that is not a memory-copy node or invalid copy parameters.
    pub unsafe fn set_memory_copy_node_1d_params(
        &self,
        params: &MemoryCopy1DNodeParams,
    ) -> Result<()> {
        self.bind_context()?;
        // SAFETY: `cudaGraphMemcpyNodeSetParams1D` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(runtime::cudaGraphMemcpyNodeSetParams1D(
                self.as_raw(),
                params.dst().cast(),
                params.src().cast(),
                params.count() as _,
                params.kind().into(),
            ))
        }
    }

    /// Replaces the child graph in this graph definition with an independent snapshot.
    /// Existing executable graphs retain their previous parameters until explicitly updated.
    /// The node identity and its incoming and outgoing dependencies are preserved.
    ///
    /// # Errors
    /// Returns an error for an unassociated node, a context mismatch or rejected native parameters.
    /// CUDA validates the replacement before changing the node.
    pub fn set_child_graph(&self, child_graph: &Graph) -> Result<()> {
        self.bind_context()?;
        if let (Some(context), Some(child_context)) = (&self.context, child_graph.context())
            && context.as_ref() != child_context.as_ref()
        {
            return Err(Error::GraphContextMismatch);
        }

        let graph = self.graph.as_ref().ok_or(Error::GraphNodeMismatch)?;
        let child_resources = child_graph.retained_resources()?;
        let mut retained = graph.retained.lock().map_err(|_| Error::ResourcePoisoned {
            name: "graph resources",
        })?;
        // Retain both snapshots before publishing native pointers. A rejected replacement
        // can retain extra owners, but cannot release resources used by an existing graph.
        retained.extend(child_resources);
        drop(retained);
        let mut params = runtime::cudaGraphNodeParams {
            type_: GraphNodeType::Graph.into(),
            ..Default::default()
        };
        // SAFETY: the graph tag selects this zero-initialized union member. Write its
        // fields separately so native-required padding and reserved bytes stay zero.
        let graph_params = unsafe { &mut params.__bindgen_anon_1.graph };
        graph_params.graph = child_graph.as_raw();
        graph_params.ownership =
            driver::CUgraphChildGraphNodeOwnership::CU_GRAPH_CHILD_GRAPH_OWNERSHIP_CLONE;
        // SAFETY: zero-initialized parameters select a cloned child graph; both owners
        // remain live through the call and retained resources cover the copied nodes.
        unsafe {
            try_ffi!(runtime::cudaGraphNodeSetParams(
                self.as_raw(),
                &raw mut params
            ))
        }
    }

    /// Returns the parameters of this memset node.
    ///
    /// Graph objects are not threadsafe.
    ///
    /// # Errors
    ///
    /// Returns an error if this is not a memset node, CUDA cannot query the parameters, a previous asynchronous launch reports an error, or CUDA reports runtime initialization diagnostics.
    pub fn memory_set_node_params(&self) -> Result<driver::CUDA_MEMSET_NODE_PARAMS> {
        self.bind_context()?;
        let mut params = driver::CUDA_MEMSET_NODE_PARAMS::default();
        // SAFETY: `cudaGraphMemsetNodeGetParams` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(runtime::cudaGraphMemsetNodeGetParams(
                self.as_raw(),
                &raw mut params,
            ))?;
        }
        Ok(params)
    }

    /// Changes a memset node in the graph definition.
    /// Existing executables keep their previous parameters until explicitly updated.
    ///
    /// # Errors
    /// Returns an error if the context cannot bind or CUDA rejects the node parameters.
    ///
    /// # Safety
    /// The destination must uphold [`MemorySetNodeParams::new`] for every affected replay.
    /// Order graph mutation and retain previous destinations through existing executable launches.
    pub unsafe fn set_memory_set_node_params(&self, params: &MemorySetNodeParams) -> Result<()> {
        self.bind_context()?;
        let params = params.into();
        // SAFETY: `cudaGraphMemsetNodeSetParams` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(runtime::cudaGraphMemsetNodeSetParams(
                self.as_raw(),
                &raw const params,
            ))
        }
    }

    /// Returns the parameters of this host node.
    ///
    /// Graph objects are not threadsafe.
    ///
    /// # Errors
    ///
    /// Returns an error if this is not a host node, CUDA cannot query the parameters, a previous asynchronous launch reports an error, or CUDA reports runtime initialization diagnostics.
    pub fn host_node_params(&self) -> Result<driver::CUDA_HOST_NODE_PARAMS> {
        self.bind_context()?;
        let mut params = driver::CUDA_HOST_NODE_PARAMS::default();
        // SAFETY: `cudaGraphHostNodeGetParams` receives live handles owned by the wrappers and valid output locations.
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
    /// Returns an error if this is not a memory-allocation node, CUDA cannot query the parameters, a previous asynchronous launch reports an error, or CUDA reports runtime initialization diagnostics.
    pub fn memory_allocation_node_info(&self) -> Result<MemoryAllocationNodeInfo> {
        self.bind_context()?;
        let mut params = runtime::cudaMemAllocNodeParams::default();
        // SAFETY: `cudaGraphMemAllocNodeGetParams` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(runtime::cudaGraphMemAllocNodeGetParams(
                self.as_raw(),
                &raw mut params,
            ))?;
        }
        Ok(MemoryAllocationNodeInfo::from_raw(
            // SAFETY: the address is only stored; dereferencing it requires further unsafe code.
            unsafe { DevicePtr::new(params.dptr as _) },
            to_usize(params.bytesize, "memory fill byte count")?,
            self.graph_id,
            self.graph.clone(),
            self.context.clone(),
        ))
    }

    /// Returns the address of this memory free node.
    ///
    /// Graph objects are not threadsafe.
    ///
    /// # Errors
    ///
    /// Returns an error if this is not a memory-free node, CUDA cannot query the pointer, a previous asynchronous launch reports an error, or CUDA reports runtime initialization diagnostics.
    ///
    /// # Safety
    ///
    /// The node must still be a valid memory-free node in a live graph, and the returned pointer must not be used after the graph frees it.
    pub unsafe fn memory_free_node_ptr(&self) -> Result<DevicePtr> {
        self.bind_context()?;
        let mut ptr: *mut c_void = ptr::null_mut();
        // SAFETY: `cudaGraphMemFreeNodeGetParams` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(runtime::cudaGraphMemFreeNodeGetParams(
                self.as_raw(),
                (&raw mut ptr).cast(),
            ))?;
        }
        // SAFETY: the caller upholds this function's safety contract.
        Ok(unsafe { DevicePtr::new(ptr.cast()) })
    }

    /// Returns the requested kernel node attribute.
    ///
    /// # Errors
    ///
    /// Returns an error if this is not a kernel node, CUDA cannot query the attribute, or a previous asynchronous launch reports an error.
    pub fn kernel_node_attribute(
        self,
        id: GraphKernelNodeAttributeId,
    ) -> Result<GraphKernelNodeAttribute> {
        self.bind_context()?;
        let mut value = runtime::cudaLaunchAttributeValue::default();
        // SAFETY: the attribute value is a writable local union sized for every attribute.
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
    /// Returns an error if this is not a kernel node, CUDA rejects the attribute update, or a previous asynchronous launch reports an error.
    pub fn set_kernel_node_attribute(&mut self, attribute: GraphKernelNodeAttribute) -> Result<()> {
        self.bind_context()?;
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
                unsafe { *attr.priority.as_mut() = value as _ };
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

        // SAFETY: `cudaGraphKernelNodeSetAttribute` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
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
    /// Returns an error if CUDA rejects the attribute copy or if a previous asynchronous launch reported an error.
    pub fn copy_kernel_node_attributes(&self, other: &Self) -> Result<()> {
        if let (Some(context), Some(other_context)) = (self.context(), other.context())
            && context != other_context
        {
            return Err(Error::GraphContextMismatch);
        }

        self.bind_context()?;
        other.bind_context()?;
        // SAFETY: `cudaGraphKernelNodeCopyAttributes` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(runtime::cudaGraphKernelNodeCopyAttributes(
                self.as_raw(),
                other.handle
            ))?;
        }
        Ok(())
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

impl GraphNode {
    /// Sets a kernel node attribute using the extended launch attribute representation.
    ///
    /// # Errors
    ///
    /// Propagates context-binding and CUDA kernel-node attribute errors, including an incompatible node type or unsupported attribute.
    pub fn set_launch_attribute(&self, attribute: LaunchAttribute) -> Result<()> {
        self.bind_context()?;
        let attribute = attribute.raw();
        // SAFETY: `cuGraphKernelNodeSetAttribute` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(driver::cuGraphKernelNodeSetAttribute(
                self.as_raw(),
                attribute.id,
                &raw const attribute.value
            ))
        }
    }

    /// Returns the shared memory configuration for this kernel node.
    ///
    /// # Errors
    ///
    /// Propagates context-binding and CUDA kernel-node attribute errors, including an incompatible node type or unsupported attribute.
    pub fn shared_memory_mode(&self) -> Result<SharedMemoryMode> {
        self.bind_context()?;
        let mut value = driver::CUlaunchAttributeValue::default();
        // SAFETY: `cuGraphKernelNodeGetAttribute` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(driver::cuGraphKernelNodeGetAttribute(
                self.as_raw(),
                driver::CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_SHARED_MEMORY_MODE,
                &raw mut value
            ))?;
            Ok(value.sharedMemoryMode.into())
        }
    }

    /// Returns the cluster scheduling policy preference for this kernel node.
    ///
    /// # Errors
    ///
    /// Propagates context-binding and CUDA kernel-node attribute errors, including an incompatible node type or unsupported attribute.
    pub fn cluster_scheduling_policy(&self) -> Result<ClusterSchedulingPolicy> {
        self.bind_context()?;
        let mut value = driver::CUlaunchAttributeValue::default();
        // SAFETY: `cuGraphKernelNodeGetAttribute` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(driver::cuGraphKernelNodeGetAttribute(self.as_raw(), driver::CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE, &raw mut value))?;
            Ok(value.clusterSchedulingPolicyPreference.into())
        }
    }
}
