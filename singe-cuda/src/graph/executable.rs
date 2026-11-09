use std::{mem::ManuallyDrop, ptr, sync::Arc};

use singe_cuda_sys::runtime;

use super::{
    ExecutableGraph, ExecutableGraphLaunchOperation, ExecutableGraphUpdate, Graph, GraphBuffer,
    GraphId, GraphInner, GraphInstantiateFlags, GraphNode, MemorySetNodeParams, RawExecutableGraph,
    RawExecutableGraphParts, RetainedResources,
    raw::{MemoryCopy1DNodeParams, MemoryCopy3DNodeParams},
};
use crate::{
    context::Context,
    error::{Error, Result},
    event::Event,
    graph::raw::{HostNodeParams, MemoryCopyFromSymbolNodeParams, MemoryCopyToSymbolNodeParams},
    memory::MemoryCopyKind,
    module::{KernelLaunchArgs, LaunchGeometry, Module},
    stream::Stream,
    try_ffi,
    types::DeviceFunction,
    view::{ByteBuffer, ByteBufferMut, DeviceOutput, DeviceRepr},
};

impl RawExecutableGraph {
    /// Wraps an existing CUDA executable graph handle and takes ownership of it.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid CUDA executable graph handle.
    /// Ownership of `handle` is transferred to the returned [`RawExecutableGraph`], and the handle must not be destroyed elsewhere after calling this function.
    ///
    /// # Errors
    ///
    /// Returns an error if the native handle is null.
    pub unsafe fn from_raw(handle: runtime::cudaGraphExec_t) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self { handle })
    }

    /// Returns the owned native executable handle without transferring ownership.
    #[must_use]
    pub const fn as_raw(&self) -> runtime::cudaGraphExec_t {
        self.handle
    }

    /// Consumes the executable graph and returns the raw CUDA executable graph handle without destroying it.
    ///
    /// The caller becomes responsible for eventually destroying the returned handle with CUDA.
    #[must_use]
    pub fn into_raw(self) -> runtime::cudaGraphExec_t {
        let graph = ManuallyDrop::new(self);
        graph.as_raw()
    }
}

impl ExecutableGraph {
    pub(crate) fn bind_context(&self) -> Result<()> {
        if let Some(context) = &self.context {
            context.bind()?;
        }
        Ok(())
    }

    pub(crate) unsafe fn from_raw_with_graph(
        handle: runtime::cudaGraphExec_t,
        context: Option<Arc<Context>>,
        source_graph_id: Option<GraphId>,
        source_graph: Option<Arc<GraphInner>>,
        retained: RetainedResources,
    ) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle,
            context,
            source_graph_id,
            source_graph,
            retained,
        })
    }

    pub(crate) fn check_node(&self, node: &GraphNode) -> Result<()> {
        self.bind_context()?;
        if !matches!((self.source_graph_id, node.graph_id), (Some(source_id), Some(node_id)) if node_id == source_id)
        {
            return Err(Error::GraphNodeMismatch);
        }
        Ok(())
    }

    pub(crate) fn retain_module(&mut self, module: &Module) {
        self.retained.retain_module(module);
    }

    fn retain_buffer<T>(&mut self, buffer: &GraphBuffer<T>)
    where
        T: DeviceRepr,
    {
        self.retained.retain_allocation(buffer.retained());
    }

    /// Returns the flags that were passed to instantiation for the given executable graph.
    /// [`GraphInstantiateFlags::UPLOAD`] is not returned because it does not affect the resulting executable graph.
    ///
    /// Graph objects are not threadsafe.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the graph operation, if a previous asynchronous launch reported an error, or if CUDA reports runtime initialization diagnostics.
    /// Callbacks must not call CUDA functions; see [`Stream::add_callback`].
    pub fn flags(&self) -> Result<GraphInstantiateFlags> {
        self.bind_context()?;
        let mut flags = 0;
        // SAFETY: `cudaGraphExecGetFlags` receives live handles owned by the wrappers and valid output locations.
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
    /// Returns an error if CUDA rejects the graph operation, if a previous asynchronous launch reported an error, or if CUDA reports runtime initialization diagnostics.
    /// Callbacks must not call CUDA functions; see [`Stream::add_callback`].
    /// # Safety
    /// Retain every referenced allocation and native owner until execution completes.
    /// Order conflicting access through external aliases and uphold every node's value contract.
    pub unsafe fn launch(&self, stream: &Stream) -> Result<()> {
        if let Some(context) = &self.context
            && stream.context() != context
        {
            return Err(Error::StreamContextMismatch);
        }

        self.bind_context()?;
        // SAFETY: `cudaGraphLaunch` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(runtime::cudaGraphLaunch(self.as_raw(), stream.as_raw()))?;
        }
        Ok(())
    }

    /// Returns a reusable operation object that launches this executable graph.
    #[must_use]
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
    /// Returns an error if CUDA rejects the graph operation, if a previous asynchronous launch reported an error, or if CUDA reports runtime initialization diagnostics.
    pub fn upload(&self, stream: &Stream) -> Result<()> {
        if let Some(context) = &self.context
            && stream.context() != context
        {
            return Err(Error::StreamContextMismatch);
        }

        self.bind_context()?;
        // SAFETY: `cudaGraphUpload` receives live handles owned by the wrappers.
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
    ///   + A node whose kernel function originally did not make device-side update calls cannot be updated to a kernel function that makes device-side update calls.
    ///   + A cooperative node cannot be updated to a non-cooperative node, and vice-versa.
    ///   + If the graph was instantiated with [`GraphInstantiateFlags::USE_NODE_PRIORITY`], the priority attribute cannot change.
    ///     Equality is checked on the originally requested priority values, before they are clamped to the device's supported range.
    ///   + If this executable graph was not instantiated for device launch, a node whose kernel function originally did not use device-side [`ExecutableGraph::launch`] cannot be updated to a kernel function that uses device-side [`ExecutableGraph::launch`] unless the node resides on the same device as nodes which contained such calls at instantiate-time.
    ///     If no such calls were present at instantiation, these updates cannot be performed at all.
    ///   + Neither the source graph nor this executable graph may contain device-updatable kernel nodes.
    /// * Memset and memcpy nodes:
    ///   + The CUDA device(s) to which the operand(s) was allocated/mapped cannot change.
    ///   + The source/destination memory must be allocated from the same contexts as the original source/destination memory.
    ///   + For 2D memsets, only address and assigned value may be updated.
    ///   + For 1D memsets, updating dimensions is also allowed, but may fail if the resulting operation does not map onto the work resources already allocated for the node.
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
    ///   The dependencies are paired based on edge order and a dependency does not match when the nodes are already paired based on other edges examined in the graph.
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
    /// Returns an error if CUDA rejects the graph update, if the update violates instantiated graph update constraints, or if a previous asynchronous launch reported an error.
    /// CUDA may also return initialization-related errors such as [`crate::error::Status::NotInitialized`], [`crate::error::Status::CallRequiresNewerDriver`], or [`crate::error::Status::NoDevice`] if this call initializes internal runtime state.
    /// Callbacks must not call CUDA functions; see [`Stream::add_callback`].
    pub fn update(&mut self, graph: &Graph) -> Result<ExecutableGraphUpdate> {
        if let (Some(exec_context), Some(graph_context)) = (&self.context, &graph.context)
            && exec_context.as_ref() != graph_context.as_ref()
        {
            return Err(Error::GraphContextMismatch);
        }

        self.bind_context()?;
        let mut result_info = runtime::cudaGraphExecUpdateResultInfo::default();
        // SAFETY: `cudaGraphExecUpdate` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(runtime::cudaGraphExecUpdate(
                self.as_raw(),
                graph.as_raw(),
                &raw mut result_info,
            ))?;
        }

        self.retained.extend(graph.retained_resources()?);
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
    /// * A node whose kernel function originally did not make device-side update calls cannot be updated to a kernel function that makes device-side update calls.
    /// * If this executable graph was not instantiated for device launch, a node whose kernel function originally did not use device-side [`ExecutableGraph::launch`] cannot be updated to a kernel function that uses device-side [`ExecutableGraph::launch`] unless the node resides on the same device as nodes which contained such calls at instantiate-time.
    ///   If no such calls were present at instantiation, these updates cannot be performed at all.
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
    /// Launch attributes remain as instantiated; changing them requires a full graph update or re-instantiation.
    ///
    /// # Safety
    ///
    /// Arguments must match `function`'s ABI and satisfy its value and access requirements.
    /// The function must belong to the node's execution context.
    /// Keep its code owner alive until every queued graph execution completes.
    /// The executable's existing launch attributes must support the new function and geometry.
    /// CUDA copies the kernel argument values during this call and stores those copied values in the executable graph for future launches.
    /// If an argument value is itself a pointer, only the pointer address is copied.
    /// The caller must ensure every copied pointer value remains valid for every future launch that can execute this node.
    /// Mutable pointer arguments must also remain exclusive for the work ordered by those launches.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the graph operation, if a previous asynchronous launch reported an error, or if CUDA reports runtime initialization diagnostics.
    /// Callbacks must not call CUDA functions; see [`Stream::add_callback`].
    pub unsafe fn set_kernel_node_params<'a, P>(
        &mut self,
        node: &GraphNode,
        function: DeviceFunction,
        geometry: &LaunchGeometry,
        params: P,
    ) -> Result<()>
    where
        P: KernelLaunchArgs<'a>,
    {
        self.check_node(node)?;
        // SAFETY: the caller upholds this function's safety contract.
        params.with_encoded_arguments(|mut arguments| unsafe {
            let params = runtime::cudaKernelNodeParams {
                func: function.as_raw().cast(),
                gridDim: geometry.grid_dim().into(),
                blockDim: geometry.block_dim().into(),
                sharedMemBytes: geometry.shared_memory_bytes_u32(),
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
    /// CUDA stores the raw source and destination addresses in the executable graph for future launches.
    /// The caller must ensure `params` remains valid according to [`MemoryCopy1DNodeParams::new`] for every future launch that can execute this node.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the graph operation, if a previous asynchronous launch reported an error, or if CUDA reports runtime initialization diagnostics.
    /// Callbacks must not call CUDA functions; see [`Stream::add_callback`].
    pub unsafe fn set_memory_copy_node_1d_params(
        &mut self,
        node: &GraphNode,
        params: &MemoryCopy1DNodeParams,
    ) -> Result<()> {
        self.check_node(node)?;
        // SAFETY: `cudaGraphExecMemcpyNodeSetParams1D` receives live handles owned by the wrappers.
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
    /// The node copies `src.byte_len()` bytes.
    /// `dst` must have at least that many bytes.
    ///
    /// # Safety
    ///
    /// CUDA stores the raw source and destination addresses in the executable graph for future launches.
    /// The caller must ensure `dst` and `src` remain valid for every future launch that can execute this node.
    /// `dst` must not be accessed through another mutable path while graph launches using this node can write it.
    ///
    /// # Errors
    ///
    /// Returns an error if `dst` is smaller than `src`, if CUDA rejects the graph operation, if a previous asynchronous launch reported an error, or if CUDA reports runtime initialization diagnostics.
    pub unsafe fn set_memory_copy_node_1d_device_to_device<D, S>(
        &mut self,
        node: &GraphNode,
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
        unsafe { self.set_memory_copy_node_1d_params(node, &params) }
    }

    /// Updates a memcpy node to copy between graph-retained buffers.
    ///
    /// The node copies `src.byte_len()` bytes.
    /// `dst` must have at least that many bytes.
    /// The executable graph retains both allocations so future launches cannot outlive the baked CUDA pointer values.
    ///
    /// # Errors
    ///
    /// Returns an error if `dst` is smaller than `src`, if `node` does not belong to the graph used to instantiate this executable graph, if CUDA rejects the graph update, if a previous asynchronous launch reported an error, or if CUDA reports runtime initialization diagnostics.
    ///
    /// # Safety
    /// Keep source elements valid for every launch, and order their producers before replay.
    /// Keep destination accesses exclusive through completion, including failed submissions.
    /// Allocation retention preserves addresses; initialization and access ordering remain caller obligations.
    /// Serialize host updates with other operations on this graph executable.
    pub unsafe fn set_buffer_memory_copy_node_1d_device_to_device<T, D>(
        &mut self,
        node: &GraphNode,
        dst: &mut GraphBuffer<D>,
        src: &GraphBuffer<T>,
    ) -> Result<()>
    where
        T: DeviceRepr,
        D: DeviceRepr,
        GraphBuffer<D>: DeviceOutput<T>,
    {
        if let Some(exec_context) = &self.context
            && exec_context != dst.context()
        {
            return Err(Error::GraphContextMismatch);
        }

        if let Some(exec_context) = &self.context
            && exec_context != src.context()
        {
            return Err(Error::GraphContextMismatch);
        }

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
    /// CUDA stores the raw source and destination addresses in the executable graph for future launches.
    /// The caller must ensure `params` remains valid according to [`MemoryCopy3DNodeParams`] for every future launch that can execute this node.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the graph operation, if a previous asynchronous launch reported an error, or if CUDA reports runtime initialization diagnostics.
    /// Callbacks must not call CUDA functions; see [`Stream::add_callback`].
    pub unsafe fn set_memory_copy_node_params(
        &mut self,
        node: &GraphNode,
        params: &MemoryCopy3DNodeParams,
    ) -> Result<()> {
        self.check_node(node)?;
        let params = params.into();
        // SAFETY: `cudaGraphExecMemcpyNodeSetParams` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
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
    /// CUDA stores the raw symbol and source pointer in the executable graph for future launches.
    /// The caller must ensure `params` remains valid according to [`MemoryCopyToSymbolNodeParams::new`] for every future launch that can execute this node.
    ///
    /// # Errors
    ///
    /// Returns an error if the node does not belong to this executable’s source graph, context binding fails, or CUDA rejects the symbol copy update.
    pub unsafe fn set_memory_copy_node_to_symbol_params(
        &mut self,
        node: &GraphNode,
        params: &MemoryCopyToSymbolNodeParams,
    ) -> Result<()> {
        self.check_node(node)?;
        // SAFETY: `cudaGraphExecMemcpyNodeSetParamsToSymbol` receives live handles owned by the wrappers.
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
    /// CUDA stores the raw destination and symbol pointer in the executable graph for future launches.
    /// The caller must ensure `params` remains valid according to [`MemoryCopyFromSymbolNodeParams::new`] for every future launch that can execute this node.
    ///
    /// # Errors
    ///
    /// Returns an error if the node does not belong to this executable’s source graph, context binding fails, or CUDA rejects the symbol copy update.
    pub unsafe fn set_memory_copy_node_from_symbol_params(
        &mut self,
        node: &GraphNode,
        params: &MemoryCopyFromSymbolNodeParams,
    ) -> Result<()> {
        self.check_node(node)?;
        // SAFETY: `cudaGraphExecMemcpyNodeSetParamsFromSymbol` receives live handles owned by the wrappers.
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
    /// CUDA stores the raw destination address in the executable graph for future launches.
    /// The caller must ensure `params` remains valid according to [`MemorySetNodeParams::new`] for every future launch that can execute this node.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the graph operation, if a previous asynchronous launch reported an error, or if CUDA reports runtime initialization diagnostics.
    /// Callbacks must not call CUDA functions; see [`Stream::add_callback`].
    pub unsafe fn set_memory_set_node_params(
        &mut self,
        node: &GraphNode,
        params: &MemorySetNodeParams,
    ) -> Result<()> {
        self.check_node(node)?;
        let params = params.into();
        // SAFETY: `cudaGraphExecMemsetNodeSetParams` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
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
    /// CUDA stores the raw callback function and user-data pointer in the executable graph for future launches.
    /// The caller must ensure `params` remains valid according to [`HostNodeParams::new`] for every future launch that can execute this node.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the graph operation, if a previous asynchronous launch reported an error, or if CUDA reports runtime initialization diagnostics.
    /// Callbacks must not call CUDA functions; see [`Stream::add_callback`].
    pub unsafe fn set_host_node_params(
        &mut self,
        node: &GraphNode,
        params: &HostNodeParams,
    ) -> Result<()> {
        self.check_node(node)?;
        let params = params.into();
        // SAFETY: `cudaGraphExecHostNodeSetParams` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
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
    /// Returns an error if CUDA rejects the graph operation, if a previous asynchronous launch reported an error, or if CUDA reports runtime initialization diagnostics.
    /// Callbacks must not call CUDA functions; see [`Stream::add_callback`].
    pub fn set_event_record_node_event(&mut self, node: &GraphNode, event: &Event) -> Result<()> {
        self.check_node(node)?;
        if let Some(context) = &self.context
            && context != event.context()
        {
            return Err(Error::GraphContextMismatch);
        }

        // SAFETY: `cudaGraphExecEventRecordNodeSetEvent` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(runtime::cudaGraphExecEventRecordNodeSetEvent(
                self.as_raw(),
                node.as_raw(),
                event.as_raw(),
            ))?;
        }
        Ok(())
    }

    /// Validates the recorded node identity and context before updating a child graph.
    /// This lets callers reject an incompatible executable before changing the source graph.
    /// CUDA determines topology and parameter compatibility when the update is applied.
    ///
    /// # Errors
    /// Returns an error if the node does not belong to this executable's source graph, the recorded CUDA contexts differ, or binding the executable's context fails.
    pub fn validate_child_graph_node_update(
        &self,
        node: &GraphNode,
        child_graph: &Graph,
    ) -> Result<()> {
        self.check_node(node)?;
        if let (Some(exec_context), Some(child_context)) = (&self.context, &child_graph.context)
            && exec_context.as_ref() != child_context.as_ref()
        {
            return Err(Error::GraphContextMismatch);
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
    /// Returns an error if CUDA rejects the graph operation, if a previous asynchronous launch reported an error, or if CUDA reports runtime initialization diagnostics.
    /// Callbacks must not call CUDA functions; see [`Stream::add_callback`].
    pub fn set_child_graph_node(&mut self, node: &GraphNode, child_graph: &Graph) -> Result<()> {
        self.validate_child_graph_node_update(node, child_graph)?;
        // SAFETY: `cudaGraphExecChildGraphNodeSetParams` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(runtime::cudaGraphExecChildGraphNodeSetParams(
                self.as_raw(),
                node.as_raw(),
                child_graph.as_raw(),
            ))?;
        }

        self.retained.extend(child_graph.retained_resources()?);
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
    /// Returns an error if CUDA rejects the graph operation, if a previous asynchronous launch reported an error, or if CUDA reports runtime initialization diagnostics.
    /// Callbacks must not call CUDA functions; see [`Stream::add_callback`].
    pub fn set_event_wait_node_event(&mut self, node: &GraphNode, event: &Event) -> Result<()> {
        self.check_node(node)?;
        // SAFETY: `cudaGraphExecEventWaitNodeSetEvent` receives live handles owned by the wrappers.
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
    /// Returns an error if CUDA rejects the graph operation, if a previous asynchronous launch reported an error, or if CUDA reports runtime initialization diagnostics.
    /// Callbacks must not call CUDA functions; see [`Stream::add_callback`].
    fn set_node_enabled(&mut self, node: &GraphNode, enabled: bool) -> Result<()> {
        self.check_node(node)?;
        // SAFETY: `cudaGraphNodeSetEnabled` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(runtime::cudaGraphNodeSetEnabled(
                self.as_raw(),
                node.as_raw(),
                u32::from(enabled),
            ))?;
        }
        Ok(())
    }

    /// Enables the specified node in this executable graph.
    ///
    /// # Errors
    ///
    /// Returns an error if the node does not belong to this executable’s source graph, context binding fails, or CUDA rejects changing the node’s enabled state.
    pub fn enable_node(&mut self, node: &GraphNode) -> Result<()> {
        self.set_node_enabled(node, true)
    }

    /// Disables the specified node in this executable graph.
    ///
    /// # Errors
    ///
    /// Returns an error if the node does not belong to this executable’s source graph, context binding fails, or CUDA rejects changing the node’s enabled state.
    pub fn disable_node(&mut self, node: &GraphNode) -> Result<()> {
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
    /// Returns an error if CUDA rejects the graph operation, if a previous asynchronous launch reported an error, or if CUDA reports runtime initialization diagnostics.
    /// Callbacks must not call CUDA functions; see [`Stream::add_callback`].
    pub fn is_node_enabled(&self, node: &GraphNode) -> Result<bool> {
        self.check_node(node)?;
        let mut enabled = 0;
        // SAFETY: `cudaGraphNodeGetEnabled` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(runtime::cudaGraphNodeGetEnabled(
                self.as_raw(),
                node.as_raw(),
                &raw mut enabled,
            ))?;
        }
        Ok(enabled != 0)
    }

    /// Returns the native executable handle without transferring ownership.
    #[must_use]
    pub const fn as_raw(&self) -> runtime::cudaGraphExec_t {
        self.handle
    }

    /// Returns the retained execution context, when present.
    #[must_use]
    pub fn context(&self) -> Option<&Arc<Context>> {
        self.context.as_ref()
    }

    /// Transfers the executable, source graph, context, and retained resource owners without destruction.
    ///
    /// Keep the returned parts alive through completed native launches and handle destruction, or re-adopt them.
    #[must_use]
    pub fn into_raw(self) -> RawExecutableGraphParts {
        let graph = ManuallyDrop::new(self);
        // SAFETY: ManuallyDrop prevents all fields from being dropped, and each owner is transferred once.
        unsafe {
            RawExecutableGraphParts {
                handle: graph.handle,
                context: ptr::read(&raw const graph.context),
                retained: ptr::read(&raw const graph.retained),
                source_graph: ptr::read(&raw const graph.source_graph),
                source_graph_id: graph.source_graph_id,
            }
        }
    }

    /// Re-adopts an executable with its original source graph, context, and retained resources.
    ///
    /// # Safety
    /// The executable handle must still be live and unchanged and have no other destroying owner.
    /// Every dependency must remain retained by these parts or its original unsafe caller contract.
    ///
    /// # Errors
    ///
    /// Returns an error if the native handle is null.
    pub unsafe fn from_raw_parts(parts: RawExecutableGraphParts) -> Result<Self> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            Self::from_raw_with_graph(
                parts.handle,
                parts.context,
                parts.source_graph_id,
                parts.source_graph,
                parts.retained,
            )
        }
    }
}

impl ExecutableGraphLaunchOperation<'_> {
    /// Enqueues this graph launch in `stream`.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the graph operation, if `stream` belongs to a different context, if a previous asynchronous launch reported an error, or if CUDA reports runtime initialization diagnostics.
    /// # Safety
    /// Follow [`ExecutableGraph::launch`]'s resource and access contract.
    pub unsafe fn enqueue(self, stream: &Stream) -> Result<()> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { self.graph.launch(stream) }
    }

    /// Returns the executable graph borrowed by this launch descriptor.
    #[must_use]
    pub const fn graph(&self) -> &ExecutableGraph {
        self.graph
    }
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
