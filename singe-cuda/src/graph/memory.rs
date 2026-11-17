//! Graph nodes for stream memory synchronization.

use std::ptr;

use singe_cuda_sys::driver as sys;

use crate::{
    error::Result,
    graph::{ExecutableGraph, Graph, GraphNode},
    stream::memory::{MemoryOperation, native_batch},
    try_ffi,
    utility::{to_u32, to_u64},
};

impl Graph {
    /// Adds a batch memory-operation node to this graph.
    ///
    /// # Safety
    ///
    /// Follow [`crate::stream::Stream::memory_operations`]'s address and progress contracts for every replay.
    ///
    /// # Errors
    ///
    /// Returns `Error::OutOfRange` for 256 or more operations, `Error::NullPointer` for null wait or write addresses, or `Error::Misaligned` for misaligned ones.
    /// Also returns an error for a dependency from another graph, an unrepresentable dependency count, or CUDA rejecting batch-node creation.
    pub unsafe fn add_memory_operations_node(
        &mut self,
        dependencies: &[GraphNode],
        operations: &[MemoryOperation],
    ) -> Result<GraphNode> {
        self.check_nodes(dependencies)?;
        let context = self.context();
        let mut native = native_batch(operations)?;
        let params = batch_params(context.as_raw(), &mut native)?;
        let dependencies: Vec<_> = dependencies.iter().map(GraphNode::as_raw).collect();
        let mut node = ptr::null_mut();
        // SAFETY: `node` is a local output, `check_nodes` confirmed the dependencies belong to this exclusively borrowed graph, `native_batch` validated the operations in `native`, which `params` points into and which outlives the call, and the caller keeps the addresses valid for every replay.
        unsafe {
            try_ffi!(sys::cuGraphAddBatchMemOpNode(
                &raw mut node,
                self.as_raw(),
                dependencies.as_ptr(),
                to_u64(dependencies.len(), "graph dependency count")?,
                &raw const params
            ))?;
        }
        Ok(self.node_from_raw(node))
    }

    /// Replaces an existing batch node's operations.
    ///
    /// # Safety
    ///
    /// The new addresses and synchronization must satisfy the replay contract of [`Self::add_memory_operations_node`].
    ///
    /// # Errors
    ///
    /// Returns `Error::OutOfRange` for 256 or more operations, `Error::NullPointer` for null wait or write addresses, or `Error::Misaligned` for misaligned ones.
    /// Also returns an error if the node belongs to another graph or CUDA rejects the new batch parameters.
    pub unsafe fn set_memory_operations_node(
        &mut self,
        node: &GraphNode,
        operations: &[MemoryOperation],
    ) -> Result<()> {
        self.check_node(node)?;
        let context = self.context();
        let mut native = native_batch(operations)?;
        let params = batch_params(context.as_raw(), &mut native)?;
        // SAFETY: `check_node` confirmed `node` belongs to this exclusively borrowed graph, `native_batch` validated the operations in `native`, which outlives the call, and the caller keeps the new addresses valid for every replay.
        unsafe {
            try_ffi!(sys::cuGraphBatchMemOpNodeSetParams(
                node.as_raw(),
                &raw const params
            ))
        }
    }
}

impl ExecutableGraph {
    /// Replaces a batch node's addresses, values and wait comparisons in this executable.
    /// Operation count, kinds, context, barriers and flush flags must remain unchanged.
    ///
    /// # Safety
    ///
    /// Keep replacement storage live and uphold synchronization for all subsequent launches.
    ///
    /// # Errors
    ///
    /// Returns `Error::OutOfRange` for 256 or more operations, `Error::NullPointer` for null wait or write addresses, or `Error::Misaligned` for misaligned ones.
    /// Also returns an error if the node does not belong to the executable's source graph or CUDA rejects the executable update, including changes forbidden for an existing batch node.
    pub unsafe fn set_memory_operations_node(
        &mut self,
        node: &GraphNode,
        operations: &[MemoryOperation],
    ) -> Result<()> {
        self.check_node(node)?;
        let context = self.context();
        let mut native = native_batch(operations)?;
        let params = batch_params(context.as_raw(), &mut native)?;
        // SAFETY: `self` owns the live executable, `check_node` confirmed `node` comes from its source graph, `native_batch` validated the operations in `native`, which outlives the call, and the caller keeps the new addresses valid for every later launch.
        unsafe {
            try_ffi!(sys::cuGraphExecBatchMemOpNodeSetParams(
                self.as_raw(),
                node.as_raw(),
                &raw const params
            ))
        }
    }
}

fn batch_params(
    context: sys::CUcontext,
    operations: &mut [sys::CUstreamBatchMemOpParams],
) -> Result<sys::CUDA_BATCH_MEM_OP_NODE_PARAMS> {
    Ok(sys::CUDA_BATCH_MEM_OP_NODE_PARAMS {
        ctx: context,
        count: to_u32(operations.len(), "stream memory operation count")?,
        paramArray: operations.as_mut_ptr(),
        flags: 0,
    })
}
