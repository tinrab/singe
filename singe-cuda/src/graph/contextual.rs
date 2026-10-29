//! Graph nodes with explicit normal or green execution contexts.
use std::{ptr, rc::Rc, slice, sync::Arc};

use singe_core::impl_enum_display;
use singe_cuda_sys::driver as sys;

use crate::{
    context::Context,
    driver::entry_points::{cuGraphAddNode_v3, cuGraphNodeSetParams_v2},
    error::Result,
    event::Event,
    graph::{ExecutableGraph, Graph, GraphInstantiateFlags, GraphNode, raw::HostNodeParams},
    green::{GreenContext, GreenStream},
    module::{KernelLaunchArgs, LaunchConfig},
    semaphore::ExternalSemaphore,
    stream::Stream,
    try_ffi,
    types::DeviceFunction,
};

/// Preferred execution context of a graph node.
#[derive(Debug, Clone)]
pub enum ExecutionContext {
    /// An ordinary CUDA context.
    Context(Arc<Context>),
    /// A green context with provisioned GPU resources.
    Green(Rc<GreenContext>),
}

impl ExecutionContext {
    fn raw(&self) -> sys::CUcontext {
        match self {
            Self::Context(context) => context.as_raw(),
            Self::Green(context) => context.as_raw().cast(),
        }
    }

    fn kernel_context(&self) -> Result<sys::CUcontext> {
        match self {
            Self::Context(context) => Ok(context.as_raw()),
            Self::Green(context) => context.context_handle(),
        }
    }
}

/// Node parameters whose resources are retained by contextual graphs.
#[derive(Debug, Clone)]
pub enum NodeParameters {
    /// Function and user data to call when the node executes.
    Host {
        /// Native callback and borrowed user-data pointer.
        parameters: HostNodeParams,
        /// Blocking or spinning behavior of the host-task execution thread.
        synchronization_mode: HostTaskSyncMode,
    },
    /// Event to record when the node executes.
    EventRecord(Rc<Event>),
    /// External semaphore and fence value to signal.
    SemaphoreSignal {
        /// External semaphore retained through graph execution.
        semaphore: Rc<ExternalSemaphore>,
        /// Timeline fence value to signal.
        value: u64,
    },
    /// External semaphore and fence value to wait for.
    SemaphoreWait {
        /// External semaphore retained through graph execution.
        semaphore: Rc<ExternalSemaphore>,
        /// Timeline fence value required before dependent graph work can proceed.
        value: u64,
    },
}

impl NodeParameters {
    fn with_raw<T>(
        &self,
        context: &ExecutionContext,
        f: impl FnOnce(&mut sys::CUgraphNodeParams) -> Result<T>,
    ) -> Result<T> {
        let mut raw = sys::CUgraphNodeParams::default();
        let mut semaphore_handle;
        let mut signal = sys::CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS::default();
        let mut wait = sys::CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS::default();
        match self {
            Self::Host {
                parameters,
                synchronization_mode,
            } => {
                raw.type_ = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_HOST;
                raw.__bindgen_anon_1.host = sys::CUDA_HOST_NODE_PARAMS_v2 {
                    fn_: parameters.function().as_raw(),
                    userData: parameters.user_data().cast(),
                    syncMode: *synchronization_mode as u32,
                    __bindgen_anon_1: sys::CUDA_HOST_NODE_PARAMS_v2_st__bindgen_ty_1 {
                        ctx: context.raw(),
                    },
                };
            }
            Self::EventRecord(event) => {
                raw.type_ = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EVENT_RECORD;
                raw.__bindgen_anon_1.eventRecord = sys::CUDA_EVENT_RECORD_NODE_PARAMS {
                    event: event.as_raw(),
                    __bindgen_anon_1: sys::CUDA_EVENT_RECORD_NODE_PARAMS_st__bindgen_ty_1 {
                        ctx: context.raw(),
                    },
                };
            }
            Self::SemaphoreSignal { semaphore, value } => {
                semaphore_handle = semaphore.as_raw();
                signal.params.fence.value = *value;
                raw.type_ = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL;
                raw.__bindgen_anon_1.extSemSignal = sys::CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v2 {
                    extSemArray: &raw mut semaphore_handle,
                    paramsArray: &raw const signal,
                    numExtSems: 1,
                    __bindgen_anon_1: sys::CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v2_st__bindgen_ty_1 {
                        ctx: context.raw(),
                    },
                };
            }
            Self::SemaphoreWait { semaphore, value } => {
                semaphore_handle = semaphore.as_raw();
                wait.params.fence.value = *value;
                raw.type_ = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT;
                raw.__bindgen_anon_1.extSemWait = sys::CUDA_EXT_SEM_WAIT_NODE_PARAMS_v2 {
                    extSemArray: &raw mut semaphore_handle,
                    paramsArray: &raw const wait,
                    numExtSems: 1,
                    __bindgen_anon_1: sys::CUDA_EXT_SEM_WAIT_NODE_PARAMS_v2_st__bindgen_ty_1 {
                        ctx: context.raw(),
                    },
                };
            }
        }
        f(&mut raw)
    }
}

/// A node retaining its chosen context and external resources.
#[derive(Debug, Clone)]
pub struct ContextNode {
    node: GraphNode,
    context: ExecutionContext,
    parameters: Option<NodeParameters>,
}

impl ContextNode {
    /// Returns the borrowed native node handle without independently retaining its graph.
    #[must_use]
    pub fn as_raw(&self) -> sys::CUgraphNode {
        self.node.as_raw()
    }
}

/// A graph that may retain thread-affine green contexts.
#[derive(Debug)]
pub struct ContextGraph {
    graph: Graph,
    nodes: Vec<ContextNode>,
}

impl ContextGraph {
    /// Creates an empty graph in `context` for nodes with explicit ordinary or green execution contexts.
    ///
    /// # Errors
    ///
    /// Propagates context binding and native graph creation errors from [`Context::create_graph`].
    pub fn create(context: &Arc<Context>) -> Result<Self> {
        Ok(Self {
            graph: context.create_graph()?,
            nodes: Vec::new(),
        })
    }

    /// Creates a new node described by its type-specific parameters.
    ///
    /// # Safety
    /// Host callbacks and their data must remain valid for every execution.
    /// Callbacks must not call CUDA. External resources must not be destroyed
    /// or manipulated through raw handles while referenced by the graph.
    ///
    /// # Errors
    ///
    /// Returns an error for dependencies belonging to another graph, an unavailable driver entry point, or a native node-creation failure such as incompatible parameters or execution context. Context and resource owners are retained only after creation succeeds.
    pub unsafe fn add_node(
        &mut self,
        context: ExecutionContext,
        dependencies: &[ContextNode],
        parameters: NodeParameters,
    ) -> Result<ContextNode> {
        let dependencies = dependencies
            .iter()
            .map(|n| n.node.clone())
            .collect::<Vec<_>>();
        self.graph.check_nodes(&dependencies)?;
        let handles = dependencies
            .iter()
            .map(GraphNode::as_raw)
            .collect::<Vec<_>>();
        let mut handle = ptr::null_mut();
        // SAFETY: the caller upholds this function's safety contract.
        parameters.with_raw(&context, |raw| unsafe {
            cuGraphAddNode_v3(
                &raw mut handle,
                self.graph.as_raw(),
                handles.as_ptr(),
                ptr::null(),
                handles.len() as _,
                raw,
            )
        })?;
        let node = ContextNode {
            node: self.graph.node_from_raw(handle),
            context,
            parameters: Some(parameters),
        };
        self.nodes.push(node.clone());
        Ok(node)
    }

    /// Updates a node's type-specific parameters and preferred context.
    ///
    /// # Safety
    /// The callback and data lifetime requirements of `add_node` also apply.
    ///
    /// # Errors
    ///
    /// Returns an error if `node` belongs to another graph, the driver entry point is unavailable, or CUDA rejects the parameter or execution-context change. Retained node parameters change only after native success.
    pub unsafe fn set_node_parameters(
        &mut self,
        node: &mut ContextNode,
        context: ExecutionContext,
        parameters: NodeParameters,
    ) -> Result<()> {
        self.graph.check_nodes(slice::from_ref(&node.node))?;
        // SAFETY: the caller upholds this function's safety contract.
        parameters.with_raw(&context, |raw| unsafe {
            cuGraphNodeSetParams_v2(node.as_raw(), raw)
        })?;
        node.context = context;
        node.parameters = Some(parameters);
        if let Some(retained) = self.nodes.iter_mut().find(|n| n.node == node.node) {
            *retained = node.clone();
        }
        Ok(())
    }

    /// Adds a kernel node with an explicit execution context.
    ///
    /// # Safety
    /// Arguments must match the function ABI and satisfy its value and access requirements.
    /// The function must be valid in `context` and its code owner must outlive every queued execution.
    /// Pointed-to storage must remain valid for every execution, with exclusive writable access.
    ///
    /// # Errors
    ///
    /// Returns an error for dependencies belonging to another graph, an unavailable driver entry point, a failed green-context conversion, or CUDA rejecting the kernel node or launch attributes. If setting an attribute fails after creation, cleanup attempts to destroy the new native node.
    pub unsafe fn add_kernel<'a, P: KernelLaunchArgs<'a>>(
        &mut self,
        context: ExecutionContext,
        dependencies: &[ContextNode],
        function: DeviceFunction,
        config: &LaunchConfig,
        parameters: P,
    ) -> Result<ContextNode> {
        let dependencies = dependencies
            .iter()
            .map(|n| n.node.clone())
            .collect::<Vec<_>>();
        self.graph.check_nodes(&dependencies)?;
        let handles = dependencies
            .iter()
            .map(GraphNode::as_raw)
            .collect::<Vec<_>>();
        let mut handle = ptr::null_mut();
        let kernel_context = context.kernel_context()?;
        // SAFETY: the caller upholds this function's safety contract.
        parameters.with_encoded_arguments(|mut arguments| unsafe {
            let mut raw = sys::CUgraphNodeParams {
                type_: sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL,
                ..Default::default()
            };
            raw.__bindgen_anon_1.kernel = sys::CUDA_KERNEL_NODE_PARAMS_v3 {
                func: function.as_raw(),
                gridDimX: config.grid_dim().x,
                gridDimY: config.grid_dim().y,
                gridDimZ: config.grid_dim().z,
                blockDimX: config.block_dim().x,
                blockDimY: config.block_dim().y,
                blockDimZ: config.block_dim().z,
                sharedMemBytes: config.shared_memory_bytes_u32(),
                kernelParams: arguments.as_mut_ptr().cast(),
                ctx: kernel_context,
                ..Default::default()
            };
            cuGraphAddNode_v3(
                &raw mut handle,
                self.graph.as_raw(),
                handles.as_ptr(),
                ptr::null(),
                handles.len() as _,
                &raw mut raw,
            )
        })?;
        let node = ContextNode {
            node: self.graph.node_from_raw(handle),
            context,
            parameters: None,
        };
        for attribute in config.attributes() {
            if let Err(error) = node.node.set_launch_attribute(*attribute) {
                // SAFETY: `cuGraphDestroyNode` receives only plain values.
                unsafe {
                    let _ = sys::cuGraphDestroyNode(handle);
                }
                return Err(error);
            }
        }
        self.nodes.push(node.clone());
        Ok(node)
    }

    /// Instantiates the graph, retaining its execution contexts and resources.
    ///
    /// # Errors
    ///
    /// Propagates native graph-instantiation errors from [`Graph::instantiate_with_flags`], including invalid nodes, incompatible contexts, or unsupported instantiation flags.
    pub fn instantiate(&self, flags: GraphInstantiateFlags) -> Result<ContextExecutableGraph> {
        Ok(ContextExecutableGraph {
            executable: self.graph.instantiate_with_flags(flags)?,
            nodes: self.nodes.clone(),
        })
    }
}

/// An executable graph retaining thread-affine contexts and external resources.
#[derive(Debug)]
pub struct ContextExecutableGraph {
    executable: ExecutableGraph,
    nodes: Vec<ContextNode>,
}

impl ContextExecutableGraph {
    /// Enqueues the retained executable on an ordinary CUDA stream.
    ///
    /// # Safety
    /// Retain node storage and exclude conflicting accesses through completion.
    ///
    /// # Errors
    ///
    /// Propagates stream-context validation, context binding, and native launch errors from [`ExecutableGraph::launch`].
    pub unsafe fn launch(&self, stream: &Stream) -> Result<()> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { self.executable.launch(stream) }
    }

    /// Enqueues the retained executable on a green-context stream.
    ///
    /// # Safety
    /// Retain node storage and exclude conflicting accesses through completion.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects launching this executable on the supplied green-context stream, including incompatible graph or stream state and prior asynchronous failures.
    pub unsafe fn launch_on_green(&self, stream: &GreenStream) -> Result<()> {
        // SAFETY: `cuGraphLaunch` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::cuGraphLaunch(
                self.executable.as_raw(),
                stream.as_raw()
            ))
        }
    }

    /// Borrows the retained node and resource snapshot from graph instantiation.
    #[must_use]
    pub fn nodes(&self) -> &[ContextNode] {
        &self.nodes
    }
}

/// Synchronization mode for a host task's execution thread.
#[derive(Debug, Clone, Copy)]
#[repr(u32)]
#[non_exhaustive]
pub enum HostTaskSyncMode {
    /// The execution thread will block until new host tasks are ready to run.
    Blocking = sys::CUhostTaskSyncMode::CU_HOST_TASK_BLOCKING as _,
    /// The execution thread will spin wait until new host tasks are ready to run.
    SpinWait = sys::CUhostTaskSyncMode::CU_HOST_TASK_SPINWAIT as _,
}

impl_enum_display!(HostTaskSyncMode, {
    HostTaskSyncMode::Blocking => "CU_HOST_TASK_BLOCKING",
    HostTaskSyncMode::SpinWait => "CU_HOST_TASK_SPINWAIT",
});
