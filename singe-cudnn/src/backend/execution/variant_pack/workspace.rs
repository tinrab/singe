use std::sync::Arc;

use singe_cuda::{
    graph::{
        Graph, GraphHostBuffer, GraphNodeType, MemorySetNodeParams, raw::MemoryCopy1DNodeParams,
    },
    memory::{DeviceMemory, MemoryCopyKind},
    types::DevicePtr,
};

use crate::{
    backend::execution::{ExecutionPlan, variant_pack::VariantPack},
    context::Context,
    error::{Error, Result},
    tensor::WorkspaceInitializer,
};

#[derive(Debug, Clone)]
pub(crate) enum WorkspaceInitialization {
    Copy {
        byte_offset: usize,
        data: Arc<GraphHostBuffer>,
    },
    Fill {
        byte_offset: usize,
        byte_len: usize,
        value: u8,
    },
}

impl WorkspaceInitialization {
    pub(crate) fn new(
        byte_offset: usize,
        byte_len: usize,
        initializer: &WorkspaceInitializer,
    ) -> Self {
        match initializer {
            WorkspaceInitializer::Copy(data) => Self::Copy {
                byte_offset,
                data: Arc::clone(data),
            },
            WorkspaceInitializer::Fill(value) => Self::Fill {
                byte_offset,
                byte_len,
                value: *value,
            },
        }
    }

    fn byte_offset(&self) -> usize {
        match self {
            Self::Copy { byte_offset, .. } | Self::Fill { byte_offset, .. } => *byte_offset,
        }
    }

    fn byte_len(&self) -> usize {
        match self {
            Self::Copy { data, .. } => data.as_bytes().len(),
            Self::Fill { byte_len, .. } => *byte_len,
        }
    }

    fn node_type(&self) -> GraphNodeType {
        match self {
            Self::Copy { .. } => GraphNodeType::Memcpy,
            Self::Fill { .. } => GraphNodeType::Memset,
        }
    }

    fn retain_host_source(&self, graph: &Graph) -> Result<()> {
        if let Self::Copy { data, .. } = self {
            data.retain_on_graph(graph)?;
        }
        Ok(())
    }
}

impl VariantPack {
    pub(crate) fn set_workspace_initializations(
        &mut self,
        initializations: Vec<WorkspaceInitialization>,
    ) -> Result<()> {
        for initialization in &initializations {
            let end = initialization
                .byte_offset()
                .checked_add(initialization.byte_len())
                .ok_or_else(|| Error::out_of_range("workspace initialization bytes"))?;
            if end > self.workspace_prefix_bytes {
                return Err(Error::BufferTooSmall {
                    name: "workspace".into(),
                    required: end,
                    actual: self.workspace_prefix_bytes,
                });
            }
        }

        self.initializations = initializations;
        Ok(())
    }

    pub(crate) fn has_workspace_initializations(&self) -> bool {
        !self.initializations.is_empty()
    }

    pub(crate) fn initialize_workspace(&self, context: &Context) -> Result<()> {
        if self.initializations.is_empty() {
            return Ok(());
        }

        let stream = context.stream_binding();
        for initialization in &self.initializations {
            let pointer = self
                .workspace_base
                .wrapping_add(initialization.byte_offset());
            match initialization {
                WorkspaceInitialization::Copy { data, .. } => {
                    // SAFETY: `set_workspace_initializations` checked that the copy ends inside the prefix of the workspace allocation.
                    // The pack's unsafe constructor requires that allocation to stay valid, and the stream orders the copy before cuDNN reads it.
                    unsafe {
                        data.copy_to_device(DevicePtr::from_raw(pointer.cast()), &stream)?;
                    }
                }
                WorkspaceInitialization::Fill {
                    byte_len, value, ..
                } => {
                    // SAFETY: `set_workspace_initializations` checked that the fill ends inside the prefix of the workspace allocation.
                    // The pack's unsafe constructor requires that allocation to stay valid, and the stream orders the fill before cuDNN reads it.
                    unsafe {
                        DeviceMemory::<u8>::set_async(pointer, *value, *byte_len, &stream)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn copy_parameters(
        &self,
        byte_offset: usize,
        data: &GraphHostBuffer,
    ) -> MemoryCopy1DNodeParams {
        // SAFETY: callers retain `data` on the graph before adding or changing the node, and the destination lies in the checked workspace prefix.
        // The pack's unsafe constructor requires the workspace allocation to stay valid through every replay.
        unsafe {
            MemoryCopy1DNodeParams::new(
                self.workspace_base.wrapping_add(byte_offset).cast(),
                data.as_bytes().as_ptr().cast(),
                data.as_bytes().len(),
                MemoryCopyKind::HostToDevice,
            )
        }
    }

    fn fill_parameters(
        &self,
        byte_offset: usize,
        byte_len: usize,
        value: u8,
    ) -> MemorySetNodeParams {
        // SAFETY: `set_workspace_initializations` checked that this range lies in the workspace prefix.
        // The pack's unsafe constructor requires the workspace allocation to stay valid through every replay.
        unsafe {
            MemorySetNodeParams::new(
                DevicePtr::from_raw(self.workspace_base.wrapping_add(byte_offset).cast()),
                1,
                byte_len,
            )
        }
        .with_value(value.into())
    }

    pub(crate) fn populate_initialized_graph(
        &self,
        context: &Context,
        plan: &ExecutionPlan,
        graph: &mut Graph,
    ) -> Result<()> {
        if !graph.nodes()?.is_empty() {
            return Err(Error::CudaGraphPopulateRequiresEmptyGraph);
        }

        // Publish the initializer chain and native work as one replaceable snapshot.
        let mut initialized = context.cuda_context().create_graph()?;
        self.populate_initializer_chain(context, plan, &mut initialized)?;
        graph.add_child_graph_node(&[], &initialized)?;
        Ok(())
    }

    fn populate_initializer_chain(
        &self,
        context: &Context,
        plan: &ExecutionPlan,
        graph: &mut Graph,
    ) -> Result<()> {
        let child = context.cuda_context().create_graph()?;
        plan.populate_native_graph(context, self, &child)?;
        for initialization in &self.initializations {
            initialization.retain_host_source(graph)?;
        }

        let mut previous = None;
        for initialization in &self.initializations {
            let node = match initialization {
                WorkspaceInitialization::Copy { byte_offset, data } => {
                    // SAFETY: the loop above retained `data` on `graph`, and the destination lies in the checked workspace prefix.
                    unsafe {
                        graph.add_memory_copy_node_1d(
                            previous.as_slice(),
                            &self.copy_parameters(*byte_offset, data),
                        )?
                    }
                }
                WorkspaceInitialization::Fill {
                    byte_offset,
                    byte_len,
                    value,
                } => {
                    // SAFETY: the fill range lies in the checked workspace prefix, and the node runs before the native child it precedes.
                    unsafe {
                        graph.add_memory_set_node(
                            previous.as_slice(),
                            &self.fill_parameters(*byte_offset, *byte_len, *value),
                        )?
                    }
                }
            };
            previous = Some(node);
        }

        graph.add_child_graph_node(previous.as_slice(), &child)?;
        Ok(())
    }

    pub(crate) fn update_initialized_graph(
        &self,
        context: &Context,
        plan: &ExecutionPlan,
        graph: &mut Graph,
    ) -> Result<()> {
        let nodes = graph.nodes()?;
        let [node] = nodes.as_slice() else {
            return Err(invalid_graph());
        };
        if graph.node_type(node)? != GraphNodeType::Graph {
            return Err(invalid_graph());
        }

        let mut replacement = graph.child_graph(node)?.try_clone()?;
        self.update_initializer_chain(context, plan, &mut replacement)?;
        // All fallible preparation mutated only the private clone.
        // One native call publishes a complete workspace/initializer/kernel snapshot without changing this node's identity.
        graph.set_child_graph(node, &replacement)?;
        Ok(())
    }

    fn update_initializer_chain(
        &self,
        context: &Context,
        plan: &ExecutionPlan,
        graph: &mut Graph,
    ) -> Result<()> {
        if graph.nodes()?.len().checked_sub(1) != Some(self.initializations.len()) {
            return Err(invalid_graph());
        }

        let roots = graph.root_nodes()?;
        let [root] = roots.as_slice() else {
            return Err(invalid_graph());
        };
        let mut child = root.clone();
        let mut initialization_nodes = Vec::with_capacity(self.initializations.len());
        // The cloned topology preserves initializer identity independently of addresses.
        for initialization in &self.initializations {
            if graph.node_type(&child)? != initialization.node_type() {
                return Err(invalid_graph());
            }

            let dependents = graph.node_dependents(&child)?;
            let [next] = dependents.as_slice() else {
                return Err(invalid_graph());
            };
            initialization_nodes.push(child);
            child = next.node.clone();
        }

        if graph.node_type(&child)? != GraphNodeType::Graph
            || !graph.node_dependents(&child)?.is_empty()
        {
            return Err(invalid_graph());
        }

        let child_graph = graph.child_graph(&child)?;
        plan.retain_native_graph_resources(context, &child_graph)?;
        // Finish owner preparation before changing any initializer or backend node.
        // Mutations remain private until the complete replacement succeeds.
        for initialization in &self.initializations {
            initialization.retain_host_source(graph)?;
        }

        for (node, initialization) in initialization_nodes.iter().zip(&self.initializations) {
            match initialization {
                WorkspaceInitialization::Copy { byte_offset, data } => {
                    // SAFETY: the loop above retained `data` on `graph`, and the destination lies in the checked workspace prefix.
                    unsafe {
                        graph.set_memory_copy_node_1d_params(
                            node,
                            &self.copy_parameters(*byte_offset, data),
                        )?;
                    }
                }
                WorkspaceInitialization::Fill {
                    byte_offset,
                    byte_len,
                    value,
                } => {
                    // SAFETY: the fill range lies in the checked workspace prefix, and `node` is the matching memset node of this clone.
                    unsafe {
                        graph.set_memory_set_node_params(
                            node,
                            &self.fill_parameters(*byte_offset, *byte_len, *value),
                        )?;
                    }
                }
            }
        }

        let child_graph = graph.child_graph(&child)?;
        // SAFETY: `retain_native_graph_resources` retained the plan code on the child graph before any mutation.
        // The pack's unsafe constructor requires its bindings and workspace to stay valid for every graph update that uses it.
        unsafe {
            plan.descriptor()
                .update_graph(context, self.descriptor(), &child_graph)
        }
    }
}

fn invalid_graph() -> Error {
    Error::DescriptorMismatch {
        name: "frontend workspace CUDA graph structure".into(),
    }
}
