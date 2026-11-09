use std::{collections::HashMap, mem::MaybeUninit, sync::Arc};

use singe_cuda::{
    graph::{Graph as NativeCudaGraph, GraphNode},
    memory::DeviceMemory,
    submission::{AccessMode, Resources},
    view::DeviceRegionProvider,
};
use singe_cudnn::context::Context as CudnnContext;

use crate::{
    backend::cuda::{
        cudnn_backend::CudnnBackendExecutable, cutensor::CachedCutensorContractionPlan,
        graph_value::CudaGraphValue,
    },
    error::Result,
    graph::ir::ValueId,
};

/// A cuDNN child graph, its parent graph node, output allocation, and executable binding state.
#[derive(Debug)]
pub struct CudnnGraphStep {
    graph_node: GraphNode,
    child_graph: NativeCudaGraph,
    pub(crate) output: CudaGraphValue,
    executable: CudnnBackendExecutable,
    update_on_replay: bool,
}

impl CudnnGraphStep {
    /// Retains a cuDNN child graph, output, executable, and replay-update policy.
    /// The constructor does not validate graph topology or binding compatibility.
    pub fn new(
        graph_node: GraphNode,
        child_graph: NativeCudaGraph,
        output: CudaGraphValue,
        executable: CudnnBackendExecutable,
        update_on_replay: bool,
    ) -> Self {
        Self {
            graph_node,
            child_graph,
            output,
            executable,
            update_on_replay,
        }
    }

    pub(crate) fn submission_resources(&self) -> Result<Resources> {
        let mut resources = self.executable.submission_resources();
        resources.extend(self.output.submission_resources(AccessMode::ReadWrite)?);
        Ok(resources)
    }

    pub(crate) fn graph_node(&self) -> GraphNode {
        self.graph_node.clone()
    }

    pub(crate) fn child_graph(&self) -> &NativeCudaGraph {
        &self.child_graph
    }

    pub(crate) fn update_on_replay(&self) -> bool {
        self.update_on_replay
    }

    pub(crate) fn update_graph(
        &mut self,
        cudnn: &mut CudnnContext,
        cache: &HashMap<ValueId, CudaGraphValue>,
    ) -> Result<()> {
        let variant_pack = self.executable.variant_pack(&self.output, cache)?;
        // SAFETY: this step retains its workspace and output, while graph values and submission resources retain the current and pending replay bindings.
        unsafe {
            self.executable
                .update_graph(cudnn, &variant_pack, &mut self.child_graph)?;
        }
        Ok(())
    }
}

/// A cuTENSOR child graph with retained output, execution plan, and optional workspace.
#[derive(Debug)]
pub struct CutensorGraphStep {
    _graph_node: GraphNode,
    _child_graph: NativeCudaGraph,
    pub(crate) output: CudaGraphValue,
    workspace: Option<DeviceMemory<MaybeUninit<u8>>>,
    plan: CachedCutensorContractionPlan,
}

impl CutensorGraphStep {
    /// Retains a cuTENSOR child graph and its output, plan, and optional pending workspace.
    /// The constructor does not validate graph topology or workspace requirements.
    #[must_use]
    pub fn new(
        graph_node: GraphNode,
        child_graph: NativeCudaGraph,
        output: CudaGraphValue,
        workspace: Option<DeviceMemory<MaybeUninit<u8>>>,
        plan: CachedCutensorContractionPlan,
    ) -> Self {
        Self {
            _graph_node: graph_node,
            _child_graph: child_graph,
            output,
            workspace,
            plan,
        }
    }

    pub(crate) fn submission_resources(&self) -> Result<Resources> {
        let mut resources = self.output.submission_resources(AccessMode::ReadWrite)?;
        resources.retain(Arc::clone(self.plan.plan()));
        if let Some(workspace) = &self.workspace {
            resources.retain(Arc::clone(workspace.allocation()));
            resources.access(workspace.region(), AccessMode::ReadWrite);
        }
        Ok(resources)
    }
}
