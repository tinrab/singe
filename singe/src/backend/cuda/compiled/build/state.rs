use std::collections::{HashMap, HashSet};

use singe_cuda::graph::{Graph as NativeCudaGraph, GraphNode};

use crate::{
    backend::cuda::{
        compiled::{
            build::{
                alias::{
                    GraphAliasLowering, GraphAliasLoweringContext, try_add_cuda_graph_alias_node,
                },
                fusion::GraphFusionLowering,
                provider::GraphProviderLowering,
            },
            executable::{
                CudaCompiledGraphStats, CudnnGraphStep, CutensorGraphStep, GraphPrivateBuffers,
                LibraryResource,
            },
            plan::{
                fusion::FusionPlan,
                kernel::NodeKernelPlans,
                lowering::{CudaGraphProvider, LoweringPlan, RealizedLoweringPlan},
                memory::GraphMemoryPlanner,
            },
        },
        context::CudaExecutionContext,
        diagnostics::GraphProviderFailures,
        graph_value::CudaGraphValue,
        kernel_plan::KernelPlanner,
    },
    error::Result,
    graph::ir::{Graph, ValueId},
};

pub struct GraphLoweringState {
    pub(crate) native_graph: NativeCudaGraph,
    pub(crate) cache: HashMap<ValueId, CudaGraphValue>,
    pub(crate) native_nodes: HashMap<ValueId, GraphNode>,
    pub(crate) values: HashMap<ValueId, CudaGraphValue>,
    pub(crate) cudnn_steps: Vec<CudnnGraphStep>,
    pub(crate) cutensor_steps: Vec<CutensorGraphStep>,
    pub(crate) library_resources: Vec<LibraryResource>,
    pub(crate) realized_lowering: RealizedLoweringPlan,
    pub(crate) provider_failure_notes: HashMap<ValueId, GraphProviderFailures>,
    pub(crate) stats: CudaCompiledGraphStats,
    pub(crate) skipped_nodes: HashSet<ValueId>,
    pub(crate) memory_planner: GraphMemoryPlanner,
    /// The status word and scratch buffers that kernels allocate as they lower.
    pub(crate) private: GraphPrivateBuffers,
    /// The fusion groups lowering builds, each when it reaches the group's first member.
    pub(crate) fusion_plan: FusionPlan,
    pub(crate) planner: KernelPlanner,
    pub(crate) node_plans: HashMap<ValueId, NodeKernelPlans>,
    // The results a kernel already wrote for nodes that lower later, and the graph node that writes them.
    pub(crate) sibling_results: HashMap<ValueId, (CudaGraphValue, GraphNode, CudaGraphProvider)>,
}

impl GraphLoweringState {
    pub(crate) fn create(
        ir: &Graph,
        execution_order: &[ValueId],
        native_graph: NativeCudaGraph,
        planner: KernelPlanner,
    ) -> Result<Self> {
        let memory_planner = GraphMemoryPlanner::create(ir, execution_order)?;
        Ok(Self {
            native_graph,
            cache: HashMap::new(),
            native_nodes: HashMap::new(),
            values: HashMap::new(),
            cudnn_steps: Vec::new(),
            cutensor_steps: Vec::new(),
            library_resources: Vec::new(),
            realized_lowering: RealizedLoweringPlan::default(),
            provider_failure_notes: HashMap::new(),
            stats: CudaCompiledGraphStats::default(),
            skipped_nodes: HashSet::new(),
            memory_planner,
            private: GraphPrivateBuffers::default(),
            fusion_plan: FusionPlan::default(),
            planner,
            node_plans: HashMap::new(),
            sibling_results: HashMap::new(),
        })
    }

    pub(crate) fn fusion_lowering<'a>(
        &'a mut self,
        context: &'a CudaExecutionContext,
        provider_failures: &'a mut GraphProviderFailures,
    ) -> GraphFusionLowering<'a> {
        GraphFusionLowering {
            context,
            native_graph: &mut self.native_graph,
            cache: &mut self.cache,
            values: &mut self.values,
            native_nodes: &mut self.native_nodes,
            skipped_nodes: &mut self.skipped_nodes,
            realized_lowering: &mut self.realized_lowering,
            memory_planner: &mut self.memory_planner,
            cudnn_steps: &mut self.cudnn_steps,
            library_resources: &mut self.library_resources,
            stats: &mut self.stats,
            provider_failures,
            planner: &self.planner,
            private: &mut self.private,
        }
    }

    pub(crate) fn try_add_alias_node(
        &mut self,
        id: ValueId,
        ir: &Graph,
        lowering_plan: &LoweringPlan,
    ) -> Result<GraphAliasLowering> {
        let mut context = GraphAliasLoweringContext {
            cache: &mut self.cache,
            values: &mut self.values,
            native_nodes: &mut self.native_nodes,
            stats: &mut self.stats,
            realized_lowering: &mut self.realized_lowering,
        };
        try_add_cuda_graph_alias_node(id, ir, lowering_plan, &mut context)
    }

    pub(crate) fn with_provider_lowering<R>(
        &mut self,
        provider_failures: &mut GraphProviderFailures,
        f: impl FnOnce(&HashMap<ValueId, CudaGraphValue>, &mut GraphProviderLowering<'_>) -> R,
    ) -> R {
        let mut lowering = GraphProviderLowering {
            native_graph: &mut self.native_graph,
            stats: &mut self.stats,
            realized_lowering: &mut self.realized_lowering,
            cudnn_steps: &mut self.cudnn_steps,
            library_resources: &mut self.library_resources,
            cutensor_steps: &mut self.cutensor_steps,
            provider_failures,
            private: &mut self.private,
            planner: &self.planner,
            node_plans: &mut self.node_plans,
        };
        f(&self.cache, &mut lowering)
    }
}
