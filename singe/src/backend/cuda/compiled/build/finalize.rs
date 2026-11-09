use std::{collections::HashMap, sync::Arc};

use singe_cuda::graph::Graph as NativeCudaGraph;

use crate::{
    backend::cuda::{
        compiled::{
            executable::{
                CudaCompiledGraph, CudaCompiledGraphParts, CudaCompiledGraphStats, CudnnGraphStep,
                CutensorGraphStep, GraphPrivateBuffers, LibraryResource,
            },
            plan::{
                cache_key::GraphCacheKey,
                fusion::FusionPlan,
                lowering::{LoweringPlan, RealizedLoweringPlan},
                memory::GraphMemoryPlanner,
            },
        },
        diagnostics::GraphProviderFailures,
        graph_value::CudaGraphValue,
        nvtx::CudaNvtxRange,
        runtime::CudaRuntime,
    },
    device::Device,
    error::Result,
    graph::{
        guards::GraphGuards,
        ir::{Graph, ValueId},
    },
};

pub struct GraphFinalizeInput<'a> {
    pub(crate) ir: Arc<Graph>,
    pub(crate) cache_key: GraphCacheKey,
    pub(crate) device: Device,
    pub(crate) guards: GraphGuards,
    pub(crate) execution_order: Vec<ValueId>,
    pub(crate) lowering_plan: LoweringPlan,
    pub(crate) realized_lowering: RealizedLoweringPlan,
    pub(crate) provider_failure_notes: HashMap<ValueId, GraphProviderFailures>,
    pub(crate) stats: CudaCompiledGraphStats,
    pub(crate) runtime: &'a Arc<CudaRuntime>,
    pub(crate) native_graph: NativeCudaGraph,
    pub(crate) has_native_nodes: bool,
    pub(crate) cudnn_steps: Vec<CudnnGraphStep>,
    pub(crate) cutensor_steps: Vec<CutensorGraphStep>,
    pub(crate) library_resources: Vec<LibraryResource>,
    pub(crate) values: HashMap<ValueId, CudaGraphValue>,
    pub(crate) memory_planner: GraphMemoryPlanner,
    pub(crate) private: GraphPrivateBuffers,
    pub(crate) fusion_plan: FusionPlan,
}

pub fn finalize_cuda_graph_compile(mut input: GraphFinalizeInput<'_>) -> Result<CudaCompiledGraph> {
    input.memory_planner.record_stats(&mut input.stats)?;
    let reusable_buffers = input.memory_planner.reusable_buffer_summaries()?;
    let reusable_buffer_allocations = input.memory_planner.reusable_buffer_allocations()?;
    let (executable, report) = {
        let _instantiate_range = CudaNvtxRange::create("singe cuda graph compile instantiate");
        let executable = input
            .has_native_nodes
            .then(|| input.native_graph.instantiate())
            .transpose()?;
        let report = CudaCompiledGraph::compile_report(
            &input.ir,
            &input.execution_order,
            &input.lowering_plan,
            &input.realized_lowering,
            &input.provider_failure_notes,
            input.stats,
            &input.cache_key,
            input.fusion_plan,
        )?;
        (executable, report)
    };
    let outputs = input.ir.output_values_vec();
    let output_buffers = CudaCompiledGraph::output_buffer_summaries(&input.values, &outputs)?;
    let output_buffer_allocations =
        CudaCompiledGraph::output_buffer_allocation_records(&input.values, &outputs)?;
    Ok(CudaCompiledGraph::new(CudaCompiledGraphParts {
        ir: input.ir,
        cache_key: input.cache_key,
        stats: input.stats,
        report,
        runtime: Arc::clone(input.runtime),
        device: input.device,
        guards: input.guards,
        execution_order: input.execution_order,
        native_graph: input.has_native_nodes.then_some(input.native_graph),
        executable,
        cudnn_steps: input.cudnn_steps,
        cutensor_steps: input.cutensor_steps,
        library_resources: input.library_resources,
        values: input.values,
        outputs,
        output_buffers,
        output_buffer_allocations,
        reusable_buffers,
        reusable_buffer_allocations,
        private: input.private,
    }))
}
