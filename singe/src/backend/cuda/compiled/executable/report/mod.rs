mod operation;
mod partition;
mod types;

use std::collections::HashMap;

#[cfg(test)]
pub use self::types::GraphCompileInputLayoutRequirement;
pub use self::types::{
    GraphCompileProviderCandidate, GraphCompileProviderDecision, GraphCompileProviderRejection,
    GraphCompileReport, GraphCompileValueMaterialization,
};
use self::{
    operation::{
        compile_report_ir_alias_groups, compile_report_operations,
        compile_report_unique_layout_input_values,
        compile_report_unique_non_contiguous_layout_input_values,
    },
    partition::{compile_report_partitions, compile_report_planned_partitions},
};
use crate::{
    backend::cuda::{
        compiled::{
            executable::stats::CudaCompiledGraphStats,
            plan::{
                cache_key::GraphCacheKey,
                fusion::FusionPlan,
                lowering::{LoweringPlan, RealizedLoweringPlan},
            },
        },
        diagnostics::GraphProviderFailures,
    },
    error::Result,
    graph::ir::{Graph, ValueId},
};

pub fn build_cuda_graph_compile_report(
    ir: &Graph,
    execution_order: &[ValueId],
    lowering_plan: &LoweringPlan,
    realized_lowering: &RealizedLoweringPlan,
    provider_failures: &HashMap<ValueId, GraphProviderFailures>,
    stats: CudaCompiledGraphStats,
    cache_key: &GraphCacheKey,
    fusion_plan: FusionPlan,
) -> Result<GraphCompileReport> {
    let nodes = compile_report_operations(
        ir,
        execution_order,
        lowering_plan,
        realized_lowering,
        provider_failures,
    )?;
    let layout_constrained_input_values =
        compile_report_unique_layout_input_values(nodes.as_slice());
    let non_contiguous_layout_input_values =
        compile_report_unique_non_contiguous_layout_input_values(nodes.as_slice());
    let partitions = compile_report_partitions(realized_lowering, ir)?;
    let planned_partitions = compile_report_planned_partitions(ir)?;
    let alias_groups = compile_report_ir_alias_groups(ir)?;
    let graph_values = ir.values().len();
    let graph_operations = ir.operations().len();
    let graph_blocks = ir.blocks().len();

    Ok(GraphCompileReport {
        graph_nodes: execution_order.len(),
        graph_values,
        graph_operations,
        graph_blocks,
        graph_partitions: partitions.len(),
        graph_outputs: ir.output_value_count(),
        cacheable: cache_key.is_cacheable(),
        operations: nodes,
        layout_constrained_input_values,
        non_contiguous_layout_input_values,
        planned_partitions,
        partitions,
        alias_groups,
        stats,
        fusion_plan,
    })
}
