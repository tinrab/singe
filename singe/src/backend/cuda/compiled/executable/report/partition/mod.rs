mod fields;
mod provider;

use self::{
    fields::compile_report_ir_partition_fields, provider::assign_cuda_schema_provider_partitions,
};
use crate::{
    backend::cuda::compiled::{
        executable::report::types::GraphCompilePartitionReport,
        plan::lowering::{LoweringDecision, RealizedLoweringPlan},
    },
    error::Result,
    graph::ir::Graph,
};

// The partitions operation support would give `ir`, before any lowering.
pub fn compile_report_planned_partitions(ir: &Graph) -> Result<Vec<GraphCompilePartitionReport>> {
    let mut ir = ir.clone();
    assign_cuda_schema_provider_partitions(&mut ir);
    partition_reports(&ir)
}

// The partitions the providers lowering chose give `ir`.
pub fn compile_report_partitions(
    realized_lowering: &RealizedLoweringPlan,
    ir: &Graph,
) -> Result<Vec<GraphCompilePartitionReport>> {
    let mut ir = ir.clone();
    let providers = ir
        .operations()
        .iter()
        .map(|operation| {
            realized_lowering
                .decision_for_operation(operation.id())
                .and_then(LoweringDecision::provider)
        })
        .collect::<Vec<_>>();
    ir.assign_provider_partitions_from_provider_list(&providers);
    partition_reports(&ir)
}

fn partition_reports(ir: &Graph) -> Result<Vec<GraphCompilePartitionReport>> {
    ir.partitions()
        .iter()
        .filter_map(|partition| partition.provider().map(|provider| (partition, provider)))
        .map(|(partition, provider)| {
            let fields = compile_report_ir_partition_fields(ir, partition.operations())?;
            Ok(GraphCompilePartitionReport {
                partition: partition.id(),
                provider,
                layout_support: fields.layout_support,
                operations: fields.operations,
                input_values: fields.input_values,
                layout_constrained_input_values: fields.layout_constrained_input_values,
                non_contiguous_layout_input_values: fields.non_contiguous_layout_input_values,
                internal_edges: fields.internal_edges,
                boundary_output_values: fields.boundary_output_values,
                graph_output_values: fields.graph_output_values,
                output_values: fields.output_values,
            })
        })
        .collect()
}
