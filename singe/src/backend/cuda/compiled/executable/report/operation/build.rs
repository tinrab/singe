use std::collections::HashMap;

use crate::{
    backend::{
        cuda::{
            compiled::{
                executable::report::{
                    operation::{
                        provider::{
                            compile_report_node_materialization,
                            compile_report_planned_provider_decision,
                            compile_report_provider_candidate,
                            compile_report_realized_provider_decision,
                            compile_report_selected_algorithm,
                        },
                        values::{
                            compile_report_ir_alias_group_root_value,
                            compile_report_ir_alias_input_value,
                            compile_report_ir_non_contiguous_layout_input_values,
                            compile_report_ir_operation_values, compile_report_ir_value_data_types,
                            compile_report_ir_value_layouts,
                            compile_report_layout_constrained_input_values,
                        },
                    },
                    types::{GraphCompileOperationReport, GraphCompileProviderFailure},
                },
                plan::lowering::{LoweringPlan, RealizedLoweringPlan},
            },
            diagnostics::GraphProviderFailures,
        },
        registry::{operation_name, operation_support},
    },
    error::Result,
    graph::ir::{Graph, ValueId},
};

pub fn compile_report_operation(
    ir: &Graph,
    id: ValueId,
    lowering_plan: &LoweringPlan,
    realized_lowering: &RealizedLoweringPlan,
    provider_failures: &HashMap<ValueId, GraphProviderFailures>,
) -> Result<Option<GraphCompileOperationReport>> {
    let ir_operation = ir.required_operation_for_value(id)?;
    let operation_id = ir_operation.id();
    let Some(operation) = ir_operation.opcode() else {
        return Ok(None);
    };
    let operation_name = operation_name(operation);
    let (input_values, output_values) = compile_report_ir_operation_values(ir, operation_id)?;
    let input_data_types = compile_report_ir_value_data_types(ir, input_values.as_slice())?;
    let input_layouts = compile_report_ir_value_layouts(ir, input_values.as_slice())?;
    let output_data_types = compile_report_ir_value_data_types(ir, output_values.as_slice())?;
    let output_layouts = compile_report_ir_value_layouts(ir, output_values.as_slice())?;
    let alias_input_value = compile_report_ir_alias_input_value(ir, output_values.as_slice())?;
    let alias_group_root_value =
        compile_report_ir_alias_group_root_value(ir, output_values.as_slice())?;
    let layout_support = operation_support(operation).layout();
    let layout_constrained_input_values =
        compile_report_layout_constrained_input_values(layout_support, input_values.as_slice());
    let non_contiguous_layout_input_values = compile_report_ir_non_contiguous_layout_input_values(
        ir,
        layout_support,
        input_values.as_slice(),
    )?;
    let output_materialization = compile_report_node_materialization(
        ir,
        output_values.as_slice(),
        realized_lowering.decision(id),
    )?;
    Ok(Some(GraphCompileOperationReport {
        operation_id,
        input_values,
        input_data_types,
        input_layouts,
        output_values,
        output_data_types,
        output_layouts,
        alias_input_value,
        alias_group_root_value,
        operation: operation_name,
        layout_support,
        layout_constrained_input_values,
        non_contiguous_layout_input_values,
        output_materialization,
        candidates: lowering_plan
            .candidates(id)
            .iter()
            .copied()
            .map(compile_report_provider_candidate)
            .collect(),
        planned_decision: lowering_plan
            .decision_for_operation(operation_id)
            .map(compile_report_planned_provider_decision),
        realized_decision: realized_lowering
            .decision(id)
            .map(compile_report_realized_provider_decision),
        selected_algorithm: compile_report_selected_algorithm(realized_lowering.decision(id)),
        provider_failures: provider_failures
            .get(&id)
            .map(|failures| {
                failures
                    .as_slice()
                    .iter()
                    .map(|(provider, reason)| GraphCompileProviderFailure::new(*provider, reason))
                    .collect()
            })
            .unwrap_or_default(),
    }))
}
