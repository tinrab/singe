mod alias;
mod build;
mod provider;
mod values;

use std::collections::HashMap;

use self::build::compile_report_operation;

pub use self::{
    alias::compile_report_ir_alias_groups,
    values::{
        compile_report_non_contiguous_layout_input_values_for_ir_values,
        compile_report_unique_layout_input_values,
        compile_report_unique_non_contiguous_layout_input_values,
    },
};
use crate::{
    backend::cuda::{
        compiled::{
            executable::report::types::GraphCompileOperationReport,
            plan::lowering::{LoweringPlan, RealizedLoweringPlan},
        },
        diagnostics::GraphProviderFailures,
    },
    error::Result,
    graph::ir::{Graph, ValueId},
};

pub fn compile_report_operations(
    ir: &Graph,
    execution_order: &[ValueId],
    lowering_plan: &LoweringPlan,
    realized_lowering: &RealizedLoweringPlan,
    provider_failures: &HashMap<ValueId, GraphProviderFailures>,
) -> Result<Vec<GraphCompileOperationReport>> {
    Ok(execution_order
        .iter()
        .map(|&id| {
            compile_report_operation(ir, id, lowering_plan, realized_lowering, provider_failures)
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>())
}
