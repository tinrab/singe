use std::collections::HashSet;

use crate::{
    backend::{
        cuda::compiled::executable::report::types::GraphCompileOperationReport,
        registry::LayoutSupport,
    },
    data_type::DataType,
    error::Result,
    graph::ir::{Graph, OperationId, ValueId},
    layout::Layout,
};

pub fn compile_report_ir_alias_input_value(
    ir: &Graph,
    output_values: &[ValueId],
) -> Result<Option<ValueId>> {
    let Some(output) = output_values.first() else {
        return Ok(None);
    };
    let Some(alias) = ir.required_value_alias(*output)? else {
        return Ok(None);
    };
    Ok(Some(
        ir.alias_group_root_value(alias.source())?
            .unwrap_or_else(|| alias.source()),
    ))
}

pub fn compile_report_ir_alias_group_root_value(
    ir: &Graph,
    output_values: &[ValueId],
) -> Result<Option<ValueId>> {
    let Some(output) = output_values.first() else {
        return Ok(None);
    };
    ir.alias_group_root_value(*output)
}

pub fn compile_report_layout_constrained_input_values(
    layout_support: LayoutSupport,
    input_values: &[ValueId],
) -> Vec<ValueId> {
    match layout_support {
        LayoutSupport::AnyStrided => Vec::new(),
        LayoutSupport::MaterializesToContiguous | LayoutSupport::RequiresContiguous => {
            input_values.to_vec()
        }
    }
}

pub fn compile_report_ir_non_contiguous_layout_input_values(
    ir: &Graph,
    layout_support: LayoutSupport,
    input_values: &[ValueId],
) -> Result<Vec<ValueId>> {
    if matches!(layout_support, LayoutSupport::AnyStrided) {
        return Ok(Vec::new());
    }
    compile_report_non_contiguous_layout_input_values_for_ir_values(ir, input_values)
}

pub fn compile_report_non_contiguous_layout_input_values_for_ir_values(
    ir: &Graph,
    input_values: &[ValueId],
) -> Result<Vec<ValueId>> {
    let mut values = Vec::new();
    for value in input_values {
        let layout = ir.required_value_layout(*value)?;
        if !layout.is_contiguous() || layout.storage_offset() != 0 {
            values.push(*value);
        }
    }
    Ok(values)
}

pub fn compile_report_ir_operation_values(
    ir: &Graph,
    operation_id: OperationId,
) -> Result<(Vec<ValueId>, Vec<ValueId>)> {
    let operation = ir.required_operation(operation_id)?;
    Ok((
        operation.inputs().to_vec(),
        operation
            .results()
            .iter()
            .map(|result| result.value())
            .collect(),
    ))
}

pub fn compile_report_ir_value_layouts(ir: &Graph, values: &[ValueId]) -> Result<Vec<Layout>> {
    values
        .iter()
        .map(|value| Ok(ir.required_value_layout(*value)?.clone()))
        .collect()
}

pub fn compile_report_ir_value_data_types(ir: &Graph, values: &[ValueId]) -> Result<Vec<DataType>> {
    values
        .iter()
        .map(|value| ir.required_value_data_type(*value))
        .collect()
}

pub fn compile_report_unique_layout_input_values(
    operations: &[GraphCompileOperationReport],
) -> Vec<ValueId> {
    unique_values(
        operations,
        GraphCompileOperationReport::layout_constrained_input_values,
    )
}

pub fn compile_report_unique_non_contiguous_layout_input_values(
    operations: &[GraphCompileOperationReport],
) -> Vec<ValueId> {
    unique_values(
        operations,
        GraphCompileOperationReport::non_contiguous_layout_input_values,
    )
}

fn unique_values(
    operations: &[GraphCompileOperationReport],
    values_of: impl Fn(&GraphCompileOperationReport) -> &[ValueId],
) -> Vec<ValueId> {
    let mut seen = HashSet::new();
    operations
        .iter()
        .flat_map(&values_of)
        .copied()
        .filter(|value| seen.insert(*value))
        .collect()
}
