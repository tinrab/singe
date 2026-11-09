use std::collections::HashSet;

use crate::{
    backend::{
        cuda::compiled::executable::report::{
            operation::compile_report_non_contiguous_layout_input_values_for_ir_values,
            types::GraphCompileIrPartitionFields,
        },
        registry::{LayoutSupport, operation_support},
    },
    error::Result,
    graph::ir::{Graph, OperationId, ValueId},
};

pub fn compile_report_ir_partition_fields(
    ir: &Graph,
    operations: &[OperationId],
) -> Result<GraphCompileIrPartitionFields> {
    let operations = operations.to_vec();
    let layout_support = compile_report_ir_partition_layout_support(ir, operations.as_slice())?;
    let input_values = compile_report_ir_partition_input_values(ir, operations.as_slice())?;
    let layout_constrained_input_values =
        compile_report_ir_partition_layout_constrained_input_values(ir, operations.as_slice())?;
    let non_contiguous_layout_input_values =
        compile_report_non_contiguous_layout_input_values_for_ir_values(
            ir,
            layout_constrained_input_values.as_slice(),
        )?;
    let internal_edges = compile_report_ir_partition_internal_edges(ir, operations.as_slice())?;
    let boundary_output_values =
        compile_report_ir_partition_boundary_output_values(ir, operations.as_slice())?;
    let graph_output_values =
        compile_report_ir_partition_graph_output_values(ir, operations.as_slice())?;
    let output_values = compile_report_ir_partition_output_values(ir, operations.as_slice())?;
    Ok(GraphCompileIrPartitionFields {
        operations,
        layout_support,
        input_values,
        layout_constrained_input_values,
        non_contiguous_layout_input_values,
        internal_edges,
        boundary_output_values,
        graph_output_values,
        output_values,
    })
}

pub fn compile_report_ir_partition_layout_support(
    ir: &Graph,
    operations: &[OperationId],
) -> Result<LayoutSupport> {
    let mut layout_support = LayoutSupport::AnyStrided;
    for operation in operations {
        let operation = ir.required_operation(*operation)?;
        if let Some(opcode) = operation.opcode() {
            layout_support =
                strongest_layout_support(layout_support, operation_support(opcode).layout());
        }
    }
    Ok(layout_support)
}

pub fn compile_report_ir_partition_layout_constrained_input_values(
    ir: &Graph,
    operations: &[OperationId],
) -> Result<Vec<ValueId>> {
    let mut inputs = Vec::new();
    let mut seen = HashSet::new();
    for operation in operations {
        let operation = ir.required_operation(*operation)?;
        if operation.opcode().is_some_and(|opcode| {
            !matches!(
                operation_support(opcode).layout(),
                LayoutSupport::AnyStrided
            )
        }) {
            for input in operation.inputs() {
                if seen.insert(*input) {
                    inputs.push(*input);
                }
            }
        }
    }
    Ok(inputs)
}

fn strongest_layout_support(current: LayoutSupport, next: LayoutSupport) -> LayoutSupport {
    match (current, next) {
        (LayoutSupport::RequiresContiguous, _) | (_, LayoutSupport::RequiresContiguous) => {
            LayoutSupport::RequiresContiguous
        }
        (LayoutSupport::MaterializesToContiguous, _)
        | (_, LayoutSupport::MaterializesToContiguous) => LayoutSupport::MaterializesToContiguous,
        (LayoutSupport::AnyStrided, LayoutSupport::AnyStrided) => LayoutSupport::AnyStrided,
    }
}

pub fn compile_report_ir_partition_input_values(
    ir: &Graph,
    operations: &[OperationId],
) -> Result<Vec<ValueId>> {
    let produced = compile_report_ir_partition_output_values(ir, operations)?
        .into_iter()
        .collect::<HashSet<_>>();
    let mut inputs = Vec::new();
    let mut seen = HashSet::new();
    for operation in operations {
        let operation = ir.required_operation(*operation)?;
        for input in operation.inputs() {
            if !produced.contains(input) && seen.insert(*input) {
                inputs.push(*input);
            }
        }
    }
    Ok(inputs)
}

pub fn compile_report_ir_partition_internal_edges(
    ir: &Graph,
    operations: &[OperationId],
) -> Result<Vec<(ValueId, OperationId)>> {
    let produced = compile_report_ir_partition_output_values(ir, operations)?
        .into_iter()
        .collect::<HashSet<_>>();
    let mut edges = Vec::new();
    for operation in operations {
        let ir_operation = ir.required_operation(*operation)?;
        for input in ir_operation.inputs() {
            if produced.contains(input) {
                edges.push((*input, *operation));
            }
        }
    }
    Ok(edges)
}

pub fn compile_report_ir_partition_output_values(
    ir: &Graph,
    operations: &[OperationId],
) -> Result<Vec<ValueId>> {
    let mut outputs = Vec::new();
    for operation in operations {
        outputs.extend(
            ir.required_operation(*operation)?
                .results()
                .iter()
                .map(|result| result.value()),
        );
    }
    Ok(outputs)
}

pub fn compile_report_ir_partition_boundary_output_values(
    ir: &Graph,
    operations: &[OperationId],
) -> Result<Vec<ValueId>> {
    let partition_operations = operations.iter().copied().collect::<HashSet<_>>();
    let graph_outputs = graph_output_values_from_ir(ir);
    let mut outputs = Vec::new();
    let mut seen = HashSet::new();
    for operation in operations {
        let operation = ir.required_operation(*operation)?;
        for result in operation.results() {
            let value = result.value();
            let used_outside = ir.operations().iter().any(|consumer| {
                !partition_operations.contains(&consumer.id()) && consumer.inputs().contains(&value)
            });
            if (graph_outputs.contains(&value) || used_outside) && seen.insert(value) {
                outputs.push(value);
            }
        }
    }
    Ok(outputs)
}

pub fn compile_report_ir_partition_graph_output_values(
    ir: &Graph,
    operations: &[OperationId],
) -> Result<Vec<ValueId>> {
    let graph_outputs = graph_output_values_from_ir(ir);
    let mut outputs = Vec::new();
    let mut seen = HashSet::new();
    for operation in operations {
        let operation = ir.required_operation(*operation)?;
        for result in operation.results() {
            let value = result.value();
            if graph_outputs.contains(&value) && seen.insert(value) {
                outputs.push(value);
            }
        }
    }
    Ok(outputs)
}

fn graph_output_values_from_ir(ir: &Graph) -> HashSet<ValueId> {
    ir.output_value_set()
}
