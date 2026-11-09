use singe_onnx::Node;

use crate::{
    backend::{registry::Operation, spec::matmul_output_shape},
    graph::{GraphOperationOutput, GraphOperationRequest},
    onnx::{
        attributes::Attributes,
        error::{Error, Result},
        importer::{ImportedValue, Importer},
        ops::{
            add_binary_intermediate, expect_input_count, expect_input_range, expect_outputs,
            register_node_outputs,
        },
        utility::node_name,
    },
};

pub fn lower_gelu(importer: &mut Importer, node: &Node) -> Result<()> {
    let operation = if Attributes::create(node)?.string("approximate", "none")? == "none" {
        Operation::GeluExact
    } else {
        Operation::Gelu
    };
    lower_unary(importer, node, operation)
}

pub fn lower_binary(importer: &mut Importer, node: &Node, operation: Operation) -> Result<()> {
    expect_input_count(importer, node, 2)?;
    expect_outputs(importer, node, 1)?;
    let a = importer.required_input(node, 0)?;
    let b = importer.required_input(node, 1)?;
    let shape = if node.operator_type == "MatMul" {
        matmul_output_shape(&a.shape, &b.shape)?
    } else {
        a.shape.broadcast_with(&b.shape)?
    };
    let device = importer.options().device;
    let id = importer
        .graph()
        .add_operation_request(GraphOperationRequest::binary(
            node_name(node),
            operation,
            [a.node, b.node],
            GraphOperationOutput::new(a.data_type, shape.clone(), device),
        ))?;
    register_node_outputs(
        importer,
        node,
        vec![ImportedValue::new(id, a.data_type, shape)],
    )
}

pub fn lower_unary(importer: &mut Importer, node: &Node, operation: Operation) -> Result<()> {
    expect_input_count(importer, node, 1)?;
    expect_outputs(importer, node, 1)?;
    let input = importer.required_input(node, 0)?;
    let device = importer.options().device;
    let id = importer
        .graph()
        .add_operation_request(GraphOperationRequest::unary(
            node_name(node),
            operation,
            input.node,
            GraphOperationOutput::new(input.data_type, input.shape.clone(), device),
        ))?;
    register_node_outputs(
        importer,
        node,
        vec![ImportedValue::new(id, input.data_type, input.shape)],
    )
}

pub fn lower_variadic_binary(
    importer: &mut Importer,
    node: &Node,
    operation: Operation,
) -> Result<()> {
    expect_input_range(importer, node, 1, usize::MAX)?;
    expect_outputs(importer, node, 1)?;
    let mut values = node
        .inputs
        .iter()
        .map(|name| importer.value(name).cloned())
        .collect::<Result<Vec<_>>>()?;
    let mut current = values.remove(0);
    for value in values {
        if current.data_type != value.data_type {
            return Err(Error::unsupported_operator(
                node.domain.as_deref(),
                &node.operator_type,
                importer.opset_version(node.domain.as_deref()),
                "input dtypes must match",
            ));
        }

        let shape = current.shape.broadcast_with(&value.shape)?;
        let id = add_binary_intermediate(
            importer,
            None,
            operation,
            [current.node, value.node],
            current.data_type,
            shape.clone(),
        )?;
        current = ImportedValue::new(id, current.data_type, shape);
    }
    if node.inputs.len() == 1 {
        importer.register_value(&node.outputs[0], current)
    } else {
        register_node_outputs(importer, node, vec![current])
    }
}
