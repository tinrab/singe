use singe_onnx::Node;

use crate::{
    backend::registry::Operation,
    data_type::DataType,
    graph::{GraphOperationOutput, GraphOperationRequest},
    onnx::{
        error::{Error, Result},
        importer::{ImportedValue, Importer},
        ops::{add_binary_intermediate, expect_input_count, expect_outputs, register_node_outputs},
        utility::node_name,
    },
};

pub fn lower_comparison(importer: &mut Importer, node: &Node, operation: Operation) -> Result<()> {
    expect_input_count(importer, node, 2)?;
    expect_outputs(importer, node, 1)?;
    let a = importer.required_input(node, 0)?;
    let b = importer.required_input(node, 1)?;
    if a.data_type != b.data_type {
        return Err(Error::unsupported_operator(
            node.domain.as_deref(),
            &node.operator_type,
            importer.opset_version(node.domain.as_deref()),
            "comparison input dtypes must match",
        ));
    }

    let shape = a.shape.broadcast_with(&b.shape)?;
    let id = add_binary_intermediate(
        importer,
        node_name(node),
        operation,
        [a.node, b.node],
        DataType::Bool,
        shape.clone(),
    )?;
    register_node_outputs(
        importer,
        node,
        vec![ImportedValue::new(id, DataType::Bool, shape)],
    )
}

pub fn lower_is_nan(importer: &mut Importer, node: &Node) -> Result<()> {
    expect_input_count(importer, node, 1)?;
    expect_outputs(importer, node, 1)?;
    let input = importer.required_input(node, 0)?;
    let device = importer.options().device;
    let id = importer
        .graph()
        .add_operation_request(GraphOperationRequest::unary(
            node_name(node),
            Operation::IsNan,
            input.node,
            GraphOperationOutput::new(DataType::Bool, input.shape.clone(), device),
        ))?;
    register_node_outputs(
        importer,
        node,
        vec![ImportedValue::new(id, DataType::Bool, input.shape)],
    )
}
