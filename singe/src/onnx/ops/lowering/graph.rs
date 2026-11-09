use crate::{
    backend::registry::Operation,
    data_type::DataType,
    graph::{GraphOperationOutput, GraphOperationRequest, ir::ValueId, node::NodeKind},
    onnx::{
        error::{Error, Result},
        importer::{ImportedValue, Importer},
        ops::lowering::{Node, unsupported_output_count},
    },
    shape::Shape,
};

pub fn register_node_outputs(
    importer: &mut Importer,
    node: &Node,
    values: Vec<ImportedValue>,
) -> Result<()> {
    if node.outputs.len() != values.len() {
        return Err(unsupported_output_count(
            importer,
            node,
            "unexpected output count",
        ));
    }

    for (name, value) in node.outputs.iter().zip(values) {
        importer.register_value(name, value)?;
    }
    Ok(())
}

pub fn add_binary_intermediate(
    importer: &mut Importer,
    name: Option<String>,
    operation: Operation,
    inputs: [ValueId; 2],
    data_type: DataType,
    shape: Shape,
) -> Result<ValueId> {
    let device = importer.options().device;
    importer
        .graph()
        .add_operation_request(GraphOperationRequest::binary(
            name,
            operation,
            inputs,
            GraphOperationOutput::new(data_type, shape, device),
        ))
        .map_err(Into::into)
}

pub fn add_unary_intermediate(
    importer: &mut Importer,
    operation: Operation,
    value: ImportedValue,
) -> Result<ImportedValue> {
    let device = importer.options().device;
    let id = importer
        .graph()
        .add_operation_request(GraphOperationRequest::unary(
            None,
            operation,
            value.node,
            GraphOperationOutput::new(value.data_type, value.shape.clone(), device),
        ))?;
    Ok(ImportedValue::new(id, value.data_type, value.shape))
}

pub fn add_transpose(
    importer: &mut Importer,
    node_name: &str,
    value: &ImportedValue,
) -> Result<ImportedValue> {
    if value.shape.dimensions().len() != 2 {
        return Err(Error::UnsupportedShape {
            name: node_name.into(),
            reason: format!(
                "transpose helper expects rank-2 input, got rank {} shape {:?}",
                value.shape.dimensions().len(),
                value.shape.dimensions()
            ),
        });
    }

    let shape = value.shape.transpose_2d()?;
    let device = importer.options().device;
    let id = importer.graph().add_named_operation(
        Some(format!("{node_name}/transpose")),
        NodeKind::transpose(value.node, 0, 1),
        value.data_type,
        shape.clone(),
        device,
    )?;
    Ok(ImportedValue::new(id, value.data_type, shape))
}

pub fn add_scale(
    importer: &mut Importer,
    value: ImportedValue,
    scalar: f32,
) -> Result<ImportedValue> {
    let device = importer.options().device;
    let id = importer
        .graph()
        .add_operation_request(GraphOperationRequest::scalar(
            None,
            Operation::Scale,
            value.node,
            scalar,
            GraphOperationOutput::new(value.data_type, value.shape.clone(), device),
        ))?;
    Ok(ImportedValue::new(id, value.data_type, value.shape))
}
