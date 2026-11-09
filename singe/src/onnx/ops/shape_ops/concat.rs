use singe_onnx::Node;

use crate::{
    error::Error as SingeError,
    graph::node::NodeKind,
    onnx::{
        attributes::Attributes,
        error::{Error, Result},
        importer::{ImportedValue, Importer},
        ops::{expect_outputs, register_node_outputs},
        shape::concat_shape,
        utility::node_name,
    },
};

pub fn lower_concat(importer: &mut Importer, node: &Node) -> Result<()> {
    if node.inputs.is_empty() {
        return Err(SingeError::MissingConcatInput.into());
    }

    expect_outputs(importer, node, 1)?;
    let attrs = Attributes::create(node)?;
    let axis = attrs.int("axis", 0)?;
    let values = node
        .inputs
        .iter()
        .map(|name| importer.value(name).cloned())
        .collect::<Result<Vec<_>>>()?;
    let shapes = values
        .iter()
        .map(|value| value.shape.clone())
        .collect::<Vec<_>>();
    let shape = concat_shape(&shapes, axis)?;
    let data_type = values[0].data_type;
    if values.iter().any(|value| value.data_type != data_type) {
        return Err(Error::unsupported_operator(
            node.domain.as_deref(),
            &node.operator_type,
            importer.opset_version(node.domain.as_deref()),
            "all concat inputs must have the same data type",
        ));
    }

    let device = importer.options().device;
    let id = importer.graph().add_named_operation(
        node_name(node),
        NodeKind::concat(values.iter().map(|value| value.node).collect(), axis),
        data_type,
        shape.clone(),
        device,
    )?;
    register_node_outputs(
        importer,
        node,
        vec![ImportedValue::new(id, data_type, shape)],
    )
}
