use singe_onnx::Node;

use crate::{
    graph::node::NodeKind,
    onnx::{
        attributes::Attributes,
        error::Result,
        importer::{ImportedValue, Importer},
        ops::{expect_input_count, expect_outputs, register_node_outputs, validate_permutation},
        utility::node_name,
    },
};

pub fn lower_transpose(importer: &mut Importer, node: &Node) -> Result<()> {
    expect_input_count(importer, node, 1)?;
    expect_outputs(importer, node, 1)?;
    let attrs = Attributes::create(node)?;
    let input = importer.required_input(node, 0)?;
    let rank = input.shape.rank();
    let order = attrs
        .ints("perm")?
        .unwrap_or_else(|| (0..rank).rev().collect::<Vec<_>>());
    validate_permutation(node, &order, rank)?;
    let shape = input.shape.permute(&order)?;
    let device = importer.options().device;
    let id = importer.graph().add_named_operation(
        node_name(node),
        NodeKind::permute(input.node, order),
        input.data_type,
        shape.clone(),
        device,
    )?;
    register_node_outputs(
        importer,
        node,
        vec![ImportedValue::new(id, input.data_type, shape)],
    )
}
