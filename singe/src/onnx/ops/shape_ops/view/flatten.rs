use singe_onnx::Node;

use crate::{
    graph::node::NodeKind,
    onnx::{
        attributes::Attributes,
        error::{Error, Result},
        importer::{ImportedValue, Importer},
        ops::{
            expect_input_count, expect_outputs, register_node_outputs,
            shape_ops::{onnx_dimension_to_usize, onnx_len_to_i64},
        },
        utility::{node_label, node_name},
    },
    shape::Shape,
    utility::checked_product_i64,
};

pub fn lower_flatten(importer: &mut Importer, node: &Node) -> Result<()> {
    expect_input_count(importer, node, 1)?;
    expect_outputs(importer, node, 1)?;
    let attrs = Attributes::create(node)?;
    let input = importer.required_input(node, 0)?;
    let axis = attrs.int("axis", 1)?;
    let rank = input.shape.dimensions().len();
    let rank_i64 = onnx_len_to_i64(node, "flatten rank", rank)?;
    if axis < -rank_i64 || axis > rank_i64 {
        return Err(Error::invalid_attribute(
            node_label(node),
            "axis",
            format!("expected axis in [-{rank}, {rank}], got {axis}"),
        ));
    }

    let normalized_axis = if axis < 0 {
        rank_i64
            .checked_add(axis)
            .ok_or_else(|| Error::UnsupportedShape {
                name: node_label(node),
                reason: "flatten normalized axis overflowed".into(),
            })?
    } else {
        axis
    };
    let axis_index = onnx_dimension_to_usize(node, "flatten axis", normalized_axis)?;
    let left = checked_product_i64(
        &input.shape.dimensions()[..axis_index],
        "onnx flatten left extent",
    )?;
    let right = checked_product_i64(
        &input.shape.dimensions()[axis_index..],
        "onnx flatten right extent",
    )?;
    let shape = Shape::create([left, right])?;
    let device = importer.options().device;
    let id = importer.graph().add_named_operation(
        node_name(node),
        NodeKind::flatten(input.node, normalized_axis, -1),
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
