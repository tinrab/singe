use singe_onnx::Node;

use crate::{
    data_type::DataType,
    graph::node::NodeKind,
    onnx::{
        attributes::Attributes,
        error::{Error, Result},
        importer::{ImportedValue, Importer},
        ops::{
            expect_input_range, expect_outputs, onnx_padding, optional_static_f32_scalar,
            padded_shape, register_node_outputs, shape_input_values,
        },
        utility::{node_label, node_name},
    },
    shape::Shape,
};

pub fn lower_pad(importer: &mut Importer, node: &Node) -> Result<()> {
    expect_input_range(importer, node, 2, 4)?;
    expect_outputs(importer, node, 1)?;
    let plan = pad_plan(importer, node)?;
    let device = importer.options().device;
    let id = importer.graph().add_named_operation(
        node_name(node),
        NodeKind::pad(plan.input.node, plan.padding, plan.value),
        plan.input.data_type,
        plan.shape.clone(),
        device,
    )?;
    register_node_outputs(
        importer,
        node,
        vec![ImportedValue::new(id, plan.input.data_type, plan.shape)],
    )
}

pub struct PadPlan {
    pub(crate) input: ImportedValue,
    pub(crate) padding: Vec<(i64, i64)>,
    pub(crate) value: f32,
    pub(crate) shape: Shape,
}

pub fn pad_plan(importer: &Importer, node: &Node) -> Result<PadPlan> {
    let attrs = Attributes::create(node)?;
    validate_mode(importer, node, &attrs)?;
    validate_axes_input(importer, node)?;
    let input = importer.required_input(node, 0)?;
    validate_input_type(importer, node, &input)?;
    let pads = shape_input_values(importer, node, 1, "pads")?;
    let padding = onnx_padding(&node_label(node), input.shape.rank(), &pads)?;
    let value = optional_static_f32_scalar(importer, node, 2, "constant_value")?.unwrap_or(0.0);
    let shape = padded_shape(&node_label(node), &input.shape, &padding)?;
    Ok(PadPlan {
        input,
        padding,
        value,
        shape,
    })
}

fn validate_mode(importer: &Importer, node: &Node, attrs: &Attributes) -> Result<()> {
    let mode = attrs.string("mode", "constant")?;
    if mode == "constant" {
        return Ok(());
    }
    Err(Error::unsupported_operator(
        node.domain.as_deref(),
        &node.operator_type,
        importer.opset_version(node.domain.as_deref()),
        "only constant pad mode is supported",
    ))
}

fn validate_axes_input(importer: &Importer, node: &Node) -> Result<()> {
    if node.inputs.get(3).is_none_or(String::is_empty) {
        return Ok(());
    }
    Err(Error::unsupported_operator(
        node.domain.as_deref(),
        &node.operator_type,
        importer.opset_version(node.domain.as_deref()),
        "pad axes input is not supported",
    ))
}

fn validate_input_type(importer: &Importer, node: &Node, input: &ImportedValue) -> Result<()> {
    if input.data_type == DataType::F32 {
        return Ok(());
    }
    Err(Error::unsupported_operator(
        node.domain.as_deref(),
        &node.operator_type,
        importer.opset_version(node.domain.as_deref()),
        "only f32 pad inputs are supported",
    ))
}
