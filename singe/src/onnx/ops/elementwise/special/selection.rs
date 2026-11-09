use singe_onnx::Node;

use crate::{
    data_type::DataType,
    graph::node::NodeKind,
    onnx::{
        attributes::Attributes,
        error::{Error, Result},
        importer::{ImportedValue, Importer},
        ops::{
            expect_input_count, expect_input_range, expect_outputs, optional_static_f32_scalar,
            register_node_outputs,
        },
        utility::node_name,
    },
    shape::Shape,
};

pub fn lower_clip(importer: &mut Importer, node: &Node) -> Result<()> {
    expect_input_range(importer, node, 1, 3)?;
    expect_outputs(importer, node, 1)?;
    let attrs = Attributes::create(node)?;
    let input = importer.required_input(node, 0)?;
    if input.data_type != DataType::F32 {
        return Err(Error::unsupported_operator(
            node.domain.as_deref(),
            &node.operator_type,
            importer.opset_version(node.domain.as_deref()),
            "only f32 clip inputs are supported",
        ));
    }

    let min = optional_static_f32_scalar(importer, node, 1, "min")?
        .unwrap_or(attrs.float("min", f32::NEG_INFINITY)?);
    let max = optional_static_f32_scalar(importer, node, 2, "max")?
        .unwrap_or(attrs.float("max", f32::INFINITY)?);
    if min > max {
        return Err(Error::unsupported_operator(
            node.domain.as_deref(),
            &node.operator_type,
            importer.opset_version(node.domain.as_deref()),
            "min bound must be less than or equal to max bound",
        ));
    }

    let device = importer.options().device;
    let id = importer.graph().add_named_operation(
        node_name(node),
        NodeKind::clamp(input.node, min, max),
        input.data_type,
        input.shape.clone(),
        device,
    )?;
    register_node_outputs(
        importer,
        node,
        vec![ImportedValue::new(id, input.data_type, input.shape)],
    )
}

pub fn lower_where(importer: &mut Importer, node: &Node) -> Result<()> {
    expect_input_count(importer, node, 3)?;
    expect_outputs(importer, node, 1)?;
    let plan = where_plan(importer, node)?;
    let device = importer.options().device;
    let id = importer.graph().add_named_operation(
        node_name(node),
        NodeKind::where_operation(plan.condition.node, plan.x.node, plan.y.node),
        plan.x.data_type,
        plan.shape.clone(),
        device,
    )?;
    register_node_outputs(
        importer,
        node,
        vec![ImportedValue::new(id, plan.x.data_type, plan.shape)],
    )
}

pub struct WherePlan {
    pub(crate) condition: ImportedValue,
    pub(crate) x: ImportedValue,
    pub(crate) y: ImportedValue,
    pub(crate) shape: Shape,
}

pub fn where_plan(importer: &Importer, node: &Node) -> Result<WherePlan> {
    let condition = importer.required_input(node, 0)?;
    let x = importer.required_input(node, 1)?;
    let y = importer.required_input(node, 2)?;
    validate_where_inputs(importer, node, &condition, &x, &y)?;
    let shape = condition.shape.broadcast_with(&x.shape).and_then(|shape| {
        let shape = Shape::create(shape.dimensions().to_vec())?;
        shape.broadcast_with(&y.shape)
    })?;
    Ok(WherePlan {
        condition,
        x,
        y,
        shape,
    })
}

fn validate_where_inputs(
    importer: &Importer,
    node: &Node,
    condition: &ImportedValue,
    x: &ImportedValue,
    y: &ImportedValue,
) -> Result<()> {
    if condition.data_type != DataType::Bool {
        return Err(Error::unsupported_operator(
            node.domain.as_deref(),
            "Where",
            importer.opset_version(node.domain.as_deref()),
            "condition must be bool",
        ));
    }

    if x.data_type != y.data_type {
        return Err(Error::unsupported_operator(
            node.domain.as_deref(),
            "Where",
            importer.opset_version(node.domain.as_deref()),
            "x and y dtypes must match",
        ));
    }

    if x.data_type != DataType::F32 {
        return Err(Error::unsupported_operator(
            node.domain.as_deref(),
            "Where",
            importer.opset_version(node.domain.as_deref()),
            "only f32 where branch inputs are supported",
        ));
    }
    Ok(())
}
