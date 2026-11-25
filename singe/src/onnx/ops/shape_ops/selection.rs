use singe_onnx::Node;

use crate::{
    data_type::DataType,
    error::Result as SingeResult,
    graph::ir::ValueId,
    onnx::{
        attributes::Attributes,
        error::{Error, Result},
        importer::{ImportedValue, Importer},
        ops::{expect_input_count, expect_outputs, register_node_outputs},
    },
    shape::transform::normalize_dim,
    tensor::{Tensor, count::Count},
};

pub fn lower_nonzero(importer: &mut Importer, node: &Node) -> Result<()> {
    expect_input_count(importer, node, 1)?;
    expect_outputs(importer, node, 1)?;
    let input = importer.required_input(node, 0)?;
    let value = counted(importer, node, &[input.node], |inputs| {
        inputs[0].nonzero()?.transpose(0, 1)
    })?;
    register_node_outputs(importer, node, vec![value])
}

pub fn lower_compress(importer: &mut Importer, node: &Node) -> Result<()> {
    expect_input_count(importer, node, 2)?;
    expect_outputs(importer, node, 1)?;
    let attributes = Attributes::create(node)?;
    let input = importer.required_input(node, 0)?;
    let condition = importer.required_input(node, 1)?;
    if condition.data_type != DataType::Bool || condition.shape.rank() != 1 {
        return Err(refuse(
            importer,
            node,
            "a condition that is not a Boolean vector",
        ));
    }

    let axis = if attributes.has("axis") {
        Some(normalize_dim(
            attributes.int("axis", 0)?,
            input.shape.rank(),
        )?)
    } else {
        None
    };
    let value = counted(importer, node, &[input.node, condition.node], |inputs| {
        let rows = inputs[1].nonzero()?.squeeze(1)?;
        match axis {
            Some(axis) => inputs[0].index_select(axis, &rows),
            None => inputs[0].reshape([-1])?.index_select(0, &rows),
        }
    })?;
    register_node_outputs(importer, node, vec![value])
}

fn counted(
    importer: &mut Importer,
    node: &Node,
    operands: &[ValueId],
    trace: impl FnOnce(&[&Tensor]) -> SingeResult<Tensor>,
) -> Result<ImportedValue> {
    if !importer.have_fixed_shapes(operands)? {
        return Err(refuse(importer, node, "inputs with run-time axes"));
    }

    let device = importer.options().device;
    let mut axis = None;
    let ids = importer.graph().append_traced(operands, device, |inputs| {
        let output = trace(inputs)?;
        axis = output.count().map(Count::axis);
        Ok(vec![output])
    })?;
    let (&[value, count], Some(axis)) = (ids.as_slice(), axis) else {
        return Err(refuse(importer, node, "a selection without its count"));
    };
    let traced = importer.graph().required_node(value)?;
    Ok(ImportedValue::new(
        value,
        traced.output_data_type(),
        traced.output_shape().clone(),
    )
    .counted(axis, count))
}

fn refuse(importer: &Importer, node: &Node, reason: &str) -> Error {
    Error::unsupported_operator(
        node.domain.as_deref(),
        &node.operator_type,
        importer.opset_version(node.domain.as_deref()),
        reason,
    )
}
