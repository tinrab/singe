use singe_onnx::Node;

use crate::{
    graph::node::NodeKind,
    onnx::{
        error::{Error, Result},
        importer::{ImportedValue, Importer},
        ops::{
            expect_input_range, expect_outputs, normalize_slice_index, normalized_axes,
            register_node_outputs, shape_input_values,
            shape_ops::{onnx_dimension_to_usize, onnx_len_to_i64},
        },
        utility::{node_label, node_name},
    },
    shape::{Shape, transform::normalize_dim},
};

pub fn lower_slice(importer: &mut Importer, node: &Node) -> Result<()> {
    expect_input_range(importer, node, 3, 5)?;
    expect_outputs(importer, node, 1)?;
    let input = importer.required_input(node, 0)?;
    let params = slice_parameters(importer, node)?;
    let current = apply_slice_parameters(importer, node, input, params)?;
    register_node_outputs(importer, node, vec![current])
}

pub fn apply_slice_parameters(
    importer: &mut Importer,
    node: &Node,
    input: ImportedValue,
    params: Vec<SliceParameter>,
) -> Result<ImportedValue> {
    let mut current = input;
    for parameter in params {
        current = apply_slice_parameter(importer, node, &current, parameter)?;
    }
    make_contiguous(importer, node, &current)
}

fn apply_slice_parameter(
    importer: &mut Importer,
    node: &Node,
    current: &ImportedValue,
    parameter: SliceParameter,
) -> Result<ImportedValue> {
    if parameter.step != 1 {
        return Err(Error::unsupported_operator(
            node.domain.as_deref(),
            "Slice",
            importer.opset_version(node.domain.as_deref()),
            "only step=1 slices are supported",
        ));
    }

    let axis = normalize_dim(parameter.axis, current.shape.rank())?;
    let axis_index = onnx_dimension_to_usize(node, "slice axis", axis)?;
    let dim = current.shape.dimensions()[axis_index];
    let start = normalize_slice_index(parameter.start, dim);
    let end = normalize_slice_index(parameter.end, dim);
    let length = end.saturating_sub(start).max(0);
    let mut dimensions = current.shape.dimensions().to_vec();
    dimensions[axis_index] = length;
    let shape = Shape::create(dimensions)?;
    let device = importer.options().device;
    let id = importer.graph().add_named_slice_operation(
        None,
        current.node,
        axis,
        start,
        length,
        current.data_type,
        &shape,
        device,
    )?;
    Ok(ImportedValue::new(id, current.data_type, shape))
}

fn make_contiguous(
    importer: &mut Importer,
    node: &Node,
    current: &ImportedValue,
) -> Result<ImportedValue> {
    let shape = Shape::create(current.shape.dimensions().to_vec())?;
    let device = importer.options().device;
    let id = importer.graph().add_named_operation(
        node_name(node),
        NodeKind::contiguous(current.node),
        current.data_type,
        shape.clone(),
        device,
    )?;
    Ok(ImportedValue::new(id, current.data_type, shape))
}

#[derive(Clone, Copy)]
pub struct SliceParameter {
    pub(crate) start: i64,
    pub(crate) end: i64,
    pub(crate) axis: i64,
    pub(crate) step: i64,
}

pub fn slice_parameters(importer: &Importer, node: &Node) -> Result<Vec<SliceParameter>> {
    let starts = shape_input_values(importer, node, 1, "starts")?;
    let ends = shape_input_values(importer, node, 2, "ends")?;
    let axes = match node.inputs.get(3) {
        Some(name) if !name.is_empty() => shape_input_values(importer, node, 3, "axes")?,
        _ => (0..onnx_len_to_i64(node, "slice starts", starts.len())?).collect(),
    };
    normalized_axes(&axes, importer.required_input(node, 0)?.shape.rank())?;
    let steps = match node.inputs.get(4) {
        Some(name) if !name.is_empty() => shape_input_values(importer, node, 4, "steps")?,
        _ => vec![1; starts.len()],
    };
    if starts.len() != ends.len() || starts.len() != axes.len() || starts.len() != steps.len() {
        return Err(Error::UnsupportedShape {
            name: node_label(node),
            reason: "slice starts, ends, axes, and steps must have equal lengths".into(),
        });
    }
    Ok(starts
        .into_iter()
        .zip(ends)
        .zip(axes.into_iter().zip(steps))
        .map(|((start, end), (axis, step))| SliceParameter {
            start,
            end,
            axis,
            step,
        })
        .collect())
}
