use singe_onnx::Node;

use crate::{
    data_type::DataType,
    graph::{DimensionExpr, SymbolicAttributes, capture, node::NodeKind},
    onnx::{
        error::{Error, Result},
        importer::{ImportedValue, Importer},
        ops::{
            expect_input_range, expect_outputs, normalized_axes, register_node_outputs,
            shape_input_expressions, shape_input_values,
            shape_ops::{onnx_dimension_to_usize, onnx_len_to_i64},
        },
        utility::{node_label, node_name},
    },
    shape::{Shape, transform::normalize_dim},
    tensor::{Tensor, count::Count},
};

pub fn lower_slice(importer: &mut Importer, node: &Node) -> Result<()> {
    expect_input_range(importer, node, 3, 5)?;
    expect_outputs(importer, node, 1)?;
    let input = importer.required_input(node, 0)?;
    if has_run_time_end(importer, node)? {
        let value = counted_slice(importer, node, &input)?;
        return register_node_outputs(importer, node, vec![value]);
    }

    let params = slice_parameters(importer, node)?;
    let current = apply_slice_parameters(importer, node, input, &params)?;
    register_node_outputs(importer, node, vec![current])
}

fn has_run_time_end(importer: &Importer, node: &Node) -> Result<bool> {
    let name = &node.inputs[2];
    Ok(importer.metadata_shape_value(name)?.is_none()
        && importer.static_input_tensor(name)?.is_none())
}

fn counted_slice(
    importer: &mut Importer,
    node: &Node,
    input: &ImportedValue,
) -> Result<ImportedValue> {
    let starts = shape_input_values(importer, node, 1, "starts")?;
    let axes = optional_values(importer, node, 3, "axes", 0)?;
    let steps = optional_values(importer, node, 4, "steps", 1)?;
    let (&[0], &[axis], &[1]) = (starts.as_slice(), axes.as_slice(), steps.as_slice()) else {
        return Err(refuse(
            importer,
            node,
            "a slice to a run-time end that does not start at zero and step by one along one axis",
        ));
    };
    if !importer.have_fixed_shapes(&[input.node])? {
        return Err(refuse(
            importer,
            node,
            "a slice to a run-time end of an input with run-time axes",
        ));
    }

    let axis =
        onnx_dimension_to_usize(node, "slice axis", normalize_dim(axis, input.shape.rank())?)?;
    let extent = input.shape.dimensions()[axis];
    let end = importer.required_input(node, 2)?;
    let device = importer.options().device;
    let ids = importer
        .graph()
        .append_traced(&[input.node, end.node], device, |inputs| {
            let end = inputs[1].cast(DataType::I64)?.reshape([])?;
            let literal =
                |value: i64| capture::literal(end.device(), |device| Tensor::scalar(device, value));
            let (zero, extent) = (literal(0)?, literal(extent)?);
            let wrapped = Tensor::where_select(&end.less(&zero)?, &end.add(&extent)?, &end)?;
            let rows = wrapped.maximum(&zero)?.minimum(&extent)?;
            Ok(vec![inputs[0].with_count(Count::create(axis, rows)?)?])
        })?;
    let &[value, count] = ids.as_slice() else {
        return Err(refuse(importer, node, "a slice without its count"));
    };
    Ok(ImportedValue::new(value, input.data_type, input.shape.clone()).counted(axis, count))
}

fn optional_values(
    importer: &Importer,
    node: &Node,
    index: usize,
    label: &str,
    default: i64,
) -> Result<Vec<i64>> {
    match node.inputs.get(index) {
        Some(name) if !name.is_empty() => shape_input_values(importer, node, index, label),
        _ => Ok(vec![default]),
    }
}

fn refuse(importer: &Importer, node: &Node, reason: &str) -> Error {
    Error::unsupported_operator(
        node.domain.as_deref(),
        "Slice",
        importer.opset_version(node.domain.as_deref()),
        reason,
    )
}

pub fn apply_slice_parameters(
    importer: &mut Importer,
    node: &Node,
    input: ImportedValue,
    params: &[SliceParameter],
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
    parameter: &SliceParameter,
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
    let extent = importer.metadata_dimension(current.node, axis_index)?;
    let start = slice_bound(importer, node, &parameter.start, &extent, dim)?;
    let end = slice_bound(importer, node, &parameter.end, &extent, dim)?;
    let (first, last) = (
        importer.evaluate_dimension(&start)?,
        importer.evaluate_dimension(&end)?,
    );
    let length = last.saturating_sub(first).max(0);
    let mut dimensions = current.shape.dimensions().to_vec();
    dimensions[axis_index] = length;
    let shape = Shape::create(dimensions)?;
    let device = importer.options().device;
    let id = importer.graph().add_named_slice_operation(
        None,
        current.node,
        axis,
        first,
        length,
        current.data_type,
        &shape,
        device,
    )?;
    if !(start.is_constant() && end.is_constant()) {
        importer.graph().shape_program.attributes.insert(
            id,
            SymbolicAttributes::Slice {
                dim: axis,
                length: end - start.clone(),
                start,
            },
        );
    }
    Ok(ImportedValue::new(id, current.data_type, shape))
}

// A slice bound as an expression over the axis's `extent`: a negative constant counts from its end, and one past `dim`, the traced extent, stands for the whole axis.
// A bound computed from shapes stays as it is, and must lie within the axis.
fn slice_bound(
    importer: &Importer,
    node: &Node,
    bound: &DimensionExpr,
    extent: &DimensionExpr,
    dim: i64,
) -> Result<DimensionExpr> {
    let value = importer.evaluate_dimension(bound)?;
    if !bound.is_constant() {
        if !(0..=dim).contains(&value) {
            return Err(Error::UnsupportedShape {
                name: node_label(node),
                reason: format!("a run-time slice bound {value} outside the axis of {dim}"),
            });
        }
        return Ok(bound.clone());
    }

    Ok(if value < 0 && value >= -dim {
        extent.clone() + value
    } else if value < 0 {
        DimensionExpr::constant(0)
    } else if value >= dim {
        extent.clone()
    } else {
        bound.clone()
    })
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

#[derive(Clone)]
pub struct SliceParameter {
    pub(crate) start: DimensionExpr,
    pub(crate) end: DimensionExpr,
    pub(crate) axis: i64,
    pub(crate) step: i64,
}

pub fn slice_parameters(importer: &Importer, node: &Node) -> Result<Vec<SliceParameter>> {
    let starts = shape_input_expressions(importer, node, 1, "starts")?;
    let ends = shape_input_expressions(importer, node, 2, "ends")?;
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
