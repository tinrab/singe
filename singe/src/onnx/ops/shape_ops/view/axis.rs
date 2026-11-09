use singe_onnx::Node;

use crate::{
    backend::registry::Operation,
    graph::{GraphOperationOutput, GraphOperationRequest},
    onnx::{
        error::{Error, Result},
        importer::{ImportedValue, Importer},
        ops::{
            axes_from_attribute_or_input, expect_input_range, expect_outputs, normalized_axes,
            normalized_unsqueeze_axes, register_node_outputs, shape_ops::onnx_dimension_to_usize,
        },
        utility::node_label,
    },
};

pub fn lower_squeeze(importer: &mut Importer, node: &Node) -> Result<()> {
    expect_input_range(importer, node, 1, 2)?;
    expect_outputs(importer, node, 1)?;
    let input = importer.required_input(node, 0)?;
    let axes = match axes_from_attribute_or_input(importer, node, "axes", 1)? {
        Some(axes) => axes,
        None => input
            .shape
            .dimensions()
            .iter()
            .enumerate()
            .filter(|(_, dimension)| **dimension == 1)
            .map(|(axis, _)| {
                i64::try_from(axis).map_err(|_| {
                    Error::invalid_attribute(node_label(node), "axes", "squeeze axis exceeds i64")
                })
            })
            .collect::<Result<Vec<_>>>()?,
    };
    let mut axes = normalized_axes(&axes, input.shape.rank())?;
    for &axis in &axes {
        let axis_index = onnx_dimension_to_usize(node, "squeeze axis", axis)?;
        if input.shape.dimensions()[axis_index] != 1 {
            return Err(Error::unsupported_operator(
                node.domain.as_deref(),
                &node.operator_type,
                importer.opset_version(node.domain.as_deref()),
                "squeeze axes must refer to dimensions with size 1",
            ));
        }
    }

    axes.sort_unstable_by(|a, b| b.cmp(a));
    let mut current = input;
    for axis in axes {
        let shape = current.shape.squeeze(axis)?;
        let device = importer.options().device;
        let id = importer
            .graph()
            .add_operation_request(GraphOperationRequest::dimension(
                None,
                Operation::Squeeze,
                current.node,
                axis,
                GraphOperationOutput::new(current.data_type, shape.clone(), device),
            ))?;
        current = ImportedValue::new(id, current.data_type, shape);
    }
    register_node_outputs(importer, node, vec![current])
}

pub fn lower_unsqueeze(importer: &mut Importer, node: &Node) -> Result<()> {
    expect_input_range(importer, node, 1, 2)?;
    expect_outputs(importer, node, 1)?;
    let input = importer.required_input(node, 0)?;
    let axes = axes_from_attribute_or_input(importer, node, "axes", 1)?
        .ok_or_else(|| Error::invalid_attribute(node_label(node), "axes", "missing axes"))?;
    let mut axes = normalized_unsqueeze_axes(&axes, input.shape.rank(), axes.len())?;
    axes.sort_unstable();
    let mut current = input;
    for axis in axes {
        let shape = current.shape.unsqueeze(axis)?;
        let device = importer.options().device;
        let id = importer
            .graph()
            .add_operation_request(GraphOperationRequest::dimension(
                None,
                Operation::Unsqueeze,
                current.node,
                axis,
                GraphOperationOutput::new(current.data_type, shape.clone(), device),
            ))?;
        current = ImportedValue::new(id, current.data_type, shape);
    }
    register_node_outputs(importer, node, vec![current])
}
