use singe_onnx::Node;

use crate::{
    backend::{
        registry::{Operation, OperationAttributes},
        spec,
    },
    data_type::DataType,
    graph::{GraphOperationInputs, GraphOperationOutput, GraphOperationRequest},
    onnx::{
        attributes::Attributes,
        error::{Error, Result},
        importer::{ImportedValue, Importer},
        ops::{expect_input_count, expect_outputs, register_node_outputs},
        utility::{node_label, node_name},
    },
    shape::{Shape, transform::normalize_dim},
};

pub fn lower_gather(importer: &mut Importer, node: &Node) -> Result<()> {
    expect_input_count(importer, node, 2)?;
    expect_outputs(importer, node, 1)?;
    if importer.lower_metadata_gather(node)? {
        return Ok(());
    }

    let plan = gather_plan(importer, node)?;
    let device = importer.options().device;
    let id = importer
        .graph()
        .add_operation_request(GraphOperationRequest::new(
            node_name(node),
            GraphOperationInputs::new(
                Operation::Take,
                OperationAttributes::IndexSelect(plan.axis),
                vec![plan.input.node, plan.indices.node],
            ),
            GraphOperationOutput::new(plan.input.data_type, plan.shape.clone(), device),
        ))?;
    register_node_outputs(
        importer,
        node,
        vec![ImportedValue::new(id, plan.input.data_type, plan.shape)],
    )
}

pub struct GatherPlan {
    pub(crate) input: ImportedValue,
    pub(crate) indices: ImportedValue,
    pub(crate) axis: i64,
    pub(crate) shape: Shape,
}

pub fn gather_plan(importer: &Importer, node: &Node) -> Result<GatherPlan> {
    let attrs = Attributes::create(node)?;
    let input = importer.required_input(node, 0)?;
    let indices = importer.required_input(node, 1)?;
    validate_indices(importer, node, &indices)?;
    let axis = normalized_axis(node, &attrs, &input)?;
    let shape =
        spec::gather_problem(&input.shape, &indices.shape, axis, "onnx gather")?.output_shape;
    Ok(GatherPlan {
        input,
        indices,
        axis,
        shape,
    })
}

fn validate_indices(importer: &Importer, node: &Node, indices: &ImportedValue) -> Result<()> {
    if !matches!(indices.data_type, DataType::I32 | DataType::I64) {
        return Err(Error::unsupported_operator(
            node.domain.as_deref(),
            "Gather",
            importer.opset_version(node.domain.as_deref()),
            "indices must be i32 or i64",
        ));
    }
    Ok(())
}

fn normalized_axis(node: &Node, attrs: &Attributes, input: &ImportedValue) -> Result<i64> {
    let axis = attrs.int("axis", 0)?;
    normalize_dim(axis, input.shape.rank()).map_err(|_| {
        Error::invalid_attribute(
            node_label(node),
            "axis",
            format!(
                "expected axis in input rank {}, got {axis}",
                input.shape.rank()
            ),
        )
    })
}
