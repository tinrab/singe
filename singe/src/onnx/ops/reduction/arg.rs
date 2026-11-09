use singe_onnx::Node;

use crate::{
    backend::registry::Operation,
    data_type::DataType,
    graph::{GraphOperationOutput, GraphOperationRequest},
    onnx::{
        attributes::Attributes,
        error::{Error, Result},
        importer::{ImportedValue, Importer},
        ops::{
            expect_input_count, expect_outputs, reduction::reduction_axis_to_usize,
            register_node_outputs,
        },
        utility::node_name,
    },
    shape::{Shape, transform::normalize_dim},
};

pub fn lower_arg_reduction(
    importer: &mut Importer,
    node: &Node,
    operation: Operation,
) -> Result<()> {
    expect_input_count(importer, node, 1)?;
    expect_outputs(importer, node, 1)?;
    let input = importer.required_input(node, 0)?;
    let plan = arg_reduction_plan(importer, node, &input)?;
    let device = importer.options().device;
    let reduced_id = importer
        .graph()
        .add_operation_request(GraphOperationRequest::dimension(
            if plan.keepdim && input.shape.rank() > 1 {
                None
            } else {
                node_name(node)
            },
            operation,
            input.node,
            plan.axis,
            GraphOperationOutput::new(DataType::I64, plan.reduced_shape.clone(), device),
        ))?;

    let (id, shape) = if plan.keepdim && input.shape.rank() > 1 {
        let shape = plan.reduced_shape.unsqueeze(plan.axis)?;
        let id = importer
            .graph()
            .add_operation_request(GraphOperationRequest::dimension(
                node_name(node),
                Operation::Unsqueeze,
                reduced_id,
                plan.axis,
                GraphOperationOutput::new(DataType::I64, shape.clone(), device),
            ))?;
        (id, shape)
    } else {
        (reduced_id, plan.reduced_shape)
    };

    register_node_outputs(
        importer,
        node,
        vec![ImportedValue::new(id, DataType::I64, shape)],
    )
}

pub struct ArgReductionPlan {
    pub(crate) axis: i64,
    pub(crate) keepdim: bool,
    pub(crate) reduced_shape: Shape,
}

pub fn arg_reduction_plan(
    importer: &Importer,
    node: &Node,
    input: &ImportedValue,
) -> Result<ArgReductionPlan> {
    let attrs = Attributes::create(node)?;
    if attrs.int("select_last_index", 0)? != 0 {
        return Err(Error::unsupported_operator(
            node.domain.as_deref(),
            &node.operator_type,
            importer.opset_version(node.domain.as_deref()),
            "select_last_index is not supported",
        ));
    }

    let axis = attrs.int("axis", 0)?;
    let axis = normalize_dim(axis, input.shape.rank())?;
    Ok(ArgReductionPlan {
        axis,
        keepdim: attrs.int("keepdims", 1)? != 0,
        reduced_shape: arg_reduction_shape(&input.shape, axis)?,
    })
}

fn arg_reduction_shape(input: &Shape, axis: i64) -> Result<Shape> {
    let mut dimensions = input.dimensions().to_vec();
    dimensions.remove(reduction_axis_to_usize(axis)?);
    Shape::create(dimensions).map_err(Into::into)
}
