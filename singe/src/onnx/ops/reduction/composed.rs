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
            add_unary_intermediate, axes_from_attribute_or_input, expect_input_range,
            expect_outputs,
            reduction::{ComposedSumReductionKind, NormReductionKind, reduction_shape},
            register_node_outputs,
        },
        utility::node_name,
    },
    shape::Shape,
};

struct ComposedReductionInput {
    value: ImportedValue,
    axes: Vec<i64>,
    keepdim: bool,
}

fn composed_reduction_input(importer: &Importer, node: &Node) -> Result<ComposedReductionInput> {
    expect_input_range(importer, node, 1, 2)?;
    expect_outputs(importer, node, 1)?;
    let attrs = Attributes::create(node)?;
    let input = importer.required_input(node, 0)?;
    if input.data_type != DataType::F32 {
        return Err(Error::unsupported_operator(
            node.domain.as_deref(),
            &node.operator_type,
            importer.opset_version(node.domain.as_deref()),
            "only f32 composed reductions are supported",
        ));
    }

    let mut axes = axes_from_attribute_or_input(importer, node, "axes", 1)?.unwrap_or_default();
    if axes.is_empty() && attrs.int("noop_with_empty_axes", 0)? == 0 {
        axes = (0..input.shape.rank()).collect();
    }
    Ok(ComposedReductionInput {
        value: input,
        axes,
        keepdim: attrs.int("keepdims", 1)? != 0,
    })
}

fn reduction_output_shape(input: &ComposedReductionInput) -> Result<Shape> {
    if input.axes.is_empty() {
        return Ok(input.value.shape.clone());
    }
    reduction_shape(&input.value.shape, &input.axes, input.keepdim)
}

pub fn lower_norm_reduction(
    importer: &mut Importer,
    node: &Node,
    kind: NormReductionKind,
) -> Result<()> {
    let input = composed_reduction_input(importer, node)?;
    let shape = reduction_output_shape(&input)?;
    let pre_reduce = match kind {
        NormReductionKind::L1 => {
            add_unary_intermediate(importer, Operation::Abs, input.value.clone())?
        }
        NormReductionKind::L2 => add_unary_intermediate(importer, Operation::Square, input.value)?,
    };
    let device = importer.options().device;
    let reduced_id = if input.axes.is_empty() {
        pre_reduce.node
    } else {
        importer
            .graph()
            .add_operation_request(GraphOperationRequest::reduction(
                if matches!(kind, NormReductionKind::L2) {
                    None
                } else {
                    node_name(node)
                },
                Operation::Sum,
                pre_reduce.node,
                input.axes,
                input.keepdim,
                GraphOperationOutput::new(pre_reduce.data_type, shape.clone(), device),
            ))?
    };

    let id = if matches!(kind, NormReductionKind::L2) {
        importer
            .graph()
            .add_operation_request(GraphOperationRequest::unary(
                node_name(node),
                Operation::Sqrt,
                reduced_id,
                GraphOperationOutput::new(pre_reduce.data_type, shape.clone(), device),
            ))?
    } else {
        reduced_id
    };

    register_node_outputs(
        importer,
        node,
        vec![ImportedValue::new(id, pre_reduce.data_type, shape)],
    )
}

pub fn lower_composed_sum_reduction(
    importer: &mut Importer,
    node: &Node,
    kind: ComposedSumReductionKind,
) -> Result<()> {
    let input = composed_reduction_input(importer, node)?;
    let shape = reduction_output_shape(&input)?;
    let pre_reduce = match kind {
        ComposedSumReductionKind::SumSquare => {
            add_unary_intermediate(importer, Operation::Square, input.value)?
        }
        ComposedSumReductionKind::LogSum => input.value,
        ComposedSumReductionKind::LogSumExp => {
            add_unary_intermediate(importer, Operation::Exp, input.value)?
        }
    };
    let needs_post_log = matches!(
        kind,
        ComposedSumReductionKind::LogSum | ComposedSumReductionKind::LogSumExp
    );
    let device = importer.options().device;
    let reduced_id = if input.axes.is_empty() {
        pre_reduce.node
    } else {
        importer
            .graph()
            .add_operation_request(GraphOperationRequest::reduction(
                if needs_post_log {
                    None
                } else {
                    node_name(node)
                },
                Operation::Sum,
                pre_reduce.node,
                input.axes,
                input.keepdim,
                GraphOperationOutput::new(pre_reduce.data_type, shape.clone(), device),
            ))?
    };
    let id = if needs_post_log {
        importer
            .graph()
            .add_operation_request(GraphOperationRequest::unary(
                node_name(node),
                Operation::Log,
                reduced_id,
                GraphOperationOutput::new(pre_reduce.data_type, shape.clone(), device),
            ))?
    } else {
        reduced_id
    };

    register_node_outputs(
        importer,
        node,
        vec![ImportedValue::new(id, pre_reduce.data_type, shape)],
    )
}
