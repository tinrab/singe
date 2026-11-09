use singe_onnx::Node;

use crate::{
    backend::registry::Operation,
    graph::{GraphOperationOutput, GraphOperationRequest},
    onnx::{
        attributes::Attributes,
        error::Result,
        importer::{ImportedValue, Importer},
        ops::{
            axes_from_attribute_or_input, expect_input_range, expect_outputs,
            reduction::{ReductionKind, reduction_shape},
            register_node_outputs,
        },
        utility::node_name,
    },
};

pub fn lower_reduction(
    importer: &mut Importer,
    node: &Node,
    reduction: ReductionKind,
) -> Result<()> {
    expect_input_range(importer, node, 1, 2)?;
    expect_outputs(importer, node, 1)?;
    let attrs = Attributes::create(node)?;
    let input = importer.required_input(node, 0)?;
    let mut axes = axes_from_attribute_or_input(importer, node, "axes", 1)?.unwrap_or_default();
    if axes.is_empty() && attrs.int("noop_with_empty_axes", 0)? != 0 {
        return register_node_outputs(importer, node, vec![input]);
    }

    if axes.is_empty() {
        axes = (0..input.shape.rank()).collect();
    }

    let keepdim = attrs.int("keepdims", 1)? != 0;
    let shape = reduction_shape(&input.shape, &axes, keepdim)?;
    let operation = match reduction {
        ReductionKind::Mean => Operation::Mean,
        ReductionKind::Sum => Operation::Sum,
        ReductionKind::Max => Operation::Max,
        ReductionKind::Min => Operation::Min,
        ReductionKind::Product => Operation::Product,
    };
    let device = importer.options().device;
    let id = importer
        .graph()
        .add_operation_request(GraphOperationRequest::reduction(
            node_name(node),
            operation,
            input.node,
            axes,
            keepdim,
            GraphOperationOutput::new(input.data_type, shape.clone(), device),
        ))?;
    register_node_outputs(
        importer,
        node,
        vec![ImportedValue::new(id, input.data_type, shape)],
    )
}
