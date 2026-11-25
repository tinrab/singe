use singe_onnx::Node;

use crate::{
    backend::registry::Operation,
    data_type::DataType,
    graph::{GraphOperationOutput, GraphOperationRequest},
    onnx::{
        attributes::Attributes,
        error::{Error, Result},
        importer::{ImportedValue, Importer},
        ops::{expect_input_count, expect_outputs, register_node_outputs},
        utility::node_name,
    },
    shape::transform::normalize_dim,
};

fn refuse(importer: &Importer, node: &Node, reason: &str) -> Error {
    Error::unsupported_operator(
        node.domain.as_deref(),
        &node.operator_type,
        importer.opset_version(node.domain.as_deref()),
        reason,
    )
}

pub fn lower_cumulative(importer: &mut Importer, node: &Node, operation: Operation) -> Result<()> {
    expect_input_count(importer, node, 2)?;
    expect_outputs(importer, node, 1)?;
    let attrs = Attributes::create(node)?;
    if attrs.int("exclusive", 0)? != 0 || attrs.int("reverse", 0)? != 0 {
        return Err(refuse(importer, node, "exclusive or reversed scans"));
    }

    let input = importer.required_input(node, 0)?;
    if !input.data_type.is_float() && input.data_type != DataType::I64 {
        return Err(refuse(
            importer,
            node,
            "integers narrower than int64, which Singe accumulates in int64",
        ));
    }

    let (_, axis) = importer.required_static_input_tensor(node, 1, "axis")?;
    let [axis] = axis.cast(DataType::I64)?.to_vec::<i64>()?[..] else {
        return Err(refuse(importer, node, "an axis other than one integer"));
    };
    let axis = normalize_dim(axis, input.shape.rank())?;
    let device = importer.options().device;
    let id = importer
        .graph()
        .add_operation_request(GraphOperationRequest::dimension(
            node_name(node),
            operation,
            input.node,
            axis,
            GraphOperationOutput::new(input.data_type, input.shape.clone(), device),
        ))?;
    register_node_outputs(
        importer,
        node,
        vec![ImportedValue::new(id, input.data_type, input.shape)],
    )
}
