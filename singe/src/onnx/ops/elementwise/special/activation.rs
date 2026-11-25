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
            add_binary_intermediate, add_scale, elementwise::lower_binary, expect_input_count,
            expect_outputs, register_node_outputs,
        },
        utility::{node_label, node_name},
    },
    shape::transform::normalize_dim,
};

pub fn lower_leaky_relu(importer: &mut Importer, node: &Node) -> Result<()> {
    expect_input_count(importer, node, 1)?;
    expect_outputs(importer, node, 1)?;
    let attrs = Attributes::create(node)?;
    let input = importer.required_input(node, 0)?;
    if input.data_type != DataType::F32 {
        return Err(Error::unsupported_operator(
            node.domain.as_deref(),
            &node.operator_type,
            importer.opset_version(node.domain.as_deref()),
            "only f32 leakyrelu inputs are supported",
        ));
    }

    let alpha = attrs.float("alpha", 0.01)?;
    if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
        return Err(Error::unsupported_operator(
            node.domain.as_deref(),
            &node.operator_type,
            importer.opset_version(node.domain.as_deref()),
            "alpha must be finite and in [0, 1]",
        ));
    }

    let data_type = input.data_type;
    let shape = input.shape.clone();
    let scaled = add_scale(importer, input.clone(), alpha)?;
    let id = add_binary_intermediate(
        importer,
        node_name(node),
        Operation::Maximum,
        [input.node, scaled.node],
        data_type,
        shape.clone(),
    )?;
    register_node_outputs(
        importer,
        node,
        vec![ImportedValue::new(id, data_type, shape)],
    )
}

pub fn lower_softmax(importer: &mut Importer, node: &Node, operation: Operation) -> Result<()> {
    expect_input_count(importer, node, 1)?;
    expect_outputs(importer, node, 1)?;
    let attrs = Attributes::create(node)?;
    let input = importer.required_input(node, 0)?;
    let axis = attrs.int("axis", -1)?;
    let axis = normalize_dim(axis, input.shape.rank()).map_err(|_| {
        Error::invalid_attribute(
            node_label(node),
            "axis",
            format!(
                "expected axis in input rank {}, got {axis}",
                input.shape.rank()
            ),
        )
    })?;
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

pub fn lower_pow(importer: &mut Importer, node: &Node) -> Result<()> {
    expect_input_count(importer, node, 2)?;
    expect_outputs(importer, node, 1)?;
    let input = importer.required_input(node, 0)?;
    let exponent = importer.required_input(node, 1)?;
    let special = match importer.optional_static_input_tensor(node, 1)? {
        Some((_, tensor))
            if input.data_type == DataType::F32
                && tensor.data_type() == DataType::F32
                && tensor.shape().rank() == 0 =>
        {
            tensor
                .to_vec::<f32>()?
                .first()
                .copied()
                .and_then(pow_operation)
        }
        _ => None,
    };
    let Some(operation) = special else {
        if exponent.data_type != input.data_type {
            return Err(Error::unsupported_operator(
                node.domain.as_deref(),
                &node.operator_type,
                importer.opset_version(node.domain.as_deref()),
                "an exponent of another type than the base",
            ));
        }
        return lower_binary(importer, node, Operation::Pow);
    };
    let device = importer.options().device;
    let id = importer
        .graph()
        .add_operation_request(GraphOperationRequest::unary(
            node_name(node),
            operation,
            input.node,
            GraphOperationOutput::new(input.data_type, input.shape.clone(), device),
        ))?;
    register_node_outputs(
        importer,
        node,
        vec![ImportedValue::new(id, input.data_type, input.shape)],
    )
}

fn pow_operation(exponent: f32) -> Option<Operation> {
    match exponent {
        2.0 => Some(Operation::Square),
        0.5 => Some(Operation::Sqrt),
        -0.5 => Some(Operation::Rsqrt),
        -1.0 => Some(Operation::Reciprocal),
        _ => None,
    }
}
