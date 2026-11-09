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
            add_binary_intermediate, add_scale, expect_input_count, expect_outputs,
            register_node_outputs, static_f32_scalar,
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

pub fn lower_softmax(importer: &mut Importer, node: &Node, log: bool) -> Result<()> {
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
    let operation = if log {
        Operation::LogSoftmax
    } else {
        Operation::Softmax
    };
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
    if input.data_type != DataType::F32 {
        return Err(Error::unsupported_operator(
            node.domain.as_deref(),
            &node.operator_type,
            importer.opset_version(node.domain.as_deref()),
            "only f32 pow inputs are supported",
        ));
    }

    let exponent = static_f32_scalar(importer, node, 1, "exponent")?;
    let operation = pow_operation(importer, node, exponent)?;
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

#[expect(
    clippy::float_cmp,
    reason = "exact equality selects an identity, specialization, or deterministic tie; approximate equality changes semantics"
)]
pub fn pow_operation(importer: &Importer, node: &Node, exponent: f32) -> Result<Operation> {
    if exponent == 2.0 {
        Ok(Operation::Square)
    } else if exponent == 0.5 {
        Ok(Operation::Sqrt)
    } else if exponent == -0.5 {
        Ok(Operation::Rsqrt)
    } else if exponent == -1.0 {
        Ok(Operation::Reciprocal)
    } else {
        Err(Error::unsupported_operator(
            node.domain.as_deref(),
            &node.operator_type,
            importer.opset_version(node.domain.as_deref()),
            format!("static exponent {exponent} is not supported"),
        ))
    }
}
