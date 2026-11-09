use singe_onnx::{Node, TensorDataType};

use crate::{
    backend::registry::Operation,
    graph::{GraphOperationOutput, GraphOperationRequest},
    onnx::{
        attributes::Attributes,
        error::{Error, Result},
        importer::{ImportedValue, Importer},
        ops::{expect_input_range, optional_input, register_node_outputs},
        utility::{node_label, node_name},
    },
    shape::normalize_dim_usize,
};

pub fn lower_batch_normalization(importer: &mut Importer, node: &Node) -> Result<()> {
    expect_input_range(importer, node, 5, 5)?;
    if node.outputs.len() != 1 {
        return Err(Error::unsupported_operator(
            node.domain.as_deref(),
            "BatchNormalization",
            importer.opset_version(node.domain.as_deref()),
            "training outputs are not supported",
        ));
    }

    let attrs = Attributes::create(node)?;
    if attrs.int("training_mode", 0)? != 0 {
        return Err(Error::unsupported_operator(
            node.domain.as_deref(),
            "BatchNormalization",
            importer.opset_version(node.domain.as_deref()),
            "training mode is not supported",
        ));
    }

    let input = importer.required_input(node, 0)?;
    let weight = importer.required_input(node, 1)?;
    let bias = importer.required_input(node, 2)?;
    let running_mean = importer.required_input(node, 3)?;
    let running_var = importer.required_input(node, 4)?;
    let device = importer.options().device;
    let id = importer
        .graph()
        .add_operation_request(GraphOperationRequest::batch_normalization(
            node_name(node),
            input.node,
            Some(weight.node),
            Some(bias.node),
            Some(running_mean.node),
            Some(running_var.node),
            false,
            attrs.float("momentum", 0.9)?,
            attrs.float("epsilon", 1e-5)?,
            GraphOperationOutput::new(input.data_type, input.shape.clone(), device),
        ))?;
    register_node_outputs(
        importer,
        node,
        vec![ImportedValue::new(id, input.data_type, input.shape)],
    )
}

pub fn lower_layer_normalization(importer: &mut Importer, node: &Node) -> Result<()> {
    expect_input_range(importer, node, 2, 3)?;
    let attrs = Attributes::create(node)?;
    let input = importer.required_input(node, 0)?;
    let weight = importer.required_input(node, 1)?;
    let bias = optional_input(importer, node, 2)?;
    let normalized_shape = normalized_shape(&attrs, &input)?;
    validate_layer_norm_io(
        importer,
        node,
        &attrs,
        &input,
        &weight,
        bias.as_ref(),
        &normalized_shape,
    )?;
    let epsilon = layer_norm_epsilon(&attrs)?;
    let device = importer.options().device;
    let id = importer
        .graph()
        .add_operation_request(GraphOperationRequest::layer_normalization(
            node_name(node),
            Operation::LayerNorm,
            input.node,
            normalized_shape,
            Some(weight.node),
            bias.as_ref().map(|bias| bias.node),
            epsilon,
            GraphOperationOutput::new(input.data_type, input.shape.clone(), device),
        ))?;
    register_node_outputs(
        importer,
        node,
        vec![ImportedValue::new(id, input.data_type, input.shape)],
    )
}

pub fn normalized_shape(attrs: &Attributes, input: &ImportedValue) -> Result<Vec<i64>> {
    let axis = attrs.int("axis", -1)?;
    let axis = normalize_dim_usize(axis, input.shape.rank())?;
    Ok(input.shape.dimensions()[axis..].to_vec())
}

pub fn validate_layer_norm_io(
    importer: &Importer,
    node: &Node,
    attrs: &Attributes,
    input: &ImportedValue,
    weight: &ImportedValue,
    bias: Option<&ImportedValue>,
    normalized_shape: &[i64],
) -> Result<()> {
    validate_supported_outputs(importer, node)?;
    validate_stash_type(importer, node, attrs)?;
    validate_data_types(node, input, weight, bias)?;
    validate_parameter_shape(node, "scale", weight, normalized_shape)?;
    if let Some(bias) = bias {
        validate_parameter_shape(node, "bias", bias, normalized_shape)?;
    }
    Ok(())
}

pub fn layer_norm_epsilon(attrs: &Attributes) -> Result<f32> {
    attrs.float("epsilon", 1e-5)
}

fn validate_supported_outputs(importer: &Importer, node: &Node) -> Result<()> {
    if node.outputs.len() == 1 {
        return Ok(());
    }
    Err(Error::unsupported_operator(
        node.domain.as_deref(),
        "LayerNormalization",
        importer.opset_version(node.domain.as_deref()),
        "mean and inverse standard deviation outputs are not supported",
    ))
}

fn validate_stash_type(importer: &Importer, node: &Node, attrs: &Attributes) -> Result<()> {
    if attrs.int("stash_type", 1)? == 1 {
        return Ok(());
    }
    Err(Error::unsupported_operator(
        node.domain.as_deref(),
        "LayerNormalization",
        importer.opset_version(node.domain.as_deref()),
        "only float stash type is supported",
    ))
}

fn validate_data_types(
    node: &Node,
    input: &ImportedValue,
    weight: &ImportedValue,
    bias: Option<&ImportedValue>,
) -> Result<()> {
    if input.data_type == weight.data_type
        && bias
            .as_ref()
            .is_none_or(|bias| bias.data_type == input.data_type)
    {
        return Ok(());
    }
    Err(Error::UnsupportedType {
        name: node_label(node),
        data_type: TensorDataType::Float,
        reason: "layer normalization input, scale, and bias dtypes must match".into(),
    })
}

fn validate_parameter_shape(
    node: &Node,
    label: &str,
    value: &ImportedValue,
    normalized_shape: &[i64],
) -> Result<()> {
    if value.shape.dimensions() == normalized_shape {
        return Ok(());
    }
    Err(Error::UnsupportedShape {
        name: node_label(node),
        reason: format!(
            "{label} shape {:?} must match normalized shape {:?}",
            value.shape.dimensions(),
            normalized_shape
        ),
    })
}
