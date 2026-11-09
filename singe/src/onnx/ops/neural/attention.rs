use singe_onnx::Node;

use crate::{
    data_type::DataType,
    graph::{GraphOperationOutput, GraphOperationRequest},
    onnx::{
        attributes::Attributes,
        error::{Error, Result},
        importer::{ImportedValue, Importer},
        ops::{expect_input_range, expect_outputs, optional_input, register_node_outputs},
        utility::{node_label, node_name},
    },
    shape::Shape,
};

pub fn lower_attention(importer: &mut Importer, node: &Node) -> Result<()> {
    expect_input_range(importer, node, 3, 7)?;
    expect_outputs(importer, node, 1)?;
    let attrs = Attributes::create(node)?;
    let query = importer.required_input(node, 0)?;
    let key = importer.required_input(node, 1)?;
    let value = importer.required_input(node, 2)?;
    validate_attention_inputs(importer, node, &query, &key, &value)?;
    reject_unsupported_attention_inputs(importer, node, &attrs)?;
    let scale = attention_scale(node, &attrs)?;
    let causal = attrs.int("is_causal", 0)? != 0;
    let shape = attention_output_shape(node, &query, &key, &value)?;
    let device = importer.options().device;
    let id = importer.graph().add_operation_request(
        GraphOperationRequest::scaled_dot_product_attention(
            node_name(node),
            query.node,
            key.node,
            value.node,
            scale,
            causal,
            GraphOperationOutput::new(DataType::F32, shape.clone(), device),
        ),
    )?;
    register_node_outputs(
        importer,
        node,
        vec![ImportedValue::new(id, DataType::F32, shape)],
    )
}

pub fn validate_attention_inputs(
    importer: &Importer,
    node: &Node,
    query: &ImportedValue,
    key: &ImportedValue,
    value: &ImportedValue,
) -> Result<()> {
    for (name, input) in [("Q", query), ("K", key), ("V", value)] {
        if input.data_type != DataType::F32 {
            return Err(Error::unsupported_operator(
                node.domain.as_deref(),
                &node.operator_type,
                importer.opset_version(node.domain.as_deref()),
                format!("{name} must be a float32 tensor"),
            ));
        }
        if input.shape.dimensions().len() != 4 {
            return Err(Error::unsupported_operator(
                node.domain.as_deref(),
                &node.operator_type,
                importer.opset_version(node.domain.as_deref()),
                format!("{name} must be a 4D tensor"),
            ));
        }
    }
    Ok(())
}

pub fn reject_unsupported_attention_inputs(
    importer: &Importer,
    node: &Node,
    attrs: &Attributes,
) -> Result<()> {
    for index in 3..node.inputs.len() {
        if optional_input(importer, node, index)?.is_some() {
            return Err(Error::unsupported_operator(
                node.domain.as_deref(),
                &node.operator_type,
                importer.opset_version(node.domain.as_deref()),
                "attention masks, kv cache, and nonpad lengths are not supported",
            ));
        }
    }

    if attrs.int("softmax_precision", 1)? != 1 {
        return Err(Error::unsupported_operator(
            node.domain.as_deref(),
            &node.operator_type,
            importer.opset_version(node.domain.as_deref()),
            "only float32 softmax precision is supported",
        ));
    }

    if attrs.int("qk_matmul_output_mode", 0)? != 0 {
        return Err(Error::unsupported_operator(
            node.domain.as_deref(),
            &node.operator_type,
            importer.opset_version(node.domain.as_deref()),
            "qk_matmul_output is not supported",
        ));
    }

    if attrs.float("softcap", 0.0)? != 0.0 {
        return Err(Error::unsupported_operator(
            node.domain.as_deref(),
            &node.operator_type,
            importer.opset_version(node.domain.as_deref()),
            "softcap is not supported",
        ));
    }
    Ok(())
}

pub fn attention_scale(node: &Node, attrs: &Attributes) -> Result<Option<f32>> {
    let scale = attrs.float("scale", f32::NAN)?;
    let scale = (!scale.is_nan()).then_some(scale);
    if scale.is_some_and(|scale| !scale.is_finite()) {
        return Err(Error::invalid_attribute(
            node_label(node),
            "scale",
            "scale must be finite",
        ));
    }
    Ok(scale)
}

pub fn attention_output_shape(
    node: &Node,
    query: &ImportedValue,
    key: &ImportedValue,
    value: &ImportedValue,
) -> Result<Shape> {
    let query_dims = query.shape.dimensions();
    let key_dims = key.shape.dimensions();
    let value_dims = value.shape.dimensions();
    if query_dims[0] != key_dims[0]
        || query_dims[0] != value_dims[0]
        || query_dims[1] != key_dims[1]
        || query_dims[1] != value_dims[1]
        || query_dims[3] != key_dims[3]
        || key_dims[2] != value_dims[2]
        || query_dims[0] < 0
        || query_dims[1] < 0
        || query_dims[2] < 0
        || query_dims[3] <= 0
        || key_dims[2] <= 0
        || value_dims[3] < 0
    {
        return Err(Error::UnsupportedShape {
            name: node_label(node),
            reason: "unsupported attention input shapes".into(),
        });
    }
    Ok(Shape::create([
        query_dims[0],
        query_dims[1],
        query_dims[2],
        value_dims[3],
    ])?)
}
