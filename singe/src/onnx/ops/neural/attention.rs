use singe_onnx::Node;

use crate::{
    backend::registry::{Operation, OperationAttributes, ScaledDotProductAttentionAttributes},
    data_type::DataType,
    error::Error as SingeError,
    graph::{GraphOperationInputs, GraphOperationOutput, GraphOperationRequest, node::NodeKind},
    onnx::{
        attributes::Attributes,
        error::{Error, Result},
        importer::{ImportedValue, Importer},
        ops::{expect_input_range, expect_outputs, optional_input, register_node_outputs},
        utility::{node_label, node_name},
    },
    shape::Shape,
    tensor::AttentionOptions,
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
    let mask = optional_input(importer, node, 3)?;
    let lengths = optional_input(importer, node, 6)?;
    // Singe aligns a causal mask to the key lengths at the bottom right, where ONNX counts it from the first key.
    if lengths.is_some() && (mask.is_some() || attrs.int("is_causal", 0)? != 0) {
        return Err(refuse(
            importer,
            node,
            "nonpad lengths with a mask or a causal mask",
        ));
    }

    let softcap = attrs.float("softcap", 0.0)?;
    // A query sees the `left_window_size` keys before it, and Singe's window hides keys `window` or more positions back.
    let left = attrs.int("left_window_size", -1)?;
    if let Some(boolean) = &mask
        && boolean.data_type == DataType::Bool
    {
        let options = AttentionOptions {
            scale: attention_scale(node, &attrs)?,
            causal: attrs.int("is_causal", 0)? != 0,
            window: (left >= 0).then_some(left + 1),
            softcap: (softcap != 0.0).then_some(softcap),
        };
        return lower_boolean_mask(importer, node, [&query, &key, &value, boolean], options);
    }

    if let Some(mask) = &mask
        && mask.data_type != query.data_type
    {
        return Err(refuse(
            importer,
            node,
            "a mask of another type than the query or bool",
        ));
    }

    let attention = ScaledDotProductAttentionAttributes {
        scale: attention_scale(node, &attrs)?,
        causal: attrs.int("is_causal", 0)? != 0,
        window: (left >= 0).then_some(left + 1),
        softcap: (softcap != 0.0).then_some(softcap),
        masked: mask.is_some(),
        sinks: false,
        key_lengths: lengths.is_some(),
    };
    let shape = attention_output_shape(node, &query, &key, &value)?;
    let mut operands = vec![query.node, key.node, value.node];
    operands.extend(mask.map(|mask| mask.node));
    let device = importer.options().device;
    if let Some(lengths) = lengths {
        if lengths.data_type != DataType::I64
            || lengths.shape.dimensions() != [shape.dimensions()[0]]
        {
            return Err(refuse(
                importer,
                node,
                "nonpad lengths that are not int64 with one per batch",
            ));
        }

        let cast = importer.graph().add_named_operation(
            node_name(node).map(|name| format!("{name}.lengths")),
            NodeKind::cast(lengths.node, DataType::I32),
            DataType::I32,
            lengths.shape,
            device,
        )?;
        operands.push(cast);
    }

    let id = importer
        .graph()
        .add_operation_request(GraphOperationRequest::new(
            node_name(node),
            GraphOperationInputs::new(
                Operation::ScaledDotProductAttention,
                OperationAttributes::ScaledDotProductAttention(attention),
                operands,
            ),
            GraphOperationOutput::new(query.data_type, shape.clone(), device),
        ))?;
    register_node_outputs(
        importer,
        node,
        vec![ImportedValue::new(id, query.data_type, shape)],
    )
}

fn lower_boolean_mask(
    importer: &mut Importer,
    node: &Node,
    operands: [&ImportedValue; 4],
    options: AttentionOptions,
) -> Result<()> {
    let nodes = operands.map(|operand| operand.node);
    if !importer.have_fixed_shapes(&nodes)? {
        return Err(refuse(importer, node, "a boolean mask with run-time axes"));
    }

    let device = importer.options().device;
    let [id] = importer.graph().append_traced(&nodes, device, |inputs| {
        let [query, key, value, mask] = inputs else {
            return Err(SingeError::missing_input("attention"));
        };
        Ok(vec![query.attend(key, value, Some(mask), None, options)?])
    })?[..] else {
        return Err(refuse(importer, node, "an attention without one result"));
    };
    let traced = importer.graph().required_node(id)?;
    let value = ImportedValue::new(id, traced.output_data_type(), traced.output_shape().clone());
    register_node_outputs(importer, node, vec![value])
}

fn refuse(importer: &Importer, node: &Node, reason: &str) -> Error {
    Error::unsupported_operator(
        node.domain.as_deref(),
        &node.operator_type,
        importer.opset_version(node.domain.as_deref()),
        reason,
    )
}

fn validate_attention_inputs(
    importer: &Importer,
    node: &Node,
    query: &ImportedValue,
    key: &ImportedValue,
    value: &ImportedValue,
) -> Result<()> {
    if !matches!(
        query.data_type,
        DataType::F16 | DataType::Bf16 | DataType::F32 | DataType::F64
    ) || key.data_type != query.data_type
        || value.data_type != query.data_type
    {
        return Err(refuse(
            importer,
            node,
            "query, key, and value must be floats of one type",
        ));
    }

    for input in [query, key, value] {
        if input.shape.dimensions().len() != 4 {
            return Err(refuse(
                importer,
                node,
                "query, key, and value must be 4D tensors",
            ));
        }
    }
    Ok(())
}

fn reject_unsupported_attention_inputs(
    importer: &Importer,
    node: &Node,
    attrs: &Attributes,
) -> Result<()> {
    for index in 4..node.inputs.len().min(6) {
        if optional_input(importer, node, index)?.is_some() {
            return Err(refuse(importer, node, "a kv cache"));
        }
    }

    if attrs.int("softmax_precision", 1)? != 1 {
        return Err(refuse(
            importer,
            node,
            "only float32 softmax precision is supported",
        ));
    }

    if attrs.int("qk_matmul_output_mode", 0)? != 0 {
        return Err(refuse(importer, node, "qk_matmul_output is not supported"));
    }

    if attrs.int("right_window_size", -1)? != -1 {
        return Err(refuse(
            importer,
            node,
            "windows that hide later keys are not supported",
        ));
    }

    if attrs.float("softcap", 0.0)? < 0.0 {
        return Err(refuse(importer, node, "a negative softcap"));
    }
    Ok(())
}

fn attention_scale(node: &Node, attrs: &Attributes) -> Result<Option<f32>> {
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

fn attention_output_shape(
    node: &Node,
    query: &ImportedValue,
    key: &ImportedValue,
    value: &ImportedValue,
) -> Result<Shape> {
    let unsupported = || Error::UnsupportedShape {
        name: node_label(node),
        reason: "unsupported attention input shapes".into(),
    };
    let (
        &[batch, heads, queries, width],
        &[key_batch, key_heads, keys, key_width],
        &[value_batch, value_heads, values, value_width],
    ) = (
        query.shape.dimensions(),
        key.shape.dimensions(),
        value.shape.dimensions(),
    )
    else {
        return Err(unsupported());
    };
    // Grouped key and value heads each serve the same number of query heads.
    if batch != key_batch
        || batch != value_batch
        || key_heads <= 0
        || heads % key_heads != 0
        || key_heads != value_heads
        || width != key_width
        || keys != values
        || batch < 0
        || heads < 0
        || queries < 0
        || width <= 0
        || keys <= 0
        || value_width < 0
    {
        return Err(unsupported());
    }
    Ok(Shape::create([batch, heads, queries, value_width])?)
}
