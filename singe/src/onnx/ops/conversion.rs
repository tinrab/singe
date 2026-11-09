use singe_onnx::{Node, TensorDataType};

use crate::{
    backend::registry::Operation,
    data_type::DataType,
    graph::node::NodeKind,
    onnx::{
        attributes::Attributes,
        error::{Error, Result},
        importer::{ImportedValue, Importer},
        ops::{
            expect_input_count, expect_input_range, expect_outputs, optional_input,
            quantization_scale, register_node_outputs, zero_point_initializer,
        },
        tensor::data_type as onnx_data_type,
        utility::{node_label, node_name},
    },
};

pub fn lower_cast(importer: &mut Importer, node: &Node) -> Result<()> {
    expect_input_count(importer, node, 1)?;
    expect_outputs(importer, node, 1)?;
    let attrs = Attributes::create(node)?;
    let input = importer.required_input(node, 0)?;
    let to = attrs.int("to", 0)?;
    let invalid = || Error::invalid_attribute(node_label(node), "to", "unknown tensor data type");
    let to = i32::try_from(to).map_err(|_| invalid())?;
    let to = TensorDataType::try_from(to).map_err(|_| invalid())?;
    let data_type = onnx_data_type(&node_label(node), to)?;
    if matches!(
        input.data_type,
        DataType::F8E4M3 | DataType::F8E5M2 | DataType::F4E2M1
    ) || matches!(
        data_type,
        DataType::F8E4M3 | DataType::F8E5M2 | DataType::F4E2M1
    ) {
        return Err(Error::unsupported_operator(
            node.domain.as_deref(),
            "Cast",
            importer.opset_version(node.domain.as_deref()),
            "float8 and float4 cast semantics are not supported",
        ));
    }

    let saturate = attrs.int("saturate", 1)?;
    if !matches!(saturate, 0 | 1) {
        return Err(Error::invalid_attribute(
            node_label(node),
            "saturate",
            "expected 0 or 1",
        ));
    }

    // Saturation applies only to float8 outputs, which are rejected above.
    let device = importer.options().device;
    let id = importer.graph().add_named_operation(
        node_name(node),
        NodeKind::cast(input.node, data_type),
        data_type,
        input.shape.clone(),
        device,
    )?;
    register_node_outputs(
        importer,
        node,
        vec![ImportedValue::new(id, data_type, input.shape)],
    )
}

fn expect_quantized_dtype(importer: &Importer, node: &Node, data_type: DataType) -> Result<()> {
    if matches!(data_type, DataType::I8 | DataType::U8) {
        Ok(())
    } else {
        Err(Error::unsupported_operator(
            node.domain.as_deref(),
            &node.operator_type,
            importer.opset_version(node.domain.as_deref()),
            "input must be int8 or uint8",
        ))
    }
}

fn unsupported_zero_point_dtype(importer: &Importer, node: &Node, name: &str) -> Error {
    Error::unsupported_operator(
        node.domain.as_deref(),
        &node.operator_type,
        importer.opset_version(node.domain.as_deref()),
        format!("zero point `{name}` must be an int8 or uint8 scalar"),
    )
}

pub fn lower_dequantize_linear(importer: &mut Importer, node: &Node) -> Result<()> {
    expect_input_range(importer, node, 2, 3)?;
    expect_outputs(importer, node, 1)?;
    let input = importer.required_input(node, 0)?;
    let scale = quantization_scale(importer, node, 1, "x_scale")?;
    let zero_point = if let Some(zero_point) = optional_input(importer, node, 2)? {
        if zero_point.data_type != input.data_type {
            return Err(Error::unsupported_operator(
                node.domain.as_deref(),
                &node.operator_type,
                importer.opset_version(node.domain.as_deref()),
                "zero point dtype must match the quantized input dtype",
            ));
        }

        expect_quantized_dtype(importer, node, input.data_type)?;
        zero_point_initializer(importer, node, 2, "x_zero_point", input.data_type)?
    } else {
        expect_quantized_dtype(importer, node, input.data_type)?;
        0
    };
    let device = importer.options().device;
    let id = importer.graph().add_named_quantization_operation(
        node_name(node),
        Operation::Dequantize,
        input.node,
        scale,
        zero_point,
        DataType::F32,
        input.shape.clone(),
        device,
    )?;
    register_node_outputs(
        importer,
        node,
        vec![ImportedValue::new(id, DataType::F32, input.shape)],
    )
}

pub fn lower_quantize_linear(importer: &mut Importer, node: &Node) -> Result<()> {
    expect_input_range(importer, node, 2, 3)?;
    expect_outputs(importer, node, 1)?;
    let input = importer.required_input(node, 0)?;
    let scale = quantization_scale(importer, node, 1, "y_scale")?;
    let (zero_point, data_type) = match optional_input(importer, node, 2)? {
        Some(zero_point) => match zero_point.data_type {
            DataType::I8 | DataType::U8 => (
                zero_point_initializer(importer, node, 2, "y_zero_point", zero_point.data_type)?,
                zero_point.data_type,
            ),
            _ => {
                return Err(unsupported_zero_point_dtype(
                    importer,
                    node,
                    &node.inputs[2],
                ));
            }
        },
        None => (0, DataType::U8),
    };
    let device = importer.options().device;
    let id = importer.graph().add_named_quantization_operation(
        node_name(node),
        Operation::Quantize,
        input.node,
        scale,
        zero_point,
        data_type,
        input.shape.clone(),
        device,
    )?;
    register_node_outputs(
        importer,
        node,
        vec![ImportedValue::new(id, data_type, input.shape)],
    )
}
