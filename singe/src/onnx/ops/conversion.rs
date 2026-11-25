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

const ONNX_FLOAT: i64 = 1;
const ONNX_UINT8: i64 = 2;
const ONNX_INT8: i64 = 3;

fn reject_block_quantization(importer: &Importer, node: &Node, attrs: &Attributes) -> Result<()> {
    if attrs.int("block_size", 0)? == 0 {
        Ok(())
    } else {
        Err(unsupported_quantization(
            importer,
            node,
            "blocked quantization is not supported",
        ))
    }
}

fn unsupported_quantization(importer: &Importer, node: &Node, reason: &str) -> Error {
    Error::unsupported_operator(
        node.domain.as_deref(),
        &node.operator_type,
        importer.opset_version(node.domain.as_deref()),
        reason,
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
    let attrs = Attributes::create(node)?;
    reject_block_quantization(importer, node, &attrs)?;
    if !matches!(attrs.int("output_dtype", 0)?, 0 | ONNX_FLOAT) {
        return Err(unsupported_quantization(
            importer,
            node,
            "only float32 dequantization output is supported",
        ));
    }

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
    let attrs = Attributes::create(node)?;
    reject_block_quantization(importer, node, &attrs)?;
    if !matches!(attrs.int("precision", 0)?, 0 | ONNX_FLOAT) {
        return Err(unsupported_quantization(
            importer,
            node,
            "only float32 quantization precision is supported",
        ));
    }

    // Saturation applies only to float8 outputs, which the output dtype check below rejects.
    if !matches!(attrs.int("saturate", 1)?, 0 | 1) {
        return Err(Error::invalid_attribute(
            node_label(node),
            "saturate",
            "expected 0 or 1",
        ));
    }

    let input = importer.required_input(node, 0)?;
    if !matches!(input.data_type, DataType::F32 | DataType::I32) {
        return Err(unsupported_quantization(
            importer,
            node,
            "input must be float32 or int32",
        ));
    }

    let scale = quantization_scale(importer, node, 1, "y_scale")?;
    let output_dtype = match attrs.int("output_dtype", 0)? {
        0 => None,
        ONNX_UINT8 => Some(DataType::U8),
        ONNX_INT8 => Some(DataType::I8),
        _ => {
            return Err(unsupported_quantization(
                importer,
                node,
                "output dtype must be int8 or uint8",
            ));
        }
    };
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
        None => (0, output_dtype.unwrap_or(DataType::U8)),
    };
    if output_dtype.is_some_and(|output_dtype| output_dtype != data_type) {
        return Err(Error::invalid_attribute(
            node_label(node),
            "output_dtype",
            "output dtype must match the zero point dtype",
        ));
    }

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
