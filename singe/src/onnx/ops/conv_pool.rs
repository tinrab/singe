use singe_onnx::{Node, TensorDataType};

use crate::{
    backend::registry::{Conv3dAttributes, Operation, OperationAttributes, OperationRequest},
    graph::{GraphOperationInputs, GraphOperationOutput, GraphOperationRequest},
    onnx::{
        attributes::Attributes,
        error::{Error, Result},
        importer::{ImportedValue, Importer},
        ops::{
            expect_input_count, expect_input_range, expect_outputs, optional_input,
            register_node_outputs,
        },
        shape::{conv_output_shape, pool_output_shape},
        utility::{node_label, node_name, pair, planar_pads},
    },
    shape::{
        Shape,
        spatial::{Padding2d, Spatial2dI64, Spatial3dI64},
    },
    value::ValueSpec,
};

pub fn lower_conv(importer: &mut Importer, node: &Node) -> Result<()> {
    expect_outputs(importer, node, 1)?;
    expect_input_range(importer, node, 2, 3)?;
    if importer.required_input(node, 1)?.shape.rank() == 5 {
        return lower_conv3d(importer, node);
    }

    let Conv2dInput {
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
    } = conv2d_input(importer, node)?;
    let shape = conv_output_shape(
        &input.shape,
        &weight.shape,
        stride,
        padding,
        dilation,
        groups,
    )?;
    let device = importer.options().device;
    let id = importer
        .graph()
        .add_operation_request(GraphOperationRequest::conv2d(
            node_name(node),
            input.node,
            weight.node,
            bias.as_ref().map(|bias| bias.node),
            stride,
            padding,
            dilation,
            groups,
            GraphOperationOutput::new(input.data_type, shape.clone(), device),
        ))?;
    register_node_outputs(
        importer,
        node,
        vec![ImportedValue::new(id, input.data_type, shape)],
    )
}

fn lower_conv3d(importer: &mut Importer, node: &Node) -> Result<()> {
    let attrs = Attributes::create(node)?;
    let refuse = |reason: &str| {
        Error::unsupported_operator(
            node.domain.as_deref(),
            &node.operator_type,
            importer.opset_version(node.domain.as_deref()),
            reason,
        )
    };
    if attrs.string("auto_pad", "NOTSET")? != "NOTSET" {
        return Err(refuse("auto_pad is not supported"));
    }

    let triple = |name: &str, default: i64| -> Result<Spatial3dI64> {
        match attrs.ints_or(name, &[default; 3])?[..] {
            [depth, height, width] => Ok(Spatial3dI64::new(depth, height, width)),
            _ => Err(Error::invalid_attribute(
                node_label(node),
                name,
                "expected three values",
            )),
        }
    };
    let stride = triple("strides", 1)?;
    let dilation = triple("dilations", 1)?;
    let pads = attrs.ints_or("pads", &[0; 6])?;
    let [first, second, third, first_end, second_end, third_end] = pads[..] else {
        return Err(Error::invalid_attribute(
            node_label(node),
            "pads",
            "expected six values",
        ));
    };
    if (first, second, third) != (first_end, second_end, third_end) {
        return Err(refuse("pads that differ before and after an axis"));
    }

    let input = importer.required_input(node, 0)?;
    let weight = importer.required_input(node, 1)?;
    let bias = optional_input(importer, node, 2)?;
    let attributes = OperationAttributes::Conv3d(Conv3dAttributes {
        has_bias: bias.is_some(),
        stride,
        padding: Spatial3dI64::new(first, second, third),
        dilation,
        groups: attrs.int("group", 1)?,
    });
    let mut operands = vec![&input, &weight];
    operands.extend(bias.as_ref());
    let specs = operands
        .iter()
        .map(|operand| ValueSpec::new(operand.data_type, operand.shape.clone()))
        .collect();
    let inference = OperationRequest::new(Operation::Conv3d, attributes.clone(), specs).infer()?;
    let output = &inference.outputs()[0];
    let device = importer.options().device;
    let id = importer
        .graph()
        .add_operation_request(GraphOperationRequest::new(
            node_name(node),
            GraphOperationInputs::new(
                Operation::Conv3d,
                attributes,
                operands.iter().map(|operand| operand.node).collect(),
            ),
            GraphOperationOutput::new(output.data_type(), output.shape().clone(), device),
        ))?;
    register_node_outputs(
        importer,
        node,
        vec![ImportedValue::new(
            id,
            output.data_type(),
            output.shape().clone(),
        )],
    )
}

pub struct Conv2dInput {
    pub(crate) input: ImportedValue,
    pub(crate) weight: ImportedValue,
    pub(crate) bias: Option<ImportedValue>,
    pub(crate) stride: Spatial2dI64,
    pub(crate) padding: Padding2d,
    pub(crate) dilation: Spatial2dI64,
    pub(crate) groups: i64,
}

pub fn conv2d_input(importer: &Importer, node: &Node) -> Result<Conv2dInput> {
    let attributes = conv2d_attributes(importer, node)?;
    let input = importer.required_input(node, 0)?;
    let weight = importer.required_input(node, 1)?;
    let bias = optional_input(importer, node, 2)?;
    if input.data_type != weight.data_type
        || bias
            .as_ref()
            .is_some_and(|bias| bias.data_type != input.data_type)
    {
        return Err(Error::UnsupportedType {
            name: node_label(node),
            data_type: TensorDataType::Float,
            reason: "conv input, weight, and bias dtypes must match".into(),
        });
    }

    let kernel_rank = weight
        .shape
        .dimensions()
        .len()
        .checked_sub(2)
        .ok_or_else(|| {
            Error::unsupported_operator(
                None,
                "Conv",
                importer.opset_version(None),
                "invalid weight rank",
            )
        })?;
    if kernel_rank != 2 {
        return Err(Error::unsupported_operator(
            node.domain.as_deref(),
            "Conv",
            importer.opset_version(node.domain.as_deref()),
            "only 2d convolution is supported",
        ));
    }
    Ok(Conv2dInput {
        input,
        weight,
        bias,
        stride: attributes.stride,
        padding: attributes.padding,
        dilation: attributes.dilation,
        groups: attributes.groups,
    })
}

pub struct Conv2dAttributes {
    pub(crate) stride: Spatial2dI64,
    pub(crate) padding: Padding2d,
    pub(crate) dilation: Spatial2dI64,
    pub(crate) groups: i64,
}

pub fn conv2d_attributes(importer: &Importer, node: &Node) -> Result<Conv2dAttributes> {
    let attrs = Attributes::create(node)?;
    let auto_pad = attrs.string("auto_pad", "NOTSET")?;
    if auto_pad != "NOTSET" {
        return Err(Error::unsupported_operator(
            node.domain.as_deref(),
            "Conv",
            importer.opset_version(node.domain.as_deref()),
            "auto_pad is not supported",
        ));
    }

    if let Some(kernel) = attrs.ints("kernel_shape")? {
        let weight = importer.required_input(node, 1)?;
        if weight.shape.dimensions().get(2..) != Some(kernel.as_slice()) {
            return Err(Error::invalid_attribute(
                node_label(node),
                "kernel_shape",
                "kernel dimensions do not match the weight tensor",
            ));
        }
    }

    Ok(Conv2dAttributes {
        stride: pair(&attrs.ints_or("strides", &[1, 1])?, "strides", node)?,
        padding: planar_pads(&attrs.ints_or("pads", &[0, 0, 0, 0])?, node)?,
        dilation: pair(&attrs.ints_or("dilations", &[1, 1])?, "dilations", node)?,
        groups: attrs.int("group", 1)?,
    })
}

pub fn lower_max_pool(importer: &mut Importer, node: &Node) -> Result<()> {
    expect_input_count(importer, node, 1)?;
    if node.outputs.len() != 1 {
        return Err(Error::unsupported_operator(
            node.domain.as_deref(),
            "MaxPool",
            importer.opset_version(node.domain.as_deref()),
            "indices output is not supported",
        ));
    }
    lower_pool2d(importer, node, Operation::MaxPool2d, false)
}

pub fn lower_avg_pool(importer: &mut Importer, node: &Node) -> Result<()> {
    expect_input_count(importer, node, 1)?;
    expect_outputs(importer, node, 1)?;
    let count_include_pad = Attributes::create(node)?.int("count_include_pad", 0)? != 0;
    lower_pool2d(importer, node, Operation::AvgPool2d, count_include_pad)
}

pub fn lower_global_avg_pool(importer: &mut Importer, node: &Node) -> Result<()> {
    expect_input_count(importer, node, 1)?;
    expect_outputs(importer, node, 1)?;
    let input = importer.required_input(node, 0)?;
    let dimensions = input.shape.dimensions();
    let &[batch, channels, height, width] = dimensions else {
        return Err(Error::UnsupportedShape {
            name: node_label(node),
            reason: format!(
                "GlobalAveragePool expects rank-4 input, got rank {} shape {:?}",
                dimensions.len(),
                dimensions
            ),
        });
    };
    let shape = Shape::create([batch, channels, 1, 1])?;
    let device = importer.options().device;
    let id = importer
        .graph()
        .add_operation_request(GraphOperationRequest::pool2d(
            node_name(node),
            Operation::AvgPool2d,
            input.node,
            Spatial2dI64::new(height, width),
            Spatial2dI64::square(1),
            Padding2d::default(),
            Spatial2dI64::square(1),
            false,
            false,
            GraphOperationOutput::new(input.data_type, shape.clone(), device),
        ))?;
    register_node_outputs(
        importer,
        node,
        vec![ImportedValue::new(id, input.data_type, shape)],
    )
}

pub fn lower_pool2d(
    importer: &mut Importer,
    node: &Node,
    operation: Operation,
    count_include_pad: bool,
) -> Result<()> {
    let input = importer.required_input(node, 0)?;
    let attributes = pool2d_attributes(node)?;
    let shape = pool_output_shape(
        &input.shape,
        attributes.kernel,
        attributes.stride,
        attributes.padding,
        attributes.dilation,
        attributes.ceil_mode,
    )?;
    let device = importer.options().device;
    let id = importer
        .graph()
        .add_operation_request(GraphOperationRequest::pool2d(
            node_name(node),
            operation,
            input.node,
            attributes.kernel,
            attributes.stride,
            attributes.padding,
            attributes.dilation,
            attributes.ceil_mode,
            count_include_pad,
            GraphOperationOutput::new(input.data_type, shape.clone(), device),
        ))?;
    register_node_outputs(
        importer,
        node,
        vec![ImportedValue::new(id, input.data_type, shape)],
    )
}

pub struct Pool2dAttributes {
    pub(crate) kernel: Spatial2dI64,
    pub(crate) stride: Spatial2dI64,
    pub(crate) padding: Padding2d,
    pub(crate) dilation: Spatial2dI64,
    pub(crate) ceil_mode: bool,
}

pub fn pool2d_attributes(node: &Node) -> Result<Pool2dAttributes> {
    let attrs = Attributes::create(node)?;
    if attrs.string("auto_pad", "NOTSET")? != "NOTSET" {
        return Err(Error::invalid_attribute(
            node_label(node),
            "auto_pad",
            "automatic pooling padding is not supported",
        ));
    }

    let kernel = pair(
        &attrs.ints("kernel_shape")?.ok_or_else(|| {
            Error::invalid_attribute(node_label(node), "kernel_shape", "missing attribute")
        })?,
        "kernel_shape",
        node,
    )?;
    Ok(Pool2dAttributes {
        kernel,
        stride: pair(&attrs.ints_or("strides", &[1, 1])?, "strides", node)?,
        padding: planar_pads(&attrs.ints_or("pads", &[0, 0, 0, 0])?, node)?,
        dilation: pair(&attrs.ints_or("dilations", &[1, 1])?, "dilations", node)?,
        ceil_mode: attrs.int("ceil_mode", 0)? != 0,
    })
}
