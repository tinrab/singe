use singe_onnx::{Node, TensorDataType};

use crate::{
    backend::registry::Operation,
    graph::{GraphOperationOutput, GraphOperationRequest},
    onnx::{
        attributes::Attributes,
        error::{Error, Result},
        importer::{ImportedValue, Importer},
        ops::{
            expect_input_count, expect_input_range, expect_outputs, optional_input,
            register_node_outputs,
        },
        shape::{conv_output_shape, pool_output_shape},
        utility::{node_label, node_name, pair},
    },
    shape::Shape,
};

pub fn lower_conv(importer: &mut Importer, node: &Node) -> Result<()> {
    expect_outputs(importer, node, 1)?;
    expect_input_range(importer, node, 2, 3)?;
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

pub struct Conv2dInput {
    pub(crate) input: ImportedValue,
    pub(crate) weight: ImportedValue,
    pub(crate) bias: Option<ImportedValue>,
    pub(crate) stride: (i64, i64),
    pub(crate) padding: (i64, i64, i64, i64),
    pub(crate) dilation: (i64, i64),
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
    pub(crate) stride: (i64, i64),
    pub(crate) padding: (i64, i64, i64, i64),
    pub(crate) dilation: (i64, i64),
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

    let stride = pair(&attrs.ints_or("strides", &[1, 1])?, "strides", node)?;
    let dilation = pair(&attrs.ints_or("dilations", &[1, 1])?, "dilations", node)?;
    let pads = attrs.ints_or("pads", &[0, 0, 0, 0])?;
    if pads.len() != 4 {
        return Err(Error::invalid_attribute(
            node_label(node),
            "pads",
            "expected four values",
        ));
    }
    Ok(Conv2dAttributes {
        stride,
        padding: (pads[0], pads[2], pads[1], pads[3]),
        dilation,
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
    if dimensions.len() != 4 {
        return Err(Error::UnsupportedShape {
            name: node_label(node),
            reason: format!(
                "GlobalAveragePool expects rank-4 input, got rank {} shape {:?}",
                dimensions.len(),
                dimensions
            ),
        });
    }

    let kernel = (dimensions[2], dimensions[3]);
    let shape = Shape::create([dimensions[0], dimensions[1], 1, 1])?;
    let device = importer.options().device;
    let id = importer
        .graph()
        .add_operation_request(GraphOperationRequest::pool2d(
            node_name(node),
            Operation::AvgPool2d,
            input.node,
            kernel,
            (1, 1),
            (0, 0, 0, 0),
            (1, 1),
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
    pub(crate) kernel: (i64, i64),
    pub(crate) stride: (i64, i64),
    pub(crate) padding: (i64, i64, i64, i64),
    pub(crate) dilation: (i64, i64),
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
    let stride = pair(&attrs.ints_or("strides", &[1, 1])?, "strides", node)?;
    let dilation = pair(&attrs.ints_or("dilations", &[1, 1])?, "dilations", node)?;
    let pads = attrs.ints_or("pads", &[0, 0, 0, 0])?;
    if pads.len() != 4 {
        return Err(Error::invalid_attribute(
            node_label(node),
            "pads",
            "expected four values",
        ));
    }
    Ok(Pool2dAttributes {
        kernel,
        stride,
        padding: (pads[0], pads[2], pads[1], pads[3]),
        dilation,
        ceil_mode: attrs.int("ceil_mode", 0)? != 0,
    })
}
