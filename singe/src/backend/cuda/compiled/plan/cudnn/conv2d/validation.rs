use crate::{
    backend::{
        cuda::{
            compiled::plan::cudnn::ensure_cudnn_value_shape,
            graph_binding::{validate_graph_node_value_device, validate_graph_node_value_operand},
        },
        registry::{BackendExecution, Operation, operation_support},
        spec::ConvolutionWindow,
    },
    error::{Error, Result},
    graph::ir::{Graph, ValueId},
    utility::to_i64,
};

pub fn validate_cudnn_conv2d_nodes(
    ir: &Graph,
    id: ValueId,
    input: ValueId,
    weight: ValueId,
    stride: (i64, i64),
    padding: (i64, i64, i64, i64),
    dilation: (i64, i64),
    groups: i64,
) -> Result<()> {
    let window = ConvolutionWindow::conv2d(stride, padding, dilation, groups);
    validate_cudnn_convolution_nodes(ir, id, input, weight, &window)
}

pub fn validate_cudnn_convolution_nodes(
    ir: &Graph,
    id: ValueId,
    input: ValueId,
    weight: ValueId,
    window: &ConvolutionWindow,
) -> Result<()> {
    validate_graph_node_value_operand(ir, id, input)?;
    validate_graph_node_value_operand(ir, id, weight)?;
    let output_shape = window.output_shape(
        ir.required_value_shape(input)?,
        ir.required_value_shape(weight)?,
    )?;
    ensure_cudnn_value_shape(
        ir.required_value_shape(id)?.dimensions(),
        output_shape.dimensions(),
        "cudnn convolution output shape",
    )
}

pub fn validate_cudnn_conv2d_bias(
    ir: &Graph,
    id: ValueId,
    weight: ValueId,
    bias: ValueId,
) -> Result<()> {
    validate_graph_node_value_device(ir, id, bias)?;
    let weight_value = weight;
    let bias_value = bias;
    if !operation_support(Operation::Conv2d).supports_input_data_type(
        BackendExecution::CudaGraph,
        ir.required_value_data_type(bias_value)?,
    ) {
        return Err(Error::unsupported_data_type("cudnn conv2d bias"));
    }

    let weight_shape = ir.required_value_shape(weight_value)?;
    let bias_shape = ir.required_value_shape(bias_value)?;
    if bias_shape.dimensions().len() != 1 {
        return Err(Error::length_mismatch(
            "cudnn conv2d bias rank",
            1,
            bias_shape.dimensions().len(),
        ));
    }

    let bias_count = to_i64(bias_shape.element_count(), "shape")?;
    if bias_count != weight_shape.dimensions()[0] {
        return Err(Error::shape_mismatch(
            "cudnn conv2d bias element count",
            vec![weight_shape.dimensions()[0]],
            vec![bias_count],
        ));
    }
    Ok(())
}
