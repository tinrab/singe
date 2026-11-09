use singe_cudnn::backend::tensor::BackendTensorUid;

use crate::{
    backend::{
        cuda::{
            cudnn_backend::{
                CUDNN_GRAPH_BIAS_TENSOR_ID, CUDNN_GRAPH_INPUT0_TENSOR_ID,
                CUDNN_GRAPH_INPUT1_TENSOR_ID, CudnnBackendGraphKind,
                build_cudnn_convolution_template, cudnn_convolution_geometry,
                execute_cudnn_template,
            },
            utility::ensure_same_data_type,
        },
        registry::{Conv3dAttributes, Operation, operation_schema},
        spec::{self, ConvolutionWindow},
    },
    error::{Error, Result},
    shape::Shape,
    tensor::Tensor,
};

pub fn conv2d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: (i64, i64),
    padding: (i64, i64, i64, i64),
    dilation: (i64, i64),
    groups: i64,
) -> Result<Tensor> {
    ensure_same_data_type(input, weight)?;
    let problem = spec::conv2d_problem(
        input.shape(),
        weight.shape(),
        stride,
        padding,
        dilation,
        groups,
        "cuda conv2d",
    )?;
    spec::validate_conv2d_bias(&problem, bias.map(Tensor::shape), "cuda conv2d")?;
    let window = ConvolutionWindow::conv2d(stride, padding, dilation, groups);
    cudnn_convolution(
        Operation::Conv2d,
        input,
        weight,
        bias,
        &window,
        &problem.output_shape,
    )
}

// One cuDNN convolution over the three spatial axes of `[N, C, D, H, W]` inputs.
pub fn conv3d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    attributes: &Conv3dAttributes,
) -> Result<Tensor> {
    ensure_same_data_type(input, weight)?;
    let window = ConvolutionWindow::conv3d(attributes);
    let output_shape = window.output_shape(input.shape(), weight.shape())?;
    if let Some(bias) = bias
        && bias.shape().dimensions() != [output_shape.dimensions()[1]]
    {
        return Err(Error::invalid_argument(
            "cuda conv3d needs one bias per output channel",
        ));
    }
    cudnn_convolution(
        Operation::Conv3d,
        input,
        weight,
        bias,
        &window,
        &output_shape,
    )
}

fn cudnn_convolution(
    operation: Operation,
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    window: &ConvolutionWindow,
    output_shape: &Shape,
) -> Result<Tensor> {
    let mut operands: Vec<(BackendTensorUid, &Tensor)> = vec![
        (CUDNN_GRAPH_INPUT0_TENSOR_ID, input),
        (CUDNN_GRAPH_INPUT1_TENSOR_ID, weight),
    ];
    if let Some(bias) = bias {
        operands.push((CUDNN_GRAPH_BIAS_TENSOR_ID, bias));
    }

    let kind = if operation == Operation::Conv2d {
        CudnnBackendGraphKind::Conv2d
    } else {
        CudnnBackendGraphKind::Conv3d
    };
    execute_cudnn_template(
        &operation_schema(operation),
        operands,
        input.data_type(),
        output_shape,
        kind,
        window.parameters(),
        |context, inputs, output| {
            build_cudnn_convolution_template(
                context,
                &inputs[0],
                &inputs[1],
                inputs.get(2),
                output,
                cudnn_convolution_geometry(window)?,
            )
        },
    )
}
