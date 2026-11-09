pub use crate::backend::cuda::eager::indexing::embedding;
use crate::{
    backend::{
        cuda::{
            eager::{
                convolution::{conv2d, conv3d},
                ensure_cuda_eager_same_device_inputs, unsupported_eager_tensor_inputs,
            },
            kernel_plan::{self, KernelInput, KernelPlanner},
        },
        decomposition,
        registry::{
            BackendExecution, BatchNormalizationAttributes, Conv2dAttributes, Conv3dAttributes,
            LayerNormalizationAttributes, Operation, OperationAttributes, Pool2dAttributes,
            ScaledDotProductAttentionAttributes, operation_support,
        },
        spec,
    },
    error::{Error, Result},
    tensor::Tensor,
};

pub fn convolution2d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: (i64, i64),
    padding: (i64, i64, i64, i64),
    dilation: (i64, i64),
    groups: i64,
) -> Result<Tensor> {
    // Depthwise convolutions, one input channel per group, are generated window reductions.
    if input.shape().dimensions().get(1) == Some(&groups) {
        let attributes = OperationAttributes::Conv2d(Conv2dAttributes {
            has_bias: bias.is_some(),
            stride,
            padding,
            dilation,
            groups,
        });
        let inputs = [Some(input), Some(weight), bias]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        return kernel_plan::execute(Operation::Conv2d, &attributes, &inputs);
    }
    conv2d(input, weight, bias, stride, padding, dilation, groups)
}

// One cuDNN convolution, or the decomposition into 2D convolutions per kernel depth tap where cuDNN rejects the operands.
pub fn convolution3d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    attributes: &Conv3dAttributes,
) -> Result<Tensor> {
    ensure_cuda_eager_same_device_inputs(input, &[weight])?;
    if let Ok(output) = conv3d(input, weight, bias, attributes) {
        return Ok(output);
    }
    input.convolution3d_by_planes(weight, bias, attributes)
}

pub fn max_pool2d(
    input: &Tensor,
    kernel_size: (i64, i64),
    stride: (i64, i64),
    padding: (i64, i64, i64, i64),
    dilation: (i64, i64),
    ceil_mode: bool,
) -> Result<Tensor> {
    backend_max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)
}

pub fn avg_pool2d(
    input: &Tensor,
    kernel_size: (i64, i64),
    stride: (i64, i64),
    padding: (i64, i64, i64, i64),
    dilation: (i64, i64),
    ceil_mode: bool,
    count_include_pad: bool,
) -> Result<Tensor> {
    backend_avg_pool2d(
        input,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        count_include_pad,
    )
}

pub fn layer_norm(
    input: &Tensor,
    normalized_shape: &[i64],
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    eps: f32,
) -> Result<Tensor> {
    normalization(
        Operation::LayerNorm,
        input,
        normalized_shape,
        weight,
        bias,
        eps,
    )
}

pub fn rms_norm(
    input: &Tensor,
    normalized_shape: &[i64],
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    eps: f32,
) -> Result<Tensor> {
    normalization(
        Operation::RmsNorm,
        input,
        normalized_shape,
        weight,
        bias,
        eps,
    )
}

// Rotates every pair of each head in one generated kernel, and decomposes rotations of part of a head.
pub fn rotary_embedding(input: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    // A rotation of no pairs leaves every value as it is.
    if cos.shape().dimensions().last() == Some(&0) {
        return input.copy();
    }

    let inputs = [input, cos, sin];
    match kernel_plan::execute(
        Operation::RotaryEmbedding,
        &OperationAttributes::None,
        &inputs,
    ) {
        Ok(output) => Ok(output),
        Err(_) => decomposition::decompose(
            Operation::RotaryEmbedding,
            &OperationAttributes::None,
            &inputs,
        )?
        .pop()
        .ok_or_else(|| Error::missing_output("rotary embedding")),
    }
}

pub fn scaled_dot_product_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    mask: Option<&Tensor>,
    sinks: Option<&Tensor>,
    key_lengths: Option<&Tensor>,
    attention: ScaledDotProductAttentionAttributes,
) -> Result<Tensor> {
    let inputs = [query, key, value]
        .into_iter()
        .chain(mask)
        .chain(sinks)
        .chain(key_lengths)
        .collect::<Vec<_>>();
    ensure_cuda_eager_same_device_inputs(query, &inputs[1..])?;
    if let Some(plan) = KernelPlanner::default().fused_attention(
        &KernelInput::from_source(query),
        &KernelInput::from_source(key),
        &KernelInput::from_source(value),
        mask.map(KernelInput::from_source).as_ref(),
        sinks.map(KernelInput::from_source).as_ref(),
        key_lengths.map(KernelInput::from_source).as_ref(),
        attention,
    )? {
        return kernel_plan::run(&plan, &inputs);
    }

    let attributes = OperationAttributes::ScaledDotProductAttention(attention);
    decomposition::decompose(Operation::ScaledDotProductAttention, &attributes, &inputs)?
        .pop()
        .ok_or_else(|| Error::missing_output("scaled dot product attention"))
}

pub fn batch_norm(
    input: &Tensor,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    running_mean: Option<&Tensor>,
    running_var: Option<&Tensor>,
    training: bool,
    momentum: f32,
    eps: f32,
) -> Result<Tensor> {
    let all_inputs = [Some(input), weight, bias, running_mean, running_var]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    ensure_cuda_eager_same_device_inputs(input, &all_inputs[1..])?;
    let problem = spec::batch_norm_problem(
        input.shape(),
        weight.map(Tensor::shape),
        bias.map(Tensor::shape),
        running_mean.map(Tensor::shape),
        running_var.map(Tensor::shape),
        "cuda batch_norm",
    )?;
    let support = operation_support(Operation::BatchNorm);
    if all_inputs.iter().any(|tensor| {
        !support.supports_input_data_type(BackendExecution::CudaEager, tensor.data_type())
    }) {
        return unsupported_eager_tensor_inputs(
            Operation::BatchNorm,
            input.data_type(),
            &problem.output_shape,
            &all_inputs,
        );
    }

    let attributes = OperationAttributes::BatchNormalization(BatchNormalizationAttributes {
        has_weight: weight.is_some(),
        has_bias: bias.is_some(),
        has_running_mean: running_mean.is_some(),
        has_running_var: running_var.is_some(),
        training,
        momentum,
        eps,
    });
    kernel_plan::execute(Operation::BatchNorm, &attributes, &all_inputs)
}

fn normalization(
    operation: Operation,
    input: &Tensor,
    normalized_shape: &[i64],
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    eps: f32,
) -> Result<Tensor> {
    let problem =
        spec::normalization_problem(input.shape(), normalized_shape, "cuda normalization")?;
    spec::validate_normalization_parameters(
        &problem,
        weight.map(Tensor::shape),
        bias.map(Tensor::shape),
        "cuda normalization",
    )?;
    let attributes = OperationAttributes::LayerNormalization(LayerNormalizationAttributes {
        normalized_shape: normalized_shape.to_vec(),
        has_weight: weight.is_some(),
        has_bias: bias.is_some(),
        eps,
    });
    let inputs = [Some(input), weight, bias]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    kernel_plan::execute(operation, &attributes, &inputs)
}

pub fn backend_max_pool2d(
    input: &Tensor,
    kernel_size: (i64, i64),
    stride: (i64, i64),
    padding: (i64, i64, i64, i64),
    dilation: (i64, i64),
    ceil_mode: bool,
) -> Result<Tensor> {
    backend_pool2d(
        Operation::MaxPool2d,
        input,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        false,
    )
}

pub fn backend_avg_pool2d(
    input: &Tensor,
    kernel_size: (i64, i64),
    stride: (i64, i64),
    padding: (i64, i64, i64, i64),
    dilation: (i64, i64),
    ceil_mode: bool,
    count_include_pad: bool,
) -> Result<Tensor> {
    backend_pool2d(
        Operation::AvgPool2d,
        input,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        count_include_pad,
    )
}

fn backend_pool2d(
    operation: Operation,
    input: &Tensor,
    kernel_size: (i64, i64),
    stride: (i64, i64),
    padding: (i64, i64, i64, i64),
    dilation: (i64, i64),
    ceil_mode: bool,
    count_include_pad: bool,
) -> Result<Tensor> {
    let attributes = OperationAttributes::Pool2d(Pool2dAttributes {
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        count_include_pad: (operation == Operation::AvgPool2d).then_some(count_include_pad),
    });
    kernel_plan::execute(operation, &attributes, &[input])
}

pub fn paged_attention(
    inputs: &[&Tensor],
    attention: ScaledDotProductAttentionAttributes,
) -> Result<Vec<Tensor>> {
    let [query, rest @ ..] = inputs else {
        return Err(Error::missing_input("paged attention"));
    };
    ensure_cuda_eager_same_device_inputs(query, rest)?;
    let metadata = inputs
        .iter()
        .map(|input| KernelInput::from_source(*input))
        .collect::<Vec<_>>();
    let plan = KernelPlanner::default().paged_attention(&metadata, attention)?;
    kernel_plan::run_outputs(&plan, inputs)
}

// Latent attention as one kernel.
pub fn paged_mla_attention(
    inputs: &[&Tensor],
    attention: ScaledDotProductAttentionAttributes,
) -> Result<Vec<Tensor>> {
    let [query, rest @ ..] = inputs else {
        return Err(Error::missing_input("paged mla attention"));
    };
    ensure_cuda_eager_same_device_inputs(query, rest)?;
    let metadata = inputs
        .iter()
        .map(|input| KernelInput::from_source(*input))
        .collect::<Vec<_>>();
    let plan = KernelPlanner::default().paged_mla_attention(&metadata, attention)?;
    kernel_plan::run_outputs(&plan, inputs)
}
