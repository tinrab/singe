use crate::{
    backend::{
        cpu::utility::{LogicalValues, cpu_device_mismatch, get_f32_data},
        spec,
    },
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    parallel,
    shape::Shape,
    tensor::Tensor,
};

pub fn batch_norm(
    input: &Tensor,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    running_mean: Option<&Tensor>,
    running_var: Option<&Tensor>,
    training: bool,
    _momentum: f32,
    eps: f32,
) -> Result<Tensor> {
    let prepared = prepare_batch_norm(input, weight, bias, running_mean, running_var)?;
    let shape = BatchNormShape {
        channels: prepared.feature_count,
        spatial: prepared.spatial_len,
    };
    // Training normalizes with the batch's statistics, which its decomposition computes before calling this kernel.
    if training {
        return Err(Error::unsupported_operation(
            "cpu batch_norm with batch statistics",
        ));
    }

    let inverse_std = prepared
        .running_var_data
        .iter()
        .map(|variance| 1.0 / (variance + eps).sqrt())
        .collect::<Vec<_>>();
    let mean = prepared.running_mean_data.to_vec();
    let output = normalize(
        &prepared.input_data,
        shape,
        &mean,
        &inverse_std,
        prepared.weight_data.as_deref(),
        prepared.bias_data.as_deref(),
    );
    Tensor::from_vec(Device::Cpu, prepared.output_shape, output)
}

/// The per-channel shape of a batch normalization input viewed as `[batch, channels, spatial]`.
#[derive(Debug, Clone, Copy)]
pub struct BatchNormShape {
    pub(crate) channels: usize,
    pub(crate) spatial: usize,
}

/// Normalizes every element with its channel's mean and reciprocal standard deviation, then applies the optional weight and bias.
pub fn normalize(
    input: &[f32],
    shape: BatchNormShape,
    mean: &[f32],
    inverse_std: &[f32],
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
) -> Vec<f32> {
    parallel::collect(input.len(), |index| {
        let channel = index / shape.spatial.max(1) % shape.channels;
        let normalized = (input[index] - mean[channel]) * inverse_std[channel];
        let scaled = weight.map_or(normalized, |weight| normalized * weight[channel]);
        bias.map_or(scaled, |bias| scaled + bias[channel])
    })
}

pub struct PreparedBatchNorm<'a> {
    pub(crate) input_data: LogicalValues<'a, f32>,
    pub(crate) output_shape: Shape,
    pub(crate) feature_count: usize,
    pub(crate) spatial_len: usize,
    pub(crate) weight_data: Option<LogicalValues<'a, f32>>,
    pub(crate) bias_data: Option<LogicalValues<'a, f32>>,
    pub(crate) running_mean_data: LogicalValues<'a, f32>,
    pub(crate) running_var_data: LogicalValues<'a, f32>,
}

pub fn prepare_batch_norm<'a>(
    input: &'a Tensor,
    weight: Option<&'a Tensor>,
    bias: Option<&'a Tensor>,
    running_mean: Option<&'a Tensor>,
    running_var: Option<&'a Tensor>,
) -> Result<PreparedBatchNorm<'a>> {
    validate_batch_norm_parameters(input, weight, bias, running_mean, running_var)?;

    let problem = spec::batch_norm_problem(
        input.shape(),
        weight.map(Tensor::shape),
        bias.map(Tensor::shape),
        running_mean.map(Tensor::shape),
        running_var.map(Tensor::shape),
        "cpu batch_norm",
    )?;

    let input_data = get_f32_data(input)?;
    let weight_data = weight.map(get_f32_data).transpose()?;
    let bias_data = bias.map(get_f32_data).transpose()?;
    let running_mean_data = running_mean
        .map(get_f32_data)
        .transpose()?
        .unwrap_or_else(|| LogicalValues::Gathered(vec![0.0f32; problem.feature_count]));
    let running_var_data = running_var
        .map(get_f32_data)
        .transpose()?
        .unwrap_or_else(|| LogicalValues::Gathered(vec![1.0f32; problem.feature_count]));

    Ok(PreparedBatchNorm {
        input_data,
        output_shape: problem.output_shape,
        feature_count: problem.feature_count,
        spatial_len: problem.spatial_len,
        weight_data,
        bias_data,
        running_mean_data,
        running_var_data,
    })
}

fn validate_batch_norm_parameters(
    input: &Tensor,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    running_mean: Option<&Tensor>,
    running_var: Option<&Tensor>,
) -> Result<()> {
    if input.device() != Device::Cpu
        || [weight, bias, running_mean, running_var]
            .into_iter()
            .flatten()
            .any(|parameter| parameter.device() != Device::Cpu)
    {
        let actual = [Some(input), weight, bias, running_mean, running_var]
            .into_iter()
            .flatten()
            .map(Tensor::device)
            .find(|device| *device != Device::Cpu)
            .unwrap_or(Device::Cpu);
        return Err(cpu_device_mismatch("cpu batch_norm input", actual));
    }

    for parameter in [weight, bias, running_mean, running_var]
        .into_iter()
        .flatten()
    {
        if parameter.data_type() != DataType::F32 {
            return Err(Error::unsupported_data_type("cpu normalization"));
        }
    }

    Ok(())
}
