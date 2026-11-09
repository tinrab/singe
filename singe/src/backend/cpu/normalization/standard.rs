use crate::{
    backend::{
        cpu::{
            normalization::{usize_to_f32, validate_parameter},
            utility::{cpu_device_mismatch, get_f32_data},
        },
        spec,
    },
    device::Device,
    error::Result,
    parallel::{self, GRAIN_SIZE},
    tensor::Tensor,
    utility::to_i64,
};

pub fn layer_norm(
    input: &Tensor,
    normalized_shape: &[i64],
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    eps: f32,
) -> Result<Tensor> {
    normalize(
        "cpu layer_norm",
        input,
        normalized_shape,
        [weight, bias],
        |row, width| {
            let mean = row.iter().sum::<f32>() / width;
            let variance = row
                .iter()
                .map(|value| {
                    let difference = *value - mean;
                    difference * difference
                })
                .sum::<f32>()
                / width;
            (mean, 1.0 / (variance + eps).sqrt())
        },
    )
}

pub fn rms_norm(
    input: &Tensor,
    normalized_shape: &[i64],
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    eps: f32,
) -> Result<Tensor> {
    normalize(
        "cpu rms_norm",
        input,
        normalized_shape,
        [weight, bias],
        |row, width| {
            let mean_square = row.iter().map(|value| value * value).sum::<f32>() / width;
            (0.0, 1.0 / (mean_square + eps).sqrt())
        },
    )
}

/// Normalizes each row of the trailing `normalized_shape` dimensions as `(value - center) * scale`, where `statistics(row, width)` returns the row's center and scale, then applies the optional weight and bias.
///
/// Rows are split across threads, and each row is computed by one thread in a fixed order.
fn normalize(
    diagnostic_prefix: &'static str,
    input: &Tensor,
    normalized_shape: &[i64],
    [weight, bias]: [Option<&Tensor>; 2],
    statistics: impl Fn(&[f32], f32) -> (f32, f32) + Sync,
) -> Result<Tensor> {
    validate_devices(diagnostic_prefix, input, weight, bias)?;
    let problem = spec::normalization_problem(input.shape(), normalized_shape, diagnostic_prefix)?;
    let width = problem.feature_size;
    let expected = to_i64(width, &format!("{diagnostic_prefix} feature size"))?;
    for parameter in [weight, bias].into_iter().flatten() {
        validate_parameter(parameter, expected)?;
    }

    let width_f32 = usize_to_f32(format!("{diagnostic_prefix} feature size"), width)?;

    let input_data = get_f32_data(input)?;
    let weight_data = weight.map(get_f32_data).transpose()?;
    let bias_data = bias.map(get_f32_data).transpose()?;
    let mut output = vec![0.0_f32; input_data.len()];
    parallel::for_each_row_mut(&mut output, width, GRAIN_SIZE, |row, values| {
        let source = &input_data[row * width..(row + 1) * width];
        let (center, scale) = statistics(source, width_f32);
        for (feature, value) in values.iter_mut().enumerate() {
            let normalized = (source[feature] - center) * scale;
            let scaled = weight_data
                .as_ref()
                .map_or(normalized, |weight| normalized * weight[feature]);
            *value = bias_data
                .as_ref()
                .map_or(scaled, |bias| scaled + bias[feature]);
        }
        Ok(())
    })?;
    drop(bias_data);
    drop(weight_data);
    drop(input_data);
    Tensor::from_vec(Device::Cpu, input.shape().clone(), output)
}

fn validate_devices(
    diagnostic_prefix: &'static str,
    input: &Tensor,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
) -> Result<()> {
    [Some(input), weight, bias]
        .into_iter()
        .flatten()
        .map(Tensor::device)
        .find(|device| *device != Device::Cpu)
        .map_or(Ok(()), |actual| {
            Err(cpu_device_mismatch(
                format!("{diagnostic_prefix} input"),
                actual,
            ))
        })
}
