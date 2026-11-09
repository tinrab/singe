use crate::{
    backend::{
        cpu::{
            iteration::Window2d,
            utility::{NativeFloat, cpu_device_mismatch, logical_values},
        },
        spec,
    },
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    parallel::{self, GRAIN_SIZE},
    tensor::Tensor,
    utility::to_usize,
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
    if input.device() != Device::Cpu
        || weight.device() != Device::Cpu
        || bias.is_some_and(|bias| bias.device() != Device::Cpu)
    {
        let actual = [Some(input), Some(weight), bias]
            .into_iter()
            .flatten()
            .map(Tensor::device)
            .find(|device| *device != Device::Cpu)
            .unwrap_or(Device::Cpu);
        return Err(cpu_device_mismatch("cpu conv2d input", actual));
    }

    let options = (stride, padding, dilation, groups);
    match input.data_type() {
        DataType::F32 => convolve::<f32>(input, weight, bias, options),
        DataType::F64 => convolve::<f64>(input, weight, bias, options),
        _ => Err(Error::unsupported_data_type("cpu conv2d")),
    }
}

type ConvolutionOptions = ((i64, i64), (i64, i64, i64, i64), (i64, i64), i64);

fn convolve<T: NativeFloat>(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    (stride, padding, dilation, groups): ConvolutionOptions,
) -> Result<Tensor> {
    let problem = spec::conv2d_problem(
        input.shape(),
        weight.shape(),
        stride,
        padding,
        dilation,
        groups,
        "cpu conv2d",
    )?;
    spec::validate_conv2d_bias(&problem, bias.map(Tensor::shape), "cpu conv2d")?;
    let window = Window2d::create(
        (problem.input_height, problem.input_width),
        (problem.kernel_height, problem.kernel_width),
        stride,
        padding,
        dilation,
    )?;
    let size = |name: &'static str, value: i64| to_usize(value, name);
    let input_channels = size("cpu conv2d input channels", problem.input_channels)?;
    let output_channels = size("cpu conv2d output channels", problem.output_channels)?;
    let groups = size("cpu conv2d groups", problem.groups)?;
    let input_height = size("cpu conv2d input height", problem.input_height)?;
    let input_width = size("cpu conv2d input width", problem.input_width)?;
    let output_height = size("cpu conv2d output height", problem.output_height)?;
    let output_width = size("cpu conv2d output width", problem.output_width)?;
    let kernel_height = size("cpu conv2d kernel height", problem.kernel_height)?;
    let kernel_width = size("cpu conv2d kernel width", problem.kernel_width)?;
    let channels_per_group = input_channels / groups;
    let output_channels_per_group = output_channels / groups;
    let plane = input_height * input_width;
    let kernel_size = kernel_height * kernel_width;

    let input = input.contiguous()?;
    let weight = weight.contiguous()?;
    let input_data = logical_values::<T>(&input)?;
    let weight_data = logical_values::<T>(&weight)?;
    let bias_data = bias.map(logical_values::<T>).transpose()?;

    // Each output element sums its group's channels and kernel taps in a fixed order on one thread.
    let work = (channels_per_group * kernel_size).max(1);
    let output = parallel::collect_with_grain(
        problem.output_shape.element_count(),
        (GRAIN_SIZE / work).max(1),
        |linear| {
            let column = linear % output_width;
            let row = linear / output_width % output_height;
            let output_channel = linear / (output_width * output_height) % output_channels;
            let batch = linear / (output_width * output_height * output_channels);
            let first_channel = output_channel / output_channels_per_group * channels_per_group;
            let mut sum = T::zero();
            for channel in 0..channels_per_group {
                let input_base = (batch * input_channels + first_channel + channel) * plane;
                let weight_base = (output_channel * channels_per_group + channel) * kernel_size;
                window.for_each_tap(
                    row,
                    column,
                    |kernel_row, kernel_column, input_row, input_column| {
                        sum = input_data[input_base + input_row * input_width + input_column]
                            .mul_add(
                                weight_data
                                    [weight_base + kernel_row * kernel_width + kernel_column],
                                sum,
                            );
                    },
                );
            }
            bias_data
                .as_ref()
                .map_or(sum, |bias| sum + bias[output_channel])
        },
    );
    drop(bias_data);
    drop(weight_data);
    drop(input_data);
    Tensor::from_vec(Device::Cpu, problem.output_shape, output)
}
