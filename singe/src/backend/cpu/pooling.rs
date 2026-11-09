use num_traits::NumCast;

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

pub fn max_pool2d(
    input: &Tensor,
    kernel_size: (i64, i64),
    stride: (i64, i64),
    padding: (i64, i64, i64, i64),
    dilation: (i64, i64),
    ceil_mode: bool,
) -> Result<Tensor> {
    fn maximum<T: NativeFloat>(taps: &Taps<'_, T>) -> T {
        let mut maximum = T::neg_infinity();
        taps.for_each(|value| {
            if value.is_nan() || value > maximum {
                maximum = value;
            }
        });
        maximum
    }
    let options = (kernel_size, stride, padding, dilation, ceil_mode);
    match input.data_type() {
        DataType::F32 => pool2d(input, options, false, "cpu max_pool2d", maximum::<f32>),
        DataType::F64 => pool2d(input, options, false, "cpu max_pool2d", maximum::<f64>),
        _ => Err(Error::unsupported_data_type("cpu max_pool2d")),
    }
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
    fn average<T: NativeFloat>(taps: &Taps<'_, T>, count_include_pad: bool) -> T {
        let (mut sum, mut count) = (T::zero(), 0_usize);
        taps.for_each(|value| {
            sum = sum + value;
            count += 1;
        });
        let divisor = if count_include_pad {
            taps.padded
        } else {
            count
        };
        match <T as NumCast>::from(divisor) {
            Some(divisor) if divisor != T::zero() => sum / divisor,
            _ => T::zero(),
        }
    }
    let options = (kernel_size, stride, padding, dilation, ceil_mode);
    match input.data_type() {
        DataType::F32 => pool2d(input, options, true, "cpu avg_pool2d", |taps| {
            average::<f32>(taps, count_include_pad)
        }),
        DataType::F64 => pool2d(input, options, true, "cpu avg_pool2d", |taps| {
            average::<f64>(taps, count_include_pad)
        }),
        _ => Err(Error::unsupported_data_type("cpu avg_pool2d")),
    }
}

/// The input values one output element of a pooling reads, in row-major kernel order.
struct Taps<'a, T> {
    window: &'a Window2d,
    values: &'a [T],
    width: usize,
    row: usize,
    column: usize,
    /// The number of taps inside the padded input, counting padding.
    padded: usize,
}

impl<T: Copy> Taps<'_, T> {
    fn for_each(&self, mut visit: impl FnMut(T)) {
        self.window
            .for_each_tap(self.row, self.column, |_, _, input_row, input_column| {
                visit(self.values[input_row * self.width + input_column]);
            });
    }
}

type PoolOptions = (
    (i64, i64),
    (i64, i64),
    (i64, i64, i64, i64),
    (i64, i64),
    bool,
);

/// Computes each output element of a pooling over `[N, C, H, W]` or `[C, H, W]` input from its window's taps, splitting output elements across threads.
fn pool2d<T: NativeFloat>(
    input: &Tensor,
    (kernel_size, stride, padding, dilation, ceil_mode): PoolOptions,
    average: bool,
    diagnostic_prefix: &'static str,
    reduce: impl Fn(&Taps<'_, T>) -> T + Sync,
) -> Result<Tensor> {
    if input.device() != Device::Cpu {
        return Err(cpu_device_mismatch("cpu operation input", input.device()));
    }

    let problem = spec::pool2d_problem(
        input.shape(),
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        average,
        diagnostic_prefix,
    )?;
    let window = Window2d::create(
        (problem.input_height, problem.input_width),
        kernel_size,
        stride,
        padding,
        dilation,
    )?;
    let size = |value: i64| to_usize(value, diagnostic_prefix);
    let input_height = size(problem.input_height)?;
    let input_width = size(problem.input_width)?;
    let output_height = size(problem.output_height)?;
    let output_width = size(problem.output_width)?;
    let taps = size(problem.kernel_height * problem.kernel_width)?.max(1);

    let input = input.contiguous()?;
    let input_data = logical_values::<T>(&input)?;
    let output = parallel::collect_with_grain(
        problem.output_shape.element_count(),
        (GRAIN_SIZE / taps).max(1),
        |linear| {
            let column = linear % output_width;
            let row = linear / output_width % output_height;
            let plane = linear / (output_width * output_height);
            let start = plane * input_height * input_width;
            reduce(&Taps {
                window: &window,
                values: &input_data[start..start + input_height * input_width],
                width: input_width,
                row,
                column,
                padded: window.padded_taps(row, column),
            })
        },
    );
    drop(input_data);
    Tensor::from_vec(Device::Cpu, problem.output_shape, output)
}
