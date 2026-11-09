//! Differentiable forms of operations whose kernels have no backward formula.
//!
//! Each form computes what the kernel computes from operations that have formulas, so backward can run through it.

use crate::{
    backend::registry::{
        BatchNormalizationAttributes, Conv2dAttributes, LayerNormalizationAttributes,
        Pool2dAttributes,
    },
    error::{Error, Result},
    tensor::Tensor,
    utility::to_i64,
};

fn trailing_axes(input: &Tensor, count: usize) -> Result<Vec<i64>> {
    let rank = input.shape().dimensions().len();
    (rank - count..rank)
        .map(|axis| to_i64(axis, "trailing axis"))
        .collect()
}

pub fn rms_norm(inputs: &[Tensor], norm: &LayerNormalizationAttributes) -> Result<Tensor> {
    let x = &inputs[0];
    let axes = trailing_axes(x, norm.normalized_shape.len())?;
    let scale = x
        .square()?
        .mean(&axes, true)?
        .add_scalar(f64::from(norm.eps))?
        .sqrt()?;
    let normed = x.div(&scale)?;
    match inputs.get(1).filter(|_| norm.has_weight) {
        Some(weight) => normed.mul(weight),
        None => Ok(normed),
    }
}

pub fn layer_norm(inputs: &[Tensor], norm: &LayerNormalizationAttributes) -> Result<Tensor> {
    let x = &inputs[0];
    let axes = trailing_axes(x, norm.normalized_shape.len())?;
    let centered = x.sub(&x.mean(&axes, true)?)?;
    let scale = centered
        .square()?
        .mean(&axes, true)?
        .add_scalar(f64::from(norm.eps))?
        .sqrt()?;
    let mut normed = centered.div(&scale)?;
    let mut next = 1;
    if norm.has_weight {
        normed = normed.mul(&inputs[next])?;
        next += 1;
    }

    if norm.has_bias {
        normed = normed.add(&inputs[next])?;
    }
    Ok(normed)
}

// Normalizes over every axis but the channels, by the running statistics where the call has them and the batch's otherwise.
pub fn batch_norm(inputs: &[Tensor], batch: &BatchNormalizationAttributes) -> Result<Tensor> {
    let x = &inputs[0];
    let dimensions = x.shape().dimensions();
    let channels = dimensions[1];
    let mut shape = vec![1; dimensions.len()];
    shape[1] = channels;
    let axes = (0..dimensions.len())
        .filter(|&dimension| dimension != 1)
        .map(|axis| to_i64(axis, "batch normalization axis"))
        .collect::<Result<Vec<_>>>()?;
    let weight = usize::from(batch.has_weight);
    let bias = usize::from(batch.has_bias);
    let running = 1 + weight + bias;
    let mean = if batch.has_running_mean && !batch.training {
        inputs[running].reshape(shape.clone())?
    } else {
        x.mean(&axes, true)?
    };
    let variance = if batch.has_running_var && !batch.training {
        inputs[running + usize::from(batch.has_running_mean)].reshape(shape.clone())?
    } else {
        x.sub(&mean)?.square()?.mean(&axes, true)?
    };
    let mut normed = x
        .sub(&mean)?
        .div(&variance.add_scalar(f64::from(batch.eps))?.sqrt()?)?;
    if batch.has_weight {
        normed = normed.mul(&inputs[1].reshape(shape.clone())?)?;
    }

    if batch.has_bias {
        normed = normed.add(&inputs[1 + weight].reshape(shape)?)?;
    }
    Ok(normed)
}

// The points of the window tap `(i, j)` of every output point, `[N, C, H, W]` for an output of `H` by `W`.
fn window_tap(
    padded: &Tensor,
    tap: (i64, i64),
    dilation: (i64, i64),
    stride: (i64, i64),
    output: (i64, i64),
) -> Result<Tensor> {
    padded
        .slice(2, tap.0 * dilation.0, output.0, stride.0)?
        .slice(3, tap.1 * dilation.1, output.1, stride.1)
}

// A convolution as the sum over its weight's taps of a matmul with each group's channels.
pub fn conv2d(inputs: &[Tensor], conv: &Conv2dAttributes, output: &[i64]) -> Result<Tensor> {
    let (x, weight) = (&inputs[0], &inputs[1]);
    let (top, bottom, left, right) = conv.padding;
    let padded = x.pad(&[(0, 0), (0, 0), (top, bottom), (left, right)], 0.0)?;
    let &[out_channels, group_inputs, kernel_height, kernel_width] = weight.shape().dimensions()
    else {
        return Err(Error::invalid_argument("conv2d needs an OIHW weight"));
    };
    let group_outputs = out_channels / conv.groups;
    let extents = (output[2], output[3]);
    let mut total: Option<Tensor> = None;
    for i in 0..kernel_height {
        for j in 0..kernel_width {
            let tap = window_tap(&padded, (i, j), conv.dilation, conv.stride, extents)?;
            let tap_weight = weight
                .narrow(2, i, 1)?
                .narrow(3, j, 1)?
                .squeeze(3)?
                .squeeze(2)?;
            let groups = (0..conv.groups)
                .map(|group| {
                    let channels = tap.narrow(1, group * group_inputs, group_inputs)?;
                    let weights = tap_weight
                        .narrow(0, group * group_outputs, group_outputs)?
                        .transpose(0, 1)?;
                    channels.permute(&[0, 2, 3, 1])?.matmul(&weights)
                })
                .collect::<Result<Vec<_>>>()?;
            let term = Tensor::concat(&groups, 3)?.permute(&[0, 3, 1, 2])?;
            total = Some(match total {
                Some(total) => total.add(&term)?,
                None => term,
            });
        }
    }

    let total = total.ok_or_else(|| Error::invalid_argument("conv2d needs a weight tap"))?;
    match inputs.get(2).filter(|_| conv.has_bias) {
        Some(bias) => total.add(&bias.reshape([1, out_channels, 1, 1])?),
        None => Ok(total),
    }
}

// The padding that holds every window of `pool`: its own, and more after the input where ceil mode reaches past it.
fn pool_padding(
    input: &[i64],
    output: &[i64],
    pool: &Pool2dAttributes,
) -> ((i64, i64), (i64, i64)) {
    let (top, bottom, left, right) = pool.padding;
    let needed = |axis: usize, kernel: i64, dilation: i64, stride: i64, before: i64, after: i64| {
        let reach = dilation * (kernel - 1) + stride * (output[axis] - 1) + 1;
        after.max(reach - before - input[axis])
    };
    let (kernel, stride, dilation) = (pool.kernel_size, pool.stride, pool.dilation);
    (
        (top, needed(2, kernel.0, dilation.0, stride.0, top, bottom)),
        (left, needed(3, kernel.1, dilation.1, stride.1, left, right)),
    )
}

fn pool_taps(padded: &Tensor, pool: &Pool2dAttributes, output: &[i64]) -> Result<Vec<Tensor>> {
    let mut taps = Vec::new();
    for i in 0..pool.kernel_size.0 {
        for j in 0..pool.kernel_size.1 {
            taps.push(window_tap(
                padded,
                (i, j),
                pool.dilation,
                pool.stride,
                (output[2], output[3]),
            )?);
        }
    }
    Ok(taps)
}

pub fn max_pool2d(input: &Tensor, pool: &Pool2dAttributes, output: &[i64]) -> Result<Tensor> {
    let (rows, columns) = pool_padding(input.shape().dimensions(), output, pool);
    let padded = input.pad(&[(0, 0), (0, 0), rows, columns], f32::NEG_INFINITY)?;
    let mut taps = pool_taps(&padded, pool, output)?.into_iter();
    let first = taps
        .next()
        .ok_or_else(|| Error::invalid_argument("max pool needs a window"))?;
    taps.try_fold(first, |largest, tap| {
        Tensor::where_select(&tap.greater(&largest)?, &tap, &largest)
    })
}

// The sum of each window over how many of its points count: those of the input, and of its explicit padding when `count_include_pad` says so, but never of what ceil mode adds.
pub fn avg_pool2d(input: &Tensor, pool: &Pool2dAttributes, output: &[i64]) -> Result<Tensor> {
    let (rows, columns) = pool_padding(input.shape().dimensions(), output, pool);
    let padded = input.pad(&[(0, 0), (0, 0), rows, columns], 0.0)?;
    let (top, bottom, left, right) = pool.padding;
    let ones = input.detach().ones_like()?;
    let counted = if pool.count_include_pad == Some(true) {
        ones.pad(&[(0, 0), (0, 0), (top, bottom), (left, right)], 1.0)?
            .pad(
                &[(0, 0), (0, 0), (0, rows.1 - bottom), (0, columns.1 - right)],
                0.0,
            )?
    } else {
        ones.pad(&[(0, 0), (0, 0), rows, columns], 0.0)?
    };
    let sum = |taps: Vec<Tensor>| -> Result<Tensor> {
        let mut taps = taps.into_iter();
        let first = taps
            .next()
            .ok_or_else(|| Error::invalid_argument("average pool needs a window"))?;
        taps.try_fold(first, |total, tap| total.add(&tap))
    };
    sum(pool_taps(&padded, pool, output)?)?.div(&sum(pool_taps(&counted, pool, output)?)?)
}
