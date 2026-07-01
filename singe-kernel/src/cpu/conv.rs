//! Small convolution and layout-specific convolution helpers.

use crate::{
    error::{Error, Result},
    utility::checked_element_count,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CausalConv1dConfig {
    pub channels_in: usize,
    pub channels_out: usize,
    pub input_length: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub dilation: usize,
    pub groups: usize,
    pub left_padding: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Conv1dConfig {
    pub channels_in: usize,
    pub channels_out: usize,
    pub input_length: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub dilation: usize,
    pub groups: usize,
    pub left_padding: usize,
    pub right_padding: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BatchedConv1dConfig {
    pub batch: usize,
    pub channels_in: usize,
    pub channels_out: usize,
    pub input_length: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub dilation: usize,
    pub groups: usize,
    pub left_padding: usize,
    pub right_padding: usize,
    pub input_batch_stride: usize,
    pub output_batch_stride: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Conv1dIm2colConfig {
    pub batch: usize,
    pub channels_in: usize,
    pub input_length: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub dilation: usize,
    pub groups: usize,
    pub left_padding: usize,
    pub right_padding: usize,
    pub input_batch_stride: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Conv1dActivation {
    None,
    Gelu,
}

impl CausalConv1dConfig {
    pub fn output_length(self) -> Result<usize> {
        validate_causal_conv1d_config(self)?;
        conv1d_output_length(
            self.input_length,
            self.kernel_size,
            self.stride,
            self.dilation,
            self.left_padding,
            0,
        )
    }
}

impl Conv1dConfig {
    pub fn output_length(self) -> Result<usize> {
        validate_conv1d_config(self)?;
        conv1d_output_length(
            self.input_length,
            self.kernel_size,
            self.stride,
            self.dilation,
            self.left_padding,
            self.right_padding,
        )
    }
}

impl BatchedConv1dConfig {
    pub fn output_length(self) -> Result<usize> {
        validate_batched_conv1d_config(self)?;
        conv1d_output_length(
            self.input_length,
            self.kernel_size,
            self.stride,
            self.dilation,
            self.left_padding,
            self.right_padding,
        )
    }
}

impl Conv1dIm2colConfig {
    pub fn output_length(self) -> Result<usize> {
        validate_conv1d_im2col_config(self)?;
        conv1d_output_length(
            self.input_length,
            self.kernel_size,
            self.stride,
            self.dilation,
            self.left_padding,
            self.right_padding,
        )
    }

    pub fn output_values_per_batch(self) -> Result<usize> {
        let output_length = self.output_length()?;
        checked_element_count(
            checked_element_count(self.channels_in, output_length)?,
            self.kernel_size,
        )
    }
}

pub fn conv1d_causal(input: &[f32], weight: &[f32], config: CausalConv1dConfig) -> Vec<f32> {
    let output_length = config.output_length().expect("output length");
    let channels_in_per_group = config.channels_in / config.groups;
    let channels_out_per_group = config.channels_out / config.groups;
    let mut output = vec![0.0; config.channels_out * output_length];
    for out_channel in 0..config.channels_out {
        let group = out_channel / channels_out_per_group;
        let input_channel_start = group * channels_in_per_group;
        for out_pos in 0..output_length {
            let mut sum = 0.0f32;
            for group_input_channel in 0..channels_in_per_group {
                let input_channel = input_channel_start + group_input_channel;
                for kernel_index in 0..config.kernel_size {
                    let raw_input_pos = out_pos * config.stride + kernel_index * config.dilation;
                    if raw_input_pos < config.left_padding {
                        continue;
                    }
                    let input_pos = raw_input_pos - config.left_padding;
                    if input_pos >= config.input_length {
                        continue;
                    }
                    let input_value = input[input_channel * config.input_length + input_pos];
                    let weight_value = weight[(out_channel * channels_in_per_group
                        + group_input_channel)
                        * config.kernel_size
                        + kernel_index];
                    sum += input_value * weight_value;
                }
            }
            output[out_channel * output_length + out_pos] = sum;
        }
    }
    output
}

pub fn conv1d(input: &[f32], weight: &[f32], config: Conv1dConfig) -> Vec<f32> {
    let output_length = config.output_length().expect("output length");
    let channels_in_per_group = config.channels_in / config.groups;
    let channels_out_per_group = config.channels_out / config.groups;
    let mut output = vec![0.0; config.channels_out * output_length];
    for out_channel in 0..config.channels_out {
        let group = out_channel / channels_out_per_group;
        let input_channel_start = group * channels_in_per_group;
        for out_pos in 0..output_length {
            let mut sum = 0.0f32;
            for group_input_channel in 0..channels_in_per_group {
                let input_channel = input_channel_start + group_input_channel;
                for kernel_index in 0..config.kernel_size {
                    let raw_input_pos = out_pos * config.stride + kernel_index * config.dilation;
                    if raw_input_pos < config.left_padding {
                        continue;
                    }
                    let input_pos = raw_input_pos - config.left_padding;
                    if input_pos >= config.input_length {
                        continue;
                    }
                    let input_value = input[input_channel * config.input_length + input_pos];
                    let weight_value = weight[(out_channel * channels_in_per_group
                        + group_input_channel)
                        * config.kernel_size
                        + kernel_index];
                    sum += input_value * weight_value;
                }
            }
            output[out_channel * output_length + out_pos] = sum;
        }
    }
    output
}

pub fn conv1d_bias_activation(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    config: Conv1dConfig,
    activation: Conv1dActivation,
) -> Vec<f32> {
    let output_length = config.output_length().expect("output length");
    let mut output = conv1d(input, weight, config);
    for out_channel in 0..config.channels_out {
        for out_pos in 0..output_length {
            let value = &mut output[out_channel * output_length + out_pos];
            *value += bias[out_channel];
            *value = apply_activation(*value, activation);
        }
    }
    output
}

pub fn conv1d_batched(input: &[f32], weight: &[f32], config: BatchedConv1dConfig) -> Vec<f32> {
    let output_length = config.output_length().expect("output length");
    let output_len = batched_reach(
        config.batch,
        config.output_batch_stride,
        config.channels_out,
        output_length,
    );
    let channels_in_per_group = config.channels_in / config.groups;
    let channels_out_per_group = config.channels_out / config.groups;
    let mut output = vec![0.0; output_len];
    for batch in 0..config.batch {
        let input_batch_base = batch * config.input_batch_stride;
        let output_batch_base = batch * config.output_batch_stride;
        for out_channel in 0..config.channels_out {
            let group = out_channel / channels_out_per_group;
            let input_channel_start = group * channels_in_per_group;
            for out_pos in 0..output_length {
                let mut sum = 0.0f32;
                for group_input_channel in 0..channels_in_per_group {
                    let input_channel = input_channel_start + group_input_channel;
                    for kernel_index in 0..config.kernel_size {
                        let raw_input_pos =
                            out_pos * config.stride + kernel_index * config.dilation;
                        if raw_input_pos < config.left_padding {
                            continue;
                        }
                        let input_pos = raw_input_pos - config.left_padding;
                        if input_pos >= config.input_length {
                            continue;
                        }
                        let input_value = input
                            [input_batch_base + input_channel * config.input_length + input_pos];
                        let weight_value = weight[(out_channel * channels_in_per_group
                            + group_input_channel)
                            * config.kernel_size
                            + kernel_index];
                        sum += input_value * weight_value;
                    }
                }
                output[output_batch_base + out_channel * output_length + out_pos] = sum;
            }
        }
    }
    output
}

pub fn conv1d_batched_im2col(input: &[f32], config: Conv1dIm2colConfig) -> Vec<f32> {
    let output_length = config.output_length().expect("output length");
    let channels_in_per_group = config.channels_in / config.groups;
    let output_values_per_batch = config.output_values_per_batch().expect("output values");
    let mut output = vec![0.0; config.batch * output_values_per_batch];
    for batch in 0..config.batch {
        let input_batch_base = batch * config.input_batch_stride;
        let output_batch_base = batch * output_values_per_batch;
        for group in 0..config.groups {
            for out_pos in 0..output_length {
                for group_input_channel in 0..channels_in_per_group {
                    let input_channel = group * channels_in_per_group + group_input_channel;
                    for kernel_index in 0..config.kernel_size {
                        let output_offset = output_batch_base
                            + (((group * output_length + out_pos) * channels_in_per_group
                                + group_input_channel)
                                * config.kernel_size
                                + kernel_index);
                        let raw_input_pos =
                            out_pos * config.stride + kernel_index * config.dilation;
                        if raw_input_pos < config.left_padding {
                            continue;
                        }
                        let input_pos = raw_input_pos - config.left_padding;
                        if input_pos >= config.input_length {
                            continue;
                        }
                        output[output_offset] = input
                            [input_batch_base + input_channel * config.input_length + input_pos];
                    }
                }
            }
        }
    }
    output
}

pub fn conv1d_batched_from_im2col(
    columns: &[f32],
    weight: &[f32],
    config: BatchedConv1dConfig,
) -> Vec<f32> {
    let output_length = config.output_length().expect("output length");
    let output_len = batched_reach(
        config.batch,
        config.output_batch_stride,
        config.channels_out,
        output_length,
    );
    let channels_in_per_group = config.channels_in / config.groups;
    let channels_out_per_group = config.channels_out / config.groups;
    let column_values_per_batch = config.channels_in * output_length * config.kernel_size;
    let mut output = vec![0.0; output_len];
    for batch in 0..config.batch {
        let column_batch_base = batch * column_values_per_batch;
        let output_batch_base = batch * config.output_batch_stride;
        for out_channel in 0..config.channels_out {
            let group = out_channel / channels_out_per_group;
            for out_pos in 0..output_length {
                let mut sum = 0.0f32;
                for group_input_channel in 0..channels_in_per_group {
                    for kernel_index in 0..config.kernel_size {
                        let column_offset = column_batch_base
                            + (((group * output_length + out_pos) * channels_in_per_group
                                + group_input_channel)
                                * config.kernel_size
                                + kernel_index);
                        let weight_offset = (out_channel * channels_in_per_group
                            + group_input_channel)
                            * config.kernel_size
                            + kernel_index;
                        sum += columns[column_offset] * weight[weight_offset];
                    }
                }
                output[output_batch_base + out_channel * output_length + out_pos] = sum;
            }
        }
    }
    output
}

pub fn conv1d_causal_bias(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    config: CausalConv1dConfig,
) -> Vec<f32> {
    let output_length = config.output_length().expect("output length");
    let mut output = conv1d_causal(input, weight, config);
    for out_channel in 0..config.channels_out {
        for out_pos in 0..output_length {
            output[out_channel * output_length + out_pos] += bias[out_channel];
        }
    }
    output
}

pub fn conv1d_causal_bias_gelu(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    config: CausalConv1dConfig,
) -> Vec<f32> {
    let mut output = conv1d_causal_bias(input, weight, bias, config);
    for value in &mut output {
        *value = gelu(*value);
    }
    output
}

pub fn conv1d_causal_bias_activation(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    config: CausalConv1dConfig,
    activation: Conv1dActivation,
) -> Vec<f32> {
    let output_length = config.output_length().expect("output length");
    let mut output = conv1d_causal_bias(input, weight, bias, config);
    for out_channel in 0..config.channels_out {
        for out_pos in 0..output_length {
            let value = &mut output[out_channel * output_length + out_pos];
            *value = apply_activation(*value, activation);
        }
    }
    output
}

fn batched_reach(batch: usize, batch_stride: usize, channels: usize, length: usize) -> usize {
    (batch - 1) * batch_stride + channels * length
}

fn gelu(value: f32) -> f32 {
    0.5 * value * (1.0 + (0.7978846 * (value + 0.044715 * value * value * value)).tanh())
}

fn apply_activation(value: f32, activation: Conv1dActivation) -> f32 {
    match activation {
        Conv1dActivation::None => value,
        Conv1dActivation::Gelu => gelu(value),
    }
}

fn validate_conv1d_config(config: Conv1dConfig) -> Result<()> {
    validate_conv1d_shape(
        config.channels_in,
        config.channels_out,
        config.input_length,
        config.kernel_size,
        config.stride,
        config.dilation,
        config.groups,
    )
}

fn validate_batched_conv1d_config(config: BatchedConv1dConfig) -> Result<()> {
    if config.batch == 0 || config.input_batch_stride == 0 || config.output_batch_stride == 0 {
        return Err(Error::InvalidLength);
    }
    validate_conv1d_shape(
        config.channels_in,
        config.channels_out,
        config.input_length,
        config.kernel_size,
        config.stride,
        config.dilation,
        config.groups,
    )?;
    let input_item_len = checked_element_count(config.channels_in, config.input_length)?;
    let output_length = conv1d_output_length(
        config.input_length,
        config.kernel_size,
        config.stride,
        config.dilation,
        config.left_padding,
        config.right_padding,
    )?;
    let output_item_len = checked_element_count(config.channels_out, output_length)?;
    if config.input_batch_stride < input_item_len || config.output_batch_stride < output_item_len {
        return Err(Error::InvalidLength);
    }
    Ok(())
}

fn validate_conv1d_im2col_config(config: Conv1dIm2colConfig) -> Result<()> {
    if config.batch == 0 || config.input_batch_stride == 0 {
        return Err(Error::InvalidLength);
    }
    validate_conv1d_shape(
        config.channels_in,
        config.channels_in,
        config.input_length,
        config.kernel_size,
        config.stride,
        config.dilation,
        config.groups,
    )?;
    let input_item_len = checked_element_count(config.channels_in, config.input_length)?;
    if config.input_batch_stride < input_item_len {
        return Err(Error::InvalidLength);
    }
    Ok(())
}

fn validate_causal_conv1d_config(config: CausalConv1dConfig) -> Result<()> {
    validate_conv1d_shape(
        config.channels_in,
        config.channels_out,
        config.input_length,
        config.kernel_size,
        config.stride,
        config.dilation,
        config.groups,
    )
}

fn validate_conv1d_shape(
    channels_in: usize,
    channels_out: usize,
    input_length: usize,
    kernel_size: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
) -> Result<()> {
    if channels_in == 0
        || channels_out == 0
        || input_length == 0
        || kernel_size == 0
        || stride == 0
        || dilation == 0
        || groups == 0
    {
        return Err(Error::InvalidLength);
    }
    if !channels_in.is_multiple_of(groups) || !channels_out.is_multiple_of(groups) {
        return Err(Error::InvalidLength);
    }
    effective_kernel_size(kernel_size, dilation)?;
    Ok(())
}

fn conv1d_output_length(
    input_length: usize,
    kernel_size: usize,
    stride: usize,
    dilation: usize,
    left_padding: usize,
    right_padding: usize,
) -> Result<usize> {
    let padded = input_length
        .checked_add(left_padding)
        .and_then(|padded| padded.checked_add(right_padding))
        .ok_or(Error::SizeOverflow)?;
    let effective_kernel_size = effective_kernel_size(kernel_size, dilation)?;
    if padded < effective_kernel_size {
        return Ok(0);
    }
    Ok((padded - effective_kernel_size) / stride + 1)
}

fn effective_kernel_size(kernel_size: usize, dilation: usize) -> Result<usize> {
    kernel_size
        .checked_sub(1)
        .ok_or(Error::SizeOverflow)?
        .checked_mul(dilation)
        .and_then(|size| size.checked_add(1))
        .ok_or(Error::SizeOverflow)
}

/// Depthwise causal conv1d decode step with in-place rolling state and SiLU.
///
/// `input` and `out` are `[batch, channels]`. `conv_state` is
/// `[batch, channels, kernel_size]` and is shifted left by one slot per
/// channel before inserting the new input at `kernel_size - 1`. `weight` is
/// `[channels, kernel_size]`, `bias` is `[channels]`, and the returned output
/// is `silu(dot(updated_state, weight) + bias)`.
pub fn causal_conv1d_update_silu(
    conv_state: &[f32],
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    batch: usize,
    channels: usize,
    kernel_size: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut out = vec![0.0f32; batch * channels];
    let mut next_state = conv_state.to_vec();
    for batch_index in 0..batch {
        for (channel, bias_value) in bias.iter().copied().enumerate().take(channels) {
            let token_offset = batch_index * channels + channel;
            let state_offset = (batch_index * channels + channel) * kernel_size;
            let weight_offset = channel * kernel_size;
            let mut dot = bias_value;
            for kernel in 0..kernel_size {
                let value = if kernel + 1 == kernel_size {
                    input[token_offset]
                } else {
                    conv_state[state_offset + kernel + 1]
                };
                dot += value * weight[weight_offset + kernel];
            }
            out[token_offset] = dot / (1.0 + (-dot).exp());
            for kernel in 0..kernel_size - 1 {
                next_state[state_offset + kernel] = conv_state[state_offset + kernel + 1];
            }
            next_state[state_offset + kernel_size - 1] = input[token_offset];
        }
    }
    (out, next_state)
}

/// Depthwise causal conv1d sequence prefill from an implicit zero state.
///
/// `input` and `out` use `[batch, channels, time]` layout. For each output
/// position, values before the beginning of the sequence are zero-padded. The
/// returned state is the final `[batch, channels, kernel_size]` rolling window,
/// also left-padded with zeros when the input is shorter than the kernel.
pub fn causal_conv1d_prefill_silu(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    batch: usize,
    channels: usize,
    kernel_size: usize,
    input_length: usize,
    output_length: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut out = vec![0.0f32; batch * channels * output_length];
    let mut final_state = vec![0.0f32; batch * channels * kernel_size];
    for batch_index in 0..batch {
        for (channel, bias_value) in bias.iter().copied().enumerate().take(channels) {
            let input_offset = (batch_index * channels + channel) * input_length;
            let output_offset = (batch_index * channels + channel) * output_length;
            let weight_offset = channel * kernel_size;
            let state_offset = (batch_index * channels + channel) * kernel_size;

            for time in 0..output_length {
                let mut dot = bias_value;
                for kernel in 0..kernel_size {
                    let input_time = time as isize + kernel as isize + 1 - kernel_size as isize;
                    let value = if input_time >= 0 && (input_time as usize) < input_length {
                        input[input_offset + input_time as usize]
                    } else {
                        0.0
                    };
                    dot += value * weight[weight_offset + kernel];
                }
                out[output_offset + time] = dot / (1.0 + (-dot).exp());
            }

            for kernel in 0..kernel_size {
                let input_time = input_length as isize + kernel as isize - kernel_size as isize;
                final_state[state_offset + kernel] =
                    if input_time >= 0 && (input_time as usize) < input_length {
                        input[input_offset + input_time as usize]
                    } else {
                        0.0
                    };
            }
        }
    }
    (out, final_state)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn causal_conv1d_prefill_matches_repeated_update_from_zero_state() {
        let batch = 2usize;
        let channels = 3usize;
        let kernel_size = 5usize;
        let input_length = 4usize;
        let input = (0..batch * channels * input_length)
            .map(|index| (index as f32 % 11.0) * 0.125 - 0.5)
            .collect::<Vec<_>>();
        let weight = (0..channels * kernel_size)
            .map(|index| (index as f32 % 7.0) * 0.25 - 0.75)
            .collect::<Vec<_>>();
        let bias = (0..channels)
            .map(|index| (index as f32 % 5.0) * 0.125 - 0.25)
            .collect::<Vec<_>>();

        let (actual_out, actual_state) = causal_conv1d_prefill_silu(
            &input,
            &weight,
            &bias,
            batch,
            channels,
            kernel_size,
            input_length,
            input_length,
        );
        let mut expected_out = vec![0.0f32; batch * channels * input_length];
        let mut state = vec![0.0f32; batch * channels * kernel_size];
        for time in 0..input_length {
            let token = (0..batch * channels)
                .map(|index| {
                    let batch_index = index / channels;
                    let channel = index % channels;
                    input[(batch_index * channels + channel) * input_length + time]
                })
                .collect::<Vec<_>>();
            let (step_out, next_state) = causal_conv1d_update_silu(
                &state,
                &token,
                &weight,
                &bias,
                batch,
                channels,
                kernel_size,
            );
            for batch_index in 0..batch {
                for channel in 0..channels {
                    expected_out[(batch_index * channels + channel) * input_length + time] =
                        step_out[batch_index * channels + channel];
                }
            }
            state = next_state;
        }

        assert_eq!(actual_out, expected_out);
        assert_eq!(actual_state, state);
    }

    #[test]
    fn causal_conv1d_prefill_final_state_zero_pads_short_input() {
        let input = [1.0f32, 2.0];
        let weight = [1.0f32, 1.0, 1.0, 1.0];
        let bias = [0.0f32];
        let (_, state) = causal_conv1d_prefill_silu(&input, &weight, &bias, 1, 1, 4, 2, 2);
        assert_eq!(state, vec![0.0, 0.0, 1.0, 2.0]);
    }
}
