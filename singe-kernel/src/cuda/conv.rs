//! Small convolution and layout-specific convolution helpers.

#[cfg(feature = "dtype-bf16")]
use singe_cuda::types::bf16;
#[cfg(feature = "dtype-f16")]
use singe_cuda::types::f16;
use singe_cuda::{
    stream::Stream,
    view::{DeviceSlice, DeviceSliceMut},
};

#[cfg(feature = "cutile")]
use crate::cuda::cutile;
use crate::{
    cuda::interop::{borrowed_stream, input_pointer, output_pointer},
    error::{Error, Result},
    utility::{checked_element_count, ensure_len},
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
pub struct Conv2dConfig {
    pub batch: usize,
    pub channels_in: usize,
    pub channels_out: usize,
    pub input_h: usize,
    pub input_w: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub padding_h: usize,
    pub padding_w: usize,
    pub dilation_h: usize,
    pub dilation_w: usize,
    pub groups: usize,
    pub input_batch_stride: usize,
    pub output_batch_stride: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Conv1dActivation {
    None,
    Gelu,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CausalConv1dWindowConfig {
    pub channels_in: usize,
    pub channels_out: usize,
    pub input_start: usize,
    pub input_length: usize,
    pub output_start: usize,
    pub output_length: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub dilation: usize,
    pub groups: usize,
    pub left_padding: usize,
}

impl Conv1dConfig {
    pub fn output_length(self) -> Result<usize> {
        validate_conv1d_config(self)?;
        let padded = self
            .input_length
            .checked_add(self.left_padding)
            .and_then(|padded| padded.checked_add(self.right_padding))
            .ok_or(Error::SizeOverflow)?;
        let effective_kernel_size = effective_kernel_size(self.kernel_size, self.dilation)?;
        if padded < effective_kernel_size {
            return Ok(0);
        }
        Ok((padded - effective_kernel_size) / self.stride + 1)
    }

    fn validate_lengths(
        self,
        out_len: usize,
        input_len: usize,
        weight_len: usize,
    ) -> Result<usize> {
        validate_conv1d_config(self)?;
        let expected_input = checked_element_count(self.channels_in, self.input_length)?;
        let expected_weight = conv1d_weight_len(
            self.channels_in,
            self.channels_out,
            self.kernel_size,
            self.groups,
        )?;
        let output_length = self.output_length()?;
        if output_length == 0 {
            return Err(Error::InvalidLength);
        }
        let expected_output = checked_element_count(self.channels_out, output_length)?;
        ensure_len(input_len, expected_input)?;
        ensure_len(weight_len, expected_weight)?;
        ensure_len(out_len, expected_output)?;
        Ok(output_length)
    }
}

impl Conv2dConfig {
    pub fn output_h(self) -> Result<usize> {
        if self.input_h == 0 || self.kernel_h == 0 || self.stride_h == 0 || self.dilation_h == 0 {
            return Err(Error::InvalidLength);
        }
        conv1d_output_length(
            self.input_h,
            self.kernel_h,
            self.stride_h,
            self.dilation_h,
            self.padding_h,
            self.padding_h,
        )
    }

    pub fn output_w(self) -> Result<usize> {
        if self.input_w == 0 || self.kernel_w == 0 || self.stride_w == 0 || self.dilation_w == 0 {
            return Err(Error::InvalidLength);
        }
        conv1d_output_length(
            self.input_w,
            self.kernel_w,
            self.stride_w,
            self.dilation_w,
            self.padding_w,
            self.padding_w,
        )
    }

    fn validate_lengths(
        self,
        out_len: usize,
        input_len: usize,
        weight_len: usize,
        bias_len: usize,
    ) -> Result<(usize, usize)> {
        validate_conv2d_config(self)?;
        let output_h = self.output_h()?;
        let output_w = self.output_w()?;
        if output_h == 0 || output_w == 0 {
            return Err(Error::InvalidLength);
        }
        let expected_input_per_batch = checked_element_count(
            checked_element_count(self.channels_in, self.input_h)?,
            self.input_w,
        )?;
        let expected_output_per_batch = checked_element_count(
            checked_element_count(self.channels_out, output_h)?,
            output_w,
        )?;
        let expected_input = self
            .batch
            .checked_sub(1)
            .ok_or(Error::SizeOverflow)?
            .checked_mul(self.input_batch_stride)
            .and_then(|span| span.checked_add(expected_input_per_batch))
            .ok_or(Error::SizeOverflow)?;
        let expected_output = self
            .batch
            .checked_sub(1)
            .ok_or(Error::SizeOverflow)?
            .checked_mul(self.output_batch_stride)
            .and_then(|span| span.checked_add(expected_output_per_batch))
            .ok_or(Error::SizeOverflow)?;
        let channels_in_per_group = self.channels_in / self.groups;
        let expected_weight = checked_element_count(
            checked_element_count(
                checked_element_count(self.channels_out, channels_in_per_group)?,
                self.kernel_h,
            )?,
            self.kernel_w,
        )?;
        ensure_len(input_len, expected_input)?;
        ensure_len(weight_len, expected_weight)?;
        ensure_len(bias_len, self.channels_out)?;
        ensure_len(out_len, expected_output)?;
        Ok((output_h, output_w))
    }
}

macro_rules! causal_conv1d_update_silu_fn {
    ($name:ident, $ty:ty) => {
        /// Depthwise causal conv1d decode step with in-place rolling state and SiLU.
        ///
        /// `input` and `out` are `[batch, channels]`. `conv_state` is
        /// `[batch, channels, kernel_size]` and is shifted left by one slot per
        /// channel before inserting the new input at `kernel_size - 1`.
        /// `weight` is `[channels, kernel_size]` and `bias` is `[channels]`.
        pub fn $name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            conv_state: &mut impl DeviceSliceMut<$ty>,
            input: &impl DeviceSlice<$ty>,
            weight: &impl DeviceSlice<$ty>,
            bias: &impl DeviceSlice<$ty>,
            batch: usize,
            channels: usize,
            kernel_size: usize,
        ) -> Result<()> {
            validate_causal_conv1d_update_silu(
                out.len(),
                conv_state.len(),
                input.len(),
                weight.len(),
                bias.len(),
                batch,
                channels,
                kernel_size,
            )?;
            let stream = borrowed_stream(stream)?;
            cutile::conv::$name(
                &stream,
                output_pointer(out),
                output_pointer(conv_state),
                input_pointer(input),
                input_pointer(weight),
                input_pointer(bias),
                batch,
                channels,
                kernel_size,
            )
        }
    };
}

macro_rules! causal_conv1d_prefill_silu_fn {
    ($name:ident, $ty:ty) => {
        /// Depthwise causal conv1d sequence prefill from an implicit zero state.
        ///
        /// `input` and `out` use `[batch, channels, time]` layout. Values before
        /// the beginning of the sequence are zero-padded, and `conv_state`
        /// receives the final `[batch, channels, kernel_size]` rolling window.
        pub fn $name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            conv_state: &mut impl DeviceSliceMut<$ty>,
            input: &impl DeviceSlice<$ty>,
            weight: &impl DeviceSlice<$ty>,
            bias: &impl DeviceSlice<$ty>,
            batch: usize,
            channels: usize,
            kernel_size: usize,
            input_length: usize,
            output_length: usize,
        ) -> Result<()> {
            validate_causal_conv1d_prefill_silu(
                out.len(),
                conv_state.len(),
                input.len(),
                weight.len(),
                bias.len(),
                batch,
                channels,
                kernel_size,
                input_length,
                output_length,
            )?;
            let stream = borrowed_stream(stream)?;
            cutile::conv::$name(
                &stream,
                output_pointer(out),
                output_pointer(conv_state),
                input_pointer(input),
                input_pointer(weight),
                input_pointer(bias),
                batch,
                channels,
                kernel_size,
                input_length,
                output_length,
            )
        }
    };
}

#[cfg(feature = "dtype-f32")]
causal_conv1d_update_silu_fn!(causal_conv1d_update_silu_f32, f32);
#[cfg(feature = "dtype-f16")]
causal_conv1d_update_silu_fn!(causal_conv1d_update_silu_f16, f16);
#[cfg(feature = "dtype-bf16")]
causal_conv1d_update_silu_fn!(causal_conv1d_update_silu_bf16, bf16);

#[cfg(feature = "dtype-f32")]
causal_conv1d_prefill_silu_fn!(causal_conv1d_prefill_silu_f32, f32);
#[cfg(feature = "dtype-f16")]
causal_conv1d_prefill_silu_fn!(causal_conv1d_prefill_silu_f16, f16);
#[cfg(feature = "dtype-bf16")]
causal_conv1d_prefill_silu_fn!(causal_conv1d_prefill_silu_bf16, bf16);

#[cfg(feature = "dtype-f32")]
pub fn conv2d_bias_gelu_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    input: &impl DeviceSlice<f32>,
    weight: &impl DeviceSlice<f32>,
    bias: &impl DeviceSlice<f32>,
    config: Conv2dConfig,
) -> Result<()> {
    let (output_h, output_w) =
        config.validate_lengths(out.len(), input.len(), weight.len(), bias.len())?;
    let stream = borrowed_stream(stream)?;
    cutile::conv::conv2d_bias_gelu_f32(
        &stream,
        output_pointer(out),
        input_pointer(input),
        input_pointer(weight),
        input_pointer(bias),
        config.batch,
        config.channels_in,
        config.channels_out,
        config.input_h,
        config.input_w,
        output_h,
        output_w,
        config.kernel_h,
        config.kernel_w,
        config.stride_h,
        config.stride_w,
        config.padding_h,
        config.padding_w,
        config.dilation_h,
        config.dilation_w,
        config.groups,
        config.input_batch_stride,
        config.output_batch_stride,
    )
}

macro_rules! conv2d_bias_gelu_half_fn {
    ($name:ident, $ty:ty, $cutile_fn:ident) => {
        pub fn $name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            input: &impl DeviceSlice<$ty>,
            weight: &impl DeviceSlice<$ty>,
            bias: &impl DeviceSlice<$ty>,
            config: Conv2dConfig,
        ) -> Result<()> {
            let (output_h, output_w) =
                config.validate_lengths(out.len(), input.len(), weight.len(), bias.len())?;
            let stream = borrowed_stream(stream)?;
            cutile::conv::$cutile_fn(
                &stream,
                output_pointer(out),
                input_pointer(input),
                input_pointer(weight),
                input_pointer(bias),
                config.batch,
                config.channels_in,
                config.channels_out,
                config.input_h,
                config.input_w,
                output_h,
                output_w,
                config.kernel_h,
                config.kernel_w,
                config.stride_h,
                config.stride_w,
                config.padding_h,
                config.padding_w,
                config.dilation_h,
                config.dilation_w,
                config.groups,
                config.input_batch_stride,
                config.output_batch_stride,
            )
        }
    };
}

#[cfg(feature = "dtype-f16")]
conv2d_bias_gelu_half_fn!(conv2d_bias_gelu_f16, f16, conv2d_bias_gelu_f16);
#[cfg(feature = "dtype-bf16")]
conv2d_bias_gelu_half_fn!(conv2d_bias_gelu_bf16, bf16, conv2d_bias_gelu_bf16);

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

    fn validate_lengths(
        self,
        out_len: usize,
        input_len: usize,
        weight_len: usize,
    ) -> Result<usize> {
        validate_batched_conv1d_config(self)?;
        let output_length = self.output_length()?;
        if output_length == 0 {
            return Err(Error::InvalidLength);
        }
        let expected_weight = conv1d_weight_len(
            self.channels_in,
            self.channels_out,
            self.kernel_size,
            self.groups,
        )?;
        let input_reach = batched_conv1d_reach(
            self.batch,
            self.input_batch_stride,
            self.channels_in,
            self.input_length,
        )?;
        let output_reach = batched_conv1d_reach(
            self.batch,
            self.output_batch_stride,
            self.channels_out,
            output_length,
        )?;
        ensure_len(input_len, input_reach)?;
        ensure_len(weight_len, expected_weight)?;
        ensure_len(out_len, output_reach)?;
        Ok(output_length)
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

    fn validate_lengths(self, out_len: usize, input_len: usize) -> Result<usize> {
        validate_conv1d_im2col_config(self)?;
        let output_length = self.output_length()?;
        if output_length == 0 {
            return Err(Error::InvalidLength);
        }
        let input_reach = batched_conv1d_reach(
            self.batch,
            self.input_batch_stride,
            self.channels_in,
            self.input_length,
        )?;
        let output_len = checked_element_count(self.batch, self.output_values_per_batch()?)?;
        ensure_len(input_len, input_reach)?;
        ensure_len(out_len, output_len)?;
        Ok(output_length)
    }
}

impl CausalConv1dConfig {
    pub fn output_length(self) -> Result<usize> {
        validate_causal_conv1d_config(self)?;
        let padded = self
            .input_length
            .checked_add(self.left_padding)
            .ok_or(Error::SizeOverflow)?;
        let effective_kernel_size = effective_kernel_size(self.kernel_size, self.dilation)?;
        if padded < effective_kernel_size {
            return Ok(0);
        }
        Ok((padded - effective_kernel_size).div_ceil(self.stride) + 1)
    }

    fn validate_lengths(
        self,
        out_len: usize,
        input_len: usize,
        weight_len: usize,
    ) -> Result<usize> {
        validate_causal_conv1d_config(self)?;
        let expected_input = checked_element_count(self.channels_in, self.input_length)?;
        let expected_weight = conv1d_weight_len(
            self.channels_in,
            self.channels_out,
            self.kernel_size,
            self.groups,
        )?;
        let output_length = self.output_length()?;
        if output_length == 0 {
            return Err(Error::InvalidLength);
        }
        let expected_output = checked_element_count(self.channels_out, output_length)?;
        ensure_len(input_len, expected_input)?;
        ensure_len(weight_len, expected_weight)?;
        ensure_len(out_len, expected_output)?;
        Ok(output_length)
    }
}

impl CausalConv1dWindowConfig {
    fn validate_lengths(self, out_len: usize, input_len: usize, weight_len: usize) -> Result<()> {
        validate_causal_conv1d_window_config(self)?;
        let expected_input = checked_element_count(self.channels_in, self.input_length)?;
        let expected_weight = conv1d_weight_len(
            self.channels_in,
            self.channels_out,
            self.kernel_size,
            self.groups,
        )?;
        let expected_output = checked_element_count(self.channels_out, self.output_length)?;
        ensure_len(input_len, expected_input)?;
        ensure_len(weight_len, expected_weight)?;
        ensure_len(out_len, expected_output)?;
        Ok(())
    }
}

#[cfg(feature = "dtype-f32")]
pub fn conv1d_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    input: &impl DeviceSlice<f32>,
    weight: &impl DeviceSlice<f32>,
    config: Conv1dConfig,
) -> Result<()> {
    let output_length = config.validate_lengths(out.len(), input.len(), weight.len())?;
    let stream = borrowed_stream(stream)?;
    cutile::conv::conv1d_causal_f32(
        &stream,
        output_pointer(out),
        input_pointer(input),
        input_pointer(weight),
        config.channels_in,
        config.channels_out,
        config.input_length,
        output_length,
        config.kernel_size,
        config.stride,
        config.dilation,
        config.groups,
        config.left_padding,
    )
}

#[cfg(feature = "dtype-f32")]
pub fn conv1d_batched_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    input: &impl DeviceSlice<f32>,
    weight: &impl DeviceSlice<f32>,
    config: BatchedConv1dConfig,
) -> Result<()> {
    let output_length = config.validate_lengths(out.len(), input.len(), weight.len())?;
    let stream = borrowed_stream(stream)?;
    cutile::conv::conv1d_batched_f32(
        &stream,
        output_pointer(out),
        input_pointer(input),
        input_pointer(weight),
        config.batch,
        config.channels_in,
        config.channels_out,
        config.input_length,
        output_length,
        config.kernel_size,
        config.stride,
        config.dilation,
        config.groups,
        config.left_padding,
        config.input_batch_stride,
        config.output_batch_stride,
    )
}

#[cfg(feature = "dtype-f16")]
#[cfg(feature = "dtype-f32")]
pub fn conv1d_f16_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    input: &impl DeviceSlice<f16>,
    weight: &impl DeviceSlice<f16>,
    config: Conv1dConfig,
) -> Result<()> {
    let output_length = config.validate_lengths(out.len(), input.len(), weight.len())?;
    let stream = borrowed_stream(stream)?;
    cutile::conv::conv1d_causal_f16_f32(
        &stream,
        output_pointer(out),
        input_pointer(input),
        input_pointer(weight),
        config.channels_in,
        config.channels_out,
        config.input_length,
        output_length,
        config.kernel_size,
        config.stride,
        config.dilation,
        config.groups,
        config.left_padding,
    )
}

#[cfg(feature = "dtype-f16")]
#[cfg(feature = "dtype-f32")]
pub fn conv1d_batched_f16_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    input: &impl DeviceSlice<f16>,
    weight: &impl DeviceSlice<f16>,
    config: BatchedConv1dConfig,
) -> Result<()> {
    let output_length = config.validate_lengths(out.len(), input.len(), weight.len())?;
    let stream = borrowed_stream(stream)?;
    cutile::conv::conv1d_batched_f16_f32(
        &stream,
        output_pointer(out),
        input_pointer(input),
        input_pointer(weight),
        config.batch,
        config.channels_in,
        config.channels_out,
        config.input_length,
        output_length,
        config.kernel_size,
        config.stride,
        config.dilation,
        config.groups,
        config.left_padding,
        config.input_batch_stride,
        config.output_batch_stride,
    )
}

#[cfg(feature = "dtype-bf16")]
#[cfg(feature = "dtype-f32")]
pub fn conv1d_bf16_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    input: &impl DeviceSlice<bf16>,
    weight: &impl DeviceSlice<bf16>,
    config: Conv1dConfig,
) -> Result<()> {
    let output_length = config.validate_lengths(out.len(), input.len(), weight.len())?;
    let stream = borrowed_stream(stream)?;
    cutile::conv::conv1d_causal_bf16_f32(
        &stream,
        output_pointer(out),
        input_pointer(input),
        input_pointer(weight),
        config.channels_in,
        config.channels_out,
        config.input_length,
        output_length,
        config.kernel_size,
        config.stride,
        config.dilation,
        config.groups,
        config.left_padding,
    )
}

#[cfg(feature = "dtype-bf16")]
#[cfg(feature = "dtype-f32")]
pub fn conv1d_batched_bf16_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    input: &impl DeviceSlice<bf16>,
    weight: &impl DeviceSlice<bf16>,
    config: BatchedConv1dConfig,
) -> Result<()> {
    let output_length = config.validate_lengths(out.len(), input.len(), weight.len())?;
    let stream = borrowed_stream(stream)?;
    cutile::conv::conv1d_batched_bf16_f32(
        &stream,
        output_pointer(out),
        input_pointer(input),
        input_pointer(weight),
        config.batch,
        config.channels_in,
        config.channels_out,
        config.input_length,
        output_length,
        config.kernel_size,
        config.stride,
        config.dilation,
        config.groups,
        config.left_padding,
        config.input_batch_stride,
        config.output_batch_stride,
    )
}

#[cfg(feature = "dtype-f32")]
pub fn conv1d_im2col_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    input: &impl DeviceSlice<f32>,
    config: Conv1dConfig,
) -> Result<()> {
    let input_values = checked_element_count(config.channels_in, config.input_length)?;
    let im2col = Conv1dIm2colConfig {
        batch: 1,
        channels_in: config.channels_in,
        input_length: config.input_length,
        kernel_size: config.kernel_size,
        stride: config.stride,
        dilation: config.dilation,
        groups: config.groups,
        left_padding: config.left_padding,
        right_padding: config.right_padding,
        input_batch_stride: input_values,
    };
    conv1d_batched_im2col_f32(stream, out, input, im2col)
}

#[cfg(feature = "dtype-f32")]
pub fn conv1d_batched_im2col_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    input: &impl DeviceSlice<f32>,
    config: Conv1dIm2colConfig,
) -> Result<()> {
    let output_length = config.validate_lengths(out.len(), input.len())?;
    let stream = borrowed_stream(stream)?;
    cutile::conv::conv1d_batched_im2col_f32(
        &stream,
        output_pointer(out),
        input_pointer(input),
        config.batch,
        config.channels_in,
        config.input_length,
        output_length,
        config.kernel_size,
        config.stride,
        config.dilation,
        config.groups,
        config.left_padding,
        config.input_batch_stride,
    )
}

#[cfg(feature = "dtype-f32")]
pub fn conv1d_bias_activation_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    input: &impl DeviceSlice<f32>,
    weight: &impl DeviceSlice<f32>,
    bias: &impl DeviceSlice<f32>,
    config: Conv1dConfig,
    activation: Conv1dActivation,
) -> Result<()> {
    let output_length = config.validate_lengths(out.len(), input.len(), weight.len())?;
    ensure_len(bias.len(), config.channels_out)?;
    let stream = borrowed_stream(stream)?;
    match activation {
        Conv1dActivation::None => cutile::conv::conv1d_causal_bias_f32(
            &stream,
            output_pointer(out),
            input_pointer(input),
            input_pointer(weight),
            input_pointer(bias),
            config.channels_in,
            config.channels_out,
            config.input_length,
            output_length,
            config.kernel_size,
            config.stride,
            config.dilation,
            config.groups,
            config.left_padding,
        ),
        Conv1dActivation::Gelu => cutile::conv::conv1d_causal_bias_gelu_f32(
            &stream,
            output_pointer(out),
            input_pointer(input),
            input_pointer(weight),
            input_pointer(bias),
            config.channels_in,
            config.channels_out,
            config.input_length,
            output_length,
            config.kernel_size,
            config.stride,
            config.dilation,
            config.groups,
            config.left_padding,
        ),
    }
}

#[cfg(feature = "dtype-f32")]
pub fn conv1d_causal_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    input: &impl DeviceSlice<f32>,
    weight: &impl DeviceSlice<f32>,
    config: CausalConv1dConfig,
) -> Result<()> {
    let output_length = config.validate_lengths(out.len(), input.len(), weight.len())?;
    let stream = borrowed_stream(stream)?;
    cutile::conv::conv1d_causal_f32(
        &stream,
        output_pointer(out),
        input_pointer(input),
        input_pointer(weight),
        config.channels_in,
        config.channels_out,
        config.input_length,
        output_length,
        config.kernel_size,
        config.stride,
        config.dilation,
        config.groups,
        config.left_padding,
    )
}

#[cfg(feature = "dtype-f32")]
pub fn conv1d_causal_bias_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    input: &impl DeviceSlice<f32>,
    weight: &impl DeviceSlice<f32>,
    bias: &impl DeviceSlice<f32>,
    config: CausalConv1dConfig,
) -> Result<()> {
    conv1d_causal_bias_activation_f32(
        stream,
        out,
        input,
        weight,
        bias,
        config,
        Conv1dActivation::None,
    )
}

#[cfg(feature = "dtype-f32")]
pub fn conv1d_causal_bias_activation_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    input: &impl DeviceSlice<f32>,
    weight: &impl DeviceSlice<f32>,
    bias: &impl DeviceSlice<f32>,
    config: CausalConv1dConfig,
    activation: Conv1dActivation,
) -> Result<()> {
    let output_length = config.validate_lengths(out.len(), input.len(), weight.len())?;
    ensure_len(bias.len(), config.channels_out)?;
    let stream = borrowed_stream(stream)?;
    match activation {
        Conv1dActivation::None => cutile::conv::conv1d_causal_bias_f32(
            &stream,
            output_pointer(out),
            input_pointer(input),
            input_pointer(weight),
            input_pointer(bias),
            config.channels_in,
            config.channels_out,
            config.input_length,
            output_length,
            config.kernel_size,
            config.stride,
            config.dilation,
            config.groups,
            config.left_padding,
        ),
        Conv1dActivation::Gelu => cutile::conv::conv1d_causal_bias_gelu_f32(
            &stream,
            output_pointer(out),
            input_pointer(input),
            input_pointer(weight),
            input_pointer(bias),
            config.channels_in,
            config.channels_out,
            config.input_length,
            output_length,
            config.kernel_size,
            config.stride,
            config.dilation,
            config.groups,
            config.left_padding,
        ),
    }
}

#[cfg(feature = "dtype-f32")]
pub fn conv1d_causal_bias_gelu_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    input: &impl DeviceSlice<f32>,
    weight: &impl DeviceSlice<f32>,
    bias: &impl DeviceSlice<f32>,
    config: CausalConv1dConfig,
) -> Result<()> {
    conv1d_causal_bias_activation_f32(
        stream,
        out,
        input,
        weight,
        bias,
        config,
        Conv1dActivation::Gelu,
    )
}

#[cfg(feature = "dtype-f32")]
pub fn conv1d_causal_bias_gelu_f32_window(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    input: &impl DeviceSlice<f32>,
    weight: &impl DeviceSlice<f32>,
    bias: &impl DeviceSlice<f32>,
    config: CausalConv1dWindowConfig,
) -> Result<()> {
    config.validate_lengths(out.len(), input.len(), weight.len())?;
    ensure_len(bias.len(), config.channels_out)?;
    let stream = borrowed_stream(stream)?;
    cutile::conv::conv1d_causal_bias_gelu_f32_window(
        &stream,
        output_pointer(out),
        input_pointer(input),
        input_pointer(weight),
        input_pointer(bias),
        config.channels_in,
        config.channels_out,
        config.input_start,
        config.input_length,
        config.output_start,
        config.output_length,
        config.kernel_size,
        config.stride,
        config.dilation,
        config.groups,
        config.left_padding,
    )
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

fn validate_conv2d_config(config: Conv2dConfig) -> Result<()> {
    if config.batch == 0
        || config.channels_in == 0
        || config.channels_out == 0
        || config.input_h == 0
        || config.input_w == 0
        || config.kernel_h == 0
        || config.kernel_w == 0
        || config.stride_h == 0
        || config.stride_w == 0
        || config.dilation_h == 0
        || config.dilation_w == 0
        || config.groups == 0
        || config.input_batch_stride == 0
        || config.output_batch_stride == 0
        || !config.channels_in.is_multiple_of(config.groups)
        || !config.channels_out.is_multiple_of(config.groups)
    {
        return Err(Error::InvalidLength);
    }
    let input_item_len = checked_element_count(
        checked_element_count(config.channels_in, config.input_h)?,
        config.input_w,
    )?;
    let output_item_len = checked_element_count(
        checked_element_count(config.channels_out, config.output_h()?)?,
        config.output_w()?,
    )?;
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

fn validate_causal_conv1d_window_config(config: CausalConv1dWindowConfig) -> Result<()> {
    if config.output_length == 0 {
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
    config
        .input_start
        .checked_add(config.input_length)
        .ok_or(Error::SizeOverflow)?;
    config
        .output_start
        .checked_add(config.output_length)
        .ok_or(Error::SizeOverflow)?;
    Ok(())
}

fn validate_causal_conv1d_update_silu(
    out_len: usize,
    conv_state_len: usize,
    input_len: usize,
    weight_len: usize,
    bias_len: usize,
    batch: usize,
    channels: usize,
    kernel_size: usize,
) -> Result<()> {
    if batch == 0 || channels == 0 || kernel_size == 0 {
        return Err(Error::InvalidLength);
    }
    let token_len = checked_element_count(batch, channels)?;
    let state_len = checked_element_count(token_len, kernel_size)?;
    let weight_len_expected = checked_element_count(channels, kernel_size)?;
    ensure_len(out_len, token_len)?;
    ensure_len(input_len, token_len)?;
    ensure_len(conv_state_len, state_len)?;
    ensure_len(weight_len, weight_len_expected)?;
    ensure_len(bias_len, channels)
}

fn validate_causal_conv1d_prefill_silu(
    out_len: usize,
    conv_state_len: usize,
    input_len: usize,
    weight_len: usize,
    bias_len: usize,
    batch: usize,
    channels: usize,
    kernel_size: usize,
    input_length: usize,
    output_length: usize,
) -> Result<()> {
    if batch == 0 || channels == 0 || kernel_size == 0 || input_length == 0 || output_length == 0 {
        return Err(Error::InvalidLength);
    }
    let batch_channels = checked_element_count(batch, channels)?;
    ensure_len(
        input_len,
        checked_element_count(batch_channels, input_length)?,
    )?;
    ensure_len(
        out_len,
        checked_element_count(batch_channels, output_length)?,
    )?;
    ensure_len(
        conv_state_len,
        checked_element_count(batch_channels, kernel_size)?,
    )?;
    ensure_len(weight_len, checked_element_count(channels, kernel_size)?)?;
    ensure_len(bias_len, channels)
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

fn conv1d_weight_len(
    channels_in: usize,
    channels_out: usize,
    kernel_size: usize,
    groups: usize,
) -> Result<usize> {
    let channels_in_per_group = channels_in / groups;
    checked_element_count(
        checked_element_count(channels_out, channels_in_per_group)?,
        kernel_size,
    )
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

fn batched_conv1d_reach(
    batch: usize,
    batch_stride: usize,
    channels: usize,
    length: usize,
) -> Result<usize> {
    let batch_span = batch
        .checked_sub(1)
        .ok_or(Error::SizeOverflow)?
        .checked_mul(batch_stride)
        .ok_or(Error::SizeOverflow)?;
    let item_len = checked_element_count(channels, length)?;
    batch_span.checked_add(item_len).ok_or(Error::SizeOverflow)
}

fn effective_kernel_size(kernel_size: usize, dilation: usize) -> Result<usize> {
    kernel_size
        .checked_sub(1)
        .ok_or(Error::SizeOverflow)?
        .checked_mul(dilation)
        .and_then(|span| span.checked_add(1))
        .ok_or(Error::SizeOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn causal_conv1d_update_silu_validation_accepts_exact_lengths() -> Result<()> {
        validate_causal_conv1d_update_silu(10, 40, 10, 20, 5, 2, 5, 4)
    }

    #[test]
    fn causal_conv1d_update_silu_validation_rejects_short_state() {
        assert!(matches!(
            validate_causal_conv1d_update_silu(10, 39, 10, 20, 5, 2, 5, 4),
            Err(Error::LengthMismatch)
        ));
    }

    #[test]
    fn causal_conv1d_update_silu_validation_rejects_zero_channels() {
        assert!(matches!(
            validate_causal_conv1d_update_silu(0, 0, 0, 0, 0, 0, 0, 0),
            Err(Error::InvalidLength)
        ));
    }

    #[test]
    fn causal_conv1d_prefill_silu_validation_accepts_exact_lengths() -> Result<()> {
        validate_causal_conv1d_prefill_silu(60, 40, 60, 20, 5, 2, 5, 4, 6, 6)
    }

    #[test]
    fn causal_conv1d_prefill_silu_validation_rejects_short_output() {
        assert!(matches!(
            validate_causal_conv1d_prefill_silu(59, 40, 60, 20, 5, 2, 5, 4, 6, 6),
            Err(Error::LengthMismatch)
        ));
    }

    #[test]
    fn causal_conv1d_prefill_silu_validation_rejects_zero_output_length() {
        assert!(matches!(
            validate_causal_conv1d_prefill_silu(0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            Err(Error::InvalidLength)
        ));
    }
}
