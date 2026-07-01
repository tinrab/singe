use std::sync::Arc;

#[cfg(feature = "dtype-bf16")]
use cutile::half::bf16;
#[cfg(feature = "dtype-f16")]
use cutile::half::f16;
use cutile::{
    cuda_async::device_buffer::DevicePointer, cuda_core::Stream, tile_kernel::TileKernel,
};

use crate::{
    cuda::cutile::{
        DeviceOpExt,
        kernel::conv as kernel_conv,
        utility::{VectorLaunch, checked_device_pointer},
    },
    error::{Error, Result},
    utility::{checked_element_count, checked_i32_value},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Conv1dWindowParams {
    channels_in: i32,
    channels_out: i32,
    input_start: i32,
    input_length: i32,
    output_start: i32,
    output_length: i32,
    stride: i32,
    dilation: i32,
    groups: i32,
    left_padding: i32,
    launch: VectorLaunch,
}

impl Conv1dWindowParams {
    fn create(
        channels_in: usize,
        channels_out: usize,
        input_start: usize,
        input_length: usize,
        output_start: usize,
        output_length: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        left_padding: usize,
    ) -> Result<Self> {
        if channels_in == 0
            || channels_out == 0
            || input_length == 0
            || output_length == 0
            || kernel_size == 0
            || stride == 0
            || dilation == 0
            || groups == 0
        {
            return Err(Error::InvalidLength);
        }
        let input_end = input_start
            .checked_add(input_length)
            .ok_or(Error::SizeOverflow)?;
        let output_end = output_start
            .checked_add(output_length)
            .ok_or(Error::SizeOverflow)?;
        checked_i32_value(input_end)?;
        checked_i32_value(output_end)?;
        let output_len = checked_element_count(channels_out, output_length)?;
        Ok(Self {
            channels_in: checked_i32_value(channels_in)?,
            channels_out: checked_i32_value(channels_out)?,
            input_start: checked_i32_value(input_start)?,
            input_length: checked_i32_value(input_length)?,
            output_start: checked_i32_value(output_start)?,
            output_length: checked_i32_value(output_length)?,
            stride: checked_i32_value(stride)?,
            dilation: checked_i32_value(dilation)?,
            groups: checked_i32_value(groups)?,
            left_padding: checked_i32_value(left_padding)?,
            launch: VectorLaunch::create(output_len)?,
        })
    }

    fn create_full(
        channels_in: usize,
        channels_out: usize,
        input_length: usize,
        output_length: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        left_padding: usize,
    ) -> Result<Self> {
        Self::create(
            channels_in,
            channels_out,
            0,
            input_length,
            0,
            output_length,
            kernel_size,
            stride,
            dilation,
            groups,
            left_padding,
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Conv1dBatchedParams {
    channels_in: i32,
    channels_out: i32,
    input_length: i32,
    output_length: i32,
    stride: i32,
    dilation: i32,
    groups: i32,
    left_padding: i32,
    input_batch_stride: i32,
    output_batch_stride: i32,
    output_values_per_batch: i32,
    launch: VectorLaunch,
}

impl Conv1dBatchedParams {
    fn create(
        batch: usize,
        channels_in: usize,
        channels_out: usize,
        input_length: usize,
        output_length: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        left_padding: usize,
        input_batch_stride: usize,
        output_batch_stride: usize,
    ) -> Result<Self> {
        if batch == 0
            || channels_in == 0
            || channels_out == 0
            || input_length == 0
            || output_length == 0
            || kernel_size == 0
            || stride == 0
            || dilation == 0
            || groups == 0
            || input_batch_stride == 0
            || output_batch_stride == 0
        {
            return Err(Error::InvalidLength);
        }
        let input_values_per_batch = checked_element_count(channels_in, input_length)?;
        let output_values_per_batch = checked_element_count(channels_out, output_length)?;
        let input_reach = batch
            .checked_sub(1)
            .ok_or(Error::SizeOverflow)?
            .checked_mul(input_batch_stride)
            .and_then(|span| span.checked_add(input_values_per_batch))
            .ok_or(Error::SizeOverflow)?;
        let output_reach = batch
            .checked_sub(1)
            .ok_or(Error::SizeOverflow)?
            .checked_mul(output_batch_stride)
            .and_then(|span| span.checked_add(output_values_per_batch))
            .ok_or(Error::SizeOverflow)?;
        checked_i32_value(input_reach)?;
        checked_i32_value(output_reach)?;
        let output_len = checked_element_count(batch, output_values_per_batch)?;
        Ok(Self {
            channels_in: checked_i32_value(channels_in)?,
            channels_out: checked_i32_value(channels_out)?,
            input_length: checked_i32_value(input_length)?,
            output_length: checked_i32_value(output_length)?,
            stride: checked_i32_value(stride)?,
            dilation: checked_i32_value(dilation)?,
            groups: checked_i32_value(groups)?,
            left_padding: checked_i32_value(left_padding)?,
            input_batch_stride: checked_i32_value(input_batch_stride)?,
            output_batch_stride: checked_i32_value(output_batch_stride)?,
            output_values_per_batch: checked_i32_value(output_values_per_batch)?,
            launch: VectorLaunch::create(output_len)?,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Conv1dIm2colParams {
    channels_in: i32,
    input_length: i32,
    output_length: i32,
    stride: i32,
    dilation: i32,
    groups: i32,
    left_padding: i32,
    input_batch_stride: i32,
    output_values_per_batch: i32,
    launch: VectorLaunch,
}

impl Conv1dIm2colParams {
    fn create(
        batch: usize,
        channels_in: usize,
        input_length: usize,
        output_length: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        left_padding: usize,
        input_batch_stride: usize,
    ) -> Result<Self> {
        if batch == 0
            || channels_in == 0
            || input_length == 0
            || output_length == 0
            || kernel_size == 0
            || stride == 0
            || dilation == 0
            || groups == 0
            || input_batch_stride == 0
        {
            return Err(Error::InvalidLength);
        }
        let output_values_per_batch = checked_element_count(
            checked_element_count(channels_in, output_length)?,
            kernel_size,
        )?;
        let output_len = checked_element_count(batch, output_values_per_batch)?;
        Ok(Self {
            channels_in: checked_i32_value(channels_in)?,
            input_length: checked_i32_value(input_length)?,
            output_length: checked_i32_value(output_length)?,
            stride: checked_i32_value(stride)?,
            dilation: checked_i32_value(dilation)?,
            groups: checked_i32_value(groups)?,
            left_padding: checked_i32_value(left_padding)?,
            input_batch_stride: checked_i32_value(input_batch_stride)?,
            output_values_per_batch: checked_i32_value(output_values_per_batch)?,
            launch: VectorLaunch::create(output_len)?,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Conv2dParams {
    batch: i32,
    channels_in: i32,
    channels_out: i32,
    input_h: i32,
    input_w: i32,
    output_h: i32,
    output_w: i32,
    kernel_h: i32,
    kernel_w: i32,
    stride_h: i32,
    stride_w: i32,
    padding_h: i32,
    padding_w: i32,
    dilation_h: i32,
    dilation_w: i32,
    groups: i32,
    input_batch_stride: i32,
    output_batch_stride: i32,
    output_len: i32,
    launch: VectorLaunch,
}

impl Conv2dParams {
    fn create(
        batch: usize,
        channels_in: usize,
        channels_out: usize,
        input_h: usize,
        input_w: usize,
        output_h: usize,
        output_w: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        padding_h: usize,
        padding_w: usize,
        dilation_h: usize,
        dilation_w: usize,
        groups: usize,
        input_batch_stride: usize,
        output_batch_stride: usize,
    ) -> Result<Self> {
        if batch == 0
            || channels_in == 0
            || channels_out == 0
            || input_h == 0
            || input_w == 0
            || output_h == 0
            || output_w == 0
            || kernel_h == 0
            || kernel_w == 0
            || stride_h == 0
            || stride_w == 0
            || dilation_h == 0
            || dilation_w == 0
            || groups == 0
            || input_batch_stride == 0
            || output_batch_stride == 0
            || !channels_in.is_multiple_of(groups)
            || !channels_out.is_multiple_of(groups)
        {
            return Err(Error::InvalidLength);
        }

        let input_values_per_batch =
            checked_element_count(checked_element_count(channels_in, input_h)?, input_w)?;
        let output_values_per_batch =
            checked_element_count(checked_element_count(channels_out, output_h)?, output_w)?;
        let input_reach = batch
            .checked_sub(1)
            .ok_or(Error::SizeOverflow)?
            .checked_mul(input_batch_stride)
            .and_then(|span| span.checked_add(input_values_per_batch))
            .ok_or(Error::SizeOverflow)?;
        let output_reach = batch
            .checked_sub(1)
            .ok_or(Error::SizeOverflow)?
            .checked_mul(output_batch_stride)
            .and_then(|span| span.checked_add(output_values_per_batch))
            .ok_or(Error::SizeOverflow)?;
        checked_i32_value(input_reach)?;
        checked_i32_value(output_reach)?;
        let output_len = checked_element_count(batch, output_values_per_batch)?;
        Ok(Self {
            batch: checked_i32_value(batch)?,
            channels_in: checked_i32_value(channels_in)?,
            channels_out: checked_i32_value(channels_out)?,
            input_h: checked_i32_value(input_h)?,
            input_w: checked_i32_value(input_w)?,
            output_h: checked_i32_value(output_h)?,
            output_w: checked_i32_value(output_w)?,
            kernel_h: checked_i32_value(kernel_h)?,
            kernel_w: checked_i32_value(kernel_w)?,
            stride_h: checked_i32_value(stride_h)?,
            stride_w: checked_i32_value(stride_w)?,
            padding_h: checked_i32_value(padding_h)?,
            padding_w: checked_i32_value(padding_w)?,
            dilation_h: checked_i32_value(dilation_h)?,
            dilation_w: checked_i32_value(dilation_w)?,
            groups: checked_i32_value(groups)?,
            input_batch_stride: checked_i32_value(input_batch_stride)?,
            output_batch_stride: checked_i32_value(output_batch_stride)?,
            output_len: checked_i32_value(output_len)?,
            launch: VectorLaunch::create(output_len)?,
        })
    }
}

macro_rules! causal_conv1d_update_silu_fn {
    ($name:ident, $ty:ty, $kernel_fn:ident, $append_kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            conv_state: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            weight: DevicePointer<$ty>,
            bias: DevicePointer<$ty>,
            batch: usize,
            channels: usize,
            kernel_size: usize,
        ) -> Result<()> {
            checked_device_pointer(out)?;
            checked_device_pointer(conv_state)?;
            checked_device_pointer(input)?;
            checked_device_pointer(weight)?;
            checked_device_pointer(bias)?;
            if batch == 0 || channels == 0 || kernel_size == 0 {
                return Err(Error::InvalidLength);
            }
            let len = checked_element_count(batch, channels)?;
            let launch = VectorLaunch::create(len)?;
            checked_i32_value(batch)?;
            let channels = checked_i32_value(channels)?;
            let kernel_size = checked_i32_value(kernel_size)?;
            let grid = launch.grid;
            unsafe {
                kernel_conv::$kernel_fn(
                    out,
                    conv_state,
                    input,
                    weight,
                    bias,
                    channels,
                    kernel_size,
                    launch.len_i32,
                )
            }
            .grid(grid)
            .enqueue_on(stream)?;
            unsafe {
                kernel_conv::$append_kernel_fn(conv_state, input, kernel_size, launch.len_i32)
            }
            .grid(grid)
            .enqueue_on(stream)?;
            Ok(())
        }
    };
}

macro_rules! causal_conv1d_prefill_silu_fn {
    ($name:ident, $ty:ty, $kernel_fn:ident, $state_kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            conv_state: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            weight: DevicePointer<$ty>,
            bias: DevicePointer<$ty>,
            batch: usize,
            channels: usize,
            kernel_size: usize,
            input_length: usize,
            output_length: usize,
        ) -> Result<()> {
            checked_device_pointer(out)?;
            checked_device_pointer(conv_state)?;
            checked_device_pointer(input)?;
            checked_device_pointer(weight)?;
            checked_device_pointer(bias)?;
            if batch == 0
                || channels == 0
                || kernel_size == 0
                || input_length == 0
                || output_length == 0
            {
                return Err(Error::InvalidLength);
            }
            let batch_channels = checked_element_count(batch, channels)?;
            let input_values = checked_element_count(batch_channels, input_length)?;
            let output_values = checked_element_count(batch_channels, output_length)?;
            let state_values = checked_element_count(batch_channels, kernel_size)?;
            let weight_values = checked_element_count(channels, kernel_size)?;
            checked_i32_value(input_values)?;
            checked_i32_value(output_values)?;
            checked_i32_value(state_values)?;
            checked_i32_value(weight_values)?;
            let time_launch = VectorLaunch::create(output_length)?;
            let state_launch = VectorLaunch::create(state_values)?;
            let batch_blocks = u32::try_from(batch).map_err(|_| Error::SizeOverflow)?;
            let channel_blocks = u32::try_from(channels).map_err(|_| Error::SizeOverflow)?;
            let channels = checked_i32_value(channels)?;
            let kernel_size = checked_i32_value(kernel_size)?;
            let input_length = checked_i32_value(input_length)?;
            let output_length = time_launch.len_i32;
            unsafe {
                kernel_conv::$kernel_fn(
                    out,
                    input,
                    weight,
                    bias,
                    channels,
                    kernel_size,
                    input_length,
                    output_length,
                )
            }
            .grid((channel_blocks, time_launch.grid.0, batch_blocks))
            .enqueue_on(stream)?;
            unsafe {
                kernel_conv::$state_kernel_fn(
                    conv_state,
                    input,
                    kernel_size,
                    input_length,
                    state_launch.len_i32,
                )
            }
            .grid(state_launch.grid)
            .enqueue_on(stream)?;
            Ok(())
        }
    };
}

#[cfg(feature = "dtype-f32")]
causal_conv1d_update_silu_fn!(
    causal_conv1d_update_silu_f32,
    f32,
    causal_conv1d_update_silu_f32,
    causal_conv1d_append_state_f32
);
#[cfg(feature = "dtype-f16")]
causal_conv1d_update_silu_fn!(
    causal_conv1d_update_silu_f16,
    f16,
    causal_conv1d_update_silu_f16,
    causal_conv1d_append_state_f16
);
#[cfg(feature = "dtype-bf16")]
causal_conv1d_update_silu_fn!(
    causal_conv1d_update_silu_bf16,
    bf16,
    causal_conv1d_update_silu_bf16,
    causal_conv1d_append_state_bf16
);

#[cfg(feature = "dtype-f32")]
causal_conv1d_prefill_silu_fn!(
    causal_conv1d_prefill_silu_f32,
    f32,
    causal_conv1d_prefill_silu_f32,
    causal_conv1d_prefill_state_f32
);
#[cfg(feature = "dtype-f16")]
causal_conv1d_prefill_silu_fn!(
    causal_conv1d_prefill_silu_f16,
    f16,
    causal_conv1d_prefill_silu_f16,
    causal_conv1d_prefill_state_f16
);
#[cfg(feature = "dtype-bf16")]
causal_conv1d_prefill_silu_fn!(
    causal_conv1d_prefill_silu_bf16,
    bf16,
    causal_conv1d_prefill_silu_bf16,
    causal_conv1d_prefill_state_bf16
);

#[cfg(feature = "dtype-f32")]
pub fn conv2d_bias_gelu_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    input: DevicePointer<f32>,
    weight: DevicePointer<f32>,
    bias: DevicePointer<f32>,
    batch: usize,
    channels_in: usize,
    channels_out: usize,
    input_h: usize,
    input_w: usize,
    output_h: usize,
    output_w: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    groups: usize,
    input_batch_stride: usize,
    output_batch_stride: usize,
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(input)?;
    checked_device_pointer(weight)?;
    checked_device_pointer(bias)?;
    let params = Conv2dParams::create(
        batch,
        channels_in,
        channels_out,
        input_h,
        input_w,
        output_h,
        output_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        groups,
        input_batch_stride,
        output_batch_stride,
    )?;
    unsafe {
        kernel_conv::conv2d_bias_gelu_f32(
            out,
            input,
            weight,
            bias,
            params.batch,
            params.channels_in,
            params.channels_out,
            params.input_h,
            params.input_w,
            params.output_h,
            params.output_w,
            params.kernel_h,
            params.kernel_w,
            params.stride_h,
            params.stride_w,
            params.padding_h,
            params.padding_w,
            params.dilation_h,
            params.dilation_w,
            params.groups,
            params.input_batch_stride,
            params.output_batch_stride,
            params.output_len,
        )
    }
    .grid(params.launch.grid)
    .enqueue_on(stream)?;
    Ok(())
}

macro_rules! conv2d_bias_gelu_half_fn {
    ($name:ident, $ty:ty, $kernel:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            weight: DevicePointer<$ty>,
            bias: DevicePointer<$ty>,
            batch: usize,
            channels_in: usize,
            channels_out: usize,
            input_h: usize,
            input_w: usize,
            output_h: usize,
            output_w: usize,
            kernel_h: usize,
            kernel_w: usize,
            stride_h: usize,
            stride_w: usize,
            padding_h: usize,
            padding_w: usize,
            dilation_h: usize,
            dilation_w: usize,
            groups: usize,
            input_batch_stride: usize,
            output_batch_stride: usize,
        ) -> Result<()> {
            checked_device_pointer(out)?;
            checked_device_pointer(input)?;
            checked_device_pointer(weight)?;
            checked_device_pointer(bias)?;
            let params = Conv2dParams::create(
                batch,
                channels_in,
                channels_out,
                input_h,
                input_w,
                output_h,
                output_w,
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                padding_h,
                padding_w,
                dilation_h,
                dilation_w,
                groups,
                input_batch_stride,
                output_batch_stride,
            )?;
            unsafe {
                kernel_conv::$kernel(
                    out,
                    input,
                    weight,
                    bias,
                    params.batch,
                    params.channels_in,
                    params.channels_out,
                    params.input_h,
                    params.input_w,
                    params.output_h,
                    params.output_w,
                    params.kernel_h,
                    params.kernel_w,
                    params.stride_h,
                    params.stride_w,
                    params.padding_h,
                    params.padding_w,
                    params.dilation_h,
                    params.dilation_w,
                    params.groups,
                    params.input_batch_stride,
                    params.output_batch_stride,
                    params.output_len,
                )
            }
            .grid(params.launch.grid)
            .enqueue_on(stream)?;
            Ok(())
        }
    };
}

#[cfg(feature = "dtype-f16")]
conv2d_bias_gelu_half_fn!(conv2d_bias_gelu_f16, f16, conv2d_bias_gelu_f16);
#[cfg(feature = "dtype-bf16")]
conv2d_bias_gelu_half_fn!(conv2d_bias_gelu_bf16, bf16, conv2d_bias_gelu_bf16);

macro_rules! dispatch_conv1d_kernel_size {
    (
        $op:literal,
        $kernel_size:expr,
        $stream:expr,
        $grid:expr,
        [$kernel_1:ident, $kernel_2:ident, $kernel_3:ident, $kernel_5:ident]
        ($($arg:expr),* $(,)?)
    ) => {{
        match $kernel_size {
            1 => unsafe { kernel_conv::$kernel_1($($arg),*) }
                .grid($grid)
                .enqueue_on($stream)?,
            2 => unsafe { kernel_conv::$kernel_2($($arg),*) }
                .grid($grid)
                .enqueue_on($stream)?,
            3 => unsafe { kernel_conv::$kernel_3($($arg),*) }
                .grid($grid)
                .enqueue_on($stream)?,
            5 => unsafe { kernel_conv::$kernel_5($($arg),*) }
                .grid($grid)
                .enqueue_on($stream)?,
            _ => {
                return Err(Error::UnsupportedKernelSize {
                    op: $op.into(),
                    kernel_size: $kernel_size,
                });
            }
        };
        Ok::<(), crate::error::Error>(())
    }};
}

#[cfg(feature = "dtype-f32")]
pub fn conv1d_causal_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    input: DevicePointer<f32>,
    weight: DevicePointer<f32>,
    channels_in: usize,
    channels_out: usize,
    input_length: usize,
    output_length: usize,
    kernel_size: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
    left_padding: usize,
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(input)?;
    checked_device_pointer(weight)?;
    let params = Conv1dWindowParams::create_full(
        channels_in,
        channels_out,
        input_length,
        output_length,
        kernel_size,
        stride,
        dilation,
        groups,
        left_padding,
    )?;
    let channels_in = params.channels_in;
    let channels_out = params.channels_out;
    let input_start = params.input_start;
    let input_length = params.input_length;
    let output_start = params.output_start;
    let output_length = params.output_length;
    let stride = params.stride;
    let dilation = params.dilation;
    let groups = params.groups;
    let left_padding = params.left_padding;
    let output_len_i32 = params.launch.len_i32;
    let grid = params.launch.grid;

    dispatch_conv1d_kernel_size!(
        "conv1d_causal_f32",
        kernel_size,
        stream,
        grid,
        [
            conv1d_causal_f32_k1,
            conv1d_causal_f32_k2,
            conv1d_causal_f32_k3,
            conv1d_causal_f32_k5
        ](
            out,
            input,
            weight,
            channels_in,
            channels_out,
            input_start,
            input_length,
            output_start,
            output_length,
            stride,
            dilation,
            groups,
            left_padding,
            output_len_i32,
        )
    )
}

#[cfg(feature = "dtype-f32")]
pub fn conv1d_batched_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    input: DevicePointer<f32>,
    weight: DevicePointer<f32>,
    batch: usize,
    channels_in: usize,
    channels_out: usize,
    input_length: usize,
    output_length: usize,
    kernel_size: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
    left_padding: usize,
    input_batch_stride: usize,
    output_batch_stride: usize,
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(input)?;
    checked_device_pointer(weight)?;
    let params = Conv1dBatchedParams::create(
        batch,
        channels_in,
        channels_out,
        input_length,
        output_length,
        kernel_size,
        stride,
        dilation,
        groups,
        left_padding,
        input_batch_stride,
        output_batch_stride,
    )?;
    let channels_in = params.channels_in;
    let channels_out = params.channels_out;
    let input_length = params.input_length;
    let output_length = params.output_length;
    let stride = params.stride;
    let dilation = params.dilation;
    let groups = params.groups;
    let left_padding = params.left_padding;
    let input_batch_stride = params.input_batch_stride;
    let output_batch_stride = params.output_batch_stride;
    let output_values_per_batch = params.output_values_per_batch;
    let output_len_i32 = params.launch.len_i32;
    let grid = params.launch.grid;

    dispatch_conv1d_kernel_size!(
        "conv1d_batched_f32",
        kernel_size,
        stream,
        grid,
        [
            conv1d_batched_f32_k1,
            conv1d_batched_f32_k2,
            conv1d_batched_f32_k3,
            conv1d_batched_f32_k5
        ](
            out,
            input,
            weight,
            channels_in,
            channels_out,
            input_length,
            output_length,
            stride,
            dilation,
            groups,
            left_padding,
            input_batch_stride,
            output_batch_stride,
            output_values_per_batch,
            output_len_i32,
        )
    )
}

#[cfg(feature = "dtype-f32")]
pub fn conv1d_batched_im2col_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    input: DevicePointer<f32>,
    batch: usize,
    channels_in: usize,
    input_length: usize,
    output_length: usize,
    kernel_size: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
    left_padding: usize,
    input_batch_stride: usize,
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(input)?;
    let params = Conv1dIm2colParams::create(
        batch,
        channels_in,
        input_length,
        output_length,
        kernel_size,
        stride,
        dilation,
        groups,
        left_padding,
        input_batch_stride,
    )?;
    let channels_in = params.channels_in;
    let input_length = params.input_length;
    let output_length = params.output_length;
    let stride = params.stride;
    let dilation = params.dilation;
    let groups = params.groups;
    let left_padding = params.left_padding;
    let input_batch_stride = params.input_batch_stride;
    let output_values_per_batch = params.output_values_per_batch;
    let output_len_i32 = params.launch.len_i32;
    let grid = params.launch.grid;

    dispatch_conv1d_kernel_size!(
        "conv1d_batched_im2col_f32",
        kernel_size,
        stream,
        grid,
        [
            conv1d_batched_im2col_f32_k1,
            conv1d_batched_im2col_f32_k2,
            conv1d_batched_im2col_f32_k3,
            conv1d_batched_im2col_f32_k5
        ](
            out,
            input,
            channels_in,
            input_length,
            output_length,
            stride,
            dilation,
            groups,
            left_padding,
            input_batch_stride,
            output_values_per_batch,
            output_len_i32,
        )
    )
}

#[cfg(feature = "dtype-f16")]
#[cfg(feature = "dtype-f32")]
pub fn conv1d_causal_f16_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    input: DevicePointer<f16>,
    weight: DevicePointer<f16>,
    channels_in: usize,
    channels_out: usize,
    input_length: usize,
    output_length: usize,
    kernel_size: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
    left_padding: usize,
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(input)?;
    checked_device_pointer(weight)?;
    let params = Conv1dWindowParams::create_full(
        channels_in,
        channels_out,
        input_length,
        output_length,
        kernel_size,
        stride,
        dilation,
        groups,
        left_padding,
    )?;
    let channels_in = params.channels_in;
    let channels_out = params.channels_out;
    let input_length = params.input_length;
    let output_length = params.output_length;
    let stride = params.stride;
    let dilation = params.dilation;
    let groups = params.groups;
    let left_padding = params.left_padding;
    let output_len_i32 = params.launch.len_i32;
    let grid = params.launch.grid;

    dispatch_conv1d_kernel_size!(
        "conv1d_causal_f16_f32",
        kernel_size,
        stream,
        grid,
        [
            conv1d_causal_f16_f32_k1,
            conv1d_causal_f16_f32_k2,
            conv1d_causal_f16_f32_k3,
            conv1d_causal_f16_f32_k5
        ](
            out,
            input,
            weight,
            channels_in,
            channels_out,
            0,
            input_length,
            0,
            output_length,
            stride,
            dilation,
            groups,
            left_padding,
            output_len_i32,
        )
    )
}

#[cfg(feature = "dtype-f16")]
#[cfg(feature = "dtype-f32")]
pub fn conv1d_batched_f16_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    input: DevicePointer<f16>,
    weight: DevicePointer<f16>,
    batch: usize,
    channels_in: usize,
    channels_out: usize,
    input_length: usize,
    output_length: usize,
    kernel_size: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
    left_padding: usize,
    input_batch_stride: usize,
    output_batch_stride: usize,
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(input)?;
    checked_device_pointer(weight)?;
    let params = Conv1dBatchedParams::create(
        batch,
        channels_in,
        channels_out,
        input_length,
        output_length,
        kernel_size,
        stride,
        dilation,
        groups,
        left_padding,
        input_batch_stride,
        output_batch_stride,
    )?;
    let channels_in = params.channels_in;
    let channels_out = params.channels_out;
    let input_length = params.input_length;
    let output_length = params.output_length;
    let stride = params.stride;
    let dilation = params.dilation;
    let groups = params.groups;
    let left_padding = params.left_padding;
    let input_batch_stride = params.input_batch_stride;
    let output_batch_stride = params.output_batch_stride;
    let output_values_per_batch = params.output_values_per_batch;
    let output_len_i32 = params.launch.len_i32;
    let grid = params.launch.grid;

    dispatch_conv1d_kernel_size!(
        "conv1d_batched_f16_f32",
        kernel_size,
        stream,
        grid,
        [
            conv1d_batched_f16_f32_k1,
            conv1d_batched_f16_f32_k2,
            conv1d_batched_f16_f32_k3,
            conv1d_batched_f16_f32_k5
        ](
            out,
            input,
            weight,
            channels_in,
            channels_out,
            input_length,
            output_length,
            stride,
            dilation,
            groups,
            left_padding,
            input_batch_stride,
            output_batch_stride,
            output_values_per_batch,
            output_len_i32,
        )
    )
}

#[cfg(feature = "dtype-bf16")]
#[cfg(feature = "dtype-f32")]
pub fn conv1d_causal_bf16_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    input: DevicePointer<bf16>,
    weight: DevicePointer<bf16>,
    channels_in: usize,
    channels_out: usize,
    input_length: usize,
    output_length: usize,
    kernel_size: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
    left_padding: usize,
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(input)?;
    checked_device_pointer(weight)?;
    let params = Conv1dWindowParams::create_full(
        channels_in,
        channels_out,
        input_length,
        output_length,
        kernel_size,
        stride,
        dilation,
        groups,
        left_padding,
    )?;
    let channels_in = params.channels_in;
    let channels_out = params.channels_out;
    let input_length = params.input_length;
    let output_length = params.output_length;
    let stride = params.stride;
    let dilation = params.dilation;
    let groups = params.groups;
    let left_padding = params.left_padding;
    let output_len_i32 = params.launch.len_i32;
    let grid = params.launch.grid;

    dispatch_conv1d_kernel_size!(
        "conv1d_causal_bf16_f32",
        kernel_size,
        stream,
        grid,
        [
            conv1d_causal_bf16_f32_k1,
            conv1d_causal_bf16_f32_k2,
            conv1d_causal_bf16_f32_k3,
            conv1d_causal_bf16_f32_k5
        ](
            out,
            input,
            weight,
            channels_in,
            channels_out,
            0,
            input_length,
            0,
            output_length,
            stride,
            dilation,
            groups,
            left_padding,
            output_len_i32,
        )
    )
}

#[cfg(feature = "dtype-bf16")]
#[cfg(feature = "dtype-f32")]
pub fn conv1d_batched_bf16_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    input: DevicePointer<bf16>,
    weight: DevicePointer<bf16>,
    batch: usize,
    channels_in: usize,
    channels_out: usize,
    input_length: usize,
    output_length: usize,
    kernel_size: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
    left_padding: usize,
    input_batch_stride: usize,
    output_batch_stride: usize,
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(input)?;
    checked_device_pointer(weight)?;
    let params = Conv1dBatchedParams::create(
        batch,
        channels_in,
        channels_out,
        input_length,
        output_length,
        kernel_size,
        stride,
        dilation,
        groups,
        left_padding,
        input_batch_stride,
        output_batch_stride,
    )?;
    let channels_in = params.channels_in;
    let channels_out = params.channels_out;
    let input_length = params.input_length;
    let output_length = params.output_length;
    let stride = params.stride;
    let dilation = params.dilation;
    let groups = params.groups;
    let left_padding = params.left_padding;
    let input_batch_stride = params.input_batch_stride;
    let output_batch_stride = params.output_batch_stride;
    let output_values_per_batch = params.output_values_per_batch;
    let output_len_i32 = params.launch.len_i32;
    let grid = params.launch.grid;

    dispatch_conv1d_kernel_size!(
        "conv1d_batched_bf16_f32",
        kernel_size,
        stream,
        grid,
        [
            conv1d_batched_bf16_f32_k1,
            conv1d_batched_bf16_f32_k2,
            conv1d_batched_bf16_f32_k3,
            conv1d_batched_bf16_f32_k5
        ](
            out,
            input,
            weight,
            channels_in,
            channels_out,
            input_length,
            output_length,
            stride,
            dilation,
            groups,
            left_padding,
            input_batch_stride,
            output_batch_stride,
            output_values_per_batch,
            output_len_i32,
        )
    )
}

#[cfg(feature = "dtype-f32")]
pub fn conv1d_causal_bias_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    input: DevicePointer<f32>,
    weight: DevicePointer<f32>,
    bias: DevicePointer<f32>,
    channels_in: usize,
    channels_out: usize,
    input_length: usize,
    output_length: usize,
    kernel_size: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
    left_padding: usize,
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(input)?;
    checked_device_pointer(weight)?;
    checked_device_pointer(bias)?;
    let params = Conv1dWindowParams::create_full(
        channels_in,
        channels_out,
        input_length,
        output_length,
        kernel_size,
        stride,
        dilation,
        groups,
        left_padding,
    )?;
    let channels_in = params.channels_in;
    let channels_out = params.channels_out;
    let input_length = params.input_length;
    let output_length = params.output_length;
    let stride = params.stride;
    let dilation = params.dilation;
    let groups = params.groups;
    let left_padding = params.left_padding;
    let output_len_i32 = params.launch.len_i32;
    let grid = params.launch.grid;

    dispatch_conv1d_kernel_size!(
        "conv1d_causal_bias_f32",
        kernel_size,
        stream,
        grid,
        [
            conv1d_causal_bias_f32_k1,
            conv1d_causal_bias_f32_k2,
            conv1d_causal_bias_f32_k3,
            conv1d_causal_bias_f32_k5
        ](
            out,
            input,
            weight,
            bias,
            channels_in,
            channels_out,
            0,
            input_length,
            0,
            output_length,
            stride,
            dilation,
            groups,
            left_padding,
            output_len_i32,
        )
    )
}

#[cfg(feature = "dtype-f32")]
pub fn conv1d_causal_bias_gelu_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    input: DevicePointer<f32>,
    weight: DevicePointer<f32>,
    bias: DevicePointer<f32>,
    channels_in: usize,
    channels_out: usize,
    input_length: usize,
    output_length: usize,
    kernel_size: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
    left_padding: usize,
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(input)?;
    checked_device_pointer(weight)?;
    checked_device_pointer(bias)?;
    let params = Conv1dWindowParams::create_full(
        channels_in,
        channels_out,
        input_length,
        output_length,
        kernel_size,
        stride,
        dilation,
        groups,
        left_padding,
    )?;
    let channels_in = params.channels_in;
    let channels_out = params.channels_out;
    let input_length = params.input_length;
    let output_length = params.output_length;
    let stride = params.stride;
    let dilation = params.dilation;
    let groups = params.groups;
    let left_padding = params.left_padding;
    let output_len_i32 = params.launch.len_i32;
    let grid = params.launch.grid;

    dispatch_conv1d_kernel_size!(
        "conv1d_causal_bias_gelu_f32",
        kernel_size,
        stream,
        grid,
        [
            conv1d_causal_bias_gelu_f32_k1,
            conv1d_causal_bias_gelu_f32_k2,
            conv1d_causal_bias_gelu_f32_k3,
            conv1d_causal_bias_gelu_f32_k5
        ](
            out,
            input,
            weight,
            bias,
            channels_in,
            channels_out,
            0,
            input_length,
            0,
            output_length,
            stride,
            dilation,
            groups,
            left_padding,
            output_len_i32,
        )
    )
}

#[cfg(feature = "dtype-f32")]
pub fn conv1d_causal_bias_gelu_f32_window(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    input: DevicePointer<f32>,
    weight: DevicePointer<f32>,
    bias: DevicePointer<f32>,
    channels_in: usize,
    channels_out: usize,
    input_start: usize,
    input_length: usize,
    output_start: usize,
    output_length: usize,
    kernel_size: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
    left_padding: usize,
) -> Result<()> {
    checked_device_pointer(out)?;
    checked_device_pointer(input)?;
    checked_device_pointer(weight)?;
    checked_device_pointer(bias)?;
    let params = Conv1dWindowParams::create(
        channels_in,
        channels_out,
        input_start,
        input_length,
        output_start,
        output_length,
        kernel_size,
        stride,
        dilation,
        groups,
        left_padding,
    )?;
    let channels_in = params.channels_in;
    let channels_out = params.channels_out;
    let input_start = params.input_start;
    let input_length = params.input_length;
    let output_start = params.output_start;
    let output_length = params.output_length;
    let stride = params.stride;
    let dilation = params.dilation;
    let groups = params.groups;
    let left_padding = params.left_padding;
    let output_len_i32 = params.launch.len_i32;
    let grid = params.launch.grid;

    dispatch_conv1d_kernel_size!(
        "conv1d_causal_bias_gelu_f32_window",
        kernel_size,
        stream,
        grid,
        [
            conv1d_causal_bias_gelu_f32_k1,
            conv1d_causal_bias_gelu_f32_k2,
            conv1d_causal_bias_gelu_f32_k3,
            conv1d_causal_bias_gelu_f32_k5
        ](
            out,
            input,
            weight,
            bias,
            channels_in,
            channels_out,
            input_start,
            input_length,
            output_start,
            output_length,
            stride,
            dilation,
            groups,
            left_padding,
            output_len_i32,
        )
    )
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use cutile::prelude::*;

    use super::*;
    use crate::{cpu::conv as cpu_conv, error::Result};

    fn erf_approx(value: f32) -> f32 {
        let sign = if value < 0.0 { -1.0 } else { 1.0 };
        let x = value.abs();
        let t = 1.0 / (1.0 + 0.327_591_1 * x);
        let poly = (((((1.061_405_4 * t - 1.453_152_1) * t) + 1.421_413_8) * t - 0.284_496_72) * t
            + 0.254_829_6)
            * t;
        sign * (1.0 - poly * (-x * x).exp())
    }

    fn gelu_approx(value: f32) -> f32 {
        0.5 * value * (1.0 + erf_approx(value * std::f32::consts::FRAC_1_SQRT_2))
    }

    fn conv2d_bias_gelu_reference_f32(
        input: &[f32],
        weight: &[f32],
        bias: &[f32],
        batch: usize,
        channels_in: usize,
        channels_out: usize,
        input_h: usize,
        input_w: usize,
        output_h: usize,
        output_w: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        padding_h: usize,
        padding_w: usize,
        dilation_h: usize,
        dilation_w: usize,
        groups: usize,
        input_batch_stride: usize,
        output_batch_stride: usize,
    ) -> Vec<f32> {
        let mut out =
            vec![0.0f32; (batch - 1) * output_batch_stride + channels_out * output_h * output_w];
        let channels_in_per_group = channels_in / groups;
        let channels_out_per_group = channels_out / groups;
        for batch_index in 0..batch {
            for out_channel in 0..channels_out {
                let group = out_channel / channels_out_per_group;
                for out_y in 0..output_h {
                    for out_x in 0..output_w {
                        let mut sum = bias[out_channel];
                        for group_input_channel in 0..channels_in_per_group {
                            let in_channel = group * channels_in_per_group + group_input_channel;
                            for kernel_y in 0..kernel_h {
                                let input_y = out_y as isize * stride_h as isize
                                    + kernel_y as isize * dilation_h as isize
                                    - padding_h as isize;
                                if input_y < 0 || input_y >= input_h as isize {
                                    continue;
                                }
                                for kernel_x in 0..kernel_w {
                                    let input_x = out_x as isize * stride_w as isize
                                        + kernel_x as isize * dilation_w as isize
                                        - padding_w as isize;
                                    if input_x < 0 || input_x >= input_w as isize {
                                        continue;
                                    }
                                    let input_index = batch_index * input_batch_stride
                                        + in_channel * input_h * input_w
                                        + input_y as usize * input_w
                                        + input_x as usize;
                                    let weight_index =
                                        out_channel * channels_in_per_group * kernel_h * kernel_w
                                            + group_input_channel * kernel_h * kernel_w
                                            + kernel_y * kernel_w
                                            + kernel_x;
                                    sum += input[input_index] * weight[weight_index];
                                }
                            }
                        }
                        let output_index = batch_index * output_batch_stride
                            + out_channel * output_h * output_w
                            + out_y * output_w
                            + out_x;
                        out[output_index] = gelu_approx(sum);
                    }
                }
            }
        }
        out
    }

    #[cfg(feature = "dtype-f32")]
    #[test]
    fn conv2d_bias_gelu_f32_matches_cpu() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (batch, channels_in, channels_out) = (2usize, 4usize, 6usize);
        let (input_h, input_w) = (5usize, 7usize);
        let (kernel_h, kernel_w) = (3usize, 3usize);
        let (stride_h, stride_w) = (2usize, 2usize);
        let (padding_h, padding_w) = (1usize, 1usize);
        let (dilation_h, dilation_w) = (1usize, 1usize);
        let groups = 2usize;
        let output_h = 3usize;
        let output_w = 4usize;
        let input_batch_stride = channels_in * input_h * input_w + 5;
        let output_batch_stride = channels_out * output_h * output_w + 7;
        let input_len = (batch - 1) * input_batch_stride + channels_in * input_h * input_w;
        let weight_len = channels_out * (channels_in / groups) * kernel_h * kernel_w;
        let bias_len = channels_out;
        let output_len = (batch - 1) * output_batch_stride + channels_out * output_h * output_w;
        let input_host = (0..input_len)
            .map(|index| (index as f32 % 17.0) * 0.03125 - 0.25)
            .collect::<Vec<_>>();
        let weight_host = (0..weight_len)
            .map(|index| (index as f32 % 13.0) * 0.046875 - 0.28125)
            .collect::<Vec<_>>();
        let bias_host = (0..bias_len)
            .map(|index| (index as f32 % 5.0) * 0.0625 - 0.125)
            .collect::<Vec<_>>();
        let expected = conv2d_bias_gelu_reference_f32(
            &input_host,
            &weight_host,
            &bias_host,
            batch,
            channels_in,
            channels_out,
            input_h,
            input_w,
            output_h,
            output_w,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
            groups,
            input_batch_stride,
            output_batch_stride,
        );

        let input = api::copy_host_vec_to_device(&Arc::new(input_host)).sync_on(&stream)?;
        let weight = api::copy_host_vec_to_device(&Arc::new(weight_host)).sync_on(&stream)?;
        let bias = api::copy_host_vec_to_device(&Arc::new(bias_host)).sync_on(&stream)?;
        let out = api::zeros::<f32>(&[output_len]).sync_on(&stream)?;

        conv2d_bias_gelu_f32(
            &stream,
            out.device_pointer(),
            input.device_pointer(),
            weight.device_pointer(),
            bias.device_pointer(),
            batch,
            channels_in,
            channels_out,
            input_h,
            input_w,
            output_h,
            output_w,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
            groups,
            input_batch_stride,
            output_batch_stride,
        )?;

        singe_core::assert_close!(&out.to_host_vec().sync_on(&stream)?, &expected, 2e-6);
        Ok(())
    }

    #[cfg(feature = "dtype-f16")]
    #[test]
    fn conv2d_bias_gelu_f16_matches_cpu() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (batch, channels_in, channels_out) = (2usize, 4usize, 6usize);
        let (input_h, input_w) = (5usize, 7usize);
        let (kernel_h, kernel_w) = (3usize, 3usize);
        let (stride_h, stride_w) = (2usize, 2usize);
        let (padding_h, padding_w) = (1usize, 1usize);
        let (dilation_h, dilation_w) = (1usize, 1usize);
        let groups = 2usize;
        let output_h = 3usize;
        let output_w = 4usize;
        let input_batch_stride = channels_in * input_h * input_w + 5;
        let output_batch_stride = channels_out * output_h * output_w + 7;
        let input_len = (batch - 1) * input_batch_stride + channels_in * input_h * input_w;
        let weight_len = channels_out * (channels_in / groups) * kernel_h * kernel_w;
        let bias_len = channels_out;
        let output_len = (batch - 1) * output_batch_stride + channels_out * output_h * output_w;
        let input_host = half_vec(
            &(0..input_len)
                .map(|index| (index as f32 % 17.0) * 0.03125 - 0.25)
                .collect::<Vec<_>>(),
        );
        let weight_host = half_vec(
            &(0..weight_len)
                .map(|index| (index as f32 % 13.0) * 0.046875 - 0.28125)
                .collect::<Vec<_>>(),
        );
        let bias_host = half_vec(
            &(0..bias_len)
                .map(|index| (index as f32 % 5.0) * 0.0625 - 0.125)
                .collect::<Vec<_>>(),
        );
        let expected = conv2d_bias_gelu_reference_f32(
            &half_to_f32(&input_host),
            &half_to_f32(&weight_host),
            &half_to_f32(&bias_host),
            batch,
            channels_in,
            channels_out,
            input_h,
            input_w,
            output_h,
            output_w,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
            groups,
            input_batch_stride,
            output_batch_stride,
        );
        let expected = round_half_vec(&expected);

        let input = api::copy_host_vec_to_device(&Arc::new(input_host)).sync_on(&stream)?;
        let weight = api::copy_host_vec_to_device(&Arc::new(weight_host)).sync_on(&stream)?;
        let bias = api::copy_host_vec_to_device(&Arc::new(bias_host)).sync_on(&stream)?;
        let out = api::zeros::<f16>(&[output_len]).sync_on(&stream)?;

        conv2d_bias_gelu_f16(
            &stream,
            out.device_pointer(),
            input.device_pointer(),
            weight.device_pointer(),
            bias.device_pointer(),
            batch,
            channels_in,
            channels_out,
            input_h,
            input_w,
            output_h,
            output_w,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
            groups,
            input_batch_stride,
            output_batch_stride,
        )?;

        singe_core::assert_close!(
            &half_to_f32(&out.to_host_vec().sync_on(&stream)?),
            &expected,
            1e-3,
        );
        Ok(())
    }

    #[cfg(feature = "dtype-bf16")]
    #[test]
    fn conv2d_bias_gelu_bf16_matches_cpu() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (batch, channels_in, channels_out) = (2usize, 4usize, 6usize);
        let (input_h, input_w) = (5usize, 7usize);
        let (kernel_h, kernel_w) = (3usize, 3usize);
        let (stride_h, stride_w) = (2usize, 2usize);
        let (padding_h, padding_w) = (1usize, 1usize);
        let (dilation_h, dilation_w) = (1usize, 1usize);
        let groups = 2usize;
        let output_h = 3usize;
        let output_w = 4usize;
        let input_batch_stride = channels_in * input_h * input_w + 5;
        let output_batch_stride = channels_out * output_h * output_w + 7;
        let input_len = (batch - 1) * input_batch_stride + channels_in * input_h * input_w;
        let weight_len = channels_out * (channels_in / groups) * kernel_h * kernel_w;
        let bias_len = channels_out;
        let output_len = (batch - 1) * output_batch_stride + channels_out * output_h * output_w;
        let input_host = bfloat_vec(
            &(0..input_len)
                .map(|index| (index as f32 % 17.0) * 0.03125 - 0.25)
                .collect::<Vec<_>>(),
        );
        let weight_host = bfloat_vec(
            &(0..weight_len)
                .map(|index| (index as f32 % 13.0) * 0.046875 - 0.28125)
                .collect::<Vec<_>>(),
        );
        let bias_host = bfloat_vec(
            &(0..bias_len)
                .map(|index| (index as f32 % 5.0) * 0.0625 - 0.125)
                .collect::<Vec<_>>(),
        );
        let expected = conv2d_bias_gelu_reference_f32(
            &bfloat_to_f32(&input_host),
            &bfloat_to_f32(&weight_host),
            &bfloat_to_f32(&bias_host),
            batch,
            channels_in,
            channels_out,
            input_h,
            input_w,
            output_h,
            output_w,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
            groups,
            input_batch_stride,
            output_batch_stride,
        );
        let expected = round_bfloat_vec(&expected);

        let input = api::copy_host_vec_to_device(&Arc::new(input_host)).sync_on(&stream)?;
        let weight = api::copy_host_vec_to_device(&Arc::new(weight_host)).sync_on(&stream)?;
        let bias = api::copy_host_vec_to_device(&Arc::new(bias_host)).sync_on(&stream)?;
        let out = api::zeros::<bf16>(&[output_len]).sync_on(&stream)?;

        conv2d_bias_gelu_bf16(
            &stream,
            out.device_pointer(),
            input.device_pointer(),
            weight.device_pointer(),
            bias.device_pointer(),
            batch,
            channels_in,
            channels_out,
            input_h,
            input_w,
            output_h,
            output_w,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
            groups,
            input_batch_stride,
            output_batch_stride,
        )?;

        singe_core::assert_close!(
            &bfloat_to_f32(&out.to_host_vec().sync_on(&stream)?),
            &expected,
            4e-3,
        );
        Ok(())
    }

    #[cfg(feature = "dtype-f32")]
    #[test]
    fn causal_conv1d_update_silu_f32_matches_cpu() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (batch, channels, kernel_size) = (2usize, 5usize, 5usize);
        let input_host = (0..batch * channels)
            .map(|index| (index as f32 % 7.0) * 0.25 - 0.75)
            .collect::<Vec<_>>();
        let conv_state_host = (0..batch * channels * kernel_size)
            .map(|index| (index as f32 % 11.0) * 0.125 - 0.5)
            .collect::<Vec<_>>();
        let weight_host = (0..channels * kernel_size)
            .map(|index| (index as f32 % 13.0) * 0.0625 - 0.375)
            .collect::<Vec<_>>();
        let bias_host = (0..channels)
            .map(|index| (index as f32 % 5.0) * 0.125 - 0.25)
            .collect::<Vec<_>>();
        let (expected_out, expected_state) = cpu_conv::causal_conv1d_update_silu(
            &conv_state_host,
            &input_host,
            &weight_host,
            &bias_host,
            batch,
            channels,
            kernel_size,
        );

        let input = api::copy_host_vec_to_device(&Arc::new(input_host)).sync_on(&stream)?;
        let conv_state =
            api::copy_host_vec_to_device(&Arc::new(conv_state_host)).sync_on(&stream)?;
        let weight = api::copy_host_vec_to_device(&Arc::new(weight_host)).sync_on(&stream)?;
        let bias = api::copy_host_vec_to_device(&Arc::new(bias_host)).sync_on(&stream)?;
        let out = api::zeros::<f32>(&[batch * channels]).sync_on(&stream)?;

        causal_conv1d_update_silu_f32(
            &stream,
            out.device_pointer(),
            conv_state.device_pointer(),
            input.device_pointer(),
            weight.device_pointer(),
            bias.device_pointer(),
            batch,
            channels,
            kernel_size,
        )?;

        singe_core::assert_close!(&out.to_host_vec().sync_on(&stream)?, &expected_out, 1e-6);
        singe_core::assert_close!(
            &conv_state.to_host_vec().sync_on(&stream)?,
            &expected_state,
            1e-6,
        );
        Ok(())
    }

    #[cfg(feature = "dtype-f16")]
    #[test]
    fn causal_conv1d_update_silu_f16_matches_cpu() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (batch, channels, kernel_size) = (2usize, 5usize, 5usize);
        let input_host = half_vec(
            &(0..batch * channels)
                .map(|index| (index as f32 % 7.0) * 0.25 - 0.75)
                .collect::<Vec<_>>(),
        );
        let conv_state_host = half_vec(
            &(0..batch * channels * kernel_size)
                .map(|index| (index as f32 % 11.0) * 0.125 - 0.5)
                .collect::<Vec<_>>(),
        );
        let weight_host = half_vec(
            &(0..channels * kernel_size)
                .map(|index| (index as f32 % 13.0) * 0.0625 - 0.375)
                .collect::<Vec<_>>(),
        );
        let bias_host = half_vec(
            &(0..channels)
                .map(|index| (index as f32 % 5.0) * 0.125 - 0.25)
                .collect::<Vec<_>>(),
        );
        let (expected_out, expected_state) = cpu_conv::causal_conv1d_update_silu(
            &half_to_f32(&conv_state_host),
            &half_to_f32(&input_host),
            &half_to_f32(&weight_host),
            &half_to_f32(&bias_host),
            batch,
            channels,
            kernel_size,
        );
        let expected_out = round_half_vec(&expected_out);
        let expected_state = round_half_vec(&expected_state);

        let input = api::copy_host_vec_to_device(&Arc::new(input_host)).sync_on(&stream)?;
        let conv_state =
            api::copy_host_vec_to_device(&Arc::new(conv_state_host)).sync_on(&stream)?;
        let weight = api::copy_host_vec_to_device(&Arc::new(weight_host)).sync_on(&stream)?;
        let bias = api::copy_host_vec_to_device(&Arc::new(bias_host)).sync_on(&stream)?;
        let out = api::zeros::<f16>(&[batch * channels]).sync_on(&stream)?;

        causal_conv1d_update_silu_f16(
            &stream,
            out.device_pointer(),
            conv_state.device_pointer(),
            input.device_pointer(),
            weight.device_pointer(),
            bias.device_pointer(),
            batch,
            channels,
            kernel_size,
        )?;

        singe_core::assert_close!(
            &half_to_f32(&out.to_host_vec().sync_on(&stream)?),
            &expected_out,
            1e-3,
        );
        singe_core::assert_close!(
            &half_to_f32(&conv_state.to_host_vec().sync_on(&stream)?),
            &expected_state,
            1e-3,
        );
        Ok(())
    }

    #[cfg(feature = "dtype-bf16")]
    #[test]
    fn causal_conv1d_update_silu_bf16_matches_cpu() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (batch, channels, kernel_size) = (2usize, 5usize, 5usize);
        let input_host = bfloat_vec(
            &(0..batch * channels)
                .map(|index| (index as f32 % 7.0) * 0.25 - 0.75)
                .collect::<Vec<_>>(),
        );
        let conv_state_host = bfloat_vec(
            &(0..batch * channels * kernel_size)
                .map(|index| (index as f32 % 11.0) * 0.125 - 0.5)
                .collect::<Vec<_>>(),
        );
        let weight_host = bfloat_vec(
            &(0..channels * kernel_size)
                .map(|index| (index as f32 % 13.0) * 0.0625 - 0.375)
                .collect::<Vec<_>>(),
        );
        let bias_host = bfloat_vec(
            &(0..channels)
                .map(|index| (index as f32 % 5.0) * 0.125 - 0.25)
                .collect::<Vec<_>>(),
        );
        let (expected_out, expected_state) = cpu_conv::causal_conv1d_update_silu(
            &bfloat_to_f32(&conv_state_host),
            &bfloat_to_f32(&input_host),
            &bfloat_to_f32(&weight_host),
            &bfloat_to_f32(&bias_host),
            batch,
            channels,
            kernel_size,
        );
        let expected_out = round_bfloat_vec(&expected_out);
        let expected_state = round_bfloat_vec(&expected_state);

        let input = api::copy_host_vec_to_device(&Arc::new(input_host)).sync_on(&stream)?;
        let conv_state =
            api::copy_host_vec_to_device(&Arc::new(conv_state_host)).sync_on(&stream)?;
        let weight = api::copy_host_vec_to_device(&Arc::new(weight_host)).sync_on(&stream)?;
        let bias = api::copy_host_vec_to_device(&Arc::new(bias_host)).sync_on(&stream)?;
        let out = api::zeros::<bf16>(&[batch * channels]).sync_on(&stream)?;

        causal_conv1d_update_silu_bf16(
            &stream,
            out.device_pointer(),
            conv_state.device_pointer(),
            input.device_pointer(),
            weight.device_pointer(),
            bias.device_pointer(),
            batch,
            channels,
            kernel_size,
        )?;

        singe_core::assert_close!(
            &bfloat_to_f32(&out.to_host_vec().sync_on(&stream)?),
            &expected_out,
            4e-3,
        );
        singe_core::assert_close!(
            &bfloat_to_f32(&conv_state.to_host_vec().sync_on(&stream)?),
            &expected_state,
            4e-3,
        );
        Ok(())
    }

    #[cfg(feature = "dtype-f32")]
    #[test]
    fn causal_conv1d_prefill_silu_f32_matches_cpu() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (batch, channels, kernel_size) = (2usize, 3usize, 5usize);
        let input_length = 4usize;
        let output_length = 4usize;
        let input_host = (0..batch * channels * input_length)
            .map(|index| (index as f32 % 11.0) * 0.125 - 0.5)
            .collect::<Vec<_>>();
        let weight_host = (0..channels * kernel_size)
            .map(|index| (index as f32 % 7.0) * 0.25 - 0.75)
            .collect::<Vec<_>>();
        let bias_host = (0..channels)
            .map(|index| (index as f32 % 5.0) * 0.125 - 0.25)
            .collect::<Vec<_>>();
        let (expected_out, expected_state) = cpu_conv::causal_conv1d_prefill_silu(
            &input_host,
            &weight_host,
            &bias_host,
            batch,
            channels,
            kernel_size,
            input_length,
            output_length,
        );

        let input = api::copy_host_vec_to_device(&Arc::new(input_host)).sync_on(&stream)?;
        let weight = api::copy_host_vec_to_device(&Arc::new(weight_host)).sync_on(&stream)?;
        let bias = api::copy_host_vec_to_device(&Arc::new(bias_host)).sync_on(&stream)?;
        let out = api::zeros::<f32>(&[batch * channels * output_length]).sync_on(&stream)?;
        let conv_state = api::zeros::<f32>(&[batch * channels * kernel_size]).sync_on(&stream)?;

        causal_conv1d_prefill_silu_f32(
            &stream,
            out.device_pointer(),
            conv_state.device_pointer(),
            input.device_pointer(),
            weight.device_pointer(),
            bias.device_pointer(),
            batch,
            channels,
            kernel_size,
            input_length,
            output_length,
        )?;

        singe_core::assert_close!(&out.to_host_vec().sync_on(&stream)?, &expected_out, 1e-6);
        singe_core::assert_close!(
            &conv_state.to_host_vec().sync_on(&stream)?,
            &expected_state,
            1e-6,
        );
        Ok(())
    }

    #[cfg(feature = "dtype-f16")]
    #[test]
    fn causal_conv1d_prefill_silu_f16_matches_cpu() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (batch, channels, kernel_size) = (2usize, 3usize, 5usize);
        let input_length = 4usize;
        let output_length = 4usize;
        let input_host = half_vec(
            &(0..batch * channels * input_length)
                .map(|index| (index as f32 % 11.0) * 0.125 - 0.5)
                .collect::<Vec<_>>(),
        );
        let weight_host = half_vec(
            &(0..channels * kernel_size)
                .map(|index| (index as f32 % 7.0) * 0.25 - 0.75)
                .collect::<Vec<_>>(),
        );
        let bias_host = half_vec(
            &(0..channels)
                .map(|index| (index as f32 % 5.0) * 0.125 - 0.25)
                .collect::<Vec<_>>(),
        );
        let (expected_out, expected_state) = cpu_conv::causal_conv1d_prefill_silu(
            &half_to_f32(&input_host),
            &half_to_f32(&weight_host),
            &half_to_f32(&bias_host),
            batch,
            channels,
            kernel_size,
            input_length,
            output_length,
        );
        let expected_out = round_half_vec(&expected_out);
        let expected_state = round_half_vec(&expected_state);

        let input = api::copy_host_vec_to_device(&Arc::new(input_host)).sync_on(&stream)?;
        let weight = api::copy_host_vec_to_device(&Arc::new(weight_host)).sync_on(&stream)?;
        let bias = api::copy_host_vec_to_device(&Arc::new(bias_host)).sync_on(&stream)?;
        let out = api::zeros::<f16>(&[batch * channels * output_length]).sync_on(&stream)?;
        let conv_state = api::zeros::<f16>(&[batch * channels * kernel_size]).sync_on(&stream)?;

        causal_conv1d_prefill_silu_f16(
            &stream,
            out.device_pointer(),
            conv_state.device_pointer(),
            input.device_pointer(),
            weight.device_pointer(),
            bias.device_pointer(),
            batch,
            channels,
            kernel_size,
            input_length,
            output_length,
        )?;

        singe_core::assert_close!(
            &half_to_f32(&out.to_host_vec().sync_on(&stream)?),
            &expected_out,
            1e-3,
        );
        singe_core::assert_close!(
            &half_to_f32(&conv_state.to_host_vec().sync_on(&stream)?),
            &expected_state,
            1e-3,
        );
        Ok(())
    }

    #[cfg(feature = "dtype-bf16")]
    #[test]
    fn causal_conv1d_prefill_silu_bf16_matches_cpu() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (batch, channels, kernel_size) = (2usize, 3usize, 5usize);
        let input_length = 4usize;
        let output_length = 4usize;
        let input_host = bfloat_vec(
            &(0..batch * channels * input_length)
                .map(|index| (index as f32 % 11.0) * 0.125 - 0.5)
                .collect::<Vec<_>>(),
        );
        let weight_host = bfloat_vec(
            &(0..channels * kernel_size)
                .map(|index| (index as f32 % 7.0) * 0.25 - 0.75)
                .collect::<Vec<_>>(),
        );
        let bias_host = bfloat_vec(
            &(0..channels)
                .map(|index| (index as f32 % 5.0) * 0.125 - 0.25)
                .collect::<Vec<_>>(),
        );
        let (expected_out, expected_state) = cpu_conv::causal_conv1d_prefill_silu(
            &bfloat_to_f32(&input_host),
            &bfloat_to_f32(&weight_host),
            &bfloat_to_f32(&bias_host),
            batch,
            channels,
            kernel_size,
            input_length,
            output_length,
        );
        let expected_out = round_bfloat_vec(&expected_out);
        let expected_state = round_bfloat_vec(&expected_state);

        let input = api::copy_host_vec_to_device(&Arc::new(input_host)).sync_on(&stream)?;
        let weight = api::copy_host_vec_to_device(&Arc::new(weight_host)).sync_on(&stream)?;
        let bias = api::copy_host_vec_to_device(&Arc::new(bias_host)).sync_on(&stream)?;
        let out = api::zeros::<bf16>(&[batch * channels * output_length]).sync_on(&stream)?;
        let conv_state = api::zeros::<bf16>(&[batch * channels * kernel_size]).sync_on(&stream)?;

        causal_conv1d_prefill_silu_bf16(
            &stream,
            out.device_pointer(),
            conv_state.device_pointer(),
            input.device_pointer(),
            weight.device_pointer(),
            bias.device_pointer(),
            batch,
            channels,
            kernel_size,
            input_length,
            output_length,
        )?;

        singe_core::assert_close!(
            &bfloat_to_f32(&out.to_host_vec().sync_on(&stream)?),
            &expected_out,
            4e-3,
        );
        singe_core::assert_close!(
            &bfloat_to_f32(&conv_state.to_host_vec().sync_on(&stream)?),
            &expected_state,
            4e-3,
        );
        Ok(())
    }

    #[cfg(feature = "dtype-f16")]
    fn half_vec(values: &[f32]) -> Vec<f16> {
        values.iter().copied().map(f16::from_f32).collect()
    }

    #[cfg(feature = "dtype-f16")]
    fn half_to_f32(values: &[f16]) -> Vec<f32> {
        values.iter().map(|value| value.to_f32()).collect()
    }

    #[cfg(feature = "dtype-f16")]
    fn round_half_vec(values: &[f32]) -> Vec<f32> {
        values
            .iter()
            .copied()
            .map(f16::from_f32)
            .map(|value| value.to_f32())
            .collect()
    }

    #[cfg(feature = "dtype-bf16")]
    fn bfloat_vec(values: &[f32]) -> Vec<bf16> {
        values.iter().copied().map(bf16::from_f32).collect()
    }

    #[cfg(feature = "dtype-bf16")]
    fn bfloat_to_f32(values: &[bf16]) -> Vec<f32> {
        values.iter().map(|value| value.to_f32()).collect()
    }

    #[cfg(feature = "dtype-bf16")]
    fn round_bfloat_vec(values: &[f32]) -> Vec<f32> {
        values
            .iter()
            .copied()
            .map(bf16::from_f32)
            .map(|value| value.to_f32())
            .collect()
    }
}
