//! Pooling operations over device tensors.

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

#[derive(Debug, Clone, Copy)]
pub struct Pool2dConfig {
    pub batch: usize,
    pub channels: usize,
    pub input_height: usize,
    pub input_width: usize,
    pub output_height: usize,
    pub output_width: usize,
    pub kernel_height: usize,
    pub kernel_width: usize,
    pub stride_height: usize,
    pub stride_width: usize,
    pub pad_height_start: usize,
    pub pad_height_end: usize,
    pub pad_width_start: usize,
    pub pad_width_end: usize,
    pub dilation_height: usize,
    pub dilation_width: usize,
}

macro_rules! pool2d_fns {
    ($max_name:ident, $avg_name:ident, $ty:ty) => {
        pub fn $max_name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            input: &impl DeviceSlice<$ty>,
            config: Pool2dConfig,
        ) -> Result<()> {
            validate_pool2d(out.len(), input.len(), config)?;
            let stream = borrowed_stream(stream)?;
            cutile::pooling::$max_name(&stream, output_pointer(out), input_pointer(input), config)
        }

        pub fn $avg_name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            input: &impl DeviceSlice<$ty>,
            config: Pool2dConfig,
            count_include_pad: bool,
        ) -> Result<()> {
            validate_pool2d(out.len(), input.len(), config)?;
            let stream = borrowed_stream(stream)?;
            cutile::pooling::$avg_name(
                &stream,
                output_pointer(out),
                input_pointer(input),
                config,
                count_include_pad,
            )
        }
    };
}

#[cfg(feature = "dtype-f32")]
pool2d_fns!(max_pool2d_f32, avg_pool2d_f32, f32);
#[cfg(feature = "dtype-f16")]
pool2d_fns!(max_pool2d_f16, avg_pool2d_f16, f16);
#[cfg(feature = "dtype-f64")]
pool2d_fns!(max_pool2d_f64, avg_pool2d_f64, f64);

fn validate_pool2d(out_len: usize, input_len: usize, config: Pool2dConfig) -> Result<()> {
    if config.batch == 0
        || config.channels == 0
        || config.input_height == 0
        || config.input_width == 0
        || config.output_height == 0
        || config.output_width == 0
        || config.kernel_height == 0
        || config.kernel_width == 0
        || config.stride_height == 0
        || config.stride_width == 0
        || config.dilation_height == 0
        || config.dilation_width == 0
    {
        return Err(Error::InvalidLength);
    }
    let input_hw = checked_element_count(config.input_height, config.input_width)?;
    let input_chw = checked_element_count(config.channels, input_hw)?;
    let expected_input = checked_element_count(config.batch, input_chw)?;
    let output_hw = checked_element_count(config.output_height, config.output_width)?;
    let output_chw = checked_element_count(config.channels, output_hw)?;
    let expected_output = checked_element_count(config.batch, output_chw)?;
    ensure_len(input_len, expected_input)?;
    ensure_len(out_len, expected_output)
}
