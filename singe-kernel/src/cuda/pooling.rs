//! Pooling operations over device tensors.

use ::cutile::cuda_async::device_buffer::DevicePointer;
#[cfg(feature = "dtype_f16")]
use singe_cuda::types::f16;
use singe_cuda::{
    stream::Stream,
    view::{DeviceInput, DeviceOutput},
};

#[cfg(feature = "cutile")]
use crate::cuda::cutile;
use crate::{
    cuda::interop::input_pointer,
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
        /// Writes the maximum of each NCHW pooling window.
        ///
        /// # Safety
        ///
        /// Input values must be valid when read; order their producers before use.
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        pub unsafe fn $max_name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            config: Pool2dConfig,
        ) -> Result<()> {
            validate_pool2d(out.len(), input.len(), config)?;
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::pooling::$max_name(
                    &stream,
                    DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                    input_pointer(input),
                    config,
                )
            }
        }

        /// Writes the mean of each NCHW pooling window using the configured padding policy.
        ///
        /// # Safety
        ///
        /// Input values must be valid when read; order their producers before use.
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        pub unsafe fn $avg_name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            config: Pool2dConfig,
            count_include_pad: bool,
        ) -> Result<()> {
            validate_pool2d(out.len(), input.len(), config)?;
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::pooling::$avg_name(
                    &stream,
                    DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                    input_pointer(input),
                    config,
                    count_include_pad,
                )
            }
        }
    };
}

#[cfg(feature = "dtype_f32")]
pool2d_fns!(max_pool2d_f32, avg_pool2d_f32, f32);

#[cfg(feature = "dtype_f16")]
pool2d_fns!(max_pool2d_f16, avg_pool2d_f16, f16);

#[cfg(feature = "dtype_f64")]
pool2d_fns!(max_pool2d_f64, avg_pool2d_f64, f64);

#[expect(
    clippy::similar_names,
    reason = "query, key, and value operands and their tensor layouts follow the mathematical kernel notation"
)]
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
