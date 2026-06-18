use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        statistics,
        view::{C1, C3, ChannelLayout, ImageView, MaskView},
    },
};

use super::super::ImagePipeline;
use super::*;

impl_mean_standard_deviation_image!(u8, C1, statistics::mean_stddev_u8_c1);
impl_masked_mean_standard_deviation_image!(u8, statistics::mean_stddev_u8_c1_masked);
impl_masked_mean_standard_deviation_image!(i8, statistics::mean_stddev_i8_c1_masked);
impl_mean_standard_deviation_image!(u16, C1, statistics::mean_stddev_u16_c1);
impl_masked_mean_standard_deviation_image!(u16, statistics::mean_stddev_u16_c1_masked);
impl_mean_standard_deviation_image!(f32, C1, statistics::mean_stddev_f32_c1);
impl_masked_mean_standard_deviation_image!(f32, statistics::mean_stddev_f32_c1_masked);

impl_channel_mean_standard_deviation_image!(u8, C3, statistics::mean_stddev_u8_c3);
impl_masked_channel_mean_standard_deviation_image!(u8, C3, statistics::mean_stddev_u8_c3_masked);
impl_masked_channel_mean_standard_deviation_image!(i8, C3, statistics::mean_stddev_i8_c3_masked);
impl_channel_mean_standard_deviation_image!(u16, C3, statistics::mean_stddev_u16_c3);
impl_masked_channel_mean_standard_deviation_image!(u16, C3, statistics::mean_stddev_u16_c3_masked);
impl_channel_mean_standard_deviation_image!(f32, C3, statistics::mean_stddev_f32_c3);
impl_masked_channel_mean_standard_deviation_image!(f32, C3, statistics::mean_stddev_f32_c3_masked);
