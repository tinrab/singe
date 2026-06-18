use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        statistics,
        view::{C1, C3, ImageView, MaskView},
    },
};

use super::super::super::ImagePipeline;
use super::super::{MaskedChannelScalarStatisticImage, MaskedScalarStatisticImage};

impl_masked_scalar_statistic_image!(
    u8,
    statistics::mean_u8_c1_masked,
    statistics::norm_inf_u8_c1_masked,
    statistics::norm_l1_u8_c1_masked,
    statistics::norm_l2_u8_c1_masked
);
impl_masked_scalar_statistic_image!(
    i8,
    statistics::mean_i8_c1_masked,
    statistics::norm_inf_i8_c1_masked,
    statistics::norm_l1_i8_c1_masked,
    statistics::norm_l2_i8_c1_masked
);
impl_masked_scalar_statistic_image!(
    u16,
    statistics::mean_u16_c1_masked,
    statistics::norm_inf_u16_c1_masked,
    statistics::norm_l1_u16_c1_masked,
    statistics::norm_l2_u16_c1_masked
);
impl_masked_scalar_statistic_image!(
    f32,
    statistics::mean_f32_c1_masked,
    statistics::norm_inf_f32_c1_masked,
    statistics::norm_l1_f32_c1_masked,
    statistics::norm_l2_f32_c1_masked
);

impl_masked_channel_scalar_statistic_image!(
    u8,
    statistics::mean_u8_c3_masked,
    statistics::norm_inf_u8_c3_masked,
    statistics::norm_l1_u8_c3_masked,
    statistics::norm_l2_u8_c3_masked
);
impl_masked_channel_scalar_statistic_image!(
    i8,
    statistics::mean_i8_c3_masked,
    statistics::norm_inf_i8_c3_masked,
    statistics::norm_l1_i8_c3_masked,
    statistics::norm_l2_i8_c3_masked
);
impl_masked_channel_scalar_statistic_image!(
    u16,
    statistics::mean_u16_c3_masked,
    statistics::norm_inf_u16_c3_masked,
    statistics::norm_l1_u16_c3_masked,
    statistics::norm_l2_u16_c3_masked
);
impl_masked_channel_scalar_statistic_image!(
    f32,
    statistics::mean_f32_c3_masked,
    statistics::norm_inf_f32_c3_masked,
    statistics::norm_l1_f32_c3_masked,
    statistics::norm_l2_f32_c3_masked
);
