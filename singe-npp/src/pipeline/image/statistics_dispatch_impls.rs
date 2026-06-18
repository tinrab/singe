use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        statistics,
        view::{C1, C3, C4, ChannelLayout, ImageView},
    },
};

use super::super::ImagePipeline;
use super::*;

#[path = "statistics_masked_dispatch_impls.rs"]
mod masked_dispatch_impls;

impl_scalar_statistic_image!(
    u8,
    C1,
    statistics::sum_u8_c1,
    statistics::mean_u8_c1,
    statistics::norm_inf_u8_c1,
    statistics::norm_l1_u8_c1,
    statistics::norm_l2_u8_c1
);

impl_scalar_statistic_image!(
    u16,
    C1,
    statistics::sum_u16_c1,
    statistics::mean_u16_c1,
    statistics::norm_inf_u16_c1,
    statistics::norm_l1_u16_c1,
    statistics::norm_l2_u16_c1
);

impl_scalar_statistic_image!(
    i16,
    C1,
    statistics::sum_i16_c1,
    statistics::mean_i16_c1,
    statistics::norm_inf_i16_c1,
    statistics::norm_l1_i16_c1,
    statistics::norm_l2_i16_c1
);

impl_scalar_statistic_image!(
    f32,
    C1,
    statistics::sum_f32_c1,
    statistics::mean_f32_c1,
    statistics::norm_inf_f32_c1,
    statistics::norm_l1_f32_c1,
    statistics::norm_l2_f32_c1
);

impl_scalar_statistic_image!(
    u8,
    C3,
    statistics::sum_u8_c3,
    statistics::mean_u8_c3,
    statistics::norm_inf_u8_c3,
    statistics::norm_l1_u8_c3,
    statistics::norm_l2_u8_c3
);

impl_scalar_statistic_image!(
    u16,
    C3,
    statistics::sum_u16_c3,
    statistics::mean_u16_c3,
    statistics::norm_inf_u16_c3,
    statistics::norm_l1_u16_c3,
    statistics::norm_l2_u16_c3
);

impl_scalar_statistic_image!(
    i16,
    C3,
    statistics::sum_i16_c3,
    statistics::mean_i16_c3,
    statistics::norm_inf_i16_c3,
    statistics::norm_l1_i16_c3,
    statistics::norm_l2_i16_c3
);

impl_scalar_statistic_image!(
    f32,
    C3,
    statistics::sum_f32_c3,
    statistics::mean_f32_c3,
    statistics::norm_inf_f32_c3,
    statistics::norm_l1_f32_c3,
    statistics::norm_l2_f32_c3
);

impl_scalar_statistic_image!(
    u8,
    C4,
    statistics::sum_u8_c4,
    statistics::mean_u8_c4,
    statistics::norm_inf_u8_c4,
    statistics::norm_l1_u8_c4,
    statistics::norm_l2_u8_c4
);

impl_scalar_statistic_image!(
    u16,
    C4,
    statistics::sum_u16_c4,
    statistics::mean_u16_c4,
    statistics::norm_inf_u16_c4,
    statistics::norm_l1_u16_c4,
    statistics::norm_l2_u16_c4
);

impl_scalar_statistic_image!(
    i16,
    C4,
    statistics::sum_i16_c4,
    statistics::mean_i16_c4,
    statistics::norm_inf_i16_c4,
    statistics::norm_l1_i16_c4,
    statistics::norm_l2_i16_c4
);

impl_scalar_statistic_image!(
    f32,
    C4,
    statistics::sum_f32_c4,
    statistics::mean_f32_c4,
    statistics::norm_inf_f32_c4,
    statistics::norm_l1_f32_c4,
    statistics::norm_l2_f32_c4
);
