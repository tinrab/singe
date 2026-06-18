use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        statistics,
        view::{AC4, C1, C3, C4, ImageView, ImageViewMut},
    },
};

use super::super::ImagePipeline;
use super::{CountInRangeImage, EveryStatisticImage};

impl_count_in_range_image!(u8, C1, u8, 1, statistics::count_in_range_u8_c1);
impl_count_in_range_image!(f32, C1, f32, 1, statistics::count_in_range_f32_c1);
impl_count_in_range_image!(u8, C3, [u8; 3], 3, statistics::count_in_range_u8_c3);
impl_count_in_range_image!(f32, C3, [f32; 3], 3, statistics::count_in_range_f32_c3);
impl_count_in_range_image!(u8, AC4, [u8; 3], 3, statistics::count_in_range_u8_ac4);
impl_count_in_range_image!(f32, AC4, [f32; 3], 3, statistics::count_in_range_f32_ac4);

impl_every_statistic_image!(
    u8,
    C1,
    statistics::max_every_u8_c1,
    statistics::min_every_u8_c1
);
impl_every_statistic_image!(
    u16,
    C1,
    statistics::max_every_u16_c1,
    statistics::min_every_u16_c1
);
impl_every_statistic_image!(
    i16,
    C1,
    statistics::max_every_i16_c1,
    statistics::min_every_i16_c1
);
impl_every_statistic_image!(
    f32,
    C1,
    statistics::max_every_f32_c1,
    statistics::min_every_f32_c1
);
impl_every_statistic_image!(
    u8,
    C3,
    statistics::max_every_u8_c3,
    statistics::min_every_u8_c3
);
impl_every_statistic_image!(
    u16,
    C3,
    statistics::max_every_u16_c3,
    statistics::min_every_u16_c3
);
impl_every_statistic_image!(
    i16,
    C3,
    statistics::max_every_i16_c3,
    statistics::min_every_i16_c3
);
impl_every_statistic_image!(
    f32,
    C3,
    statistics::max_every_f32_c3,
    statistics::min_every_f32_c3
);
impl_every_statistic_image!(
    u8,
    C4,
    statistics::max_every_u8_c4,
    statistics::min_every_u8_c4
);
impl_every_statistic_image!(
    u16,
    C4,
    statistics::max_every_u16_c4,
    statistics::min_every_u16_c4
);
impl_every_statistic_image!(
    i16,
    C4,
    statistics::max_every_i16_c4,
    statistics::min_every_i16_c4
);
impl_every_statistic_image!(
    f32,
    C4,
    statistics::max_every_f32_c4,
    statistics::min_every_f32_c4
);
impl_every_statistic_image!(
    u8,
    AC4,
    statistics::max_every_u8_ac4,
    statistics::min_every_u8_ac4
);
impl_every_statistic_image!(
    u16,
    AC4,
    statistics::max_every_u16_ac4,
    statistics::min_every_u16_ac4
);
impl_every_statistic_image!(
    i16,
    AC4,
    statistics::max_every_i16_ac4,
    statistics::min_every_i16_ac4
);
impl_every_statistic_image!(
    f32,
    AC4,
    statistics::max_every_f32_ac4,
    statistics::min_every_f32_ac4
);
