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
use super::ExtremumStatisticImage;

impl_extremum_statistic_image!(
    u8,
    C1,
    statistics::min_u8_c1,
    statistics::max_u8_c1,
    statistics::min_max_u8_c1
);

impl_extremum_statistic_image!(
    u16,
    C1,
    statistics::min_u16_c1,
    statistics::max_u16_c1,
    statistics::min_max_u16_c1
);

impl_extremum_statistic_image!(
    i16,
    C1,
    statistics::min_i16_c1,
    statistics::max_i16_c1,
    statistics::min_max_i16_c1
);

impl_extremum_statistic_image!(
    f32,
    C1,
    statistics::min_f32_c1,
    statistics::max_f32_c1,
    statistics::min_max_f32_c1
);

impl_extremum_statistic_image!(
    u8,
    C3,
    statistics::min_u8_c3,
    statistics::max_u8_c3,
    statistics::min_max_u8_c3
);

impl_extremum_statistic_image!(
    u16,
    C3,
    statistics::min_u16_c3,
    statistics::max_u16_c3,
    statistics::min_max_u16_c3
);

impl_extremum_statistic_image!(
    i16,
    C3,
    statistics::min_i16_c3,
    statistics::max_i16_c3,
    statistics::min_max_i16_c3
);

impl_extremum_statistic_image!(
    f32,
    C3,
    statistics::min_f32_c3,
    statistics::max_f32_c3,
    statistics::min_max_f32_c3
);

impl_extremum_statistic_image!(
    u8,
    C4,
    statistics::min_u8_c4,
    statistics::max_u8_c4,
    statistics::min_max_u8_c4
);

impl_extremum_statistic_image!(
    u16,
    C4,
    statistics::min_u16_c4,
    statistics::max_u16_c4,
    statistics::min_max_u16_c4
);

impl_extremum_statistic_image!(
    i16,
    C4,
    statistics::min_i16_c4,
    statistics::max_i16_c4,
    statistics::min_max_i16_c4
);

impl_extremum_statistic_image!(
    f32,
    C4,
    statistics::min_f32_c4,
    statistics::max_f32_c4,
    statistics::min_max_f32_c4
);
