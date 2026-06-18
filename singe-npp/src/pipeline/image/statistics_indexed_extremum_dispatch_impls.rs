use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        statistics,
        view::{C1, C3, C4, ChannelLayout, ImageView, MaskView},
    },
    types::Point,
};

use super::super::ImagePipeline;
use super::{
    ChannelPairIndexedExtremumStatisticImage, IndexedExtremumStatisticImage,
    MaskedChannelPairIndexedExtremumStatisticImage, MaskedPairIndexedExtremumStatisticImage,
    PairIndexedExtremumStatisticImage,
};

impl_indexed_extremum_statistic_image!(
    u8,
    C1,
    statistics::min_indx_u8_c1,
    statistics::max_indx_u8_c1
);

impl_indexed_extremum_statistic_image!(
    u16,
    C1,
    statistics::min_indx_u16_c1,
    statistics::max_indx_u16_c1
);

impl_indexed_extremum_statistic_image!(
    i16,
    C1,
    statistics::min_indx_i16_c1,
    statistics::max_indx_i16_c1
);

impl_indexed_extremum_statistic_image!(
    f32,
    C1,
    statistics::min_indx_f32_c1,
    statistics::max_indx_f32_c1
);

impl_indexed_extremum_statistic_image!(
    u8,
    C3,
    statistics::min_indx_u8_c3,
    statistics::max_indx_u8_c3
);

impl_indexed_extremum_statistic_image!(
    u16,
    C3,
    statistics::min_indx_u16_c3,
    statistics::max_indx_u16_c3
);

impl_indexed_extremum_statistic_image!(
    i16,
    C3,
    statistics::min_indx_i16_c3,
    statistics::max_indx_i16_c3
);

impl_indexed_extremum_statistic_image!(
    f32,
    C3,
    statistics::min_indx_f32_c3,
    statistics::max_indx_f32_c3
);

impl_indexed_extremum_statistic_image!(
    u8,
    C4,
    statistics::min_indx_u8_c4,
    statistics::max_indx_u8_c4
);

impl_indexed_extremum_statistic_image!(
    u16,
    C4,
    statistics::min_indx_u16_c4,
    statistics::max_indx_u16_c4
);

impl_indexed_extremum_statistic_image!(
    i16,
    C4,
    statistics::min_indx_i16_c4,
    statistics::max_indx_i16_c4
);

impl_indexed_extremum_statistic_image!(
    f32,
    C4,
    statistics::min_indx_f32_c4,
    statistics::max_indx_f32_c4
);

impl_pair_indexed_extremum_statistic_image!(u8, C1, statistics::min_max_indx_u8_c1);
impl_masked_pair_indexed_extremum_statistic_image!(u8, statistics::min_max_indx_u8_c1_masked);

impl_masked_pair_indexed_extremum_statistic_image!(i8, statistics::min_max_indx_i8_c1_masked);

impl_pair_indexed_extremum_statistic_image!(u16, C1, statistics::min_max_indx_u16_c1);
impl_masked_pair_indexed_extremum_statistic_image!(u16, statistics::min_max_indx_u16_c1_masked);

impl_pair_indexed_extremum_statistic_image!(f32, C1, statistics::min_max_indx_f32_c1);
impl_masked_pair_indexed_extremum_statistic_image!(f32, statistics::min_max_indx_f32_c1_masked);

impl_channel_pair_indexed_extremum_statistic_image!(u8, C3, statistics::min_max_indx_u8_c3);
impl_masked_channel_pair_indexed_extremum_statistic_image!(
    u8,
    statistics::min_max_indx_u8_c3_masked
);

impl_masked_channel_pair_indexed_extremum_statistic_image!(
    i8,
    statistics::min_max_indx_i8_c3_masked
);

impl_channel_pair_indexed_extremum_statistic_image!(u16, C3, statistics::min_max_indx_u16_c3);
impl_masked_channel_pair_indexed_extremum_statistic_image!(
    u16,
    statistics::min_max_indx_u16_c3_masked
);

impl_channel_pair_indexed_extremum_statistic_image!(f32, C3, statistics::min_max_indx_f32_c3);
impl_masked_channel_pair_indexed_extremum_statistic_image!(
    f32,
    statistics::min_max_indx_f32_c3_masked
);
