use crate::{
    context::StreamContext,
    error::Result,
    image::{
        threshold as npp_threshold,
        view::{AC4, C3, ImageView, ImageViewMut},
    },
    types::ComparisonOperation,
};

use super::super::{ImagePipeline, threshold_dispatch::PackedThresholdImage};

impl_packed_threshold_image!(
    u8,
    C3,
    3,
    npp_threshold::threshold_u8_c3,
    npp_threshold::threshold_u8_c3_in_place
);
impl_packed_threshold_image!(
    u8,
    AC4,
    3,
    npp_threshold::threshold_u8_ac4,
    npp_threshold::threshold_u8_ac4_in_place
);
impl_packed_threshold_image!(
    u16,
    C3,
    3,
    npp_threshold::threshold_u16_c3,
    npp_threshold::threshold_u16_c3_in_place
);
impl_packed_threshold_image!(
    u16,
    AC4,
    3,
    npp_threshold::threshold_u16_ac4,
    npp_threshold::threshold_u16_ac4_in_place
);
impl_packed_threshold_image!(
    i16,
    C3,
    3,
    npp_threshold::threshold_i16_c3,
    npp_threshold::threshold_i16_c3_in_place
);
impl_packed_threshold_image!(
    i16,
    AC4,
    3,
    npp_threshold::threshold_i16_ac4,
    npp_threshold::threshold_i16_ac4_in_place
);
impl_packed_threshold_image!(
    f32,
    C3,
    3,
    npp_threshold::threshold_f32_c3,
    npp_threshold::threshold_f32_c3_in_place
);
impl_packed_threshold_image!(
    f32,
    AC4,
    3,
    npp_threshold::threshold_f32_ac4,
    npp_threshold::threshold_f32_ac4_in_place
);
