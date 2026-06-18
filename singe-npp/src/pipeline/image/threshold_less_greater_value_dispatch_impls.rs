use crate::{
    context::StreamContext,
    error::Result,
    image::{
        threshold as npp_threshold,
        view::{AC4, C1, C3, ImageView, ImageViewMut},
    },
};

use super::super::{
    ImagePipeline,
    threshold_dispatch::{LessGreaterValueThresholdImage, PackedLessGreaterValueThresholdImage},
};

#[macro_use]
#[path = "threshold_less_greater_value_dispatch_macros.rs"]
mod macros;

impl_less_greater_value_threshold_image!(
    u8,
    npp_threshold::threshold_less_greater_value_u8_c1,
    npp_threshold::threshold_less_greater_value_u8_c1_in_place
);
impl_less_greater_value_threshold_image!(
    u16,
    npp_threshold::threshold_less_greater_value_u16_c1,
    npp_threshold::threshold_less_greater_value_u16_c1_in_place
);
impl_less_greater_value_threshold_image!(
    i16,
    npp_threshold::threshold_less_greater_value_i16_c1,
    npp_threshold::threshold_less_greater_value_i16_c1_in_place
);
impl_less_greater_value_threshold_image!(
    f32,
    npp_threshold::threshold_less_greater_value_f32_c1,
    npp_threshold::threshold_less_greater_value_f32_c1_in_place
);

impl_packed_less_greater_value_threshold_image!(
    u8,
    C3,
    3,
    npp_threshold::threshold_less_greater_value_u8_c3,
    npp_threshold::threshold_less_greater_value_u8_c3_in_place
);
impl_packed_less_greater_value_threshold_image!(
    u8,
    AC4,
    3,
    npp_threshold::threshold_less_greater_value_u8_ac4,
    npp_threshold::threshold_less_greater_value_u8_ac4_in_place
);
impl_packed_less_greater_value_threshold_image!(
    u16,
    C3,
    3,
    npp_threshold::threshold_less_greater_value_u16_c3,
    npp_threshold::threshold_less_greater_value_u16_c3_in_place
);
impl_packed_less_greater_value_threshold_image!(
    u16,
    AC4,
    3,
    npp_threshold::threshold_less_greater_value_u16_ac4,
    npp_threshold::threshold_less_greater_value_u16_ac4_in_place
);
impl_packed_less_greater_value_threshold_image!(
    i16,
    C3,
    3,
    npp_threshold::threshold_less_greater_value_i16_c3,
    npp_threshold::threshold_less_greater_value_i16_c3_in_place
);
impl_packed_less_greater_value_threshold_image!(
    i16,
    AC4,
    3,
    npp_threshold::threshold_less_greater_value_i16_ac4,
    npp_threshold::threshold_less_greater_value_i16_ac4_in_place
);
impl_packed_less_greater_value_threshold_image!(
    f32,
    C3,
    3,
    npp_threshold::threshold_less_greater_value_f32_c3,
    npp_threshold::threshold_less_greater_value_f32_c3_in_place
);
impl_packed_less_greater_value_threshold_image!(
    f32,
    AC4,
    3,
    npp_threshold::threshold_less_greater_value_f32_ac4,
    npp_threshold::threshold_less_greater_value_f32_ac4_in_place
);
