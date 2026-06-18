use crate::{
    context::StreamContext,
    error::Result,
    image::{
        statistics,
        view::{AC4, C1, C3, C4, ImageView, ImageViewMut},
    },
    pipeline::image::ImagePipeline,
};

use super::traits::*;

#[path = "template_square_distance_to_dispatch.rs"]
mod to_dispatch;

impl_square_distance_norm_image!(
    f32,
    C1,
    statistics::square_distance_full_norm_f32_c1,
    statistics::square_distance_same_norm_f32_c1,
    statistics::square_distance_valid_norm_f32_c1
);
impl_square_distance_norm_image!(
    f32,
    C3,
    statistics::square_distance_full_norm_f32_c3,
    statistics::square_distance_same_norm_f32_c3,
    statistics::square_distance_valid_norm_f32_c3
);
impl_square_distance_norm_image!(
    f32,
    C4,
    statistics::square_distance_full_norm_f32_c4,
    statistics::square_distance_same_norm_f32_c4,
    statistics::square_distance_valid_norm_f32_c4
);
impl_square_distance_norm_image!(
    f32,
    AC4,
    statistics::square_distance_full_norm_f32_ac4,
    statistics::square_distance_same_norm_f32_ac4,
    statistics::square_distance_valid_norm_f32_ac4
);

impl_square_distance_norm_scaled_image!(
    u8,
    C1,
    statistics::square_distance_full_norm_u8_c1,
    statistics::square_distance_same_norm_u8_c1,
    statistics::square_distance_valid_norm_u8_c1
);
impl_square_distance_norm_scaled_image!(
    u8,
    C3,
    statistics::square_distance_full_norm_u8_c3,
    statistics::square_distance_same_norm_u8_c3,
    statistics::square_distance_valid_norm_u8_c3
);
impl_square_distance_norm_scaled_image!(
    u8,
    C4,
    statistics::square_distance_full_norm_u8_c4,
    statistics::square_distance_same_norm_u8_c4,
    statistics::square_distance_valid_norm_u8_c4
);
impl_square_distance_norm_scaled_image!(
    u8,
    AC4,
    statistics::square_distance_full_norm_u8_ac4,
    statistics::square_distance_same_norm_u8_ac4,
    statistics::square_distance_valid_norm_u8_ac4
);
