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

impl_cross_correlation_norm_image!(
    f32,
    C1,
    statistics::cross_correlation_full_norm_f32_c1,
    statistics::cross_correlation_same_norm_f32_c1,
    statistics::cross_correlation_valid_norm_f32_c1
);
impl_cross_correlation_norm_image!(
    f32,
    C3,
    statistics::cross_correlation_full_norm_f32_c3,
    statistics::cross_correlation_same_norm_f32_c3,
    statistics::cross_correlation_valid_norm_f32_c3
);
impl_cross_correlation_norm_image!(
    f32,
    C4,
    statistics::cross_correlation_full_norm_f32_c4,
    statistics::cross_correlation_same_norm_f32_c4,
    statistics::cross_correlation_valid_norm_f32_c4
);
impl_cross_correlation_norm_image!(
    f32,
    AC4,
    statistics::cross_correlation_full_norm_f32_ac4,
    statistics::cross_correlation_same_norm_f32_ac4,
    statistics::cross_correlation_valid_norm_f32_ac4
);
impl_cross_correlation_norm_image!(
    f64,
    C1,
    statistics::cross_correlation_full_norm_f64_c1,
    statistics::cross_correlation_same_norm_f64_c1,
    statistics::cross_correlation_valid_norm_f64_c1
);
impl_cross_correlation_norm_image!(
    f64,
    C3,
    statistics::cross_correlation_full_norm_f64_c3,
    statistics::cross_correlation_same_norm_f64_c3,
    statistics::cross_correlation_valid_norm_f64_c3
);
impl_cross_correlation_norm_image!(
    f64,
    C4,
    statistics::cross_correlation_full_norm_f64_c4,
    statistics::cross_correlation_same_norm_f64_c4,
    statistics::cross_correlation_valid_norm_f64_c4
);
impl_cross_correlation_norm_image!(
    f64,
    AC4,
    statistics::cross_correlation_full_norm_f64_ac4,
    statistics::cross_correlation_same_norm_f64_ac4,
    statistics::cross_correlation_valid_norm_f64_ac4
);

#[path = "template_cross_correlation_norm_to_dispatch.rs"]
mod norm_to_dispatch;

impl_cross_correlation_norm_scaled_image!(
    u8,
    C1,
    statistics::cross_correlation_full_norm_u8_c1,
    statistics::cross_correlation_same_norm_u8_c1,
    statistics::cross_correlation_valid_norm_u8_c1
);
impl_cross_correlation_norm_scaled_image!(
    u8,
    C3,
    statistics::cross_correlation_full_norm_u8_c3,
    statistics::cross_correlation_same_norm_u8_c3,
    statistics::cross_correlation_valid_norm_u8_c3
);
impl_cross_correlation_norm_scaled_image!(
    u8,
    C4,
    statistics::cross_correlation_full_norm_u8_c4,
    statistics::cross_correlation_same_norm_u8_c4,
    statistics::cross_correlation_valid_norm_u8_c4
);
impl_cross_correlation_norm_scaled_image!(
    u8,
    AC4,
    statistics::cross_correlation_full_norm_u8_ac4,
    statistics::cross_correlation_same_norm_u8_ac4,
    statistics::cross_correlation_valid_norm_u8_ac4
);

#[path = "template_cross_correlation_norm_level_dispatch.rs"]
mod norm_level_dispatch;

#[path = "template_cross_correlation_norm_level_to_dispatch.rs"]
mod norm_level_to_dispatch;

#[path = "template_cross_correlation_norm_level_advanced_dispatch.rs"]
mod norm_level_advanced_dispatch;

impl_cross_correlation_image!(f32, C1, statistics::cross_correlation_valid_f32_c1);
impl_cross_correlation_image!(f64, C1, statistics::cross_correlation_valid_f64_c1);

impl_cross_correlation_to_image!(
    u8,
    f32,
    C1,
    statistics::cross_correlation_valid_u8_to_f32_c1
);
impl_cross_correlation_to_image!(
    i8,
    f32,
    C1,
    statistics::cross_correlation_valid_i8_to_f32_c1
);
impl_cross_correlation_to_image!(
    u16,
    f32,
    C1,
    statistics::cross_correlation_valid_u16_to_f32_c1
);
