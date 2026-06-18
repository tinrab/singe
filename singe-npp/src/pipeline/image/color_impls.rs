use crate::{
    context::StreamContext,
    error::Result,
    image::{
        color,
        view::{AC4, C1, C2, C3, C4, ImageView, ImageViewMut, PlanarImageView, PlanarImageViewMut},
    },
};

use super::ImagePipeline;
use super::color_dispatch::*;

impl_color_twist_image!(
    u8,
    C1,
    color::color_twist_u8_c1,
    color::color_twist_u8_c1_in_place
);
impl_color_twist_image!(
    u8,
    C2,
    color::color_twist_u8_c2,
    color::color_twist_u8_c2_in_place
);
impl_color_twist_image!(
    u8,
    C3,
    color::color_twist_u8_c3,
    color::color_twist_u8_c3_in_place
);
impl_color_twist_image!(
    u8,
    C4,
    color::color_twist_u8_c4,
    color::color_twist_u8_c4_in_place
);
impl_color_twist_image!(
    u8,
    AC4,
    color::color_twist_u8_ac4,
    color::color_twist_u8_ac4_in_place
);
impl_planar_color_twist_image!(
    u8,
    color::color_twist_u8_p3,
    color::color_twist_u8_p3_in_place
);
impl_color_twist_image!(
    u16,
    C1,
    color::color_twist_u16_c1,
    color::color_twist_u16_c1_in_place
);
impl_color_twist_image!(
    u16,
    C2,
    color::color_twist_u16_c2,
    color::color_twist_u16_c2_in_place
);
impl_color_twist_image!(
    u16,
    C3,
    color::color_twist_u16_c3,
    color::color_twist_u16_c3_in_place
);
impl_color_twist_image!(
    u16,
    AC4,
    color::color_twist_u16_ac4,
    color::color_twist_u16_ac4_in_place
);
impl_planar_color_twist_image!(
    u16,
    color::color_twist_u16_p3,
    color::color_twist_u16_p3_in_place
);
impl_color_twist_image!(
    i16,
    C1,
    color::color_twist_i16_c1,
    color::color_twist_i16_c1_in_place
);
impl_color_twist_image!(
    i16,
    C2,
    color::color_twist_i16_c2,
    color::color_twist_i16_c2_in_place
);
impl_color_twist_image!(
    i16,
    C3,
    color::color_twist_i16_c3,
    color::color_twist_i16_c3_in_place
);
impl_color_twist_image!(
    i16,
    AC4,
    color::color_twist_i16_ac4,
    color::color_twist_i16_ac4_in_place
);
impl_planar_color_twist_image!(
    i16,
    color::color_twist_i16_p3,
    color::color_twist_i16_p3_in_place
);
#[path = "color_twist_f16_impls.rs"]
mod f16_impls;

#[path = "color_twist_i8_impls.rs"]
mod i8_impls;

#[path = "color_twist_constants_impls.rs"]
mod constants_impls;

#[path = "color_batch_impls.rs"]
mod batch_impls;

#[path = "color_twist_f32_impls.rs"]
mod f32_impls;
