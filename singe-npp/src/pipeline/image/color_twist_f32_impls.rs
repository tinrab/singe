use crate::{
    context::StreamContext,
    error::Result,
    image::{
        color,
        view::{AC4, C1, C2, C3, C4, ImageView, ImageViewMut, PlanarImageView, PlanarImageViewMut},
    },
};

use super::super::{ImagePipeline, color_dispatch::*};

impl_color_twist_image!(
    f32,
    C1,
    color::color_twist_f32_c1,
    color::color_twist_f32_c1_in_place
);
impl_color_twist_image!(
    f32,
    C2,
    color::color_twist_f32_c2,
    color::color_twist_f32_c2_in_place
);
impl_color_twist_image!(
    f32,
    C3,
    color::color_twist_f32_c3,
    color::color_twist_f32_c3_in_place
);
impl_color_twist_image!(
    f32,
    C4,
    color::color_twist_f32_c4,
    color::color_twist_f32_c4_in_place
);
impl_color_twist_image!(
    f32,
    AC4,
    color::color_twist_f32_ac4,
    color::color_twist_f32_ac4_in_place
);
impl_planar_color_twist_image!(
    f32,
    color::color_twist_f32_p3,
    color::color_twist_f32_p3_in_place
);
