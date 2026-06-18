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
    i8,
    C1,
    color::color_twist_i8_c1,
    color::color_twist_i8_c1_in_place
);
impl_color_twist_image!(
    i8,
    C2,
    color::color_twist_i8_c2,
    color::color_twist_i8_c2_in_place
);
impl_color_twist_image!(
    i8,
    C3,
    color::color_twist_i8_c3,
    color::color_twist_i8_c3_in_place
);
impl_color_twist_image!(
    i8,
    C4,
    color::color_twist_i8_c4,
    color::color_twist_i8_c4_in_place
);
impl_color_twist_image!(
    i8,
    AC4,
    color::color_twist_i8_ac4,
    color::color_twist_i8_ac4_in_place
);
impl_planar_color_twist_image!(
    i8,
    color::color_twist_i8_p3,
    color::color_twist_i8_p3_in_place
);
