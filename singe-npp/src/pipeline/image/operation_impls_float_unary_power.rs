use crate::{
    context::StreamContext,
    error::Result,
    image::{
        arithmetic,
        view::{AC4, C1, C3, C4, ImageView, ImageViewMut},
    },
};

use super::{ImagePipeline, SquareImage, SquareRootImage};

impl_float_unary_power_image!(
    C1,
    arithmetic::square_f32_c1,
    arithmetic::square_f32_c1_in_place,
    arithmetic::square_root_f32_c1,
    arithmetic::square_root_f32_c1_in_place
);
impl_float_unary_power_image!(
    C3,
    arithmetic::square_f32_c3,
    arithmetic::square_f32_c3_in_place,
    arithmetic::square_root_f32_c3,
    arithmetic::square_root_f32_c3_in_place
);
impl_float_unary_power_image!(
    C4,
    arithmetic::square_f32_c4,
    arithmetic::square_f32_c4_in_place,
    arithmetic::square_root_f32_c4,
    arithmetic::square_root_f32_c4_in_place
);
impl_float_unary_power_image!(
    AC4,
    arithmetic::square_f32_ac4,
    arithmetic::square_f32_ac4_in_place,
    arithmetic::square_root_f32_ac4,
    arithmetic::square_root_f32_ac4_in_place
);
