use crate::{
    context::StreamContext,
    error::Result,
    image::{
        arithmetic,
        view::{AC4, C1, C3, C4, ImageView, ImageViewMut},
    },
};

use super::{AlphaPremultiplyConstantImage, ImagePipeline};

impl_alpha_premultiply_constant_image!(
    u16,
    C1,
    u16,
    arithmetic::alpha_premultiply_constant_u16_c1,
    arithmetic::alpha_premultiply_constant_u16_c1_in_place
);
impl_alpha_premultiply_constant_image!(
    u16,
    C3,
    u16,
    arithmetic::alpha_premultiply_constant_u16_c3,
    arithmetic::alpha_premultiply_constant_u16_c3_in_place
);
impl_alpha_premultiply_constant_image!(
    u16,
    C4,
    u16,
    arithmetic::alpha_premultiply_constant_u16_c4,
    arithmetic::alpha_premultiply_constant_u16_c4_in_place
);
impl_alpha_premultiply_constant_image!(
    u16,
    AC4,
    u16,
    arithmetic::alpha_premultiply_constant_u16_ac4,
    arithmetic::alpha_premultiply_constant_u16_ac4_in_place
);
