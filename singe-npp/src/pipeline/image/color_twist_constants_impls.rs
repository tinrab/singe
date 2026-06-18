use singe_cuda::types::f16;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        color,
        view::{C4, ImageView, ImageViewMut},
    },
};

use super::super::{ImagePipeline, color_dispatch::*};

impl_color_twist_with_constants_image!(
    u8,
    color::color_twist_u8_c4_with_constants,
    color::color_twist_u8_c4_with_constants_in_place
);
impl_color_twist_with_constants_image!(
    f32,
    color::color_twist_f32_c4_with_constants,
    color::color_twist_f32_c4_with_constants_in_place
);
impl_color_twist_with_constants_image!(
    f16,
    color::color_twist_f16_c4_with_constants,
    color::color_twist_f16_c4_with_constants_in_place
);
