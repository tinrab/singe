use singe_cuda::types::f16;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        color,
        view::{C1, C2, C3, C4, ImageView, ImageViewMut},
    },
};

use super::super::{ImagePipeline, color_dispatch::*};

impl_color_twist_image!(
    f16,
    C1,
    color::color_twist_f16_c1,
    color::color_twist_f16_c1_in_place
);
impl_color_twist_image!(
    f16,
    C2,
    color::color_twist_f16_c2,
    color::color_twist_f16_c2_in_place
);
impl_color_twist_image!(
    f16,
    C3,
    color::color_twist_f16_c3,
    color::color_twist_f16_c3_in_place
);
impl_color_twist_image!(
    f16,
    C4,
    color::color_twist_f16_c4,
    color::color_twist_f16_c4_in_place
);
