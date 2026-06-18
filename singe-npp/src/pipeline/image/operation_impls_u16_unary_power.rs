use crate::{
    context::StreamContext,
    error::Result,
    image::{
        arithmetic,
        view::{C1, C3, ImageView, ImageViewMut},
    },
};

use super::super::super::{ImagePipeline, operation_traits::*};

impl_scaled_unary_power_image!(
    u16,
    C1,
    arithmetic::square_u16_c1,
    arithmetic::square_u16_c1_in_place,
    arithmetic::square_root_u16_c1,
    arithmetic::square_root_u16_c1_in_place
);
impl_scaled_unary_power_image!(
    u16,
    C3,
    arithmetic::square_u16_c3,
    arithmetic::square_u16_c3_in_place,
    arithmetic::square_root_u16_c3,
    arithmetic::square_root_u16_c3_in_place
);
