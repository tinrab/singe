use crate::{
    context::StreamContext,
    error::Result,
    image::{
        arithmetic,
        view::{C1, C3, C4, ImageView, ImageViewMut},
    },
};

use super::super::{
    ImagePipeline,
    operation_traits::{LeftShiftConstantImage, RightShiftConstantImage},
};

impl_shift_constant_image!(
    u16,
    C1,
    u32,
    arithmetic::right_shift_constant_u16_c1,
    arithmetic::right_shift_constant_u16_c1_in_place,
    arithmetic::left_shift_constant_u16_c1,
    arithmetic::left_shift_constant_u16_c1_in_place
);
impl_shift_constant_image!(
    u16,
    C3,
    [u32; 3],
    arithmetic::right_shift_constant_u16_c3,
    arithmetic::right_shift_constant_u16_c3_in_place,
    arithmetic::left_shift_constant_u16_c3,
    arithmetic::left_shift_constant_u16_c3_in_place
);
impl_shift_constant_image!(
    u16,
    C4,
    [u32; 4],
    arithmetic::right_shift_constant_u16_c4,
    arithmetic::right_shift_constant_u16_c4_in_place,
    arithmetic::left_shift_constant_u16_c4,
    arithmetic::left_shift_constant_u16_c4_in_place
);
