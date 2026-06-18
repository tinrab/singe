use crate::{
    context::StreamContext,
    error::Result,
    image::{
        arithmetic,
        view::{AC4, C1, C3, C4, ImageView, ImageViewMut},
    },
};

use super::super::{ImagePipeline, operation_traits::RightShiftConstantImage};

impl_right_shift_constant_image!(
    i8,
    C1,
    u32,
    arithmetic::right_shift_constant_i8_c1,
    arithmetic::right_shift_constant_i8_c1_in_place
);
impl_right_shift_constant_image!(
    i8,
    C3,
    [u32; 3],
    arithmetic::right_shift_constant_i8_c3,
    arithmetic::right_shift_constant_i8_c3_in_place
);
impl_right_shift_constant_image!(
    i8,
    C4,
    [u32; 4],
    arithmetic::right_shift_constant_i8_c4,
    arithmetic::right_shift_constant_i8_c4_in_place
);
impl_right_shift_constant_image!(
    i8,
    AC4,
    [u32; 3],
    arithmetic::right_shift_constant_i8_ac4,
    arithmetic::right_shift_constant_i8_ac4_in_place
);
impl_right_shift_constant_image!(
    i16,
    C1,
    u32,
    arithmetic::right_shift_constant_i16_c1,
    arithmetic::right_shift_constant_i16_c1_in_place
);
impl_right_shift_constant_image!(
    i16,
    C3,
    [u32; 3],
    arithmetic::right_shift_constant_i16_c3,
    arithmetic::right_shift_constant_i16_c3_in_place
);
impl_right_shift_constant_image!(
    i16,
    C4,
    [u32; 4],
    arithmetic::right_shift_constant_i16_c4,
    arithmetic::right_shift_constant_i16_c4_in_place
);
impl_right_shift_constant_image!(
    i16,
    AC4,
    [u32; 3],
    arithmetic::right_shift_constant_i16_ac4,
    arithmetic::right_shift_constant_i16_ac4_in_place
);
