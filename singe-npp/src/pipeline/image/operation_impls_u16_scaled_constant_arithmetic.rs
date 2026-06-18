use crate::{
    context::StreamContext,
    error::Result,
    image::{
        arithmetic,
        view::{AC4, C1, C3, C4, ImageView, ImageViewMut},
    },
};

use super::super::super::{ImagePipeline, operation_traits::ConstantArithmeticImage};

impl_scaled_constant_arithmetic_image!(
    u16,
    C1,
    u16,
    arithmetic::add_constant_u16_c1,
    arithmetic::add_constant_u16_c1_in_place,
    arithmetic::subtract_constant_u16_c1,
    arithmetic::subtract_constant_u16_c1_in_place,
    arithmetic::multiply_constant_u16_c1,
    arithmetic::multiply_constant_u16_c1_in_place,
    arithmetic::divide_constant_u16_c1,
    arithmetic::divide_constant_u16_c1_in_place
);
impl_scaled_constant_arithmetic_image!(
    u16,
    C3,
    [u16; 3],
    arithmetic::add_constant_u16_c3,
    arithmetic::add_constant_u16_c3_in_place,
    arithmetic::subtract_constant_u16_c3,
    arithmetic::subtract_constant_u16_c3_in_place,
    arithmetic::multiply_constant_u16_c3,
    arithmetic::multiply_constant_u16_c3_in_place,
    arithmetic::divide_constant_u16_c3,
    arithmetic::divide_constant_u16_c3_in_place
);
impl_scaled_constant_arithmetic_image!(
    u16,
    C4,
    [u16; 4],
    arithmetic::add_constant_u16_c4,
    arithmetic::add_constant_u16_c4_in_place,
    arithmetic::subtract_constant_u16_c4,
    arithmetic::subtract_constant_u16_c4_in_place,
    arithmetic::multiply_constant_u16_c4,
    arithmetic::multiply_constant_u16_c4_in_place,
    arithmetic::divide_constant_u16_c4,
    arithmetic::divide_constant_u16_c4_in_place
);
impl_scaled_constant_arithmetic_image!(
    u16,
    AC4,
    [u16; 3],
    arithmetic::add_constant_u16_ac4,
    arithmetic::add_constant_u16_ac4_in_place,
    arithmetic::subtract_constant_u16_ac4,
    arithmetic::subtract_constant_u16_ac4_in_place,
    arithmetic::multiply_constant_u16_ac4,
    arithmetic::multiply_constant_u16_ac4_in_place,
    arithmetic::divide_constant_u16_ac4,
    arithmetic::divide_constant_u16_ac4_in_place
);
