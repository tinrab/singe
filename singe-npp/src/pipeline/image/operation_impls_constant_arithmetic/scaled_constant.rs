use crate::{
    context::StreamContext,
    error::Result,
    image::{
        arithmetic,
        view::{AC4, C1, C3, C4, ImageOutputView, ImageView, ImageViewMut},
    },
    pipeline::image::{ImagePipeline, operation_traits::ConstantArithmeticImage},
};

mod i32_impls;
mod u16_impls;

impl_scaled_constant_arithmetic_image!(
    u8,
    C1,
    u8,
    arithmetic::add_constant_u8_c1,
    arithmetic::add_constant_u8_c1_in_place,
    arithmetic::subtract_constant_u8_c1,
    arithmetic::subtract_constant_u8_c1_in_place,
    arithmetic::multiply_constant_u8_c1,
    arithmetic::multiply_constant_u8_c1_in_place,
    arithmetic::divide_constant_u8_c1,
    arithmetic::divide_constant_u8_c1_in_place
);

impl_scaled_constant_arithmetic_image!(
    u8,
    C3,
    [u8; 3],
    arithmetic::add_constant_u8_c3,
    arithmetic::add_constant_u8_c3_in_place,
    arithmetic::subtract_constant_u8_c3,
    arithmetic::subtract_constant_u8_c3_in_place,
    arithmetic::multiply_constant_u8_c3,
    arithmetic::multiply_constant_u8_c3_in_place,
    arithmetic::divide_constant_u8_c3,
    arithmetic::divide_constant_u8_c3_in_place
);

impl_scaled_constant_arithmetic_image!(
    u8,
    C4,
    [u8; 4],
    arithmetic::add_constant_u8_c4,
    arithmetic::add_constant_u8_c4_in_place,
    arithmetic::subtract_constant_u8_c4,
    arithmetic::subtract_constant_u8_c4_in_place,
    arithmetic::multiply_constant_u8_c4,
    arithmetic::multiply_constant_u8_c4_in_place,
    arithmetic::divide_constant_u8_c4,
    arithmetic::divide_constant_u8_c4_in_place
);

impl_scaled_constant_arithmetic_image!(
    u8,
    AC4,
    [u8; 3],
    arithmetic::add_constant_u8_ac4,
    arithmetic::add_constant_u8_ac4_in_place,
    arithmetic::subtract_constant_u8_ac4,
    arithmetic::subtract_constant_u8_ac4_in_place,
    arithmetic::multiply_constant_u8_ac4,
    arithmetic::multiply_constant_u8_ac4_in_place,
    arithmetic::divide_constant_u8_ac4,
    arithmetic::divide_constant_u8_ac4_in_place
);

impl_scaled_constant_arithmetic_image!(
    i16,
    C1,
    i16,
    arithmetic::add_constant_i16_c1,
    arithmetic::add_constant_i16_c1_in_place,
    arithmetic::subtract_constant_i16_c1,
    arithmetic::subtract_constant_i16_c1_in_place,
    arithmetic::multiply_constant_i16_c1,
    arithmetic::multiply_constant_i16_c1_in_place,
    arithmetic::divide_constant_i16_c1,
    arithmetic::divide_constant_i16_c1_in_place
);

impl_scaled_constant_arithmetic_image!(
    i16,
    C3,
    [i16; 3],
    arithmetic::add_constant_i16_c3,
    arithmetic::add_constant_i16_c3_in_place,
    arithmetic::subtract_constant_i16_c3,
    arithmetic::subtract_constant_i16_c3_in_place,
    arithmetic::multiply_constant_i16_c3,
    arithmetic::multiply_constant_i16_c3_in_place,
    arithmetic::divide_constant_i16_c3,
    arithmetic::divide_constant_i16_c3_in_place
);

impl_scaled_constant_arithmetic_image!(
    i16,
    C4,
    [i16; 4],
    arithmetic::add_constant_i16_c4,
    arithmetic::add_constant_i16_c4_in_place,
    arithmetic::subtract_constant_i16_c4,
    arithmetic::subtract_constant_i16_c4_in_place,
    arithmetic::multiply_constant_i16_c4,
    arithmetic::multiply_constant_i16_c4_in_place,
    arithmetic::divide_constant_i16_c4,
    arithmetic::divide_constant_i16_c4_in_place
);

impl_scaled_constant_arithmetic_image!(
    i16,
    AC4,
    [i16; 3],
    arithmetic::add_constant_i16_ac4,
    arithmetic::add_constant_i16_ac4_in_place,
    arithmetic::subtract_constant_i16_ac4,
    arithmetic::subtract_constant_i16_ac4_in_place,
    arithmetic::multiply_constant_i16_ac4,
    arithmetic::multiply_constant_i16_ac4_in_place,
    arithmetic::divide_constant_i16_ac4,
    arithmetic::divide_constant_i16_ac4_in_place
);
