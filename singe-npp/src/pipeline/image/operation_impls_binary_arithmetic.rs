use crate::{
    context::StreamContext,
    error::Result,
    image::{
        arithmetic,
        view::{C1, C3, C4, ImageView, ImageViewMut},
    },
};

use super::{ImagePipeline, operation_traits::*};

#[macro_use]
#[path = "operation_impls_binary_arithmetic_macros.rs"]
mod macros;

#[path = "operation_impls_binary_arithmetic_ac4.rs"]
mod ac4_impls;
#[path = "operation_impls_binary_arithmetic_i16.rs"]
mod i16_impls;
#[path = "operation_impls_binary_arithmetic_i32.rs"]
mod i32_impls;

impl_scaled_binary_arithmetic_image!(
    u8,
    C1,
    arithmetic::add_u8_c1,
    arithmetic::add_u8_c1_in_place,
    arithmetic::subtract_u8_c1,
    arithmetic::subtract_u8_c1_in_place,
    arithmetic::multiply_u8_c1,
    arithmetic::multiply_u8_c1_in_place,
    arithmetic::divide_u8_c1,
    arithmetic::divide_u8_c1_in_place
);
impl_scaled_binary_arithmetic_image!(
    u8,
    C3,
    arithmetic::add_u8_c3,
    arithmetic::add_u8_c3_in_place,
    arithmetic::subtract_u8_c3,
    arithmetic::subtract_u8_c3_in_place,
    arithmetic::multiply_u8_c3,
    arithmetic::multiply_u8_c3_in_place,
    arithmetic::divide_u8_c3,
    arithmetic::divide_u8_c3_in_place
);
impl_scaled_binary_arithmetic_image!(
    u8,
    C4,
    arithmetic::add_u8_c4,
    arithmetic::add_u8_c4_in_place,
    arithmetic::subtract_u8_c4,
    arithmetic::subtract_u8_c4_in_place,
    arithmetic::multiply_u8_c4,
    arithmetic::multiply_u8_c4_in_place,
    arithmetic::divide_u8_c4,
    arithmetic::divide_u8_c4_in_place
);
impl_scaled_binary_arithmetic_image!(
    u16,
    C1,
    arithmetic::add_u16_c1,
    arithmetic::add_u16_c1_in_place,
    arithmetic::subtract_u16_c1,
    arithmetic::subtract_u16_c1_in_place,
    arithmetic::multiply_u16_c1,
    arithmetic::multiply_u16_c1_in_place,
    arithmetic::divide_u16_c1,
    arithmetic::divide_u16_c1_in_place
);
impl_scaled_binary_arithmetic_image!(
    u16,
    C3,
    arithmetic::add_u16_c3,
    arithmetic::add_u16_c3_in_place,
    arithmetic::subtract_u16_c3,
    arithmetic::subtract_u16_c3_in_place,
    arithmetic::multiply_u16_c3,
    arithmetic::multiply_u16_c3_in_place,
    arithmetic::divide_u16_c3,
    arithmetic::divide_u16_c3_in_place
);
impl_scaled_binary_arithmetic_image!(
    u16,
    C4,
    arithmetic::add_u16_c4,
    arithmetic::add_u16_c4_in_place,
    arithmetic::subtract_u16_c4,
    arithmetic::subtract_u16_c4_in_place,
    arithmetic::multiply_u16_c4,
    arithmetic::multiply_u16_c4_in_place,
    arithmetic::divide_u16_c4,
    arithmetic::divide_u16_c4_in_place
);
