use crate::{
    context::StreamContext,
    error::Result,
    image::{
        arithmetic,
        view::{AC4, C1, C3, ImageView, ImageViewMut},
    },
    types::ComplexI32,
};

use super::super::{ImagePipeline, operation_traits::BinaryArithmeticImage};

impl_scaled_binary_arithmetic_image!(
    ComplexI32,
    C1,
    arithmetic::add_i32_complex_c1,
    arithmetic::add_i32_complex_c1_in_place,
    arithmetic::subtract_i32_complex_c1,
    arithmetic::subtract_i32_complex_c1_in_place,
    arithmetic::multiply_i32_complex_c1,
    arithmetic::multiply_i32_complex_c1_in_place,
    arithmetic::divide_i32_complex_c1,
    arithmetic::divide_i32_complex_c1_in_place
);
impl_scaled_binary_arithmetic_image!(
    ComplexI32,
    C3,
    arithmetic::add_i32_complex_c3,
    arithmetic::add_i32_complex_c3_in_place,
    arithmetic::subtract_i32_complex_c3,
    arithmetic::subtract_i32_complex_c3_in_place,
    arithmetic::multiply_i32_complex_c3,
    arithmetic::multiply_i32_complex_c3_in_place,
    arithmetic::divide_i32_complex_c3,
    arithmetic::divide_i32_complex_c3_in_place
);
impl_scaled_binary_arithmetic_image!(
    ComplexI32,
    AC4,
    arithmetic::add_i32_complex_ac4,
    arithmetic::add_i32_complex_ac4_in_place,
    arithmetic::subtract_i32_complex_ac4,
    arithmetic::subtract_i32_complex_ac4_in_place,
    arithmetic::multiply_i32_complex_ac4,
    arithmetic::multiply_i32_complex_ac4_in_place,
    arithmetic::divide_i32_complex_ac4,
    arithmetic::divide_i32_complex_ac4_in_place
);
