use crate::{
    context::StreamContext,
    error::Result,
    image::{
        arithmetic,
        view::{AC4, C1, C3, ImageView, ImageViewMut},
    },
    types::{ComplexI16, ComplexI32},
};

use super::super::{ImagePipeline, operation_traits::ConstantArithmeticImage};

#[path = "operation_impls_complex_f32_constant_arithmetic.rs"]
mod complex_f32_constant_arithmetic;

impl_scaled_constant_arithmetic_image!(
    ComplexI16,
    C1,
    ComplexI16,
    arithmetic::add_constant_i16_complex_c1,
    arithmetic::add_constant_i16_complex_c1_in_place,
    arithmetic::subtract_constant_i16_complex_c1,
    arithmetic::subtract_constant_i16_complex_c1_in_place,
    arithmetic::multiply_constant_i16_complex_c1,
    arithmetic::multiply_constant_i16_complex_c1_in_place,
    arithmetic::divide_constant_i16_complex_c1,
    arithmetic::divide_constant_i16_complex_c1_in_place
);
impl_scaled_constant_arithmetic_image!(
    ComplexI16,
    C3,
    [ComplexI16; 3],
    arithmetic::add_constant_i16_complex_c3,
    arithmetic::add_constant_i16_complex_c3_in_place,
    arithmetic::subtract_constant_i16_complex_c3,
    arithmetic::subtract_constant_i16_complex_c3_in_place,
    arithmetic::multiply_constant_i16_complex_c3,
    arithmetic::multiply_constant_i16_complex_c3_in_place,
    arithmetic::divide_constant_i16_complex_c3,
    arithmetic::divide_constant_i16_complex_c3_in_place
);
impl_scaled_constant_arithmetic_image!(
    ComplexI16,
    AC4,
    [ComplexI16; 3],
    arithmetic::add_constant_i16_complex_ac4,
    arithmetic::add_constant_i16_complex_ac4_in_place,
    arithmetic::subtract_constant_i16_complex_ac4,
    arithmetic::subtract_constant_i16_complex_ac4_in_place,
    arithmetic::multiply_constant_i16_complex_ac4,
    arithmetic::multiply_constant_i16_complex_ac4_in_place,
    arithmetic::divide_constant_i16_complex_ac4,
    arithmetic::divide_constant_i16_complex_ac4_in_place
);
impl_scaled_constant_arithmetic_image!(
    ComplexI32,
    C1,
    ComplexI32,
    arithmetic::add_constant_i32_complex_c1,
    arithmetic::add_constant_i32_complex_c1_in_place,
    arithmetic::subtract_constant_i32_complex_c1,
    arithmetic::subtract_constant_i32_complex_c1_in_place,
    arithmetic::multiply_constant_i32_complex_c1,
    arithmetic::multiply_constant_i32_complex_c1_in_place,
    arithmetic::divide_constant_i32_complex_c1,
    arithmetic::divide_constant_i32_complex_c1_in_place
);
impl_scaled_constant_arithmetic_image!(
    ComplexI32,
    C3,
    [ComplexI32; 3],
    arithmetic::add_constant_i32_complex_c3,
    arithmetic::add_constant_i32_complex_c3_in_place,
    arithmetic::subtract_constant_i32_complex_c3,
    arithmetic::subtract_constant_i32_complex_c3_in_place,
    arithmetic::multiply_constant_i32_complex_c3,
    arithmetic::multiply_constant_i32_complex_c3_in_place,
    arithmetic::divide_constant_i32_complex_c3,
    arithmetic::divide_constant_i32_complex_c3_in_place
);
impl_scaled_constant_arithmetic_image!(
    ComplexI32,
    AC4,
    [ComplexI32; 3],
    arithmetic::add_constant_i32_complex_ac4,
    arithmetic::add_constant_i32_complex_ac4_in_place,
    arithmetic::subtract_constant_i32_complex_ac4,
    arithmetic::subtract_constant_i32_complex_ac4_in_place,
    arithmetic::multiply_constant_i32_complex_ac4,
    arithmetic::multiply_constant_i32_complex_ac4_in_place,
    arithmetic::divide_constant_i32_complex_ac4,
    arithmetic::divide_constant_i32_complex_ac4_in_place
);
