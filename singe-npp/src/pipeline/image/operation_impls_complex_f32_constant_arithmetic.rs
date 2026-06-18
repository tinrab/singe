use singe_cuda::types::Complex32;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        arithmetic,
        view::{AC4, C1, C3, C4, ImageView, ImageViewMut},
    },
};

use super::super::super::{ImagePipeline, operation_traits::ConstantArithmeticImage};

impl_unscaled_constant_arithmetic_image!(
    Complex32,
    C1,
    Complex32,
    arithmetic::add_constant_f32_complex_c1,
    arithmetic::add_constant_f32_complex_c1_in_place,
    arithmetic::subtract_constant_f32_complex_c1,
    arithmetic::subtract_constant_f32_complex_c1_in_place,
    arithmetic::multiply_constant_f32_complex_c1,
    arithmetic::multiply_constant_f32_complex_c1_in_place,
    arithmetic::divide_constant_f32_complex_c1,
    arithmetic::divide_constant_f32_complex_c1_in_place
);
impl_unscaled_constant_arithmetic_image!(
    Complex32,
    C3,
    [Complex32; 3],
    arithmetic::add_constant_f32_complex_c3,
    arithmetic::add_constant_f32_complex_c3_in_place,
    arithmetic::subtract_constant_f32_complex_c3,
    arithmetic::subtract_constant_f32_complex_c3_in_place,
    arithmetic::multiply_constant_f32_complex_c3,
    arithmetic::multiply_constant_f32_complex_c3_in_place,
    arithmetic::divide_constant_f32_complex_c3,
    arithmetic::divide_constant_f32_complex_c3_in_place
);
impl_unscaled_constant_arithmetic_image!(
    Complex32,
    C4,
    [Complex32; 4],
    arithmetic::add_constant_f32_complex_c4,
    arithmetic::add_constant_f32_complex_c4_in_place,
    arithmetic::subtract_constant_f32_complex_c4,
    arithmetic::subtract_constant_f32_complex_c4_in_place,
    arithmetic::multiply_constant_f32_complex_c4,
    arithmetic::multiply_constant_f32_complex_c4_in_place,
    arithmetic::divide_constant_f32_complex_c4,
    arithmetic::divide_constant_f32_complex_c4_in_place
);
impl_unscaled_constant_arithmetic_image!(
    Complex32,
    AC4,
    [Complex32; 3],
    arithmetic::add_constant_f32_complex_ac4,
    arithmetic::add_constant_f32_complex_ac4_in_place,
    arithmetic::subtract_constant_f32_complex_ac4,
    arithmetic::subtract_constant_f32_complex_ac4_in_place,
    arithmetic::multiply_constant_f32_complex_ac4,
    arithmetic::multiply_constant_f32_complex_ac4_in_place,
    arithmetic::divide_constant_f32_complex_ac4,
    arithmetic::divide_constant_f32_complex_ac4_in_place
);
