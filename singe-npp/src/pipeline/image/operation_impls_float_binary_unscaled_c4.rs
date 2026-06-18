use singe_cuda::types::Complex32;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        arithmetic,
        view::{C4, ImageView, ImageViewMut},
    },
};

use super::super::{ImagePipeline, operation_traits::BinaryArithmeticImage};

impl_unscaled_binary_arithmetic_image!(
    Complex32,
    C4,
    arithmetic::add_f32_complex_c4,
    arithmetic::add_f32_complex_c4_in_place,
    arithmetic::subtract_f32_complex_c4,
    arithmetic::subtract_f32_complex_c4_in_place,
    arithmetic::multiply_f32_complex_c4,
    arithmetic::multiply_f32_complex_c4_in_place,
    arithmetic::divide_f32_complex_c4,
    arithmetic::divide_f32_complex_c4_in_place
);
