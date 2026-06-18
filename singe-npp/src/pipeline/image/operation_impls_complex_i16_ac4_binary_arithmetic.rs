use crate::{
    context::StreamContext,
    error::Result,
    image::{
        arithmetic,
        view::{AC4, ImageView, ImageViewMut},
    },
    types::ComplexI16,
};

use super::super::{ImagePipeline, operation_traits::BinaryArithmeticImage};

impl_scaled_binary_arithmetic_image!(
    ComplexI16,
    AC4,
    arithmetic::add_i16_complex_ac4,
    arithmetic::add_i16_complex_ac4_in_place,
    arithmetic::subtract_i16_complex_ac4,
    arithmetic::subtract_i16_complex_ac4_in_place,
    arithmetic::multiply_i16_complex_ac4,
    arithmetic::multiply_i16_complex_ac4_in_place,
    arithmetic::divide_i16_complex_ac4,
    arithmetic::divide_i16_complex_ac4_in_place
);
