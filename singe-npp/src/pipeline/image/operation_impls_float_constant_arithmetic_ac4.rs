use crate::{
    context::StreamContext,
    error::Result,
    image::{
        arithmetic,
        view::{AC4, ImageView, ImageViewMut},
    },
};

use super::super::{ImagePipeline, operation_traits::ConstantArithmeticImage};

impl_float_constant_arithmetic_image!(
    AC4,
    [f32; 3],
    arithmetic::add_constant_f32_ac4,
    arithmetic::add_constant_f32_ac4_in_place,
    arithmetic::subtract_constant_f32_ac4,
    arithmetic::subtract_constant_f32_ac4_in_place,
    arithmetic::multiply_constant_f32_ac4,
    arithmetic::multiply_constant_f32_ac4_in_place,
    arithmetic::divide_constant_f32_ac4,
    arithmetic::divide_constant_f32_ac4_in_place
);
