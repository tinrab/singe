use crate::{
    context::StreamContext,
    error::Result,
    image::{
        arithmetic,
        view::{C1, C3, ImageView, ImageViewMut},
    },
};

use super::{BinaryArithmeticImage, ImagePipeline};

impl_scaled_binary_arithmetic_image!(
    i32,
    C1,
    arithmetic::add_i32_c1,
    arithmetic::add_i32_c1_in_place,
    arithmetic::subtract_i32_c1,
    arithmetic::subtract_i32_c1_in_place,
    arithmetic::multiply_i32_c1,
    arithmetic::multiply_i32_c1_in_place,
    arithmetic::divide_i32_c1,
    arithmetic::divide_i32_c1_in_place
);
impl_scaled_binary_arithmetic_image!(
    i32,
    C3,
    arithmetic::add_i32_c3,
    arithmetic::add_i32_c3_in_place,
    arithmetic::subtract_i32_c3,
    arithmetic::subtract_i32_c3_in_place,
    arithmetic::multiply_i32_c3,
    arithmetic::multiply_i32_c3_in_place,
    arithmetic::divide_i32_c3,
    arithmetic::divide_i32_c3_in_place
);
