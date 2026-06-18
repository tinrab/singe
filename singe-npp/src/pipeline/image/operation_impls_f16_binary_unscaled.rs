use singe_cuda::types::f16;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        arithmetic,
        view::{C1, C3, C4, ImageView, ImageViewMut},
    },
};

use super::super::{ImagePipeline, operation_traits::BinaryArithmeticImage};

impl_unscaled_binary_arithmetic_image!(
    f16,
    C1,
    arithmetic::add_f16_c1,
    arithmetic::add_f16_c1_in_place,
    arithmetic::subtract_f16_c1,
    arithmetic::subtract_f16_c1_in_place,
    arithmetic::multiply_f16_c1,
    arithmetic::multiply_f16_c1_in_place,
    arithmetic::divide_f16_c1,
    arithmetic::divide_f16_c1_in_place
);
impl_unscaled_binary_arithmetic_image!(
    f16,
    C3,
    arithmetic::add_f16_c3,
    arithmetic::add_f16_c3_in_place,
    arithmetic::subtract_f16_c3,
    arithmetic::subtract_f16_c3_in_place,
    arithmetic::multiply_f16_c3,
    arithmetic::multiply_f16_c3_in_place,
    arithmetic::divide_f16_c3,
    arithmetic::divide_f16_c3_in_place
);
impl_unscaled_binary_arithmetic_image!(
    f16,
    C4,
    arithmetic::add_f16_c4,
    arithmetic::add_f16_c4_in_place,
    arithmetic::subtract_f16_c4,
    arithmetic::subtract_f16_c4_in_place,
    arithmetic::multiply_f16_c4,
    arithmetic::multiply_f16_c4_in_place,
    arithmetic::divide_f16_c4,
    arithmetic::divide_f16_c4_in_place
);
