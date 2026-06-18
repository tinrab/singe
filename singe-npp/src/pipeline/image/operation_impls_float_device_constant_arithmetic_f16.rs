use singe_cuda::{memory::DeviceMemory, types::f16};

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        arithmetic,
        view::{C1, C3, C4, ImageView, ImageViewMut},
    },
};

use super::super::{ImagePipeline, operation_traits::DeviceConstantArithmeticImage};

impl_unscaled_device_constant_arithmetic_image!(
    f16,
    C1,
    f32,
    arithmetic::add_device_constant_f16_c1,
    arithmetic::add_device_constant_f16_c1_in_place,
    arithmetic::subtract_device_constant_f16_c1,
    arithmetic::subtract_device_constant_f16_c1_in_place,
    arithmetic::multiply_device_constant_f16_c1,
    arithmetic::multiply_device_constant_f16_c1_in_place,
    arithmetic::divide_device_constant_f16_c1,
    arithmetic::divide_device_constant_f16_c1_in_place
);
impl_unscaled_device_constant_arithmetic_image!(
    f16,
    C3,
    f32,
    arithmetic::add_device_constant_f16_c3,
    arithmetic::add_device_constant_f16_c3_in_place,
    arithmetic::subtract_device_constant_f16_c3,
    arithmetic::subtract_device_constant_f16_c3_in_place,
    arithmetic::multiply_device_constant_f16_c3,
    arithmetic::multiply_device_constant_f16_c3_in_place,
    arithmetic::divide_device_constant_f16_c3,
    arithmetic::divide_device_constant_f16_c3_in_place
);
impl_unscaled_device_constant_arithmetic_image!(
    f16,
    C4,
    f32,
    arithmetic::add_device_constant_f16_c4,
    arithmetic::add_device_constant_f16_c4_in_place,
    arithmetic::subtract_device_constant_f16_c4,
    arithmetic::subtract_device_constant_f16_c4_in_place,
    arithmetic::multiply_device_constant_f16_c4,
    arithmetic::multiply_device_constant_f16_c4_in_place,
    arithmetic::divide_device_constant_f16_c4,
    arithmetic::divide_device_constant_f16_c4_in_place
);
