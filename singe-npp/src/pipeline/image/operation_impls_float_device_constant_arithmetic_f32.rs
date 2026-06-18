use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        arithmetic,
        view::{AC4, C1, C3, C4, ImageView, ImageViewMut},
    },
};

use super::super::{ImagePipeline, operation_traits::DeviceConstantArithmeticImage};

impl_float_device_constant_arithmetic_image!(
    C1,
    f32,
    arithmetic::add_device_constant_f32_c1,
    arithmetic::add_device_constant_f32_c1_in_place,
    arithmetic::subtract_device_constant_f32_c1,
    arithmetic::subtract_device_constant_f32_c1_in_place,
    arithmetic::multiply_device_constant_f32_c1,
    arithmetic::multiply_device_constant_f32_c1_in_place,
    arithmetic::divide_device_constant_f32_c1,
    arithmetic::divide_device_constant_f32_c1_in_place
);
impl_float_device_constant_arithmetic_image!(
    C3,
    f32,
    arithmetic::add_device_constant_f32_c3,
    arithmetic::add_device_constant_f32_c3_in_place,
    arithmetic::subtract_device_constant_f32_c3,
    arithmetic::subtract_device_constant_f32_c3_in_place,
    arithmetic::multiply_device_constant_f32_c3,
    arithmetic::multiply_device_constant_f32_c3_in_place,
    arithmetic::divide_device_constant_f32_c3,
    arithmetic::divide_device_constant_f32_c3_in_place
);
impl_float_device_constant_arithmetic_image!(
    C4,
    f32,
    arithmetic::add_device_constant_f32_c4,
    arithmetic::add_device_constant_f32_c4_in_place,
    arithmetic::subtract_device_constant_f32_c4,
    arithmetic::subtract_device_constant_f32_c4_in_place,
    arithmetic::multiply_device_constant_f32_c4,
    arithmetic::multiply_device_constant_f32_c4_in_place,
    arithmetic::divide_device_constant_f32_c4,
    arithmetic::divide_device_constant_f32_c4_in_place
);
impl_float_device_constant_arithmetic_image!(
    AC4,
    f32,
    arithmetic::add_device_constant_f32_ac4,
    arithmetic::add_device_constant_f32_ac4_in_place,
    arithmetic::subtract_device_constant_f32_ac4,
    arithmetic::subtract_device_constant_f32_ac4_in_place,
    arithmetic::multiply_device_constant_f32_ac4,
    arithmetic::multiply_device_constant_f32_ac4_in_place,
    arithmetic::divide_device_constant_f32_ac4,
    arithmetic::divide_device_constant_f32_ac4_in_place
);
