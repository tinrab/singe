use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        arithmetic,
        view::{C1, C3, ImageView, ImageViewMut},
    },
};

use super::super::{ImagePipeline, operation_traits::DeviceConstantArithmeticImage};

impl_scaled_device_constant_arithmetic_image!(
    i32,
    C1,
    i32,
    arithmetic::add_device_constant_i32_c1,
    arithmetic::add_device_constant_i32_c1_in_place,
    arithmetic::subtract_device_constant_i32_c1,
    arithmetic::subtract_device_constant_i32_c1_in_place,
    arithmetic::multiply_device_constant_i32_c1,
    arithmetic::multiply_device_constant_i32_c1_in_place,
    arithmetic::divide_device_constant_i32_c1,
    arithmetic::divide_device_constant_i32_c1_in_place
);
impl_scaled_device_constant_arithmetic_image!(
    i32,
    C3,
    i32,
    arithmetic::add_device_constant_i32_c3,
    arithmetic::add_device_constant_i32_c3_in_place,
    arithmetic::subtract_device_constant_i32_c3,
    arithmetic::subtract_device_constant_i32_c3_in_place,
    arithmetic::multiply_device_constant_i32_c3,
    arithmetic::multiply_device_constant_i32_c3_in_place,
    arithmetic::divide_device_constant_i32_c3,
    arithmetic::divide_device_constant_i32_c3_in_place
);
