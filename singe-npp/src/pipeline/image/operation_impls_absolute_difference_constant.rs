use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        arithmetic,
        view::{C1, ImageView, ImageViewMut},
    },
};

use super::super::{ImagePipeline, operation_traits::AbsoluteDifferenceConstantImage};

impl_absolute_difference_constant_image!(
    u8,
    arithmetic::absolute_difference_constant_u8_c1,
    arithmetic::absolute_difference_device_constant_u8_c1
);
impl_absolute_difference_constant_image!(
    u16,
    arithmetic::absolute_difference_constant_u16_c1,
    arithmetic::absolute_difference_device_constant_u16_c1
);
impl_absolute_difference_constant_image!(
    f32,
    arithmetic::absolute_difference_constant_f32_c1,
    arithmetic::absolute_difference_device_constant_f32_c1
);
