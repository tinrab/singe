use singe_cuda::types::f16;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        arithmetic,
        view::{C1, C3, C4, ImageView, ImageViewMut},
    },
};

use super::super::{ImagePipeline, operation_traits::AbsoluteDifferenceImage};

impl_absolute_difference_image!(u8, C1, arithmetic::absolute_difference_u8_c1);
impl_absolute_difference_image!(u8, C3, arithmetic::absolute_difference_u8_c3);
impl_absolute_difference_image!(u8, C4, arithmetic::absolute_difference_u8_c4);
impl_absolute_difference_image!(u16, C1, arithmetic::absolute_difference_u16_c1);
impl_absolute_difference_image!(f16, C1, arithmetic::absolute_difference_f16_c1);
impl_absolute_difference_image!(f32, C1, arithmetic::absolute_difference_f32_c1);
