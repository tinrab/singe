use singe_cuda::types::f16;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        arithmetic,
        view::{C1, C3, C4, ImageView, ImageViewMut},
    },
};

use super::super::{ImagePipeline, operation_traits::*};

impl_absolute_image!(
    f16,
    C1,
    arithmetic::absolute_f16_c1,
    arithmetic::absolute_f16_c1_in_place
);
impl_absolute_image!(
    f16,
    C3,
    arithmetic::absolute_f16_c3,
    arithmetic::absolute_f16_c3_in_place
);
impl_absolute_image!(
    f16,
    C4,
    arithmetic::absolute_f16_c4,
    arithmetic::absolute_f16_c4_in_place
);
