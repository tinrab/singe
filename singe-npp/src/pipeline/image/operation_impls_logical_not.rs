use crate::{
    context::StreamContext,
    error::Result,
    image::{
        arithmetic,
        view::{AC4, C1, C3, C4, ImageView, ImageViewMut},
    },
};

use super::super::{ImagePipeline, operation_traits::LogicalNotImage};

impl_logical_not_image!(
    C1,
    arithmetic::logical_not_u8_c1,
    arithmetic::logical_not_u8_c1_in_place
);
impl_logical_not_image!(
    C3,
    arithmetic::logical_not_u8_c3,
    arithmetic::logical_not_u8_c3_in_place
);
impl_logical_not_image!(
    C4,
    arithmetic::logical_not_u8_c4,
    arithmetic::logical_not_u8_c4_in_place
);
impl_logical_not_image!(
    AC4,
    arithmetic::logical_not_u8_ac4,
    arithmetic::logical_not_u8_ac4_in_place
);
