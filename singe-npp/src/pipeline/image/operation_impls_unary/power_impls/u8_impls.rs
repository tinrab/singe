use crate::{
    context::StreamContext,
    error::Result,
    image::{
        arithmetic,
        view::{C1, C3, ImageOutputView, ImageView, ImageViewMut},
    },
    pipeline::image::{
        ImagePipeline,
        operation_traits::{SquareImage, SquareRootImage},
    },
};

impl_scaled_unary_power_image!(
    u8,
    C1,
    arithmetic::square_u8_c1,
    arithmetic::square_u8_c1_in_place,
    arithmetic::square_root_u8_c1,
    arithmetic::square_root_u8_c1_in_place
);

impl_scaled_unary_power_image!(
    u8,
    C3,
    arithmetic::square_u8_c3,
    arithmetic::square_u8_c3_in_place,
    arithmetic::square_root_u8_c3,
    arithmetic::square_root_u8_c3_in_place
);
