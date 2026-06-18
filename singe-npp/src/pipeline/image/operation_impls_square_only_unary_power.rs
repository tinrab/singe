use crate::{
    context::StreamContext,
    error::Result,
    image::{
        arithmetic,
        view::{C4, ImageView, ImageViewMut},
    },
};

use super::{ImagePipeline, SquareImage};

impl_scaled_square_image!(
    u8,
    C4,
    arithmetic::square_u8_c4,
    arithmetic::square_u8_c4_in_place
);
impl_scaled_square_image!(
    u16,
    C4,
    arithmetic::square_u16_c4,
    arithmetic::square_u16_c4_in_place
);
impl_scaled_square_image!(
    i16,
    C4,
    arithmetic::square_i16_c4,
    arithmetic::square_i16_c4_in_place
);
