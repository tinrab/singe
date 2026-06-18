use crate::{
    context::StreamContext,
    error::Result,
    image::{
        arithmetic,
        view::{AC4, ImageView, ImageViewMut},
    },
};

use super::super::{ImagePipeline, operation_traits::AlphaPremultiplyImage};

impl_alpha_premultiply_image!(
    u8,
    arithmetic::alpha_premultiply_u8_ac4,
    arithmetic::alpha_premultiply_u8_ac4_in_place
);
impl_alpha_premultiply_image!(
    u16,
    arithmetic::alpha_premultiply_u16_ac4,
    arithmetic::alpha_premultiply_u16_ac4_in_place
);
