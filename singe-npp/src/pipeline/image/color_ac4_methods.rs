use crate::{
    context::StreamContext,
    error::Result,
    image::{
        color,
        view::{AC4, C2, ImageView, ImageViewMut, PlanarImageViewMut},
    },
    pipeline::{ImageAllocator, Workspace},
};

use super::ImagePipeline;

#[path = "color_ac4_same_layout_methods.rs"]
mod same_layout_methods;

impl<'a> ImagePipeline<'a, u8, AC4>
where
    Workspace: ImageAllocator<u8, AC4>,
{
    impl_static_different_layout_convert!(
        bgr_to_cbycr422_into,
        u8,
        AC4,
        C2,
        color::bgr_to_cbycr422_u8_ac4_to_c2
    );
    impl_static_different_layout_convert!(
        bgr_to_cbycr422_709hdtv_into,
        u8,
        AC4,
        C2,
        color::bgr_to_cbycr422_709hdtv_u8_ac4_to_c2
    );
    impl_static_different_layout_convert!(
        bgr_to_ycbcr422_into,
        u8,
        AC4,
        C2,
        color::bgr_to_ycbcr422_u8_ac4_to_c2
    );
    impl_static_packed_to_planar_convert!(
        rgb_to_ycbcr_planar_into,
        u8,
        AC4,
        3,
        color::rgb_to_ycbcr_u8_ac4_to_p3
    );
    impl_static_packed_to_planar_convert!(
        bgr_to_ycbcr_planar_into,
        u8,
        AC4,
        3,
        color::bgr_to_ycbcr_u8_ac4_to_p3
    );
    impl_static_packed_to_planar_convert!(
        bgr_to_ycbcr_planar4_into,
        u8,
        AC4,
        4,
        color::bgr_to_ycbcr_u8_ac4_to_p4
    );
}
