use crate::{
    context::StreamContext,
    error::Result,
    image::{
        color,
        view::{C1, C2, C3, C4, ImageView, ImageViewMut, PlanarImageView, PlanarImageViewMut},
    },
    pipeline::{ImageAllocator, Workspace},
    types::{ColorSpace, Point},
};

use super::ImagePipeline;

#[path = "color_planar_subsampled_methods.rs"]
mod subsampled_methods;

impl<'a> ImagePipeline<'a, u8, C3>
where
    Workspace: ImageAllocator<u8, C3>,
{
    impl_static_packed_to_planar_convert!(
        rgb_to_ycbcr_planar_into,
        u8,
        C3,
        3,
        color::rgb_to_ycbcr_u8_c3_to_p3
    );
    impl_static_packed_to_planar_convert!(
        bgr_to_ycbcr_planar_into,
        u8,
        C3,
        3,
        color::bgr_to_ycbcr_u8_c3_to_p3
    );
    impl_static_packed_to_planar_convert!(
        bgr_to_yuv_planar_into,
        u8,
        C3,
        3,
        color::bgr_to_yuv_u8_c3_to_p3
    );
    impl_static_packed_to_planar_convert!(
        bgr_to_hls_planar_into,
        u8,
        C3,
        3,
        color::bgr_to_hls_u8_c3_to_p3
    );
    impl_static_planar_to_packed_convert!(
        bgr_planar_to_hls_into,
        u8,
        3,
        C3,
        color::bgr_to_hls_u8_p3_to_c3
    );
    impl_static_planar_to_packed_convert!(
        hls_planar_to_bgr_into,
        u8,
        3,
        C3,
        color::hls_to_bgr_u8_p3_to_c3
    );
    impl_static_planar_to_packed_convert!(
        ycbcr_planar_to_bgr_709csc_into,
        u8,
        3,
        C3,
        color::ycbcr_to_bgr_709csc_u8_p3_to_c3
    );
    impl_static_planar_to_packed_constant_alpha_convert!(
        ycbcr_planar_to_rgb_c4_into,
        u8,
        3,
        color::ycbcr_to_rgb_u8_p3_to_c4
    );
    impl_static_planar_to_packed_constant_alpha_convert!(
        ycbcr_planar_to_bgr_c4_into,
        u8,
        3,
        color::ycbcr_to_bgr_u8_p3_to_c4
    );
    impl_static_planar_to_packed_constant_alpha_convert!(
        ycbcr_planar_to_bgr_709csc_c4_into,
        u8,
        3,
        color::ycbcr_to_bgr_709csc_u8_p3_to_c4
    );
    impl_static_planar_to_c2_convert!(rgb_planar_to_ycbcr422_into, color::rgb_to_ycbcr422_p3_to_c2);
    impl_static_rgb_to_uyvp_planar!(
        rgb_to_uyvp_10u_planes_into,
        u8,
        C3,
        color::rgb_to_uyvp_10u_u8_c3_to_p3
    );
    impl_static_rgb_planar_to_uyvp!(
        rgb_planar_to_uyvp_10u_into,
        u8,
        color::rgb_to_uyvp_10u_u8_p3
    );
    impl_static_uyvp_to_rgb_planar!(
        uyvp_10u_to_rgb_planar_into,
        u8,
        color::uyvp_10u_to_rgb_u8_p3
    );
}
