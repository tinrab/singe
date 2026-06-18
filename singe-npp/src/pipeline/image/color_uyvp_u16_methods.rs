use crate::{
    context::StreamContext,
    error::Result,
    image::{
        color,
        view::{AC4, C1, C2, C3, C4, ImageView, ImageViewMut},
    },
    pipeline::{ImageAllocator, Workspace},
    types::{ColorSpace, Point},
};

use super::{ColorTwistMatrix, ImageBacking, ImagePipeline};

#[path = "color_uyvp_u16_packed_methods.rs"]
mod packed_methods;

impl<'a> ImagePipeline<'a, u16, C3>
where
    Workspace: ImageAllocator<u8, C3>,
{
    impl_static_rgb_planar_to_uyvp!(
        rgb_planar_to_uyvp_10u_into,
        u16,
        color::rgb_to_uyvp_10u_u16_p3
    );
    impl_static_uyvp_to_rgb_planar!(
        uyvp_10u_to_rgb_planar_into,
        u16,
        color::uyvp_10u_to_rgb_u16_p3
    );
    impl_static_different_layout_twist!(
        rgb_to_yuv422_color_twist_into,
        u16,
        C3,
        C2,
        color::rgb_to_yuv422_u16_color_twist_c3_to_c2
    );
    impl_static_packed_to_planar3_twist!(
        rgb_to_yuv422_color_twist_planar_into,
        u16,
        C3,
        color::rgb_to_yuv422_u16_color_twist_c3_to_p3
    );
    impl_static_packed_to_planar3_twist!(
        rgb_to_yuv420_color_twist_planar_into,
        u16,
        C3,
        color::rgb_to_yuv420_u16_color_twist_c3_to_p3
    );
    impl_static_planar3_to_planar3_twist!(
        rgb_planar_to_yuv422_color_twist_into,
        u16,
        color::rgb_to_yuv422_u16_color_twist_p3
    );
    impl_static_planar3_to_planar3_twist!(
        rgb_planar_to_yuv420_color_twist_into,
        u16,
        color::rgb_to_yuv420_u16_color_twist_p3
    );
    impl_static_planar3_to_planar3_twist!(
        yuv422_planar_to_rgb_color_twist_into,
        u16,
        color::yuv422_to_rgb_u16_color_twist_p3
    );
    impl_static_planar3_to_packed_twist!(
        yuv422_planar_to_rgb_color_twist_packed_into,
        u16,
        C3,
        color::yuv422_to_rgb_u16_color_twist_p3_to_c3
    );
    impl_static_planar3_to_ac4_twist!(
        yuv422_planar_to_rgba_color_twist_into,
        u16,
        color::yuv422_to_rgb_u16_color_twist_p3_to_ac4
    );
    impl_static_planar3_to_planar3_twist!(
        yuv420_planar_to_rgb_color_twist_into,
        u16,
        color::yuv420_to_rgb_u16_color_twist_p3
    );
    impl_static_planar3_to_packed_twist!(
        yuv420_planar_to_rgb_color_twist_packed_into,
        u16,
        C3,
        color::yuv420_to_rgb_u16_color_twist_p3_to_c3
    );
    impl_static_planar3_to_packed_twist!(
        yuv420_planar_to_rgb_color_twist_c4_into,
        u16,
        C4,
        color::yuv420_to_rgb_u16_color_twist_p3_to_c4
    );
    impl_static_planar3_to_ac4_twist!(
        yuv420_planar_to_rgba_color_twist_into,
        u16,
        color::yuv420_to_rgb_u16_color_twist_p3_to_ac4
    );
}
