use crate::{
    context::StreamContext,
    error::Result,
    image::{
        color,
        view::{C1, C3, ImageView, ImageViewMut},
    },
    pipeline::{ImageAllocator, Workspace},
};

use super::super::super::ImagePipeline;

impl<'a> ImagePipeline<'a, u8, C3>
where
    Workspace: ImageAllocator<u8, C3>,
{
    impl_static_planar3_to_planar3_convert!(
        rgb_planar_to_ycbcr411_jpeg_into,
        u8,
        color::rgb_to_ycbcr411_jpeg_u8_p3
    );
    impl_static_planar3_to_planar3_convert!(
        rgb_planar_to_ycbcr420_jpeg_into,
        u8,
        color::rgb_to_ycbcr420_jpeg_u8_p3
    );
    impl_static_planar3_to_planar3_convert!(
        rgb_planar_to_ycbcr422_jpeg_into,
        u8,
        color::rgb_to_ycbcr422_jpeg_u8_p3
    );
    impl_static_planar3_to_planar3_convert!(
        bgr_planar_to_ycbcr411_jpeg_into,
        u8,
        color::bgr_to_ycbcr411_jpeg_u8_p3
    );
    impl_static_planar3_to_planar3_convert!(
        bgr_planar_to_ycbcr420_jpeg_into,
        u8,
        color::bgr_to_ycbcr420_jpeg_u8_p3
    );
    impl_static_planar3_to_planar3_convert!(
        bgr_planar_to_ycbcr422_jpeg_into,
        u8,
        color::bgr_to_ycbcr422_jpeg_u8_p3
    );
    impl_static_planar3_to_planar3_convert!(
        rgb_planar_to_yuv420_into,
        u8,
        color::rgb_to_yuv420_u8_p3
    );
    impl_static_planar3_to_planar3_convert!(
        rgb_planar_to_yuv422_into,
        u8,
        color::rgb_to_yuv422_u8_p3
    );
}
