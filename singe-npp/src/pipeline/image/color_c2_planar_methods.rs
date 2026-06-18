use crate::{
    context::StreamContext,
    error::Result,
    image::{
        color,
        view::{C1, C2, ImageView, ImageViewMut},
    },
    pipeline::{ImageAllocator, Workspace},
};

use super::ImagePipeline;

impl<'a> ImagePipeline<'a, u8, C2>
where
    Workspace: ImageAllocator<u8, C2>,
{
    impl_static_c2_to_planar_convert!(ycbcr422_to_rgb_planar_into, color::ycbcr422_to_rgb_c2_to_p3);
    impl_static_c2_to_planar_convert!(ycrcb422_to_rgb_planar_into, color::ycrcb422_to_rgb_c2_to_p3);
    impl_static_c2_to_planar3_convert!(
        cbycr422_to_ycbcr411_into,
        color::cbycr422_to_ycbcr411_u8_c2_to_p3
    );
    impl_static_c2_to_planar2_convert!(
        cbycr422_to_ycbcr420_p2_into,
        color::cbycr422_to_ycbcr420_u8_c2_to_p2
    );
    impl_static_c2_to_planar3_convert!(
        cbycr422_to_ycbcr420_into,
        color::cbycr422_to_ycbcr420_u8_c2_to_p3
    );
    impl_static_c2_to_planar3_convert!(
        cbycr422_to_ycbcr422_planar_into,
        color::cbycr422_to_ycbcr422_u8_c2_to_p3
    );
    impl_static_c2_to_planar3_convert!(
        cbycr422_to_ycrcb420_into,
        color::cbycr422_to_ycrcb420_u8_c2_to_p3
    );
    impl_static_c2_to_planar2_convert!(
        ycbcr422_to_ycbcr411_p2_into,
        color::ycbcr422_to_ycbcr411_u8_c2_to_p2
    );
    impl_static_c2_to_planar3_convert!(
        ycbcr422_to_ycbcr411_into,
        color::ycbcr422_to_ycbcr411_u8_c2_to_p3
    );
    impl_static_c2_to_planar2_convert!(
        ycbcr422_to_ycbcr420_p2_into,
        color::ycbcr422_to_ycbcr420_u8_c2_to_p2
    );
    impl_static_c2_to_planar3_convert!(
        ycbcr422_to_ycbcr420_into,
        color::ycbcr422_to_ycbcr420_u8_c2_to_p3
    );
    impl_static_c2_to_planar3_convert!(
        ycbcr422_to_ycrcb420_into,
        color::ycbcr422_to_ycrcb420_u8_c2_to_p3
    );
    impl_static_c2_to_planar3_convert!(ycbcr422_to_planar_into, color::ycbcr422_u8_c2_to_p3);
    impl_static_c2_to_planar3_convert!(
        ycrcb422_to_ycbcr411_into,
        color::ycrcb422_to_ycbcr411_u8_c2_to_p3
    );
    impl_static_c2_to_planar3_convert!(
        ycrcb422_to_ycbcr420_into,
        color::ycrcb422_to_ycbcr420_u8_c2_to_p3
    );
    impl_static_c2_to_planar3_convert!(
        ycrcb422_to_ycbcr422_into,
        color::ycrcb422_to_ycbcr422_u8_c2_to_p3
    );
}
