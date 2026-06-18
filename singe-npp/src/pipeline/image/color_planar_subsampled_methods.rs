use crate::{
    context::StreamContext,
    error::Result,
    image::{
        color,
        view::{C1, C2, C3, ImageView, ImageViewMut},
    },
    pipeline::{ImageAllocator, Workspace},
};

use super::super::ImagePipeline;

#[path = "color_planar_subsampled_source_methods.rs"]
mod source_methods;

impl<'a> ImagePipeline<'a, u8, C3>
where
    Workspace: ImageAllocator<u8, C3>,
{
    impl_static_planar3_to_planar2_convert!(
        ycbcr411_planar_to_ycbcr420_p2_into,
        color::ycbcr411_to_ycbcr420_u8_p3_to_p2
    );
    impl_static_planar3_to_planar3_convert!(
        ycbcr411_planar_to_ycbcr420_into,
        u8,
        color::ycbcr411_to_ycbcr420_u8_p3_to_p3
    );
    impl_static_planar_to_c2_convert!(
        ycbcr411_planar_to_ycbcr422_into,
        color::ycbcr411_to_ycbcr422_u8_p3_to_c2
    );
    impl_static_planar3_to_planar3_convert!(
        ycbcr411_planar_to_ycbcr422_p3_into,
        u8,
        color::ycbcr411_to_ycbcr422_u8_p3_to_p3
    );
    impl_static_planar_to_c2_convert!(
        ycbcr411_planar_to_ycrcb422_into,
        color::ycbcr411_to_ycrcb422_u8_p3_to_c2
    );
    impl_static_planar3_to_planar3_convert!(
        ycbcr411_planar_to_ycrcb422_p3_into,
        u8,
        color::ycbcr411_to_ycrcb422_u8_p3_to_p3
    );
    impl_static_planar3_to_planar2_convert!(
        ycbcr411_planar_to_p2_into,
        color::ycbcr411_u8_p3_to_p2
    );
    impl_static_planar3_to_planar2_convert!(
        ycbcr420_planar_to_ycbcr411_into,
        color::ycbcr420_to_ycbcr411_u8_p3_to_p2
    );
    impl_static_planar3_to_planar3_convert!(
        ycbcr420_planar_to_ycbcr422_into,
        u8,
        color::ycbcr420_to_ycbcr422_u8_p3_to_p3
    );
    impl_static_planar3_to_planar2_convert!(
        ycbcr420_planar_to_p2_into,
        color::ycbcr420_u8_p3_to_p2
    );
    impl_static_planar3_to_planar2_convert!(
        ycbcr422_planar_to_ycbcr411_p2_into,
        color::ycbcr422_to_ycbcr411_u8_p3_to_p2
    );
    impl_static_planar3_to_planar3_convert!(
        ycbcr422_planar_to_ycbcr411_into,
        u8,
        color::ycbcr422_to_ycbcr411_u8_p3_to_p3
    );
    impl_static_planar3_to_planar2_convert!(
        ycbcr422_planar_to_ycbcr420_p2_into,
        color::ycbcr422_to_ycbcr420_u8_p3_to_p2
    );
    impl_static_planar3_to_planar3_convert!(
        ycbcr422_planar_to_ycbcr420_into,
        u8,
        color::ycbcr422_to_ycbcr420_u8_p3_to_p3
    );
    impl_static_planar_to_c2_convert!(
        ycbcr422_planar_to_ycrcb422_into,
        color::ycbcr422_to_ycrcb422_u8_p3_to_c2
    );
    impl_static_planar_to_c2_convert!(
        ycrcb420_planar_to_cbycr422_into,
        color::ycrcb420_to_cbycr422_u8_p3_to_c2
    );
    impl_static_planar3_to_planar2_convert!(
        ycrcb420_planar_to_ycbcr411_into,
        color::ycrcb420_to_ycbcr411_u8_p3_to_p2
    );
    impl_static_planar3_to_planar2_convert!(
        ycrcb420_planar_to_ycbcr420_into,
        color::ycrcb420_to_ycbcr420_u8_p3_to_p2
    );
    impl_static_planar_to_c2_convert!(
        ycrcb420_planar_to_ycbcr422_into,
        color::ycrcb420_to_ycbcr422_u8_p3_to_c2
    );
    impl_static_planar3_to_planar3_convert!(
        ycrcb420_planar_to_ycbcr422_p3_into,
        u8,
        color::ycrcb420_to_ycbcr422_u8_p3_to_p3
    );
}
