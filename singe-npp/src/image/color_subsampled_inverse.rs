use super::*;

impl_jpeg_subsampled_planar_inverse!(yuv420_to_rgb_u8_p3, nppiYUV420ToRGB_8u_P3R_Ctx, 2, 2);
impl_jpeg_subsampled_planar_inverse!(yuv422_to_rgb_u8_p3, nppiYUV422ToRGB_8u_P3R_Ctx, 2, 1);
impl_jpeg_subsampled_packed_inverse!(
    yuv420_to_rgb_u8_p3_to_c3,
    C3,
    nppiYUV420ToRGB_8u_P3C3R_Ctx,
    2,
    2
);
impl_jpeg_subsampled_packed_inverse!(
    yuv422_to_rgb_u8_p3_to_c3,
    C3,
    nppiYUV422ToRGB_8u_P3C3R_Ctx,
    2,
    1
);
impl_jpeg_subsampled_packed_inverse!(
    yuv422_to_rgb_u8_p3_to_ac4,
    AC4,
    nppiYUV422ToRGB_8u_P3AC4R_Ctx,
    2,
    1
);
impl_jpeg_subsampled_packed_inverse!(
    yuv420_to_bgr_u8_p3_to_c3,
    C3,
    nppiYUV420ToBGR_8u_P3C3R_Ctx,
    2,
    2
);
impl_jpeg_subsampled_planar_inverse!(
    ycbcr420_to_rgb_jpeg_u8_p3,
    nppiYCbCr420ToRGB_JPEG_8u_P3R_Ctx,
    2,
    2
);
impl_jpeg_subsampled_planar_inverse!(
    ycbcr422_to_rgb_jpeg_u8_p3,
    nppiYCbCr422ToRGB_JPEG_8u_P3R_Ctx,
    2,
    1
);
impl_jpeg_subsampled_planar_inverse!(
    ycbcr411_to_rgb_jpeg_u8_p3,
    nppiYCbCr411ToRGB_JPEG_8u_P3R_Ctx,
    4,
    1
);
impl_jpeg_subsampled_planar_inverse!(
    ycbcr420_to_bgr_jpeg_u8_p3,
    nppiYCbCr420ToBGR_JPEG_8u_P3R_Ctx,
    2,
    2
);
impl_jpeg_subsampled_planar_inverse!(
    ycbcr422_to_bgr_jpeg_u8_p3,
    nppiYCbCr422ToBGR_JPEG_8u_P3R_Ctx,
    2,
    1
);
impl_jpeg_subsampled_planar_inverse!(
    ycbcr411_to_bgr_jpeg_u8_p3,
    nppiYCbCr411ToBGR_JPEG_8u_P3R_Ctx,
    4,
    1
);
impl_jpeg_subsampled_packed_inverse!(
    ycbcr420_to_rgb_jpeg_u8_p3_to_c3,
    C3,
    nppiYCbCr420ToRGB_JPEG_8u_P3C3R_Ctx,
    2,
    2
);
impl_jpeg_subsampled_packed_inverse!(
    ycbcr422_to_rgb_jpeg_u8_p3_to_c3,
    C3,
    nppiYCbCr422ToRGB_JPEG_8u_P3C3R_Ctx,
    2,
    1
);
impl_jpeg_subsampled_packed_inverse!(
    ycbcr411_to_rgb_jpeg_u8_p3_to_c3,
    C3,
    nppiYCbCr411ToRGB_JPEG_8u_P3C3R_Ctx,
    4,
    1
);
impl_jpeg_subsampled_packed_inverse!(
    ycbcr420_to_bgr_jpeg_u8_p3_to_c3,
    C3,
    nppiYCbCr420ToBGR_JPEG_8u_P3C3R_Ctx,
    2,
    2
);
impl_jpeg_subsampled_packed_inverse!(
    ycbcr422_to_bgr_jpeg_u8_p3_to_c3,
    C3,
    nppiYCbCr422ToBGR_JPEG_8u_P3C3R_Ctx,
    2,
    1
);
impl_jpeg_subsampled_packed_inverse!(
    ycbcr411_to_bgr_jpeg_u8_p3_to_c3,
    C3,
    nppiYCbCr411ToBGR_JPEG_8u_P3C3R_Ctx,
    4,
    1
);
impl_jpeg_subsampled_packed_inverse!(
    ycbcr422_to_rgb_u8_p3_to_c3,
    C3,
    nppiYCbCr422ToRGB_8u_P3C3R_Ctx,
    2,
    1
);
impl_jpeg_subsampled_packed_inverse!(
    ycbcr422_to_bgr_u8_p3_to_c3,
    C3,
    nppiYCbCr422ToBGR_8u_P3C3R_Ctx,
    2,
    1
);
impl_jpeg_subsampled_packed_inverse!(
    ycbcr411_to_rgb_u8_p3_to_c3,
    C3,
    nppiYCbCr411ToRGB_8u_P3C3R_Ctx,
    4,
    1
);
impl_jpeg_subsampled_packed_inverse!(
    ycbcr411_to_bgr_u8_p3_to_c3,
    C3,
    nppiYCbCr411ToBGR_8u_P3C3R_Ctx,
    4,
    1
);
impl_jpeg_subsampled_packed_inverse_constant_alpha!(
    ycbcr411_to_rgb_u8_p3_to_c4,
    nppiYCbCr411ToRGB_8u_P3C4R_Ctx,
    4,
    1
);
impl_jpeg_subsampled_packed_inverse_constant_alpha!(
    ycbcr411_to_bgr_u8_p3_to_c4,
    nppiYCbCr411ToBGR_8u_P3C4R_Ctx,
    4,
    1
);
impl_jpeg_subsampled_packed_inverse!(
    ycbcr420_to_rgb_u8_p3_to_c3,
    C3,
    nppiYCbCr420ToRGB_8u_P3C3R_Ctx,
    2,
    2
);
impl_jpeg_subsampled_packed_inverse!(
    ycbcr420_to_bgr_u8_p3_to_c3,
    C3,
    nppiYCbCr420ToBGR_8u_P3C3R_Ctx,
    2,
    2
);
impl_jpeg_subsampled_packed_inverse_constant_alpha!(
    ycbcr420_to_bgr_u8_p3_to_c4,
    nppiYCbCr420ToBGR_8u_P3C4R_Ctx,
    2,
    2
);
impl_jpeg_subsampled_packed_inverse!(
    ycbcr420_to_bgr_709csc_u8_p3_to_c3,
    C3,
    nppiYCbCr420ToBGR_709CSC_8u_P3C3R_Ctx,
    2,
    2
);
impl_jpeg_subsampled_packed_inverse_constant_alpha!(
    ycbcr420_to_bgr_709hdtv_u8_p3_to_c4,
    nppiYCbCr420ToBGR_709HDTV_8u_P3C4R_Ctx,
    2,
    2
);
impl_jpeg_subsampled_packed_inverse!(
    yuv420_to_rgb_u8_p3_to_c4,
    C4,
    nppiYUV420ToRGB_8u_P3C4R_Ctx,
    2,
    2
);
impl_jpeg_subsampled_packed_inverse!(
    yuv420_to_bgr_u8_p3_to_c4,
    C4,
    nppiYUV420ToBGR_8u_P3C4R_Ctx,
    2,
    2
);

macro_rules! impl_jpeg_subsampled_planar_inverse_dispatch {
    ($trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait: DataTypeLike {
            fn $method(
                stream_context: &StreamContext,
                source_y: &ImageView<'_, Self, C1>,
                source_cb: &ImageView<'_, Self, C1>,
                source_cr: &ImageView<'_, Self, C1>,
                destination_0: &mut ImageViewMut<'_, Self, C1>,
                destination_1: &mut ImageViewMut<'_, Self, C1>,
                destination_2: &mut ImageViewMut<'_, Self, C1>,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source_y: &ImageView<'_, T, C1>,
            source_cb: &ImageView<'_, T, C1>,
            source_cr: &ImageView<'_, T, C1>,
            destination_0: &mut ImageViewMut<'_, T, C1>,
            destination_1: &mut ImageViewMut<'_, T, C1>,
            destination_2: &mut ImageViewMut<'_, T, C1>,
        ) -> Result<()>
        where
            T: $trait,
        {
            T::$method(
                stream_context,
                source_y,
                source_cb,
                source_cr,
                destination_0,
                destination_1,
                destination_2,
            )
        }

        $(
            impl $trait for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source_y: &ImageView<'_, Self, C1>,
                    source_cb: &ImageView<'_, Self, C1>,
                    source_cr: &ImageView<'_, Self, C1>,
                    destination_0: &mut ImageViewMut<'_, Self, C1>,
                    destination_1: &mut ImageViewMut<'_, Self, C1>,
                    destination_2: &mut ImageViewMut<'_, Self, C1>,
                ) -> Result<()> {
                    $direct(
                        stream_context,
                        source_y,
                        source_cb,
                        source_cr,
                        destination_0,
                        destination_1,
                        destination_2,
                    )
                }
            }
        )*
    };
}

macro_rules! impl_jpeg_subsampled_packed_inverse_dispatch {
    ($trait:ident, $method:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait: DataTypeLike {
            fn $method(
                stream_context: &StreamContext,
                source_y: &ImageView<'_, Self, C1>,
                source_cb: &ImageView<'_, Self, C1>,
                source_cr: &ImageView<'_, Self, C1>,
                destination: &mut ImageViewMut<'_, Self, $layout>,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source_y: &ImageView<'_, T, C1>,
            source_cb: &ImageView<'_, T, C1>,
            source_cr: &ImageView<'_, T, C1>,
            destination: &mut ImageViewMut<'_, T, $layout>,
        ) -> Result<()>
        where
            T: $trait,
        {
            T::$method(stream_context, source_y, source_cb, source_cr, destination)
        }

        $(
            impl $trait for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source_y: &ImageView<'_, Self, C1>,
                    source_cb: &ImageView<'_, Self, C1>,
                    source_cr: &ImageView<'_, Self, C1>,
                    destination: &mut ImageViewMut<'_, Self, $layout>,
                ) -> Result<()> {
                    $direct(stream_context, source_y, source_cb, source_cr, destination)
                }
            }
        )*
    };
}

impl_jpeg_subsampled_planar_inverse_dispatch!(Yuv420ToRgbP3, yuv420_to_rgb_p3, yuv420_to_rgb_p3, [
    u8 => yuv420_to_rgb_u8_p3
]);
impl_jpeg_subsampled_planar_inverse_dispatch!(Yuv422ToRgbP3, yuv422_to_rgb_p3, yuv422_to_rgb_p3, [
    u8 => yuv422_to_rgb_u8_p3
]);
impl_jpeg_subsampled_planar_inverse_dispatch!(Ycbcr420ToRgbJpegP3, ycbcr420_to_rgb_jpeg_p3, ycbcr420_to_rgb_jpeg_p3, [
    u8 => ycbcr420_to_rgb_jpeg_u8_p3
]);
impl_jpeg_subsampled_planar_inverse_dispatch!(Ycbcr422ToRgbJpegP3, ycbcr422_to_rgb_jpeg_p3, ycbcr422_to_rgb_jpeg_p3, [
    u8 => ycbcr422_to_rgb_jpeg_u8_p3
]);
impl_jpeg_subsampled_planar_inverse_dispatch!(Ycbcr411ToRgbJpegP3, ycbcr411_to_rgb_jpeg_p3, ycbcr411_to_rgb_jpeg_p3, [
    u8 => ycbcr411_to_rgb_jpeg_u8_p3
]);
impl_jpeg_subsampled_planar_inverse_dispatch!(Ycbcr420ToBgrJpegP3, ycbcr420_to_bgr_jpeg_p3, ycbcr420_to_bgr_jpeg_p3, [
    u8 => ycbcr420_to_bgr_jpeg_u8_p3
]);
impl_jpeg_subsampled_planar_inverse_dispatch!(Ycbcr422ToBgrJpegP3, ycbcr422_to_bgr_jpeg_p3, ycbcr422_to_bgr_jpeg_p3, [
    u8 => ycbcr422_to_bgr_jpeg_u8_p3
]);
impl_jpeg_subsampled_planar_inverse_dispatch!(Ycbcr411ToBgrJpegP3, ycbcr411_to_bgr_jpeg_p3, ycbcr411_to_bgr_jpeg_p3, [
    u8 => ycbcr411_to_bgr_jpeg_u8_p3
]);

impl_jpeg_subsampled_packed_inverse_dispatch!(Yuv420ToRgbP3ToC3, yuv420_to_rgb_p3_to_c3, yuv420_to_rgb_p3_to_c3, C3, [
    u8 => yuv420_to_rgb_u8_p3_to_c3
]);
impl_jpeg_subsampled_packed_inverse_dispatch!(Yuv422ToRgbP3ToC3, yuv422_to_rgb_p3_to_c3, yuv422_to_rgb_p3_to_c3, C3, [
    u8 => yuv422_to_rgb_u8_p3_to_c3
]);
impl_jpeg_subsampled_packed_inverse_dispatch!(Yuv420ToBgrP3ToC3, yuv420_to_bgr_p3_to_c3, yuv420_to_bgr_p3_to_c3, C3, [
    u8 => yuv420_to_bgr_u8_p3_to_c3
]);
impl_jpeg_subsampled_packed_inverse_dispatch!(Ycbcr420ToRgbJpegP3ToC3, ycbcr420_to_rgb_jpeg_p3_to_c3, ycbcr420_to_rgb_jpeg_p3_to_c3, C3, [
    u8 => ycbcr420_to_rgb_jpeg_u8_p3_to_c3
]);
impl_jpeg_subsampled_packed_inverse_dispatch!(Ycbcr422ToRgbJpegP3ToC3, ycbcr422_to_rgb_jpeg_p3_to_c3, ycbcr422_to_rgb_jpeg_p3_to_c3, C3, [
    u8 => ycbcr422_to_rgb_jpeg_u8_p3_to_c3
]);
impl_jpeg_subsampled_packed_inverse_dispatch!(Ycbcr411ToRgbJpegP3ToC3, ycbcr411_to_rgb_jpeg_p3_to_c3, ycbcr411_to_rgb_jpeg_p3_to_c3, C3, [
    u8 => ycbcr411_to_rgb_jpeg_u8_p3_to_c3
]);
impl_jpeg_subsampled_packed_inverse_dispatch!(Ycbcr420ToBgrJpegP3ToC3, ycbcr420_to_bgr_jpeg_p3_to_c3, ycbcr420_to_bgr_jpeg_p3_to_c3, C3, [
    u8 => ycbcr420_to_bgr_jpeg_u8_p3_to_c3
]);
impl_jpeg_subsampled_packed_inverse_dispatch!(Ycbcr422ToBgrJpegP3ToC3, ycbcr422_to_bgr_jpeg_p3_to_c3, ycbcr422_to_bgr_jpeg_p3_to_c3, C3, [
    u8 => ycbcr422_to_bgr_jpeg_u8_p3_to_c3
]);
impl_jpeg_subsampled_packed_inverse_dispatch!(Ycbcr411ToBgrJpegP3ToC3, ycbcr411_to_bgr_jpeg_p3_to_c3, ycbcr411_to_bgr_jpeg_p3_to_c3, C3, [
    u8 => ycbcr411_to_bgr_jpeg_u8_p3_to_c3
]);
impl_jpeg_subsampled_packed_inverse_dispatch!(Ycbcr422ToRgbP3ToC3, ycbcr422_to_rgb_p3_to_c3, ycbcr422_to_rgb_p3_to_c3, C3, [
    u8 => ycbcr422_to_rgb_u8_p3_to_c3
]);
impl_jpeg_subsampled_packed_inverse_dispatch!(Ycbcr422ToBgrP3ToC3, ycbcr422_to_bgr_p3_to_c3, ycbcr422_to_bgr_p3_to_c3, C3, [
    u8 => ycbcr422_to_bgr_u8_p3_to_c3
]);
impl_jpeg_subsampled_packed_inverse_dispatch!(Ycbcr411ToRgbP3ToC3, ycbcr411_to_rgb_p3_to_c3, ycbcr411_to_rgb_p3_to_c3, C3, [
    u8 => ycbcr411_to_rgb_u8_p3_to_c3
]);
impl_jpeg_subsampled_packed_inverse_dispatch!(Ycbcr411ToBgrP3ToC3, ycbcr411_to_bgr_p3_to_c3, ycbcr411_to_bgr_p3_to_c3, C3, [
    u8 => ycbcr411_to_bgr_u8_p3_to_c3
]);
impl_jpeg_subsampled_packed_inverse_dispatch!(Ycbcr420ToRgbP3ToC3, ycbcr420_to_rgb_p3_to_c3, ycbcr420_to_rgb_p3_to_c3, C3, [
    u8 => ycbcr420_to_rgb_u8_p3_to_c3
]);
impl_jpeg_subsampled_packed_inverse_dispatch!(Ycbcr420ToBgrP3ToC3, ycbcr420_to_bgr_p3_to_c3, ycbcr420_to_bgr_p3_to_c3, C3, [
    u8 => ycbcr420_to_bgr_u8_p3_to_c3
]);
impl_jpeg_subsampled_packed_inverse_dispatch!(
    Ycbcr420ToBgr709CscP3ToC3,
    ycbcr420_to_bgr_709csc_p3_to_c3,
    ycbcr420_to_bgr_709csc_p3_to_c3,
    C3,
    [u8 => ycbcr420_to_bgr_709csc_u8_p3_to_c3]
);
