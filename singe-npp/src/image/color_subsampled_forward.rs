use super::*;

impl_jpeg_subsampled_planar_forward!(
    rgb_to_ycbcr420_jpeg_u8_p3,
    nppiRGBToYCbCr420_JPEG_8u_P3R_Ctx,
    2,
    2
);
impl_jpeg_subsampled_planar_forward!(
    rgb_to_ycbcr422_jpeg_u8_p3,
    nppiRGBToYCbCr422_JPEG_8u_P3R_Ctx,
    2,
    1
);
impl_jpeg_subsampled_planar_forward!(
    rgb_to_ycbcr411_jpeg_u8_p3,
    nppiRGBToYCbCr411_JPEG_8u_P3R_Ctx,
    4,
    1
);
impl_jpeg_subsampled_planar_forward!(
    bgr_to_ycbcr420_jpeg_u8_p3,
    nppiBGRToYCbCr420_JPEG_8u_P3R_Ctx,
    2,
    2
);
impl_jpeg_subsampled_planar_forward!(
    bgr_to_ycbcr422_jpeg_u8_p3,
    nppiBGRToYCbCr422_JPEG_8u_P3R_Ctx,
    2,
    1
);
impl_jpeg_subsampled_planar_forward!(
    bgr_to_ycbcr411_jpeg_u8_p3,
    nppiBGRToYCbCr411_JPEG_8u_P3R_Ctx,
    4,
    1
);
impl_jpeg_subsampled_packed_forward!(
    rgb_to_ycbcr420_jpeg_u8_c3_to_p3,
    C3,
    nppiRGBToYCbCr420_JPEG_8u_C3P3R_Ctx,
    2,
    2
);
impl_jpeg_subsampled_packed_forward!(
    rgb_to_ycbcr422_jpeg_u8_c3_to_p3,
    C3,
    nppiRGBToYCbCr422_JPEG_8u_C3P3R_Ctx,
    2,
    1
);
impl_jpeg_subsampled_packed_forward!(
    rgb_to_ycbcr411_jpeg_u8_c3_to_p3,
    C3,
    nppiRGBToYCbCr411_JPEG_8u_C3P3R_Ctx,
    4,
    1
);
impl_jpeg_subsampled_packed_forward!(
    bgr_to_ycbcr420_jpeg_u8_c3_to_p3,
    C3,
    nppiBGRToYCbCr420_JPEG_8u_C3P3R_Ctx,
    2,
    2
);
impl_jpeg_subsampled_packed_forward!(
    bgr_to_ycbcr422_jpeg_u8_c3_to_p3,
    C3,
    nppiBGRToYCbCr422_JPEG_8u_C3P3R_Ctx,
    2,
    1
);
impl_jpeg_subsampled_packed_forward!(
    bgr_to_ycbcr411_jpeg_u8_c3_to_p3,
    C3,
    nppiBGRToYCbCr411_JPEG_8u_C3P3R_Ctx,
    4,
    1
);
impl_jpeg_subsampled_planar_forward!(rgb_to_yuv420_u8_p3, nppiRGBToYUV420_8u_P3R_Ctx, 2, 2);
impl_jpeg_subsampled_packed_forward!(
    rgb_to_yuv420_u8_c3_to_p3,
    C3,
    nppiRGBToYUV420_8u_C3P3R_Ctx,
    2,
    2
);
impl_jpeg_subsampled_packed_forward!(
    bgr_to_yuv420_u8_ac4_to_p3,
    AC4,
    nppiBGRToYUV420_8u_AC4P3R_Ctx,
    2,
    2
);
impl_jpeg_subsampled_planar_forward!(rgb_to_yuv422_u8_p3, nppiRGBToYUV422_8u_P3R_Ctx, 2, 1);
impl_jpeg_subsampled_packed_forward!(
    rgb_to_yuv422_u8_c3_to_p3,
    C3,
    nppiRGBToYUV422_8u_C3P3R_Ctx,
    2,
    1
);

macro_rules! impl_jpeg_subsampled_planar_forward_dispatch {
    ($trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait: DataTypeLike {
            fn $method(
                stream_context: &StreamContext,
                source_0: &ImageView<'_, Self, C1>,
                source_1: &ImageView<'_, Self, C1>,
                source_2: &ImageView<'_, Self, C1>,
                destination_y: &mut ImageViewMut<'_, Self, C1>,
                destination_cb: &mut ImageViewMut<'_, Self, C1>,
                destination_cr: &mut ImageViewMut<'_, Self, C1>,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source_0: &ImageView<'_, T, C1>,
            source_1: &ImageView<'_, T, C1>,
            source_2: &ImageView<'_, T, C1>,
            destination_y: &mut ImageViewMut<'_, T, C1>,
            destination_cb: &mut ImageViewMut<'_, T, C1>,
            destination_cr: &mut ImageViewMut<'_, T, C1>,
        ) -> Result<()>
        where
            T: $trait,
        {
            T::$method(
                stream_context,
                source_0,
                source_1,
                source_2,
                destination_y,
                destination_cb,
                destination_cr,
            )
        }

        $(
            impl $trait for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source_0: &ImageView<'_, Self, C1>,
                    source_1: &ImageView<'_, Self, C1>,
                    source_2: &ImageView<'_, Self, C1>,
                    destination_y: &mut ImageViewMut<'_, Self, C1>,
                    destination_cb: &mut ImageViewMut<'_, Self, C1>,
                    destination_cr: &mut ImageViewMut<'_, Self, C1>,
                ) -> Result<()> {
                    $direct(
                        stream_context,
                        source_0,
                        source_1,
                        source_2,
                        destination_y,
                        destination_cb,
                        destination_cr,
                    )
                }
            }
        )*
    };
}

macro_rules! impl_jpeg_subsampled_packed_forward_dispatch {
    ($trait:ident, $method:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait: DataTypeLike {
            fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, $layout>,
                destination_y: &mut ImageViewMut<'_, Self, C1>,
                destination_cb: &mut ImageViewMut<'_, Self, C1>,
                destination_cr: &mut ImageViewMut<'_, Self, C1>,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination_y: &mut ImageViewMut<'_, T, C1>,
            destination_cb: &mut ImageViewMut<'_, T, C1>,
            destination_cr: &mut ImageViewMut<'_, T, C1>,
        ) -> Result<()>
        where
            T: $trait,
        {
            T::$method(
                stream_context,
                source,
                destination_y,
                destination_cb,
                destination_cr,
            )
        }

        $(
            impl $trait for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination_y: &mut ImageViewMut<'_, Self, C1>,
                    destination_cb: &mut ImageViewMut<'_, Self, C1>,
                    destination_cr: &mut ImageViewMut<'_, Self, C1>,
                ) -> Result<()> {
                    $direct(
                        stream_context,
                        source,
                        destination_y,
                        destination_cb,
                        destination_cr,
                    )
                }
            }
        )*
    };
}

impl_jpeg_subsampled_planar_forward_dispatch!(
    RgbToYcbcr420JpegP3,
    rgb_to_ycbcr420_jpeg_p3,
    rgb_to_ycbcr420_jpeg_p3,
    [u8 => rgb_to_ycbcr420_jpeg_u8_p3]
);
impl_jpeg_subsampled_planar_forward_dispatch!(
    RgbToYcbcr422JpegP3,
    rgb_to_ycbcr422_jpeg_p3,
    rgb_to_ycbcr422_jpeg_p3,
    [u8 => rgb_to_ycbcr422_jpeg_u8_p3]
);
impl_jpeg_subsampled_planar_forward_dispatch!(
    RgbToYcbcr411JpegP3,
    rgb_to_ycbcr411_jpeg_p3,
    rgb_to_ycbcr411_jpeg_p3,
    [u8 => rgb_to_ycbcr411_jpeg_u8_p3]
);
impl_jpeg_subsampled_planar_forward_dispatch!(
    BgrToYcbcr420JpegP3,
    bgr_to_ycbcr420_jpeg_p3,
    bgr_to_ycbcr420_jpeg_p3,
    [u8 => bgr_to_ycbcr420_jpeg_u8_p3]
);
impl_jpeg_subsampled_planar_forward_dispatch!(
    BgrToYcbcr422JpegP3,
    bgr_to_ycbcr422_jpeg_p3,
    bgr_to_ycbcr422_jpeg_p3,
    [u8 => bgr_to_ycbcr422_jpeg_u8_p3]
);
impl_jpeg_subsampled_planar_forward_dispatch!(
    BgrToYcbcr411JpegP3,
    bgr_to_ycbcr411_jpeg_p3,
    bgr_to_ycbcr411_jpeg_p3,
    [u8 => bgr_to_ycbcr411_jpeg_u8_p3]
);
impl_jpeg_subsampled_planar_forward_dispatch!(RgbToYuv420P3, rgb_to_yuv420_p3, rgb_to_yuv420_p3, [
    u8 => rgb_to_yuv420_u8_p3
]);
impl_jpeg_subsampled_planar_forward_dispatch!(RgbToYuv422P3, rgb_to_yuv422_p3, rgb_to_yuv422_p3, [
    u8 => rgb_to_yuv422_u8_p3
]);

impl_jpeg_subsampled_packed_forward_dispatch!(
    RgbToYcbcr420JpegC3ToP3,
    rgb_to_ycbcr420_jpeg_c3_to_p3,
    rgb_to_ycbcr420_jpeg_c3_to_p3,
    C3,
    [u8 => rgb_to_ycbcr420_jpeg_u8_c3_to_p3]
);
impl_jpeg_subsampled_packed_forward_dispatch!(
    RgbToYcbcr422JpegC3ToP3,
    rgb_to_ycbcr422_jpeg_c3_to_p3,
    rgb_to_ycbcr422_jpeg_c3_to_p3,
    C3,
    [u8 => rgb_to_ycbcr422_jpeg_u8_c3_to_p3]
);
impl_jpeg_subsampled_packed_forward_dispatch!(
    RgbToYcbcr411JpegC3ToP3,
    rgb_to_ycbcr411_jpeg_c3_to_p3,
    rgb_to_ycbcr411_jpeg_c3_to_p3,
    C3,
    [u8 => rgb_to_ycbcr411_jpeg_u8_c3_to_p3]
);
impl_jpeg_subsampled_packed_forward_dispatch!(
    BgrToYcbcr420JpegC3ToP3,
    bgr_to_ycbcr420_jpeg_c3_to_p3,
    bgr_to_ycbcr420_jpeg_c3_to_p3,
    C3,
    [u8 => bgr_to_ycbcr420_jpeg_u8_c3_to_p3]
);
impl_jpeg_subsampled_packed_forward_dispatch!(
    BgrToYcbcr422JpegC3ToP3,
    bgr_to_ycbcr422_jpeg_c3_to_p3,
    bgr_to_ycbcr422_jpeg_c3_to_p3,
    C3,
    [u8 => bgr_to_ycbcr422_jpeg_u8_c3_to_p3]
);
impl_jpeg_subsampled_packed_forward_dispatch!(
    BgrToYcbcr411JpegC3ToP3,
    bgr_to_ycbcr411_jpeg_c3_to_p3,
    bgr_to_ycbcr411_jpeg_c3_to_p3,
    C3,
    [u8 => bgr_to_ycbcr411_jpeg_u8_c3_to_p3]
);
impl_jpeg_subsampled_packed_forward_dispatch!(RgbToYuv420C3ToP3, rgb_to_yuv420_c3_to_p3, rgb_to_yuv420_c3_to_p3, C3, [
    u8 => rgb_to_yuv420_u8_c3_to_p3
]);
impl_jpeg_subsampled_packed_forward_dispatch!(RgbToYuv422C3ToP3, rgb_to_yuv422_c3_to_p3, rgb_to_yuv422_c3_to_p3, C3, [
    u8 => rgb_to_yuv422_u8_c3_to_p3
]);
impl_jpeg_subsampled_packed_forward_dispatch!(BgrToYuv420Ac4ToP3, bgr_to_yuv420_ac4_to_p3, bgr_to_yuv420_ac4_to_p3, AC4, [
    u8 => bgr_to_yuv420_u8_ac4_to_p3
]);
