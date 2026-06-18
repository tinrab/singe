use super::*;

impl_color_convert_planar_to_planar!(
    rgb_to_ycbcr444_jpeg_u8_p3,
    u8,
    3,
    nppiRGBToYCbCr444_JPEG_8u_P3R_Ctx
);
impl_color_convert_planar_to_planar!(
    bgr_to_ycbcr444_jpeg_u8_p3,
    u8,
    3,
    nppiBGRToYCbCr444_JPEG_8u_P3R_Ctx
);
impl_color_convert_planar_to_planar!(
    ycbcr444_to_rgb_jpeg_u8_p3,
    u8,
    3,
    nppiYCbCr444ToRGB_JPEG_8u_P3R_Ctx
);
impl_color_convert_planar_to_planar!(
    ycbcr444_to_bgr_jpeg_u8_p3,
    u8,
    3,
    nppiYCbCr444ToBGR_JPEG_8u_P3R_Ctx
);
impl_color_convert_packed_to_planar!(
    rgb_to_ycbcr444_jpeg_u8_c3_to_p3,
    u8,
    C3,
    3,
    nppiRGBToYCbCr444_JPEG_8u_C3P3R_Ctx
);
impl_color_convert_packed_to_planar!(
    bgr_to_ycbcr444_jpeg_u8_c3_to_p3,
    u8,
    C3,
    3,
    nppiBGRToYCbCr444_JPEG_8u_C3P3R_Ctx
);
impl_color_convert_planar_to_packed!(
    ycbcr444_to_rgb_jpeg_u8_p3_to_c3,
    u8,
    3,
    C3,
    nppiYCbCr444ToRGB_JPEG_8u_P3C3R_Ctx
);
impl_color_convert_planar_to_packed!(
    ycbcr444_to_bgr_jpeg_u8_p3_to_c3,
    u8,
    3,
    C3,
    nppiYCbCr444ToBGR_JPEG_8u_P3C3R_Ctx
);

impl_color_convert_planar_to_planar_dispatch!(
    RgbToYcbcr444JpegP3,
    rgb_to_ycbcr444_jpeg_p3,
    rgb_to_ycbcr444_jpeg_p3,
    3,
    [u8 => rgb_to_ycbcr444_jpeg_u8_p3]
);
impl_color_convert_planar_to_planar_dispatch!(
    BgrToYcbcr444JpegP3,
    bgr_to_ycbcr444_jpeg_p3,
    bgr_to_ycbcr444_jpeg_p3,
    3,
    [u8 => bgr_to_ycbcr444_jpeg_u8_p3]
);
impl_color_convert_planar_to_planar_dispatch!(
    Ycbcr444ToRgbJpegP3,
    ycbcr444_to_rgb_jpeg_p3,
    ycbcr444_to_rgb_jpeg_p3,
    3,
    [u8 => ycbcr444_to_rgb_jpeg_u8_p3]
);
impl_color_convert_planar_to_planar_dispatch!(
    Ycbcr444ToBgrJpegP3,
    ycbcr444_to_bgr_jpeg_p3,
    ycbcr444_to_bgr_jpeg_p3,
    3,
    [u8 => ycbcr444_to_bgr_jpeg_u8_p3]
);
impl_color_convert_packed_c3_to_planar_dispatch!(
    RgbToYcbcr444JpegC3ToP3,
    rgb_to_ycbcr444_jpeg_c3_to_p3,
    rgb_to_ycbcr444_jpeg_c3_to_p3,
    3,
    [u8 => rgb_to_ycbcr444_jpeg_u8_c3_to_p3]
);
impl_color_convert_packed_c3_to_planar_dispatch!(
    BgrToYcbcr444JpegC3ToP3,
    bgr_to_ycbcr444_jpeg_c3_to_p3,
    bgr_to_ycbcr444_jpeg_c3_to_p3,
    3,
    [u8 => bgr_to_ycbcr444_jpeg_u8_c3_to_p3]
);
impl_color_convert_planar_to_packed_c3_dispatch!(
    Ycbcr444ToRgbJpegP3ToC3,
    ycbcr444_to_rgb_jpeg_p3_to_c3,
    ycbcr444_to_rgb_jpeg_p3_to_c3,
    3,
    [u8 => ycbcr444_to_rgb_jpeg_u8_p3_to_c3]
);
impl_color_convert_planar_to_packed_c3_dispatch!(
    Ycbcr444ToBgrJpegP3ToC3,
    ycbcr444_to_bgr_jpeg_p3_to_c3,
    ycbcr444_to_bgr_jpeg_p3_to_c3,
    3,
    [u8 => ycbcr444_to_bgr_jpeg_u8_p3_to_c3]
);

impl_color_convert_planar_to_planar!(
    ycck_to_cmyk_jpeg_601_u8_p4,
    u8,
    4,
    nppiYCCKToCMYK_JPEG_601_8u_P4R_Ctx
);
impl_color_convert_planar_to_planar_different_planes!(
    cmyk_or_ycck_to_rgb_jpeg_u8_p4_to_p3,
    u8,
    4,
    3,
    nppiCMYKOrYCCKToRGB_JPEG_8u_P4P3R_Ctx
);
impl_color_convert_planar_to_packed!(
    cmyk_or_ycck_to_rgb_jpeg_u8_p4_to_c3,
    u8,
    4,
    C3,
    nppiCMYKOrYCCKToRGB_JPEG_8u_P4C3R_Ctx
);
impl_color_convert_planar_to_planar_different_planes!(
    cmyk_or_ycck_to_bgr_jpeg_u8_p4_to_p3,
    u8,
    4,
    3,
    nppiCMYKOrYCCKToBGR_JPEG_8u_P4P3R_Ctx
);
impl_color_convert_planar_to_packed!(
    cmyk_or_ycck_to_bgr_jpeg_u8_p4_to_c3,
    u8,
    4,
    C3,
    nppiCMYKOrYCCKToBGR_JPEG_8u_P4C3R_Ctx
);

macro_rules! impl_color_convert_planar_to_planar_different_planes_dispatch {
    ($trait:ident, $method:ident, $function:ident, $source_planes:literal, $destination_planes:literal, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait: DataTypeLike {
            fn $method(
                stream_context: &StreamContext,
                source: &PlanarImageView<'_, Self, $source_planes>,
                destination: &mut PlanarImageViewMut<'_, Self, $destination_planes>,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &PlanarImageView<'_, T, $source_planes>,
            destination: &mut PlanarImageViewMut<'_, T, $destination_planes>,
        ) -> Result<()>
        where
            T: $trait,
        {
            T::$method(stream_context, source, destination)
        }

        $(
            impl $trait for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source: &PlanarImageView<'_, Self, $source_planes>,
                    destination: &mut PlanarImageViewMut<'_, Self, $destination_planes>,
                ) -> Result<()> {
                    $direct(stream_context, source, destination)
                }
            }
        )*
    };
}

impl_color_convert_planar_to_planar_dispatch!(
    YcckToCmykJpeg601P4,
    ycck_to_cmyk_jpeg_601_p4,
    ycck_to_cmyk_jpeg_601_p4,
    4,
    [u8 => ycck_to_cmyk_jpeg_601_u8_p4]
);
impl_color_convert_planar_to_planar_different_planes_dispatch!(
    CmykOrYcckToRgbJpegP4ToP3,
    cmyk_or_ycck_to_rgb_jpeg_p4_to_p3,
    cmyk_or_ycck_to_rgb_jpeg_p4_to_p3,
    4,
    3,
    [u8 => cmyk_or_ycck_to_rgb_jpeg_u8_p4_to_p3]
);
impl_color_convert_planar_to_planar_different_planes_dispatch!(
    CmykOrYcckToBgrJpegP4ToP3,
    cmyk_or_ycck_to_bgr_jpeg_p4_to_p3,
    cmyk_or_ycck_to_bgr_jpeg_p4_to_p3,
    4,
    3,
    [u8 => cmyk_or_ycck_to_bgr_jpeg_u8_p4_to_p3]
);
impl_color_convert_planar_to_packed_c3_dispatch!(
    CmykOrYcckToRgbJpegP4ToC3,
    cmyk_or_ycck_to_rgb_jpeg_p4_to_c3,
    cmyk_or_ycck_to_rgb_jpeg_p4_to_c3,
    4,
    [u8 => cmyk_or_ycck_to_rgb_jpeg_u8_p4_to_c3]
);
impl_color_convert_planar_to_packed_c3_dispatch!(
    CmykOrYcckToBgrJpegP4ToC3,
    cmyk_or_ycck_to_bgr_jpeg_p4_to_c3,
    cmyk_or_ycck_to_bgr_jpeg_p4_to_c3,
    4,
    [u8 => cmyk_or_ycck_to_bgr_jpeg_u8_p4_to_c3]
);
