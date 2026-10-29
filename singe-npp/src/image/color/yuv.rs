use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        color::ColorTwistMatrix,
        color_descriptors::{
            color_batch_advanced_descriptors, color_batch_descriptors,
            color_planar_batch_advanced_descriptors, color_planar_batch_descriptors,
            subsampled_color_batch_advanced_descriptors, subsampled_color_batch_descriptors,
            validate_jpeg_subsampled_destination_planes, validate_jpeg_subsampled_source_planes,
            validate_nv12_destination_planes, validate_nv12_source_planes,
            validate_rgb_destination_planes, validate_rgb_source_planes, validate_same_size,
            validate_subsampled_p2_destination_planes, validate_subsampled_p2_source_planes,
        },
        view::{
            AC4, C1, C2, C3, C4, ImageOutputView, ImageView, PlanarImageOutputView, PlanarImageView,
        },
    },
    try_ffi,
    types::DataTypeLike,
};

impl_color_convert_same_layout!(
    "Converts RGB color values to XYZ color values.",
    rgb_to_xyz_u8_c3,
    u8,
    C3,
    nppiRGBToXYZ_8u_C3R_Ctx
);

impl_color_convert_same_layout!(
    "Converts RGB color values to XYZ color values. The destination alpha channel is not processed.",
    rgb_to_xyz_u8_ac4,
    u8,
    AC4,
    nppiRGBToXYZ_8u_AC4R_Ctx
);

impl_color_convert_same_layout!(
    "Converts XYZ color values to RGB color values.",
    xyz_to_rgb_u8_c3,
    u8,
    C3,
    nppiXYZToRGB_8u_C3R_Ctx
);

impl_color_convert_same_layout!(
    "Converts XYZ color values to RGB color values. The destination alpha channel is not processed.",
    xyz_to_rgb_u8_ac4,
    u8,
    AC4,
    nppiXYZToRGB_8u_AC4R_Ctx
);

impl_color_convert_same_layout!(
    "Converts RGB color values to LUV color values.",
    rgb_to_luv_u8_c3,
    u8,
    C3,
    nppiRGBToLUV_8u_C3R_Ctx
);

impl_color_convert_same_layout!(
    "Converts RGB color values to LUV color values. The destination alpha channel is not processed.",
    rgb_to_luv_u8_ac4,
    u8,
    AC4,
    nppiRGBToLUV_8u_AC4R_Ctx
);

impl_color_convert_same_layout!(
    "Converts LUV color values to RGB color values.",
    luv_to_rgb_u8_c3,
    u8,
    C3,
    nppiLUVToRGB_8u_C3R_Ctx
);

impl_color_convert_same_layout!(
    "Converts LUV color values to RGB color values. The destination alpha channel is not processed.",
    luv_to_rgb_u8_ac4,
    u8,
    AC4,
    nppiLUVToRGB_8u_AC4R_Ctx
);

impl_color_convert_same_layout!(
    "Converts RGB color values to HSV color values.",
    rgb_to_hsv_u8_c3,
    u8,
    C3,
    nppiRGBToHSV_8u_C3R_Ctx
);

impl_color_convert_same_layout!(
    "Converts RGB color values to HSV color values. The destination alpha channel is not processed.",
    rgb_to_hsv_u8_ac4,
    u8,
    AC4,
    nppiRGBToHSV_8u_AC4R_Ctx
);

impl_color_convert_same_layout!(
    "Converts HSV color values to RGB color values.",
    hsv_to_rgb_u8_c3,
    u8,
    C3,
    nppiHSVToRGB_8u_C3R_Ctx
);

impl_color_convert_same_layout!(
    "Converts HSV color values to RGB color values. The destination alpha channel is not processed.",
    hsv_to_rgb_u8_ac4,
    u8,
    AC4,
    nppiHSVToRGB_8u_AC4R_Ctx
);

impl_color_convert_same_layout!(
    "Converts RGB color values to HLS color values.",
    rgb_to_hls_u8_c3,
    u8,
    C3,
    nppiRGBToHLS_8u_C3R_Ctx
);

impl_color_convert_same_layout!(
    "Converts RGB color values to HLS color values. The destination alpha channel is not processed.",
    rgb_to_hls_u8_ac4,
    u8,
    AC4,
    nppiRGBToHLS_8u_AC4R_Ctx
);

impl_color_convert_same_layout!(
    "Converts HLS color values to RGB color values.",
    hls_to_rgb_u8_c3,
    u8,
    C3,
    nppiHLSToRGB_8u_C3R_Ctx
);

impl_color_convert_same_layout!(
    "Converts HLS color values to RGB color values. The destination alpha channel is not processed.",
    hls_to_rgb_u8_ac4,
    u8,
    AC4,
    nppiHLSToRGB_8u_AC4R_Ctx
);

impl_color_convert_same_layout!(
    "Converts BGR color values to Lab color values.",
    bgr_to_lab_u8_c3,
    u8,
    C3,
    nppiBGRToLab_8u_C3R_Ctx
);

impl_color_convert_same_layout!(
    "Converts Lab color values to BGR color values.",
    lab_to_bgr_u8_c3,
    u8,
    C3,
    nppiLabToBGR_8u_C3R_Ctx
);

impl_color_convert_same_layout!(
    "Converts RGB color values to YUV color values.",
    rgb_to_yuv_u8_c3,
    u8,
    C3,
    nppiRGBToYUV_8u_C3R_Ctx
);

impl_color_convert_same_layout!(
    "Converts RGB color values to YUV color values. The destination alpha channel is not processed.",
    rgb_to_yuv_u8_ac4,
    u8,
    AC4,
    nppiRGBToYUV_8u_AC4R_Ctx
);

impl_generic_color_convert_same_layout!(
    "Converts RGB color values to XYZ color values.",
    RgbToXyzC3, rgb_to_xyz, rgb_to_xyz_c3, C3, [
    u8 => rgb_to_xyz_u8_c3,
    ]
);

impl_generic_color_convert_same_layout!(
    "Converts RGB color values to XYZ color values. The destination alpha channel is not processed.",
    RgbToXyzAC4, rgb_to_xyz, rgb_to_xyz_ac4, AC4, [
    u8 => rgb_to_xyz_u8_ac4,
    ]
);

impl_generic_color_convert_same_layout!(
    "Converts XYZ color values to RGB color values.",
    XyzToRgbC3, xyz_to_rgb, xyz_to_rgb_c3, C3, [
    u8 => xyz_to_rgb_u8_c3,
    ]
);

impl_generic_color_convert_same_layout!(
    "Converts XYZ color values to RGB color values. The destination alpha channel is not processed.",
    XyzToRgbAC4, xyz_to_rgb, xyz_to_rgb_ac4, AC4, [
    u8 => xyz_to_rgb_u8_ac4,
    ]
);

impl_generic_color_convert_same_layout!(
    "Converts RGB color values to LUV color values.",
    RgbToLuvC3, rgb_to_luv, rgb_to_luv_c3, C3, [
    u8 => rgb_to_luv_u8_c3,
    ]
);

impl_generic_color_convert_same_layout!(
    "Converts RGB color values to LUV color values. The destination alpha channel is not processed.",
    RgbToLuvAC4, rgb_to_luv, rgb_to_luv_ac4, AC4, [
    u8 => rgb_to_luv_u8_ac4,
    ]
);

impl_generic_color_convert_same_layout!(
    "Converts LUV color values to RGB color values.",
    LuvToRgbC3, luv_to_rgb, luv_to_rgb_c3, C3, [
    u8 => luv_to_rgb_u8_c3,
    ]
);

impl_generic_color_convert_same_layout!(
    "Converts LUV color values to RGB color values. The destination alpha channel is not processed.",
    LuvToRgbAC4, luv_to_rgb, luv_to_rgb_ac4, AC4, [
    u8 => luv_to_rgb_u8_ac4,
    ]
);

impl_generic_color_convert_same_layout!(
    "Converts RGB color values to HSV color values.",
    RgbToHsvC3, rgb_to_hsv, rgb_to_hsv_c3, C3, [
    u8 => rgb_to_hsv_u8_c3,
    ]
);

impl_generic_color_convert_same_layout!(
    "Converts RGB color values to HSV color values. The destination alpha channel is not processed.",
    RgbToHsvAC4, rgb_to_hsv, rgb_to_hsv_ac4, AC4, [
    u8 => rgb_to_hsv_u8_ac4,
    ]
);

impl_generic_color_convert_same_layout!(
    "Converts HSV color values to RGB color values.",
    HsvToRgbC3, hsv_to_rgb, hsv_to_rgb_c3, C3, [
    u8 => hsv_to_rgb_u8_c3,
    ]
);

impl_generic_color_convert_same_layout!(
    "Converts HSV color values to RGB color values. The destination alpha channel is not processed.",
    HsvToRgbAC4, hsv_to_rgb, hsv_to_rgb_ac4, AC4, [
    u8 => hsv_to_rgb_u8_ac4,
    ]
);

impl_generic_color_convert_same_layout!(
    "Converts RGB color values to HLS color values.",
    RgbToHlsC3, rgb_to_hls, rgb_to_hls_c3, C3, [
    u8 => rgb_to_hls_u8_c3,
    ]
);

impl_generic_color_convert_same_layout!(
    "Converts RGB color values to HLS color values. The destination alpha channel is not processed.",
    RgbToHlsAC4, rgb_to_hls, rgb_to_hls_ac4, AC4, [
    u8 => rgb_to_hls_u8_ac4,
    ]
);

impl_generic_color_convert_same_layout!(
    "Converts HLS color values to RGB color values.",
    HlsToRgbC3, hls_to_rgb, hls_to_rgb_c3, C3, [
    u8 => hls_to_rgb_u8_c3,
    ]
);

impl_generic_color_convert_same_layout!(
    "Converts HLS color values to RGB color values. The destination alpha channel is not processed.",
    HlsToRgbAC4, hls_to_rgb, hls_to_rgb_ac4, AC4, [
    u8 => hls_to_rgb_u8_ac4,
    ]
);

impl_generic_color_convert_same_layout!(
    "Converts BGR color values to Lab color values.",
    BgrToLabC3, bgr_to_lab, bgr_to_lab_c3, C3, [
    u8 => bgr_to_lab_u8_c3,
    ]
);

impl_generic_color_convert_same_layout!(
    "Converts Lab color values to BGR color values.",
    LabToBgrC3, lab_to_bgr, lab_to_bgr_c3, C3, [
    u8 => lab_to_bgr_u8_c3,
    ]
);

impl_generic_color_convert_same_layout!(
    "Converts RGB color values to YUV color values.",
    RgbToYuvC3, rgb_to_yuv, rgb_to_yuv_c3, C3, [
    u8 => rgb_to_yuv_u8_c3,
    ]
);

impl_generic_color_convert_same_layout!(
    "Converts RGB color values to YUV color values. The destination alpha channel is not processed.",
    RgbToYuvAC4, rgb_to_yuv, rgb_to_yuv_ac4, AC4, [
    u8 => rgb_to_yuv_u8_ac4,
    ]
);

impl_color_convert_different_layout!(
    "Converts RGB color values to YUV422 color values.",
    rgb_to_yuv422_u8_c3_to_c2,
    u8,
    C3,
    C2,
    nppiRGBToYUV422_8u_C3C2R_Ctx
);

impl_color_twist_different_layout!(
    "Three channel 8-bit unsigned packed RGB convertion to two channel 8-bit unsigned packed YUV 4:2:2, using a Color Twist to compute the exact color space arithmetic.",
    rgb_to_yuv422_u8_color_twist_c3_to_c2,
    u8,
    C3,
    C2,
    nppiRGBToYUV422_8u_ColorTwist32f_C3C2R_Ctx
);

impl_color_twist_different_layout!(
    "Three channel 16-bit unsigned packed RGB convertion to two channel 16-bit unsigned packed YUV 4:2:2, using a Color Twist to compute the exact color space arithmetic.",
    rgb_to_yuv422_u16_color_twist_c3_to_c2,
    u16,
    C3,
    C2,
    nppiRGBToYUV422_16u_ColorTwist32f_C3C2R_Ctx
);

impl_color_convert_same_layout!(
    "Converts YUV color values to RGB color values.",
    yuv_to_rgb_u8_c3,
    u8,
    C3,
    nppiYUVToRGB_8u_C3R_Ctx
);

impl_color_convert_same_layout!(
    "Converts YUV color values to RGB color values. The destination alpha channel is not processed.",
    yuv_to_rgb_u8_ac4,
    u8,
    AC4,
    nppiYUVToRGB_8u_AC4R_Ctx
);

impl_generic_color_convert_same_layout!(
    "Converts YUV color values to RGB color values.",
    YuvToRgbC3, yuv_to_rgb, yuv_to_rgb_c3, C3, [
    u8 => yuv_to_rgb_u8_c3,
    ]
);

impl_generic_color_convert_same_layout!(
    "Converts YUV color values to RGB color values. The destination alpha channel is not processed.",
    YuvToRgbAC4, yuv_to_rgb, yuv_to_rgb_ac4, AC4, [
    u8 => yuv_to_rgb_u8_ac4,
    ]
);

impl_color_convert_different_layout!(
    "Converts YUV422 color values to RGB color values.",
    yuv422_to_rgb_u8_c2_to_c3,
    u8,
    C2,
    C3,
    nppiYUV422ToRGB_8u_C2C3R_Ctx
);

impl_color_twist_different_layout!(
    "Two channel 8-bit unsigned packed YUV4:2:2 convertion to three channel 8-bit unsigned packed RGB, using a Color Twist to compute the exact color space arithmetic.",
    yuv422_to_rgb_u8_color_twist_c2_to_c3,
    u8,
    C2,
    C3,
    nppiYUV422ToRGB_8u_ColorTwist32f_C2C3R_Ctx
);

impl_color_twist_different_layout!(
    "Two channel 16-bit unsigned packed YUV4:2:2 convertion to three channel 16-bit unsigned packed RGB, using a Color Twist to compute the exact color space arithmetic.",
    yuv422_to_rgb_u16_color_twist_c2_to_c3,
    u16,
    C2,
    C3,
    nppiYUV422ToRGB_16u_ColorTwist32f_C2C3R_Ctx
);

impl_subsampled_color_twist_planar_forward!(
    "Three channel 8-bit unsigned planar RGB conversion to three channel 8-bit unsigned planar YUV 4:2:2, using a Color Twist to compute the exact color space arithmetic.",
    rgb_to_yuv422_u8_color_twist_p3,
    u8,
    nppiRGBToYUV422_8u_ColorTwist32f_P3R_Ctx,
    2,
    1
);

impl_subsampled_color_twist_planar_forward!(
    "Three channel 16-bit unsigned planar RGB conversion to three channel 16-bit unsigned planar YUV 4:2:2, using a Color Twist to compute the exact color space arithmetic.",
    rgb_to_yuv422_u16_color_twist_p3,
    u16,
    nppiRGBToYUV422_16u_ColorTwist32f_P3R_Ctx,
    2,
    1
);

impl_subsampled_color_twist_packed_forward!(
    "Three channel 8-bit unsigned packed RGB conversion to three channel 8-bit unsigned planar YUV 4:2:2, using a Color Twist to compute the exact color space arithmetic.",
    rgb_to_yuv422_u8_color_twist_c3_to_p3,
    u8,
    C3,
    nppiRGBToYUV422_8u_ColorTwist32f_C3P3R_Ctx,
    2,
    1
);

impl_subsampled_color_twist_packed_forward!(
    "Three channel 16-bit unsigned packed RGB conversion to three channel 16-bit unsigned planar YUV 4:2:2, using a Color Twist to compute the exact color space arithmetic.",
    rgb_to_yuv422_u16_color_twist_c3_to_p3,
    u16,
    C3,
    nppiRGBToYUV422_16u_ColorTwist32f_C3P3R_Ctx,
    2,
    1
);

impl_subsampled_color_twist_planar_inverse!(
    "Three channel 8-bit unsigned planar YUV4:2:2 conversion to three channel 8-bit unsigned planar RGB, using a Color Twist to compute the exact color space arithmetic.",
    yuv422_to_rgb_u8_color_twist_p3,
    u8,
    nppiYUV422ToRGB_8u_ColorTwist32f_P3R_Ctx,
    2,
    1
);

impl_subsampled_color_twist_planar_inverse!(
    "Three channel 16-bit unsigned planar YUV4:2:2 conversion to three channel 16-bit unsigned planar RGB, using a Color Twist to compute the exact color space arithmetic.",
    yuv422_to_rgb_u16_color_twist_p3,
    u16,
    nppiYUV422ToRGB_16u_ColorTwist32f_P3R_Ctx,
    2,
    1
);

impl_subsampled_color_twist_packed_inverse!(
    "Three channel 8-bit unsigned planar YUV4:2:2 conversion to three channel 8-bit unsigned packed RGB, using a Color Twist to compute the exact color space arithmetic.",
    yuv422_to_rgb_u8_color_twist_p3_to_c3,
    u8,
    C3,
    nppiYUV422ToRGB_8u_ColorTwist32f_P3C3R_Ctx,
    2,
    1
);

impl_subsampled_color_twist_packed_inverse!(
    "Three channel 8-bit unsigned planar YUV4:2:2 conversion to three channel 8-bit unsigned packed RGB, using a Color Twist to compute the exact color space arithmetic.",
    yuv422_to_rgb_u16_color_twist_p3_to_c3,
    u16,
    C3,
    nppiYUV422ToRGB_16u_ColorTwist32f_P3C3R_Ctx,
    2,
    1
);

impl_subsampled_color_twist_packed_inverse_constant_alpha!(
    "Three channel 8-bit unsigned planar YUV4:2:2 conversion to four channel 8-bit unsigned packed RGB, using a Color Twist to compute the exact color space arithmetic, with user set alpha.",
    yuv422_to_rgb_u8_color_twist_p3_to_ac4,
    u8,
    nppiYUV422ToRGB_8u_ColorTwist32f_P3AC4R_Ctx,
    2,
    1
);

impl_subsampled_color_twist_packed_inverse_constant_alpha!(
    "Three channel 16-bit unsigned planar YUV4:2:2 conversion to four channel 16-bit unsigned packed RGB, using a Color Twist to compute the exact color space arithmetic, with user set alpha.",
    yuv422_to_rgb_u16_color_twist_p3_to_ac4,
    u16,
    nppiYUV422ToRGB_16u_ColorTwist32f_P3AC4R_Ctx,
    2,
    1
);

impl_subsampled_color_twist_planar_forward!(
    "Three channel 8-bit unsigned planar RGB conversion to three channel 8-bit unsigned planar YUV 4:2:0, using a Color Twist to compute the exact color space arithmetic.",
    rgb_to_yuv420_u8_color_twist_p3,
    u8,
    nppiRGBToYUV420_8u_ColorTwist32f_P3R_Ctx,
    2,
    2
);

impl_subsampled_color_twist_planar_forward!(
    "Three channel 16-bit unsigned planar RGB conversion to three channel 16-bit unsigned planar YUV 4:2:0, using a Color Twist to compute the exact color space arithmetic.",
    rgb_to_yuv420_u16_color_twist_p3,
    u16,
    nppiRGBToYUV420_16u_ColorTwist32f_P3R_Ctx,
    2,
    2
);

impl_subsampled_color_twist_packed_forward!(
    "Three channel 8-bit unsigned packed RGB conversion to three channel 8-bit unsigned planar YUV 4:2:0, using a Color Twist to compute the exact color space arithmetic.",
    rgb_to_yuv420_u8_color_twist_c3_to_p3,
    u8,
    C3,
    nppiRGBToYUV420_8u_ColorTwist32f_C3P3R_Ctx,
    2,
    2
);

impl_subsampled_color_twist_packed_forward!(
    "Three channel 16-bit unsigned packed RGB conversion to three channel 16-bit unsigned planar YUV 4:2:0, using a Color Twist to compute the exact color space arithmetic.",
    rgb_to_yuv420_u16_color_twist_c3_to_p3,
    u16,
    C3,
    nppiRGBToYUV420_16u_ColorTwist32f_C3P3R_Ctx,
    2,
    2
);

impl_subsampled_color_twist_planar_inverse!(
    "Three channel 8-bit unsigned planar YUV4:2:0 conversion to three channel 8-bit unsigned planar RGB, using a Color Twist to compute the exact color space arithmetic.",
    yuv420_to_rgb_u8_color_twist_p3,
    u8,
    nppiYUV420ToRGB_8u_ColorTwist32f_P3R_Ctx,
    2,
    2
);

impl_subsampled_color_twist_planar_inverse!(
    "Three channel 16-bit unsigned planar YUV4:2:0 conversion to three channel 16-bit unsigned planar RGB, using a Color Twist to compute the exact color space arithmetic.",
    yuv420_to_rgb_u16_color_twist_p3,
    u16,
    nppiYUV420ToRGB_16u_ColorTwist32f_P3R_Ctx,
    2,
    2
);

impl_subsampled_color_twist_packed_inverse!(
    "Three channel 8-bit unsigned planar YUV4:2:0 conversion to three channel 8-bit unsigned packed RGB, using a Color Twist to compute the exact color space arithmetic.",
    yuv420_to_rgb_u8_color_twist_p3_to_c3,
    u8,
    C3,
    nppiYUV420ToRGB_8u_ColorTwist32f_P3C3R_Ctx,
    2,
    2
);

impl_subsampled_color_twist_packed_inverse!(
    "Three channel 16-bit unsigned planar YUV4:2:0 conversion to three channel 16-bit unsigned packed RGB, using a Color Twist to compute the exact color space arithmetic.",
    yuv420_to_rgb_u16_color_twist_p3_to_c3,
    u16,
    C3,
    nppiYUV420ToRGB_16u_ColorTwist32f_P3C3R_Ctx,
    2,
    2
);

impl_subsampled_color_twist_packed_inverse!(
    "Three channel 8-bit unsigned planar YUV4:2:0 conversion to four channel 8-bit unsigned packed RGB, using a Color Twist to compute the exact color space arithmetic, with constant alpha (0xFF).",
    yuv420_to_rgb_u8_color_twist_p3_to_c4,
    u8,
    C4,
    nppiYUV420ToRGB_8u_ColorTwist32f_P3C4R_Ctx,
    2,
    2
);

impl_subsampled_color_twist_packed_inverse!(
    "Three channel 16-bit unsigned planar YUV4:2:0 conversion to four channel 16-bit unsigned packed RGB, using a Color Twist to compute the exact color space arithmetic, with constant alpha (0xFFFF).",
    yuv420_to_rgb_u16_color_twist_p3_to_c4,
    u16,
    C4,
    nppiYUV420ToRGB_16u_ColorTwist32f_P3C4R_Ctx,
    2,
    2
);

impl_subsampled_color_twist_packed_inverse_constant_alpha!(
    "Three channel 8-bit unsigned planar YUV4:2:0 conversion to four channel 8-bit unsigned packed RGB, using a Color Twist to compute the exact color space arithmetic, with user set alpha.",
    yuv420_to_rgb_u8_color_twist_p3_to_ac4,
    u8,
    nppiYUV420ToRGB_8u_ColorTwist32f_P3AC4R_Ctx,
    2,
    2
);

impl_subsampled_color_twist_packed_inverse_constant_alpha!(
    "Three channel 16-bit unsigned planar YUV4:2:0 conversion to four channel 16-bit unsigned packed RGB, using a Color Twist to compute the exact color space arithmetic, with user set alpha.",
    yuv420_to_rgb_u16_color_twist_p3_to_ac4,
    u16,
    nppiYUV420ToRGB_16u_ColorTwist32f_P3AC4R_Ctx,
    2,
    2
);

impl_nv12_color_twist_planar_forward!(
    "Three channel 8-bit unsigned planar RGB conversion to two channel 8-bit unsigned planar NV12, using a Color Twist to compute the exact color space arithmetic.",
    rgb_to_nv12_u8_color_twist_p3,
    u8,
    nppiRGBToNV12_8u_ColorTwist32f_P3P2R_Ctx
);

impl_nv12_color_twist_planar_forward!(
    "Three channel 16-bit unsigned planar RGB conversion to two channel 16-bit unsigned planar NV12, using a Color Twist to compute the exact color space arithmetic.",
    rgb_to_nv12_u16_color_twist_p3,
    u16,
    nppiRGBToNV12_16u_ColorTwist32f_P3P2R_Ctx
);

impl_nv12_color_twist_packed_forward!(
    "Three channel 8-bit unsigned packed RGB conversion to two channel 8-bit unsigned planar NV12, using a Color Twist to compute the exact color space arithmetic.",
    rgb_to_nv12_u8_color_twist_c3_to_p2,
    u8,
    C3,
    nppiRGBToNV12_8u_ColorTwist32f_C3P2R_Ctx
);

impl_nv12_color_twist_packed_forward!(
    "Three channel 16-bit unsigned packed RGB conversion to two channel 16-bit unsigned planar NV12, using a Color Twist to compute the exact color space arithmetic.",
    rgb_to_nv12_u16_color_twist_c3_to_p2,
    u16,
    C3,
    nppiRGBToNV12_16u_ColorTwist32f_C3P2R_Ctx
);

impl_nv12_color_twist_packed_inverse!(
    "Two channel 8-bit unsigned planar NV12 conversion to three channel 8-bit unsigned packed RGB, using a Color Twist to compute the exact color space arithmetic.",
    nv12_to_rgb_u8_color_twist_p2_to_c3,
    u8,
    C3,
    nppiNV12ToRGB_8u_ColorTwist32f_P2C3R_Ctx
);

impl_nv12_color_twist_packed_inverse!(
    "Two channel 16-bit unsigned planar NV12 conversion to three channel 16-bit unsigned packed RGB, using a Color Twist to compute the exact color space arithmetic.",
    nv12_to_rgb_u16_color_twist_p2_to_c3,
    u16,
    C3,
    nppiNV12ToRGB_16u_ColorTwist32f_P2C3R_Ctx
);

impl_nv12_packed_inverse!(
    "2 channel 8-bit unsigned planar NV12 to 3 channel 8-bit unsigned packed RGB color conversion.",
    nv12_to_rgb_u8_p2_to_c3,
    u8,
    C3,
    nppiNV12ToRGB_8u_P2C3R_Ctx
);

impl_nv12_packed_inverse!(
    "2 channel 8-bit unsigned planar NV12 to 3 channel 8-bit unsigned packed RGB 709 HDTV full color conversion.",
    nv12_to_rgb_709hdtv_u8_p2_to_c3,
    u8,
    C3,
    nppiNV12ToRGB_709HDTV_8u_P2C3R_Ctx
);

impl_nv12_packed_inverse!(
    "2 channel 8-bit unsigned planar NV12 to 3 channel 8-bit unsigned packed RGB 709 CSC color conversion.",
    nv12_to_rgb_709csc_u8_p2_to_c3,
    u8,
    C3,
    nppiNV12ToRGB_709CSC_8u_P2C3R_Ctx
);

impl_nv12_packed_inverse!(
    "2 channel 8-bit unsigned planar NV12 to 3 channel 8-bit unsigned packed BGR color conversion.",
    nv12_to_bgr_u8_p2_to_c3,
    u8,
    C3,
    nppiNV12ToBGR_8u_P2C3R_Ctx
);

impl_nv12_packed_inverse!(
    "2 channel 8-bit unsigned planar NV12 to 3 channel 8-bit unsigned packed RGB 709 HDTV full color conversion.",
    nv12_to_bgr_709hdtv_u8_p2_to_c3,
    u8,
    C3,
    nppiNV12ToBGR_709HDTV_8u_P2C3R_Ctx
);

impl_nv12_packed_inverse!(
    "2 channel 8-bit unsigned planar NV12 to 3 channel 8-bit unsigned packed RGB 709 CSC color conversion.",
    nv12_to_bgr_709csc_u8_p2_to_c3,
    u8,
    C3,
    nppiNV12ToBGR_709CSC_8u_P2C3R_Ctx
);

impl_nv12_to_planar!(
    "2 channel 8-bit unsigned planar NV12 to 3 channel 8-bit unsigned planar YUV420 color conversion.",
    nv12_to_yuv420_u8_p2_to_p3,
    nppiNV12ToYUV420_8u_P2P3R_Ctx
);

impl_jpeg_subsampled_packed_inverse!(
    "3 channel 8-bit unsigned planar YUV420 to 4 channel 8-bit unsigned packed RGB color conversion with alpha.",
    yuv420_to_rgb_u8_p3_to_ac4,
    AC4,
    nppiYUV420ToRGB_8u_P3AC4R_Ctx,
    2,
    2
);

impl_jpeg_subsampled_packed_inverse_constant_alpha!(
    "3 channel 8-bit unsigned planar `YCrCb420` to 4 channel 8-bit unsigned packed RGB color conversion with constant alpha.",
    ycrcb420_to_rgb_u8_p3_to_c4,
    nppiYCrCb420ToRGB_8u_P3C4R_Ctx,
    2,
    2
);

impl_subsampled_p2_to_p3!(
    "2 channel 8-bit unsigned planar `YCbCr420` to 3 channel 8-bit unsigned planar `YCbCr420` sampling format conversion.",
    ycbcr420_u8_p2_to_p3,
    nppiYCbCr420_8u_P2P3R_Ctx,
    2,
    2,
    2,
    2
);

impl_subsampled_p2_to_p3!(
    "2 channel 8-bit unsigned planar `YCbCr420` to 3 channel 8-bit unsigned planar `YCbCr422` sampling format conversion.",
    ycbcr420_to_ycbcr422_u8_p2_to_p3,
    nppiYCbCr420ToYCbCr422_8u_P2P3R_Ctx,
    2,
    2,
    2,
    1
);

impl_subsampled_p2_to_p3!(
    "2 channel 8-bit unsigned planar `YCbCr420` to 3 channel 8-bit unsigned planar `YCbCr411` sampling format conversion.",
    ycbcr420_to_ycbcr411_u8_p2_to_p3,
    nppiYCbCr420ToYCbCr411_8u_P2P3R_Ctx,
    2,
    2,
    4,
    1
);

impl_subsampled_p2_to_p3!(
    "2 channel 8-bit unsigned planar `YCbCr420` to 3 channel 8-bit unsigned planar `YCrCb420` sampling format conversion.",
    ycbcr420_to_ycrcb420_u8_p2_to_p3,
    nppiYCbCr420ToYCrCb420_8u_P2P3R_Ctx,
    2,
    2,
    2,
    2
);

impl_subsampled_p2_to_p3!(
    "2 channel 8-bit unsigned planar `YCbCr411` to 3 channel 8-bit unsigned planar `YCbCr411` sampling format conversion.",
    ycbcr411_u8_p2_to_p3,
    nppiYCbCr411_8u_P2P3R_Ctx,
    4,
    1,
    4,
    1
);

impl_subsampled_p2_to_p3!(
    "2 channel 8-bit unsigned planar `YCbCr411` to 3 channel 8-bit unsigned planar `YCbCr422` sampling format conversion.",
    ycbcr411_to_ycbcr422_u8_p2_to_p3,
    nppiYCbCr411ToYCbCr422_8u_P2P3R_Ctx,
    4,
    1,
    2,
    1
);

impl_subsampled_p2_to_p3!(
    "2 channel 8-bit unsigned planar `YCbCr411` to 3 channel 8-bit unsigned planar `YCbCr420` sampling format conversion.",
    ycbcr411_to_ycbcr420_u8_p2_to_p3,
    nppiYCbCr411ToYCbCr420_8u_P2P3R_Ctx,
    4,
    1,
    2,
    2
);

impl_subsampled_p2_to_p3!(
    "2 channel 8-bit unsigned planar `YCbCr411` to 3 channel 8-bit unsigned planar `YCrCb420` sampling format conversion.",
    ycbcr411_to_ycrcb420_u8_p2_to_p3,
    nppiYCbCr411ToYCrCb420_8u_P2P3R_Ctx,
    4,
    1,
    2,
    2
);

impl_subsampled_c2_to_p3!(
    "2 channel 8-bit unsigned packed `YCbCr422` to 3 channel 8-bit unsigned planar `YCbCr422` sampling format conversion.",
    ycbcr422_u8_c2_to_p3,
    nppiYCbCr422_8u_C2P3R_Ctx,
    2,
    1
);

impl_subsampled_c2_to_p3!(
    "2 channel 8-bit unsigned packed `YCbCr422` to 3 channel 8-bit unsigned planar `YCbCr420` sampling format conversion.",
    ycbcr422_to_ycbcr420_u8_c2_to_p3,
    nppiYCbCr422ToYCbCr420_8u_C2P3R_Ctx,
    2,
    2
);

impl_subsampled_c2_to_p2!(
    "2 channel 8-bit unsigned packed `YCbCr422` to 2 channel 8-bit unsigned planar `YCbCr420` sampling format conversion.",
    ycbcr422_to_ycbcr420_u8_c2_to_p2,
    nppiYCbCr422ToYCbCr420_8u_C2P2R_Ctx,
    2,
    2
);

impl_subsampled_c2_to_p3!(
    "2 channel 8-bit unsigned packed `YCbCr422` to 3 channel 8-bit unsigned planar `YCbCr411` sampling format conversion.",
    ycbcr422_to_ycbcr411_u8_c2_to_p3,
    nppiYCbCr422ToYCbCr411_8u_C2P3R_Ctx,
    4,
    1
);

impl_subsampled_c2_to_p2!(
    "2 channel 8-bit unsigned packed `YCbCr422` to 2 channel 8-bit unsigned planar `YCbCr411` sampling format conversion.",
    ycbcr422_to_ycbcr411_u8_c2_to_p2,
    nppiYCbCr422ToYCbCr411_8u_C2P2R_Ctx,
    4,
    1
);

impl_subsampled_c2_to_p3!(
    "2 channel 8-bit unsigned packed `YCbCr422` to 3 channel 8-bit unsigned planar `YCrCb420` sampling format conversion.",
    ycbcr422_to_ycrcb420_u8_c2_to_p3,
    nppiYCbCr422ToYCrCb420_8u_C2P3R_Ctx,
    2,
    2
);

impl_subsampled_c2_to_p3!(
    "2 channel 8-bit unsigned packed `YCrCb422` to 3 channel 8-bit unsigned planar `YCbCr422` sampling format conversion.",
    ycrcb422_to_ycbcr422_u8_c2_to_p3,
    nppiYCrCb422ToYCbCr422_8u_C2P3R_Ctx,
    2,
    1
);

impl_subsampled_c2_to_p3!(
    "2 channel 8-bit unsigned packed `YCrCb422` to 3 channel 8-bit unsigned planar `YCbCr420` sampling format conversion.",
    ycrcb422_to_ycbcr420_u8_c2_to_p3,
    nppiYCrCb422ToYCbCr420_8u_C2P3R_Ctx,
    2,
    2
);

impl_subsampled_c2_to_p3!(
    "2 channel 8-bit unsigned packed `YCrCb422` to 3 channel 8-bit unsigned planar `YCbCr411` sampling format conversion.",
    ycrcb422_to_ycbcr411_u8_c2_to_p3,
    nppiYCrCb422ToYCbCr411_8u_C2P3R_Ctx,
    4,
    1
);

impl_subsampled_c2_to_p3!(
    "2 channel 8-bit unsigned packed `CbYCr422` to 3 channel 8-bit unsigned planar `YCbCr422` sampling format conversion.",
    cbycr422_to_ycbcr422_u8_c2_to_p3,
    nppiCbYCr422ToYCbCr422_8u_C2P3R_Ctx,
    2,
    1
);

impl_subsampled_c2_to_p3!(
    "2 channel 8-bit unsigned packed `CbYCr422` to 3 channel 8-bit unsigned planar `YCbCr420` sampling format conversion.",
    cbycr422_to_ycbcr420_u8_c2_to_p3,
    nppiCbYCr422ToYCbCr420_8u_C2P3R_Ctx,
    2,
    2
);

impl_subsampled_c2_to_p2!(
    "2 channel 8-bit unsigned packed `CbYCr422` to 2 channel 8-bit unsigned planar `YCbCr420` sampling format conversion.",
    cbycr422_to_ycbcr420_u8_c2_to_p2,
    nppiCbYCr422ToYCbCr420_8u_C2P2R_Ctx,
    2,
    2
);

impl_subsampled_c2_to_p3!(
    "2 channel 8-bit unsigned packed `CbYCr422` to 3 channel 8-bit unsigned planar `YCbCr411` sampling format conversion.",
    cbycr422_to_ycbcr411_u8_c2_to_p3,
    nppiCbYCr422ToYCbCr411_8u_C2P3R_Ctx,
    4,
    1
);

impl_subsampled_c2_to_p3!(
    "2 channel 8-bit unsigned packed `CbYCr422` to 3 channel 8-bit unsigned planar `YCrCb420` sampling format conversion.",
    cbycr422_to_ycrcb420_u8_c2_to_p3,
    nppiCbYCr422ToYCrCb420_8u_C2P3R_Ctx,
    2,
    2
);

impl_subsampled_p3_to_p2!(
    "3 channel 8-bit unsigned planar `YCbCr420` to 2 channel 8-bit unsigned planar `YCbCr420` sampling format conversion.",
    ycbcr420_u8_p3_to_p2,
    nppiYCbCr420_8u_P3P2R_Ctx,
    2,
    2,
    2,
    2
);

impl_subsampled_p3_to_p2!(
    "3 channel 8-bit unsigned planar `YCbCr420` to 2 channel 8-bit unsigned planar `YCbCr411` sampling format conversion.",
    ycbcr420_to_ycbcr411_u8_p3_to_p2,
    nppiYCbCr420ToYCbCr411_8u_P3P2R_Ctx,
    2,
    2,
    4,
    1
);

impl_subsampled_p3_to_p2!(
    "3 channel 8-bit unsigned planar `YCbCr422` to 2 channel 8-bit unsigned planar `YCbCr420` sampling format conversion.",
    ycbcr422_to_ycbcr420_u8_p3_to_p2,
    nppiYCbCr422ToYCbCr420_8u_P3P2R_Ctx,
    2,
    1,
    2,
    2
);

impl_subsampled_p3_to_p2!(
    "3 channel 8-bit unsigned planar `YCbCr422` to 2 channel 8-bit unsigned planar `YCbCr411` sampling format conversion.",
    ycbcr422_to_ycbcr411_u8_p3_to_p2,
    nppiYCbCr422ToYCbCr411_8u_P3P2R_Ctx,
    2,
    1,
    4,
    1
);

impl_subsampled_p3_to_p2!(
    "3 channel 8-bit unsigned planar `YCrCb420` to 2 channel 8-bit unsigned planar `YCbCr420` sampling format conversion.",
    ycrcb420_to_ycbcr420_u8_p3_to_p2,
    nppiYCrCb420ToYCbCr420_8u_P3P2R_Ctx,
    2,
    2,
    2,
    2
);

impl_subsampled_p3_to_p2!(
    "3 channel 8-bit unsigned planar `YCrCb420` to 2 channel 8-bit unsigned planar `YCbCr411` sampling format conversion.",
    ycrcb420_to_ycbcr411_u8_p3_to_p2,
    nppiYCrCb420ToYCbCr411_8u_P3P2R_Ctx,
    2,
    2,
    4,
    1
);

impl_subsampled_p3_to_p2!(
    "3 channel 8-bit unsigned planar `YCbCr411` to 2 channel 8-bit unsigned planar `YCbCr411` sampling format conversion.",
    ycbcr411_u8_p3_to_p2,
    nppiYCbCr411_8u_P3P2R_Ctx,
    4,
    1,
    4,
    1
);

impl_subsampled_p3_to_p2!(
    "3 channel 8-bit unsigned planar `YCbCr411` to 2 channel 8-bit unsigned planar `YCbCr420` sampling format conversion.",
    ycbcr411_to_ycbcr420_u8_p3_to_p2,
    nppiYCbCr411ToYCbCr420_8u_P3P2R_Ctx,
    4,
    1,
    2,
    2
);

impl_subsampled_p3_to_c2!(
    "3 channel 8-bit unsigned planar `YCbCr422` to 2 channel 8-bit unsigned packed `YCbCr422` sampling format conversion.",
    ycbcr422_u8_p3_to_c2,
    nppiYCbCr422_8u_P3C2R_Ctx,
    2,
    1
);

impl_subsampled_p3_to_c2!(
    "3 channel 8-bit unsigned planar `YCbCr422` to 2 channel 8-bit unsigned packed `YCrCb422` sampling format conversion.",
    ycbcr422_to_ycrcb422_u8_p3_to_c2,
    nppiYCbCr422ToYCrCb422_8u_P3C2R_Ctx,
    2,
    1
);

impl_subsampled_p3_to_c2!(
    "3 channel 8-bit unsigned planar `YCrCb420` to 2 channel 8-bit unsigned packed `YCbCr422` sampling format conversion.",
    ycrcb420_to_ycbcr422_u8_p3_to_c2,
    nppiYCrCb420ToYCbCr422_8u_P3C2R_Ctx,
    2,
    2
);

impl_subsampled_p3_to_c2!(
    "3 channel 8-bit unsigned planar `YCrCb420` to 2 channel 8-bit unsigned packed `CbYCr422` sampling format conversion.",
    ycrcb420_to_cbycr422_u8_p3_to_c2,
    nppiYCrCb420ToCbYCr422_8u_P3C2R_Ctx,
    2,
    2
);

impl_subsampled_p3_to_c2!(
    "3 channel 8-bit unsigned planar `YCbCr411` to 2 channel 8-bit unsigned packed `YCbCr422` sampling format conversion.",
    ycbcr411_to_ycbcr422_u8_p3_to_c2,
    nppiYCbCr411ToYCbCr422_8u_P3C2R_Ctx,
    4,
    1
);

impl_subsampled_p3_to_c2!(
    "3 channel 8-bit unsigned planar `YCbCr411` to 2 channel 8-bit unsigned packed `YCrCb422` sampling format conversion.",
    ycbcr411_to_ycrcb422_u8_p3_to_c2,
    nppiYCbCr411ToYCrCb422_8u_P3C2R_Ctx,
    4,
    1
);

impl_subsampled_p3_to_p3!(
    "3 channel 8-bit unsigned planar `YCbCr420` to 3 channel 8-bit unsigned planar `YCbCr422` sampling format conversion.",
    ycbcr420_to_ycbcr422_u8_p3_to_p3,
    nppiYCbCr420ToYCbCr422_8u_P3R_Ctx,
    2,
    2,
    2,
    1
);

impl_subsampled_p3_to_p3!(
    "3 channel 8-bit unsigned planar `YCbCr422` to 3 channel 8-bit unsigned planar `YCbCr420` sampling format conversion.",
    ycbcr422_to_ycbcr420_u8_p3_to_p3,
    nppiYCbCr422ToYCbCr420_8u_P3R_Ctx,
    2,
    1,
    2,
    2
);

impl_subsampled_p3_to_p3!(
    "3 channel 8-bit unsigned planar `YCbCr422` to 3 channel 8-bit unsigned planar `YCbCr411` sampling format conversion.",
    ycbcr422_to_ycbcr411_u8_p3_to_p3,
    nppiYCbCr422ToYCbCr411_8u_P3R_Ctx,
    2,
    1,
    4,
    1
);

impl_subsampled_p3_to_p3!(
    "3 channel 8-bit unsigned planar `YCrCb420` to 3 channel 8-bit unsigned planar `YCbCr422` sampling format conversion.",
    ycrcb420_to_ycbcr422_u8_p3_to_p3,
    nppiYCrCb420ToYCbCr422_8u_P3R_Ctx,
    2,
    2,
    2,
    1
);

impl_subsampled_p3_to_p3!(
    "3 channel 8-bit unsigned planar `YCbCr411` to 3 channel 8-bit unsigned planar `YCbCr422` sampling format conversion.",
    ycbcr411_to_ycbcr422_u8_p3_to_p3,
    nppiYCbCr411ToYCbCr422_8u_P3R_Ctx,
    4,
    1,
    2,
    1
);

impl_subsampled_p3_to_p3!(
    "3 channel 8-bit unsigned planar `YCbCr411` to 3 channel 8-bit unsigned planar `YCbCr420` sampling format conversion.",
    ycbcr411_to_ycbcr420_u8_p3_to_p3,
    nppiYCbCr411ToYCbCr420_8u_P3R_Ctx,
    4,
    1,
    2,
    2
);

impl_subsampled_p3_to_p3!(
    "3 channel 8-bit unsigned planar `YCbCr411` to 3 channel 8-bit unsigned planar `YCrCb422` sampling format conversion.",
    ycbcr411_to_ycrcb422_u8_p3_to_p3,
    nppiYCbCr411ToYCrCb422_8u_P3R_Ctx,
    4,
    1,
    2,
    1
);

impl_subsampled_p2_to_c2!(
    "2 channel 8-bit unsigned planar `YCbCr420` to 2 channel 8-bit unsigned packed `YCbCr422` sampling format conversion.",
    ycbcr420_to_ycbcr422_u8_p2_to_c2,
    nppiYCbCr420ToYCbCr422_8u_P2C2R_Ctx,
    2,
    2
);

impl_subsampled_p2_to_c2!(
    "2 channel 8-bit unsigned planar `YCbCr420` to 2 channel 8-bit unsigned packed `CbYCr422` sampling format conversion.",
    ycbcr420_to_cbycr422_u8_p2_to_c2,
    nppiYCbCr420ToCbYCr422_8u_P2C2R_Ctx,
    2,
    2
);

impl_subsampled_p2_to_c2!(
    "2 channel 8-bit unsigned planar `YCbCr411` to 2 channel 8-bit unsigned packed `YCbCr422` sampling format conversion.",
    ycbcr411_to_ycbcr422_u8_p2_to_c2,
    nppiYCbCr411ToYCbCr422_8u_P2C2R_Ctx,
    4,
    1
);

impl_subsampled_packed_to_p3!(
    "3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned planar `YCbCr420` color conversion.",
    rgb_to_ycbcr420_u8_c3_to_p3,
    C3,
    nppiRGBToYCbCr420_8u_C3P3R_Ctx,
    2,
    2
);

impl_subsampled_packed_to_p3!(
    "4 channel 8-bit unsigned packed RGB with alpha to 3 channel 8-bit unsigned planar `YCrCb420` color conversion.",
    rgb_to_ycrcb420_u8_ac4_to_p3,
    AC4,
    nppiRGBToYCrCb420_8u_AC4P3R_Ctx,
    2,
    2
);

impl_subsampled_packed_to_p3!(
    "3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar `YCbCr420` color conversion.",
    bgr_to_ycbcr420_u8_c3_to_p3,
    C3,
    nppiBGRToYCbCr420_8u_C3P3R_Ctx,
    2,
    2
);

impl_subsampled_packed_to_p3!(
    "4 channel 8-bit unsigned packed BGR with alpha to 3 channel 8-bit unsigned planar `YCbCr420` color conversion.",
    bgr_to_ycbcr420_u8_ac4_to_p3,
    AC4,
    nppiBGRToYCbCr420_8u_AC4P3R_Ctx,
    2,
    2
);

impl_subsampled_packed_to_p3!(
    "3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar `YCrCb420` color conversion.",
    bgr_to_ycrcb420_u8_c3_to_p3,
    C3,
    nppiBGRToYCrCb420_8u_C3P3R_Ctx,
    2,
    2
);

impl_subsampled_packed_to_p3!(
    "4 channel 8-bit unsigned packed BGR with alpha to 3 channel 8-bit unsigned planar `YCrCb420` color conversion.",
    bgr_to_ycrcb420_u8_ac4_to_p3,
    AC4,
    nppiBGRToYCrCb420_8u_AC4P3R_Ctx,
    2,
    2
);

impl_subsampled_packed_to_p3!(
    "3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar `YCbCr420_709CSC` color conversion.",
    bgr_to_ycbcr420_709csc_u8_c3_to_p3,
    C3,
    nppiBGRToYCbCr420_709CSC_8u_C3P3R_Ctx,
    2,
    2
);

impl_subsampled_packed_to_p3!(
    "4 channel 8-bit unsigned packed BGR with alpha to 3 channel 8-bit unsigned planar `YCbCr420_709CSC` color conversion.",
    bgr_to_ycbcr420_709csc_u8_ac4_to_p3,
    AC4,
    nppiBGRToYCbCr420_709CSC_8u_AC4P3R_Ctx,
    2,
    2
);

impl_subsampled_packed_to_p3!(
    "4 channel 8-bit unsigned packed BGR with alpha to 3 channel 8-bit unsigned planar `YCbCr420_709HDTV` color conversion.",
    bgr_to_ycbcr420_709hdtv_u8_ac4_to_p3,
    AC4,
    nppiBGRToYCbCr420_709HDTV_8u_AC4P3R_Ctx,
    2,
    2
);

impl_subsampled_packed_to_p3!(
    "3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar `YCrCb420_709CSC` color conversion.",
    bgr_to_ycrcb420_709csc_u8_c3_to_p3,
    C3,
    nppiBGRToYCrCb420_709CSC_8u_C3P3R_Ctx,
    2,
    2
);

impl_subsampled_packed_to_p3!(
    "4 channel 8-bit unsigned packed BGR with alpha to 3 channel 8-bit unsigned planar `YCrCb420_709CSC` color conversion.",
    bgr_to_ycrcb420_709csc_u8_ac4_to_p3,
    AC4,
    nppiBGRToYCrCb420_709CSC_8u_AC4P3R_Ctx,
    2,
    2
);

impl_subsampled_packed_to_p3!(
    "RGB to `YCbCr411` color conversion.",
    rgb_to_ycbcr411_u8_c3_to_p3,
    C3,
    nppiRGBToYCbCr411_8u_C3P3R_Ctx,
    4,
    1
);

impl_subsampled_packed_to_p3!(
    "4 channel 8-bit unsigned packed RGB with alpha to 3 channel 8-bit unsigned planar `YCbCr411` color conversion. images.",
    rgb_to_ycbcr411_u8_ac4_to_p3,
    AC4,
    nppiRGBToYCbCr411_8u_AC4P3R_Ctx,
    4,
    1
);

impl_subsampled_packed_to_p3!(
    "3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar `YCbCr411` color conversion.",
    bgr_to_ycbcr411_u8_c3_to_p3,
    C3,
    nppiBGRToYCbCr411_8u_C3P3R_Ctx,
    4,
    1
);

impl_subsampled_packed_to_p3!(
    "4 channel 8-bit unsigned packed BGR with alpha to 3 channel 8-bit unsigned planar `YCbCr411` color conversion.",
    bgr_to_ycbcr411_u8_ac4_to_p3,
    AC4,
    nppiBGRToYCbCr411_8u_AC4P3R_Ctx,
    4,
    1
);

impl_subsampled_packed_to_p3!(
    "3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned planar `YCbCr422` color conversion.",
    rgb_to_ycbcr422_u8_c3_to_p3,
    C3,
    nppiRGBToYCbCr422_8u_C3P3R_Ctx,
    2,
    1
);

impl_subsampled_packed_to_p3!(
    "3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar `YCbCr422` color conversion. images.",
    bgr_to_ycbcr422_u8_c3_to_p3,
    C3,
    nppiBGRToYCbCr422_8u_C3P3R_Ctx,
    2,
    1
);

impl_subsampled_packed_to_p3!(
    "4 channel 8-bit unsigned packed BGR with alpha to 3 channel 8-bit unsigned planar `YCbCr422` color conversion. images.",
    bgr_to_ycbcr422_u8_ac4_to_p3,
    AC4,
    nppiBGRToYCbCr422_8u_AC4P3R_Ctx,
    2,
    1
);

impl_nv12_packed_inverse!(
    "2 channel 8-bit unsigned planar NV21 to 4 channel 8-bit unsigned packed RGBA color conversion with constant alpha (0xFF).",
    nv21_to_rgb_u8_p2_to_c4,
    u8,
    C4,
    nppiNV21ToRGB_8u_P2C4R_Ctx
);

impl_nv12_packed_inverse!(
    "2 channel 8-bit unsigned planar NV21 to 4 channel 8-bit unsigned packed BGRA color conversion with constant alpha (0xFF).",
    nv21_to_bgr_u8_p2_to_c4,
    u8,
    C4,
    nppiNV21ToBGR_8u_P2C4R_Ctx
);

impl_color_convert_batch_same_layout!(
    "Converts YUV color values to RGB color values.",
    yuv_to_rgb_batch_u8_c3,
    u8,
    C3,
    nppiYUVToRGBBatch_8u_C3R_Ctx
);

impl_color_convert_batch_same_layout_advanced!(
    "Converts YUV color values to RGB color values.",
    yuv_to_rgb_batch_u8_c3_advanced,
    u8,
    C3,
    nppiYUVToRGBBatch_8u_C3R_Advanced_Ctx
);

impl_color_convert_batch_planar_to_packed!(
    "Converts YUV color values to RGB color values.",
    yuv_to_rgb_batch_u8_p3_to_c3,
    u8,
    3,
    C3,
    nppiYUVToRGBBatch_8u_P3C3R_Ctx
);

impl_color_convert_batch_planar_to_packed_advanced!(
    "Converts YUV color values to RGB color values.",
    yuv_to_rgb_batch_u8_p3_to_c3_advanced,
    u8,
    3,
    C3,
    nppiYUVToRGBBatch_8u_P3C3R_Advanced_Ctx
);

impl_subsampled_color_convert_batch_planar_to_packed!(
    "Converts YUV422 color values to RGB color values.",
    yuv422_to_rgb_batch_u8_p3_to_c3,
    u8,
    C3,
    nppiYUV422ToRGBBatch_8u_P3C3R_Ctx,
    2,
    1
);

impl_subsampled_color_convert_batch_planar_to_packed_advanced!(
    "Converts YUV422 color values to RGB color values.",
    yuv422_to_rgb_batch_u8_p3_to_c3_advanced,
    u8,
    C3,
    nppiYUV422ToRGBBatch_8u_P3C3R_Advanced_Ctx,
    2,
    1
);

impl_subsampled_color_convert_batch_planar_to_packed!(
    "Converts YUV420 color values to RGB color values.",
    yuv420_to_rgb_batch_u8_p3_to_c3,
    u8,
    C3,
    nppiYUV420ToRGBBatch_8u_P3C3R_Ctx,
    2,
    2
);

impl_subsampled_color_convert_batch_planar_to_packed_advanced!(
    "Converts YUV420 color values to RGB color values.",
    yuv420_to_rgb_batch_u8_p3_to_c3_advanced,
    u8,
    C3,
    nppiYUV420ToRGBBatch_8u_P3C3R_Advanced_Ctx,
    2,
    2
);

impl_color_convert_packed_to_planar!(
    "Converts RGB color values to YUV color values.",
    rgb_to_yuv_u8_c3_to_p3,
    u8,
    C3,
    3,
    nppiRGBToYUV_8u_C3P3R_Ctx
);

impl_color_convert_packed_to_planar!(
    "Converts RGB color values to YUV color values.",
    rgb_to_yuv_u8_ac4_to_p4,
    u8,
    AC4,
    4,
    nppiRGBToYUV_8u_AC4P4R_Ctx
);

impl_color_convert_planar_to_packed!(
    "Converts YUV color values to RGB color values.",
    yuv_to_rgb_u8_p3_to_c3,
    u8,
    3,
    C3,
    nppiYUVToRGB_8u_P3C3R_Ctx
);
