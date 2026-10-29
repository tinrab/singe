use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        color_descriptors::validate_same_size,
        view::{C3, ImageOutputView, ImageView, PlanarImageOutputView, PlanarImageView},
    },
    try_ffi,
    types::DataTypeLike,
};

impl_color_convert_planar_to_planar!(
    "Converts RGB color values to `YCbCr444` color values.",
    rgb_to_ycbcr444_jpeg_u8_p3,
    u8,
    3,
    nppiRGBToYCbCr444_JPEG_8u_P3R_Ctx
);

impl_color_convert_planar_to_planar!(
    "Converts BGR color values to `YCbCr444` color values.",
    bgr_to_ycbcr444_jpeg_u8_p3,
    u8,
    3,
    nppiBGRToYCbCr444_JPEG_8u_P3R_Ctx
);

impl_color_convert_planar_to_planar!(
    "Converts `YCbCr444` color values to RGB color values.",
    ycbcr444_to_rgb_jpeg_u8_p3,
    u8,
    3,
    nppiYCbCr444ToRGB_JPEG_8u_P3R_Ctx
);

impl_color_convert_planar_to_planar!(
    "Converts `YCbCr444` color values to BGR color values.",
    ycbcr444_to_bgr_jpeg_u8_p3,
    u8,
    3,
    nppiYCbCr444ToBGR_JPEG_8u_P3R_Ctx
);

impl_color_convert_packed_to_planar!(
    "Converts RGB color values to `YCbCr444` color values.",
    rgb_to_ycbcr444_jpeg_u8_c3_to_p3,
    u8,
    C3,
    3,
    nppiRGBToYCbCr444_JPEG_8u_C3P3R_Ctx
);

impl_color_convert_packed_to_planar!(
    "Converts BGR color values to `YCbCr444` color values.",
    bgr_to_ycbcr444_jpeg_u8_c3_to_p3,
    u8,
    C3,
    3,
    nppiBGRToYCbCr444_JPEG_8u_C3P3R_Ctx
);

impl_color_convert_planar_to_packed!(
    "Converts `YCbCr444` color values to RGB color values.",
    ycbcr444_to_rgb_jpeg_u8_p3_to_c3,
    u8,
    3,
    C3,
    nppiYCbCr444ToRGB_JPEG_8u_P3C3R_Ctx
);

impl_color_convert_planar_to_packed!(
    "Converts `YCbCr444` color values to BGR color values.",
    ycbcr444_to_bgr_jpeg_u8_p3_to_c3,
    u8,
    3,
    C3,
    nppiYCbCr444ToBGR_JPEG_8u_P3C3R_Ctx
);

impl_color_convert_planar_to_planar_dispatch!(
    "Converts RGB color values to `YCbCr444` color values.",

    RgbToYcbcr444JpegP3,
    rgb_to_ycbcr444_jpeg_p3,
    rgb_to_ycbcr444_jpeg_p3,
    3,
    [u8 => rgb_to_ycbcr444_jpeg_u8_p3]

);

impl_color_convert_planar_to_planar_dispatch!(
    "Converts BGR color values to `YCbCr444` color values.",

    BgrToYcbcr444JpegP3,
    bgr_to_ycbcr444_jpeg_p3,
    bgr_to_ycbcr444_jpeg_p3,
    3,
    [u8 => bgr_to_ycbcr444_jpeg_u8_p3]

);

impl_color_convert_planar_to_planar_dispatch!(
    "Converts `YCbCr444` color values to RGB color values.",

    Ycbcr444ToRgbJpegP3,
    ycbcr444_to_rgb_jpeg_p3,
    ycbcr444_to_rgb_jpeg_p3,
    3,
    [u8 => ycbcr444_to_rgb_jpeg_u8_p3]

);

impl_color_convert_planar_to_planar_dispatch!(
    "Converts `YCbCr444` color values to BGR color values.",

    Ycbcr444ToBgrJpegP3,
    ycbcr444_to_bgr_jpeg_p3,
    ycbcr444_to_bgr_jpeg_p3,
    3,
    [u8 => ycbcr444_to_bgr_jpeg_u8_p3]

);

impl_color_convert_packed_c3_to_planar_dispatch!(
    "Converts RGB color values to `YCbCr444` color values.",

    RgbToYcbcr444JpegC3ToP3,
    rgb_to_ycbcr444_jpeg_c3_to_p3,
    rgb_to_ycbcr444_jpeg_c3_to_p3,
    3,
    [u8 => rgb_to_ycbcr444_jpeg_u8_c3_to_p3]

);

impl_color_convert_packed_c3_to_planar_dispatch!(
    "Converts BGR color values to `YCbCr444` color values.",

    BgrToYcbcr444JpegC3ToP3,
    bgr_to_ycbcr444_jpeg_c3_to_p3,
    bgr_to_ycbcr444_jpeg_c3_to_p3,
    3,
    [u8 => bgr_to_ycbcr444_jpeg_u8_c3_to_p3]

);

impl_color_convert_planar_to_packed_c3_dispatch!(
    "Converts `YCbCr444` color values to RGB color values.",

    Ycbcr444ToRgbJpegP3ToC3,
    ycbcr444_to_rgb_jpeg_p3_to_c3,
    ycbcr444_to_rgb_jpeg_p3_to_c3,
    3,
    [u8 => ycbcr444_to_rgb_jpeg_u8_p3_to_c3]

);

impl_color_convert_planar_to_packed_c3_dispatch!(
    "Converts `YCbCr444` color values to BGR color values.",

    Ycbcr444ToBgrJpegP3ToC3,
    ycbcr444_to_bgr_jpeg_p3_to_c3,
    ycbcr444_to_bgr_jpeg_p3_to_c3,
    3,
    [u8 => ycbcr444_to_bgr_jpeg_u8_p3_to_c3]

);

impl_color_convert_planar_to_planar!(
    "Converts JPEG YCCK planes to CMYK using 601 RGB coefficients and CMY inversion.",
    ycck_to_cmyk_jpeg_601_u8_p4,
    u8,
    4,
    nppiYCCKToCMYK_JPEG_601_8u_P4R_Ctx
);

impl_color_convert_planar_to_planar_different_planes!(
    "Converts JPEG CMYK or YCCK planes to RGB planes without color-profile conversion.",
    cmyk_or_ycck_to_rgb_jpeg_u8_p4_to_p3,
    u8,
    4,
    3,
    nppiCMYKOrYCCKToRGB_JPEG_8u_P4P3R_Ctx
);

impl_color_convert_planar_to_packed!(
    "Converts JPEG CMYK or YCCK planes to packed RGB pixels without color-profile conversion.",
    cmyk_or_ycck_to_rgb_jpeg_u8_p4_to_c3,
    u8,
    4,
    C3,
    nppiCMYKOrYCCKToRGB_JPEG_8u_P4C3R_Ctx
);

impl_color_convert_planar_to_planar_different_planes!(
    "Converts JPEG CMYK or YCCK planes to BGR planes without color-profile conversion.",
    cmyk_or_ycck_to_bgr_jpeg_u8_p4_to_p3,
    u8,
    4,
    3,
    nppiCMYKOrYCCKToBGR_JPEG_8u_P4P3R_Ctx
);

impl_color_convert_planar_to_packed!(
    "Converts JPEG CMYK or YCCK planes to packed BGR pixels without color-profile conversion.",
    cmyk_or_ycck_to_bgr_jpeg_u8_p4_to_c3,
    u8,
    4,
    C3,
    nppiCMYKOrYCCKToBGR_JPEG_8u_P4C3R_Ctx
);

macro_rules! impl_color_convert_planar_to_planar_different_planes_dispatch {
    ($trait:ident, $method:ident, $function:ident, $source_planes:literal, $destination_planes:literal, [$($ty:ty => $direct:ident),* $(,)?]) => {
        /// Dispatches JPEG CMYK or YCCK conversion into three RGB or BGR planes.
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and plane-layout ABI, color semantics,
        /// and the method's geometry validation, memory-access rules, and queued lifetimes.
        pub unsafe trait $trait: DataTypeLike {
            /// Converts four JPEG CMYK or YCCK source planes to three RGB or BGR output planes.
            ///
            /// The function name selects RGB or BGR component order. This operation does not perform
            /// color-profile conversion.
            ///
            /// Enqueues work on the selected stream.
            /// Plane order follows the named color components.
            /// Ignored channels, pixels outside the processed ROI, and row padding may remain unwritten.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
            /// Source and destination image sizes must match.
            /// Returns an error if NPP rejects the image geometry or conversion parameters.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source samples must be initialized and ready when read by the selected stream.
            /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destinations during native use.
            /// Retain image allocations through queued completion, including submission errors and captured graph replay.
            /// Planes must satisfy the native common-size, common-byte-stride, and pointer-array requirements.
            /// Image layouts, component ranges, subsampling geometry, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &PlanarImageView<'_, Self, $source_planes>,
                destination: &mut PlanarImageOutputView<'_, Self, $destination_planes>,
            ) -> Result<()>;
        }

        /// Converts four JPEG CMYK or YCCK source planes to three RGB or BGR output planes.
        ///
        /// The function name selects RGB or BGR component order. This operation does not perform
        /// color-profile conversion.
        ///
        /// Enqueues work on the selected stream.
        /// Plane order follows the named color components.
        /// Ignored channels, pixels outside the processed ROI, and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or a participating image belongs to another CUDA context.
        /// Source and destination image sizes must match.
        /// Returns an error if NPP rejects the image geometry or conversion parameters.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs and other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Planes must satisfy the native common-size, common-byte-stride, and pointer-array requirements.
        /// Image layouts, component ranges, subsampling geometry, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &PlanarImageView<'_, T, $source_planes>,
            destination: &mut PlanarImageOutputView<'_, T, $destination_planes>,
        ) -> Result<()>
        where
            T: $trait,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, source, destination)
            }
}

        $(
            unsafe impl $trait for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &PlanarImageView<'_, Self, $source_planes>,
                    destination: &mut PlanarImageOutputView<'_, Self, $destination_planes>,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, destination)
                    }
}
            }
        )*
    };
}

impl_color_convert_planar_to_planar_dispatch!(
    "Converts JPEG YCCK planes to CMYK using 601 RGB coefficients and CMY inversion.",

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
    "Converts JPEG CMYK or YCCK planes to RGB planes without color-profile conversion.",

    CmykOrYcckToRgbJpegP4ToC3,
    cmyk_or_ycck_to_rgb_jpeg_p4_to_c3,
    cmyk_or_ycck_to_rgb_jpeg_p4_to_c3,
    4,
    [u8 => cmyk_or_ycck_to_rgb_jpeg_u8_p4_to_c3]

);

impl_color_convert_planar_to_packed_c3_dispatch!(
    "Converts JPEG CMYK or YCCK planes to BGR planes without color-profile conversion.",

    CmykOrYcckToBgrJpegP4ToC3,
    cmyk_or_ycck_to_bgr_jpeg_p4_to_c3,
    cmyk_or_ycck_to_bgr_jpeg_p4_to_c3,
    4,
    [u8 => cmyk_or_ycck_to_bgr_jpeg_u8_p4_to_c3]

);
