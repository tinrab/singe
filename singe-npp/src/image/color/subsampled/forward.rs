use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        color_descriptors::{
            validate_jpeg_subsampled_destination_planes, validate_rgb_source_planes,
        },
        view::{AC4, C1, C3, ImageOutputView, ImageView},
    },
    try_ffi,
    types::DataTypeLike,
};

impl_jpeg_subsampled_planar_forward!(
    "3 channel 8-bit unsigned planar RGB to 3 channel 8-bit unsigned planar `YCbCr420` color conversion.",
    rgb_to_ycbcr420_jpeg_u8_p3,
    nppiRGBToYCbCr420_JPEG_8u_P3R_Ctx,
    2,
    2
);

impl_jpeg_subsampled_planar_forward!(
    "3 channel 8-bit unsigned planar RGB to 3 channel 8-bit unsigned planar `YCbCr422` color conversion.",
    rgb_to_ycbcr422_jpeg_u8_p3,
    nppiRGBToYCbCr422_JPEG_8u_P3R_Ctx,
    2,
    1
);

impl_jpeg_subsampled_planar_forward!(
    "3 channel 8-bit unsigned planar RGB to 3 channel 8-bit unsigned planar `YCbCr411` color conversion.",
    rgb_to_ycbcr411_jpeg_u8_p3,
    nppiRGBToYCbCr411_JPEG_8u_P3R_Ctx,
    4,
    1
);

impl_jpeg_subsampled_planar_forward!(
    "3 channel 8-bit unsigned planar BGR to 3 channel 8-bit unsigned planar `YCbCr420` color conversion.",
    bgr_to_ycbcr420_jpeg_u8_p3,
    nppiBGRToYCbCr420_JPEG_8u_P3R_Ctx,
    2,
    2
);

impl_jpeg_subsampled_planar_forward!(
    "3 channel 8-bit unsigned planar BGR to 3 channel 8-bit unsigned planar `YCbCr422` color conversion.",
    bgr_to_ycbcr422_jpeg_u8_p3,
    nppiBGRToYCbCr422_JPEG_8u_P3R_Ctx,
    2,
    1
);

impl_jpeg_subsampled_planar_forward!(
    "3 channel 8-bit unsigned planar BGR to 3 channel 8-bit unsigned planar `YCbCr411` color conversion.",
    bgr_to_ycbcr411_jpeg_u8_p3,
    nppiBGRToYCbCr411_JPEG_8u_P3R_Ctx,
    4,
    1
);

impl_jpeg_subsampled_packed_forward!(
    "3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned planar `YCbCr420` color conversion.",
    rgb_to_ycbcr420_jpeg_u8_c3_to_p3,
    C3,
    nppiRGBToYCbCr420_JPEG_8u_C3P3R_Ctx,
    2,
    2
);

impl_jpeg_subsampled_packed_forward!(
    "3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned planar `YCbCr422` color conversion.",
    rgb_to_ycbcr422_jpeg_u8_c3_to_p3,
    C3,
    nppiRGBToYCbCr422_JPEG_8u_C3P3R_Ctx,
    2,
    1
);

impl_jpeg_subsampled_packed_forward!(
    "3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned planar `YCbCr411` color conversion.",
    rgb_to_ycbcr411_jpeg_u8_c3_to_p3,
    C3,
    nppiRGBToYCbCr411_JPEG_8u_C3P3R_Ctx,
    4,
    1
);

impl_jpeg_subsampled_packed_forward!(
    "3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar `YCbCr420` color conversion.",
    bgr_to_ycbcr420_jpeg_u8_c3_to_p3,
    C3,
    nppiBGRToYCbCr420_JPEG_8u_C3P3R_Ctx,
    2,
    2
);

impl_jpeg_subsampled_packed_forward!(
    "3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar `YCbCr422` color conversion.",
    bgr_to_ycbcr422_jpeg_u8_c3_to_p3,
    C3,
    nppiBGRToYCbCr422_JPEG_8u_C3P3R_Ctx,
    2,
    1
);

impl_jpeg_subsampled_packed_forward!(
    "3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar `YCbCr411` color conversion.",
    bgr_to_ycbcr411_jpeg_u8_c3_to_p3,
    C3,
    nppiBGRToYCbCr411_JPEG_8u_C3P3R_Ctx,
    4,
    1
);

impl_jpeg_subsampled_planar_forward!(
    "3 channel 8-bit unsigned planar RGB to 3 channel 8-bit unsigned planar YUV420 color conversion.",
    rgb_to_yuv420_u8_p3,
    nppiRGBToYUV420_8u_P3R_Ctx,
    2,
    2
);

impl_jpeg_subsampled_packed_forward!(
    "3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned planar YUV420 color conversion.",
    rgb_to_yuv420_u8_c3_to_p3,
    C3,
    nppiRGBToYUV420_8u_C3P3R_Ctx,
    2,
    2
);

impl_jpeg_subsampled_packed_forward!(
    "4 channel 8-bit unsigned packed BGR with alpha to 3 channel 8-bit unsigned planar YUV420 color conversion. Input alpha is ignored.",
    bgr_to_yuv420_u8_ac4_to_p3,
    AC4,
    nppiBGRToYUV420_8u_AC4P3R_Ctx,
    2,
    2
);

impl_jpeg_subsampled_planar_forward!(
    "3 channel 8-bit unsigned planar RGB to 3 channel 8-bit unsigned planar YUV422 color conversion.",
    rgb_to_yuv422_u8_p3,
    nppiRGBToYUV422_8u_P3R_Ctx,
    2,
    1
);

impl_jpeg_subsampled_packed_forward!(
    "3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned planar YUV422 color conversion.",
    rgb_to_yuv422_u8_c3_to_p3,
    C3,
    nppiRGBToYUV422_8u_C3P3R_Ctx,
    2,
    1
);

macro_rules! impl_jpeg_subsampled_planar_forward_dispatch {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        /// Dispatches this subsampled color conversion for a native element type.
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element representation, color semantics, and plane geometry.
        /// Uphold the documented conversion and raw image execution contracts.
        pub unsafe trait $trait: DataTypeLike {
            #[doc = $doc]
            unsafe fn $method(
                stream_context: &StreamContext,
                source_0: &ImageView<'_, Self, C1>,
                source_1: &ImageView<'_, Self, C1>,
                source_2: &ImageView<'_, Self, C1>,
                destination_y: &mut ImageOutputView<'_, Self, C1>,
                destination_cb: &mut ImageOutputView<'_, Self, C1>,
                destination_cr: &mut ImageOutputView<'_, Self, C1>,
            ) -> Result<()>;
        }

        #[doc = $doc]
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source_0: &ImageView<'_, T, C1>,
            source_1: &ImageView<'_, T, C1>,
            source_2: &ImageView<'_, T, C1>,
            destination_y: &mut ImageOutputView<'_, T, C1>,
            destination_cb: &mut ImageOutputView<'_, T, C1>,
            destination_cr: &mut ImageOutputView<'_, T, C1>,
        ) -> Result<()>
        where
            T: $trait,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
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
        }

        $(
            unsafe impl $trait for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source_0: &ImageView<'_, Self, C1>,
                    source_1: &ImageView<'_, Self, C1>,
                    source_2: &ImageView<'_, Self, C1>,
                    destination_y: &mut ImageOutputView<'_, Self, C1>,
                    destination_cb: &mut ImageOutputView<'_, Self, C1>,
                    destination_cr: &mut ImageOutputView<'_, Self, C1>,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
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
            }
        )*
    };
}

macro_rules! impl_jpeg_subsampled_packed_forward_dispatch {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),* $(,)?]) => {
        /// Dispatches this subsampled color conversion for a native element type.
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element representation, color semantics, and plane geometry.
        /// Uphold the documented conversion and raw image execution contracts.
        pub unsafe trait $trait: DataTypeLike {
            #[doc = $doc]
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, $layout>,
                destination_y: &mut ImageOutputView<'_, Self, C1>,
                destination_cb: &mut ImageOutputView<'_, Self, C1>,
                destination_cr: &mut ImageOutputView<'_, Self, C1>,
            ) -> Result<()>;
        }

        #[doc = $doc]
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination_y: &mut ImageOutputView<'_, T, C1>,
            destination_cb: &mut ImageOutputView<'_, T, C1>,
            destination_cr: &mut ImageOutputView<'_, T, C1>,
        ) -> Result<()>
        where
            T: $trait,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(
                    stream_context,
                    source,
                    destination_y,
                    destination_cb,
                    destination_cr,
                )
            }
        }

        $(
            unsafe impl $trait for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination_y: &mut ImageOutputView<'_, Self, C1>,
                    destination_cb: &mut ImageOutputView<'_, Self, C1>,
                    destination_cr: &mut ImageOutputView<'_, Self, C1>,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(
                            stream_context,
                            source,
                            destination_y,
                            destination_cb,
                            destination_cr,
                        )
                    }
                }
            }
        )*
    };
}

impl_jpeg_subsampled_planar_forward_dispatch!(
    "3 channel planar RGB to 3 channel planar `YCbCr420` color conversion.\n\nEnqueues work on the selected stream.\nRow padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nChroma width and height must equal the full-resolution dimensions divided by 2 and 2, respectively.\nFull-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.\nThe three full-resolution source color planes must have equal byte strides.\nChroma plane strides are passed independently.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples must be initialized and ready when read by the selected stream.\nKeep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.\nRetain image allocations through queued completion, including submission errors and captured graph replay.\nColor-component order, packed layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.",

    RgbToYcbcr420JpegP3,
    rgb_to_ycbcr420_jpeg_p3,
    rgb_to_ycbcr420_jpeg_p3,
    [u8 => rgb_to_ycbcr420_jpeg_u8_p3]

);

impl_jpeg_subsampled_planar_forward_dispatch!(
    "3 channel planar RGB to 3 channel planar `YCbCr422` color conversion.\n\nEnqueues work on the selected stream.\nRow padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nChroma width and height must equal the full-resolution dimensions divided by 2 and 1, respectively.\nFull-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.\nThe three full-resolution source color planes must have equal byte strides.\nChroma plane strides are passed independently.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples must be initialized and ready when read by the selected stream.\nKeep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.\nRetain image allocations through queued completion, including submission errors and captured graph replay.\nColor-component order, packed layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.",

    RgbToYcbcr422JpegP3,
    rgb_to_ycbcr422_jpeg_p3,
    rgb_to_ycbcr422_jpeg_p3,
    [u8 => rgb_to_ycbcr422_jpeg_u8_p3]

);

impl_jpeg_subsampled_planar_forward_dispatch!(
    "3 channel planar RGB to 3 channel planar `YCbCr411` color conversion.\n\nEnqueues work on the selected stream.\nRow padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nChroma width and height must equal the full-resolution dimensions divided by 4 and 1, respectively.\nFull-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.\nThe three full-resolution source color planes must have equal byte strides.\nChroma plane strides are passed independently.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples must be initialized and ready when read by the selected stream.\nKeep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.\nRetain image allocations through queued completion, including submission errors and captured graph replay.\nColor-component order, packed layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.",

    RgbToYcbcr411JpegP3,
    rgb_to_ycbcr411_jpeg_p3,
    rgb_to_ycbcr411_jpeg_p3,
    [u8 => rgb_to_ycbcr411_jpeg_u8_p3]

);

impl_jpeg_subsampled_planar_forward_dispatch!(
    "3 channel planar BGR to 3 channel planar `YCbCr420` color conversion.\n\nEnqueues work on the selected stream.\nRow padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nChroma width and height must equal the full-resolution dimensions divided by 2 and 2, respectively.\nFull-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.\nThe three full-resolution source color planes must have equal byte strides.\nChroma plane strides are passed independently.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples must be initialized and ready when read by the selected stream.\nKeep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.\nRetain image allocations through queued completion, including submission errors and captured graph replay.\nColor-component order, packed layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.",

    BgrToYcbcr420JpegP3,
    bgr_to_ycbcr420_jpeg_p3,
    bgr_to_ycbcr420_jpeg_p3,
    [u8 => bgr_to_ycbcr420_jpeg_u8_p3]

);

impl_jpeg_subsampled_planar_forward_dispatch!(
    "3 channel planar BGR to 3 channel planar `YCbCr422` color conversion.\n\nEnqueues work on the selected stream.\nRow padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nChroma width and height must equal the full-resolution dimensions divided by 2 and 1, respectively.\nFull-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.\nThe three full-resolution source color planes must have equal byte strides.\nChroma plane strides are passed independently.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples must be initialized and ready when read by the selected stream.\nKeep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.\nRetain image allocations through queued completion, including submission errors and captured graph replay.\nColor-component order, packed layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.",

    BgrToYcbcr422JpegP3,
    bgr_to_ycbcr422_jpeg_p3,
    bgr_to_ycbcr422_jpeg_p3,
    [u8 => bgr_to_ycbcr422_jpeg_u8_p3]

);

impl_jpeg_subsampled_planar_forward_dispatch!(
    "3 channel planar BGR to 3 channel planar `YCbCr411` color conversion.\n\nEnqueues work on the selected stream.\nRow padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nChroma width and height must equal the full-resolution dimensions divided by 4 and 1, respectively.\nFull-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.\nThe three full-resolution source color planes must have equal byte strides.\nChroma plane strides are passed independently.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples must be initialized and ready when read by the selected stream.\nKeep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.\nRetain image allocations through queued completion, including submission errors and captured graph replay.\nColor-component order, packed layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.",

    BgrToYcbcr411JpegP3,
    bgr_to_ycbcr411_jpeg_p3,
    bgr_to_ycbcr411_jpeg_p3,
    [u8 => bgr_to_ycbcr411_jpeg_u8_p3]

);

impl_jpeg_subsampled_planar_forward_dispatch!(
    "3 channel planar RGB to 3 channel planar YUV420 color conversion.\n\nEnqueues work on the selected stream.\nRow padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nChroma width and height must equal the full-resolution dimensions divided by 2 and 2, respectively.\nFull-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.\nThe three full-resolution source color planes must have equal byte strides.\nChroma plane strides are passed independently.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples must be initialized and ready when read by the selected stream.\nKeep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.\nRetain image allocations through queued completion, including submission errors and captured graph replay.\nColor-component order, packed layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.",
RgbToYuv420P3, rgb_to_yuv420_p3, rgb_to_yuv420_p3, [
    u8 => rgb_to_yuv420_u8_p3
]
);

impl_jpeg_subsampled_planar_forward_dispatch!(
    "3 channel planar RGB to 3 channel planar YUV422 color conversion.\n\nEnqueues work on the selected stream.\nRow padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nChroma width and height must equal the full-resolution dimensions divided by 2 and 1, respectively.\nFull-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.\nThe three full-resolution source color planes must have equal byte strides.\nChroma plane strides are passed independently.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples must be initialized and ready when read by the selected stream.\nKeep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.\nRetain image allocations through queued completion, including submission errors and captured graph replay.\nColor-component order, packed layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.",
RgbToYuv422P3, rgb_to_yuv422_p3, rgb_to_yuv422_p3, [
    u8 => rgb_to_yuv422_u8_p3
]
);

impl_jpeg_subsampled_packed_forward_dispatch!(
    "3 channel packed RGB to 3 channel planar `YCbCr420` color conversion.\n\nEnqueues work on the selected stream.\nRow padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nChroma width and height must equal the full-resolution dimensions divided by 2 and 2, respectively.\nFull-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.\nChroma plane strides are passed independently.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples must be initialized and ready when read by the selected stream.\nKeep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.\nRetain image allocations through queued completion, including submission errors and captured graph replay.\nColor-component order, packed layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.",

    RgbToYcbcr420JpegC3ToP3,
    rgb_to_ycbcr420_jpeg_c3_to_p3,
    rgb_to_ycbcr420_jpeg_c3_to_p3,
    C3,
    [u8 => rgb_to_ycbcr420_jpeg_u8_c3_to_p3]

);

impl_jpeg_subsampled_packed_forward_dispatch!(
    "3 channel packed RGB to 3 channel planar `YCbCr422` color conversion.\n\nEnqueues work on the selected stream.\nRow padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nChroma width and height must equal the full-resolution dimensions divided by 2 and 1, respectively.\nFull-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.\nChroma plane strides are passed independently.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples must be initialized and ready when read by the selected stream.\nKeep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.\nRetain image allocations through queued completion, including submission errors and captured graph replay.\nColor-component order, packed layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.",

    RgbToYcbcr422JpegC3ToP3,
    rgb_to_ycbcr422_jpeg_c3_to_p3,
    rgb_to_ycbcr422_jpeg_c3_to_p3,
    C3,
    [u8 => rgb_to_ycbcr422_jpeg_u8_c3_to_p3]

);

impl_jpeg_subsampled_packed_forward_dispatch!(
    "3 channel packed RGB to 3 channel planar `YCbCr411` color conversion.\n\nEnqueues work on the selected stream.\nRow padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nChroma width and height must equal the full-resolution dimensions divided by 4 and 1, respectively.\nFull-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.\nChroma plane strides are passed independently.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples must be initialized and ready when read by the selected stream.\nKeep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.\nRetain image allocations through queued completion, including submission errors and captured graph replay.\nColor-component order, packed layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.",

    RgbToYcbcr411JpegC3ToP3,
    rgb_to_ycbcr411_jpeg_c3_to_p3,
    rgb_to_ycbcr411_jpeg_c3_to_p3,
    C3,
    [u8 => rgb_to_ycbcr411_jpeg_u8_c3_to_p3]

);

impl_jpeg_subsampled_packed_forward_dispatch!(
    "3 channel packed BGR to 3 channel planar `YCbCr420` color conversion.\n\nEnqueues work on the selected stream.\nRow padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nChroma width and height must equal the full-resolution dimensions divided by 2 and 2, respectively.\nFull-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.\nChroma plane strides are passed independently.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples must be initialized and ready when read by the selected stream.\nKeep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.\nRetain image allocations through queued completion, including submission errors and captured graph replay.\nColor-component order, packed layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.",

    BgrToYcbcr420JpegC3ToP3,
    bgr_to_ycbcr420_jpeg_c3_to_p3,
    bgr_to_ycbcr420_jpeg_c3_to_p3,
    C3,
    [u8 => bgr_to_ycbcr420_jpeg_u8_c3_to_p3]

);

impl_jpeg_subsampled_packed_forward_dispatch!(
    "3 channel packed BGR to 3 channel planar `YCbCr422` color conversion.\n\nEnqueues work on the selected stream.\nRow padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nChroma width and height must equal the full-resolution dimensions divided by 2 and 1, respectively.\nFull-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.\nChroma plane strides are passed independently.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples must be initialized and ready when read by the selected stream.\nKeep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.\nRetain image allocations through queued completion, including submission errors and captured graph replay.\nColor-component order, packed layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.",

    BgrToYcbcr422JpegC3ToP3,
    bgr_to_ycbcr422_jpeg_c3_to_p3,
    bgr_to_ycbcr422_jpeg_c3_to_p3,
    C3,
    [u8 => bgr_to_ycbcr422_jpeg_u8_c3_to_p3]

);

impl_jpeg_subsampled_packed_forward_dispatch!(
    "3 channel packed BGR to 3 channel planar `YCbCr411` color conversion.\n\nEnqueues work on the selected stream.\nRow padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nChroma width and height must equal the full-resolution dimensions divided by 4 and 1, respectively.\nFull-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.\nChroma plane strides are passed independently.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples must be initialized and ready when read by the selected stream.\nKeep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.\nRetain image allocations through queued completion, including submission errors and captured graph replay.\nColor-component order, packed layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.",

    BgrToYcbcr411JpegC3ToP3,
    bgr_to_ycbcr411_jpeg_c3_to_p3,
    bgr_to_ycbcr411_jpeg_c3_to_p3,
    C3,
    [u8 => bgr_to_ycbcr411_jpeg_u8_c3_to_p3]

);

impl_jpeg_subsampled_packed_forward_dispatch!(
    "3 channel packed RGB to 3 channel planar YUV420 color conversion.\n\nEnqueues work on the selected stream.\nRow padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nChroma width and height must equal the full-resolution dimensions divided by 2 and 2, respectively.\nFull-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.\nChroma plane strides are passed independently.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples must be initialized and ready when read by the selected stream.\nKeep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.\nRetain image allocations through queued completion, including submission errors and captured graph replay.\nColor-component order, packed layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.",
RgbToYuv420C3ToP3, rgb_to_yuv420_c3_to_p3, rgb_to_yuv420_c3_to_p3, C3, [
    u8 => rgb_to_yuv420_u8_c3_to_p3
]
);

impl_jpeg_subsampled_packed_forward_dispatch!(
    "3 channel packed RGB to 3 channel planar YUV422 color conversion.\n\nEnqueues work on the selected stream.\nRow padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nChroma width and height must equal the full-resolution dimensions divided by 2 and 1, respectively.\nFull-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.\nChroma plane strides are passed independently.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples must be initialized and ready when read by the selected stream.\nKeep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.\nRetain image allocations through queued completion, including submission errors and captured graph replay.\nColor-component order, packed layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.",
RgbToYuv422C3ToP3, rgb_to_yuv422_c3_to_p3, rgb_to_yuv422_c3_to_p3, C3, [
    u8 => rgb_to_yuv422_u8_c3_to_p3
]
);

impl_jpeg_subsampled_packed_forward_dispatch!(
    "4 channel packed BGR with alpha to 3 channel planar YUV420 color conversion.\n\nEnqueues work on the selected stream.\nRow padding remains unwritten.\nInput alpha is ignored.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nChroma width and height must equal the full-resolution dimensions divided by 2 and 2, respectively.\nFull-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.\nChroma plane strides are passed independently.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples must be initialized and ready when read by the selected stream.\nKeep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.\nRetain image allocations through queued completion, including submission errors and captured graph replay.\nColor-component order, packed layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.",
BgrToYuv420Ac4ToP3, bgr_to_yuv420_ac4_to_p3, bgr_to_yuv420_ac4_to_p3, AC4, [
    u8 => bgr_to_yuv420_u8_ac4_to_p3
]
);
