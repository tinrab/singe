use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        color_descriptors::{
            validate_jpeg_subsampled_source_planes, validate_rgb_destination_planes,
            validate_same_size,
        },
        view::{AC4, C1, C3, C4, ImageOutputView, ImageView},
    },
    try_ffi,
    types::DataTypeLike,
};

impl_jpeg_subsampled_planar_inverse!(
    "3 channel 8-bit unsigned planar YUV420 to 3 channel 8-bit unsigned planar RGB color conversion.",
    yuv420_to_rgb_u8_p3,
    nppiYUV420ToRGB_8u_P3R_Ctx,
    2,
    2
);

impl_jpeg_subsampled_planar_inverse!(
    "3 channel 8-bit unsigned planar YUV422 to 3 channel 8-bit unsigned planar RGB color conversion.",
    yuv422_to_rgb_u8_p3,
    nppiYUV422ToRGB_8u_P3R_Ctx,
    2,
    1
);

impl_jpeg_subsampled_packed_inverse!(
    "3 channel 8-bit unsigned planar YUV420 to 3 channel 8-bit unsigned packed RGB color conversion.",
    yuv420_to_rgb_u8_p3_to_c3,
    C3,
    nppiYUV420ToRGB_8u_P3C3R_Ctx,
    2,
    2
);

impl_jpeg_subsampled_packed_inverse!(
    "3 channel 8-bit unsigned planar YUV422 to 3 channel 8-bit unsigned packed RGB color conversion.",
    yuv422_to_rgb_u8_p3_to_c3,
    C3,
    nppiYUV422ToRGB_8u_P3C3R_Ctx,
    2,
    1
);

impl_jpeg_subsampled_packed_inverse!(
    "3 channel 8-bit unsigned planar YUV422 to 4 channel 8-bit unsigned packed RGB color conversion with alpha.",
    yuv422_to_rgb_u8_p3_to_ac4,
    AC4,
    nppiYUV422ToRGB_8u_P3AC4R_Ctx,
    2,
    1
);

impl_jpeg_subsampled_packed_inverse!(
    "3 channel 8-bit unsigned planar YUV420 to 3 channel 8-bit unsigned packed BGR color conversion.",
    yuv420_to_bgr_u8_p3_to_c3,
    C3,
    nppiYUV420ToBGR_8u_P3C3R_Ctx,
    2,
    2
);

impl_jpeg_subsampled_planar_inverse!(
    "3 channel 8-bit unsigned planar `YCbCr420` to 3 channel 8-bit unsigned planar RGB color conversion.",
    ycbcr420_to_rgb_jpeg_u8_p3,
    nppiYCbCr420ToRGB_JPEG_8u_P3R_Ctx,
    2,
    2
);

impl_jpeg_subsampled_planar_inverse!(
    "3 channel 8-bit unsigned planar `YCbCr422` to 3 channel 8-bit unsigned planar RGB color conversion.",
    ycbcr422_to_rgb_jpeg_u8_p3,
    nppiYCbCr422ToRGB_JPEG_8u_P3R_Ctx,
    2,
    1
);

impl_jpeg_subsampled_planar_inverse!(
    "3 channel 8-bit unsigned planar `YCbCr411` to 3 channel 8-bit unsigned planar RGB color conversion.",
    ycbcr411_to_rgb_jpeg_u8_p3,
    nppiYCbCr411ToRGB_JPEG_8u_P3R_Ctx,
    4,
    1
);

impl_jpeg_subsampled_planar_inverse!(
    "3 channel 8-bit unsigned planar `YCbCr420` to 3 channel 8-bit unsigned planar BGR color conversion.",
    ycbcr420_to_bgr_jpeg_u8_p3,
    nppiYCbCr420ToBGR_JPEG_8u_P3R_Ctx,
    2,
    2
);

impl_jpeg_subsampled_planar_inverse!(
    "3 channel 8-bit unsigned planar `YCbCr422` to 3 channel 8-bit unsigned planar BGR color conversion.",
    ycbcr422_to_bgr_jpeg_u8_p3,
    nppiYCbCr422ToBGR_JPEG_8u_P3R_Ctx,
    2,
    1
);

impl_jpeg_subsampled_planar_inverse!(
    "3 channel 8-bit unsigned planar `YCbCr411` to 3 channel 8-bit unsigned planar BGR color conversion.",
    ycbcr411_to_bgr_jpeg_u8_p3,
    nppiYCbCr411ToBGR_JPEG_8u_P3R_Ctx,
    4,
    1
);

impl_jpeg_subsampled_packed_inverse!(
    "3 channel 8-bit unsigned planar `YCbCr420` to 3 channel 8-bit unsigned packed RGB color conversion.",
    ycbcr420_to_rgb_jpeg_u8_p3_to_c3,
    C3,
    nppiYCbCr420ToRGB_JPEG_8u_P3C3R_Ctx,
    2,
    2
);

impl_jpeg_subsampled_packed_inverse!(
    "3 channel 8-bit unsigned planar `YCbCr422` to 3 channel 8-bit unsigned packed RGB color conversion.",
    ycbcr422_to_rgb_jpeg_u8_p3_to_c3,
    C3,
    nppiYCbCr422ToRGB_JPEG_8u_P3C3R_Ctx,
    2,
    1
);

impl_jpeg_subsampled_packed_inverse!(
    "3 channel 8-bit unsigned planar `YCbCr411` to 3 channel 8-bit unsigned packed RGB color conversion.",
    ycbcr411_to_rgb_jpeg_u8_p3_to_c3,
    C3,
    nppiYCbCr411ToRGB_JPEG_8u_P3C3R_Ctx,
    4,
    1
);

impl_jpeg_subsampled_packed_inverse!(
    "3 channel 8-bit unsigned planar `YCbCr420` to 3 channel 8-bit unsigned packed BGR color conversion.",
    ycbcr420_to_bgr_jpeg_u8_p3_to_c3,
    C3,
    nppiYCbCr420ToBGR_JPEG_8u_P3C3R_Ctx,
    2,
    2
);

impl_jpeg_subsampled_packed_inverse!(
    "3 channel 8-bit unsigned planar `YCbCr422` to 3 channel 8-bit unsigned packed BGR color conversion.",
    ycbcr422_to_bgr_jpeg_u8_p3_to_c3,
    C3,
    nppiYCbCr422ToBGR_JPEG_8u_P3C3R_Ctx,
    2,
    1
);

impl_jpeg_subsampled_packed_inverse!(
    "3 channel 8-bit unsigned planar `YCbCr411` to 3 channel 8-bit unsigned packed BGR color conversion.",
    ycbcr411_to_bgr_jpeg_u8_p3_to_c3,
    C3,
    nppiYCbCr411ToBGR_JPEG_8u_P3C3R_Ctx,
    4,
    1
);

impl_jpeg_subsampled_packed_inverse!(
    "3 channel 8-bit unsigned planar `YCbCr422` to 3 channel 8-bit unsigned packed RGB color conversion.",
    ycbcr422_to_rgb_u8_p3_to_c3,
    C3,
    nppiYCbCr422ToRGB_8u_P3C3R_Ctx,
    2,
    1
);

impl_jpeg_subsampled_packed_inverse!(
    "3 channel 8-bit unsigned planar `YCbCr422` to 3 channel 8-bit unsigned packed BGR color conversion.",
    ycbcr422_to_bgr_u8_p3_to_c3,
    C3,
    nppiYCbCr422ToBGR_8u_P3C3R_Ctx,
    2,
    1
);

impl_jpeg_subsampled_packed_inverse!(
    "3 channel 8-bit unsigned planar `YCbCr411` to 3 channel 8-bit unsigned packed RGB color conversion.",
    ycbcr411_to_rgb_u8_p3_to_c3,
    C3,
    nppiYCbCr411ToRGB_8u_P3C3R_Ctx,
    4,
    1
);

impl_jpeg_subsampled_packed_inverse!(
    "3 channel 8-bit unsigned planar `YCbCr411` to 3 channel 8-bit unsigned packed BGR color conversion.",
    ycbcr411_to_bgr_u8_p3_to_c3,
    C3,
    nppiYCbCr411ToBGR_8u_P3C3R_Ctx,
    4,
    1
);

impl_jpeg_subsampled_packed_inverse_constant_alpha!(
    "3 channel 8-bit unsigned planar `YCbCr411` to 4 channel 8-bit unsigned packed RGB color conversion with constant alpha.",
    ycbcr411_to_rgb_u8_p3_to_c4,
    nppiYCbCr411ToRGB_8u_P3C4R_Ctx,
    4,
    1
);

impl_jpeg_subsampled_packed_inverse_constant_alpha!(
    "3 channel 8-bit unsigned planar `YCbCr411` to 4 channel 8-bit unsigned packed BGR color conversion with constant alpha.",
    ycbcr411_to_bgr_u8_p3_to_c4,
    nppiYCbCr411ToBGR_8u_P3C4R_Ctx,
    4,
    1
);

impl_jpeg_subsampled_packed_inverse!(
    "3 channel 8-bit unsigned planar `YCbCr420` to 3 channel 8-bit unsigned packed RGB color conversion.",
    ycbcr420_to_rgb_u8_p3_to_c3,
    C3,
    nppiYCbCr420ToRGB_8u_P3C3R_Ctx,
    2,
    2
);

impl_jpeg_subsampled_packed_inverse!(
    "3 channel 8-bit unsigned planar `YCbCr420` to 3 channel 8-bit unsigned packed BGR color conversion.",
    ycbcr420_to_bgr_u8_p3_to_c3,
    C3,
    nppiYCbCr420ToBGR_8u_P3C3R_Ctx,
    2,
    2
);

impl_jpeg_subsampled_packed_inverse_constant_alpha!(
    "3 channel 8-bit unsigned planar `YCbCr420` to 4 channel 8-bit unsigned packed BGR color conversion with constant alpha.",
    ycbcr420_to_bgr_u8_p3_to_c4,
    nppiYCbCr420ToBGR_8u_P3C4R_Ctx,
    2,
    2
);

impl_jpeg_subsampled_packed_inverse!(
    "3 channel 8-bit unsigned planar `YCbCr420` to 3 channel 8-bit unsigned packed `BGR_709CSC` color conversion.",
    ycbcr420_to_bgr_709csc_u8_p3_to_c3,
    C3,
    nppiYCbCr420ToBGR_709CSC_8u_P3C3R_Ctx,
    2,
    2
);

impl_jpeg_subsampled_packed_inverse_constant_alpha!(
    "3 channel 8-bit unsigned planar `YCbCr420` to 4 channel 8-bit unsigned packed `BGR_709HDTV` color conversion with constant alpha.",
    ycbcr420_to_bgr_709hdtv_u8_p3_to_c4,
    nppiYCbCr420ToBGR_709HDTV_8u_P3C4R_Ctx,
    2,
    2
);

impl_jpeg_subsampled_packed_inverse!(
    "3 channel 8-bit unsigned planar YUV420 to 4 channel 8-bit unsigned packed RGB color conversion with constant alpha (0xFF).",
    yuv420_to_rgb_u8_p3_to_c4,
    C4,
    nppiYUV420ToRGB_8u_P3C4R_Ctx,
    2,
    2
);

impl_jpeg_subsampled_packed_inverse!(
    "3 channel 8-bit unsigned planar YUV420 to 4 channel 8-bit unsigned packed BGR color conversion with constant alpha (0xFF).",
    yuv420_to_bgr_u8_p3_to_c4,
    C4,
    nppiYUV420ToBGR_8u_P3C4R_Ctx,
    2,
    2
);

macro_rules! impl_jpeg_subsampled_planar_inverse_dispatch {
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
                source_y: &ImageView<'_, Self, C1>,
                source_cb: &ImageView<'_, Self, C1>,
                source_cr: &ImageView<'_, Self, C1>,
                destination_0: &mut ImageOutputView<'_, Self, C1>,
                destination_1: &mut ImageOutputView<'_, Self, C1>,
                destination_2: &mut ImageOutputView<'_, Self, C1>,
            ) -> Result<()>;
        }

        #[doc = $doc]
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source_y: &ImageView<'_, T, C1>,
            source_cb: &ImageView<'_, T, C1>,
            source_cr: &ImageView<'_, T, C1>,
            destination_0: &mut ImageOutputView<'_, T, C1>,
            destination_1: &mut ImageOutputView<'_, T, C1>,
            destination_2: &mut ImageOutputView<'_, T, C1>,
        ) -> Result<()>
        where
            T: $trait,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
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
        }

        $(
            unsafe impl $trait for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source_y: &ImageView<'_, Self, C1>,
                    source_cb: &ImageView<'_, Self, C1>,
                    source_cr: &ImageView<'_, Self, C1>,
                    destination_0: &mut ImageOutputView<'_, Self, C1>,
                    destination_1: &mut ImageOutputView<'_, Self, C1>,
                    destination_2: &mut ImageOutputView<'_, Self, C1>,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
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
            }
        )*
    };
}

macro_rules! impl_jpeg_subsampled_packed_inverse_dispatch {
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
                source_y: &ImageView<'_, Self, C1>,
                source_cb: &ImageView<'_, Self, C1>,
                source_cr: &ImageView<'_, Self, C1>,
                destination: &mut ImageOutputView<'_, Self, $layout>,
            ) -> Result<()>;
        }

        #[doc = $doc]
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source_y: &ImageView<'_, T, C1>,
            source_cb: &ImageView<'_, T, C1>,
            source_cr: &ImageView<'_, T, C1>,
            destination: &mut ImageOutputView<'_, T, $layout>,
        ) -> Result<()>
        where
            T: $trait,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, source_y, source_cb, source_cr, destination)
            }
}

        $(
            unsafe impl $trait for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source_y: &ImageView<'_, Self, C1>,
                    source_cb: &ImageView<'_, Self, C1>,
                    source_cr: &ImageView<'_, Self, C1>,
                    destination: &mut ImageOutputView<'_, Self, $layout>,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source_y, source_cb, source_cr, destination)
                    }
}
            }
        )*
    };
}

impl_jpeg_subsampled_planar_inverse_dispatch!(
    "3 channel planar YUV420 to 3 channel planar RGB color conversion.\n\nEnqueues work on the selected stream.\nRow padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nChroma width and height must equal the full-resolution dimensions divided by 2 and 2, respectively.\nFull-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.\nThe three full-resolution destination color planes must have equal byte strides.\nChroma plane strides are passed independently.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples must be initialized and ready when read by the selected stream.\nKeep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.\nRetain image allocations through queued completion, including submission errors and captured graph replay.\nColor-component order, packed layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.",
Yuv420ToRgbP3, yuv420_to_rgb_p3, yuv420_to_rgb_p3, [
    u8 => yuv420_to_rgb_u8_p3
]
);

impl_jpeg_subsampled_planar_inverse_dispatch!(
    "3 channel planar YUV422 to 3 channel planar RGB color conversion.\n\nEnqueues work on the selected stream.\nRow padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nChroma width and height must equal the full-resolution dimensions divided by 2 and 1, respectively.\nFull-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.\nThe three full-resolution destination color planes must have equal byte strides.\nChroma plane strides are passed independently.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples must be initialized and ready when read by the selected stream.\nKeep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.\nRetain image allocations through queued completion, including submission errors and captured graph replay.\nColor-component order, packed layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.",
Yuv422ToRgbP3, yuv422_to_rgb_p3, yuv422_to_rgb_p3, [
    u8 => yuv422_to_rgb_u8_p3
]
);

impl_jpeg_subsampled_planar_inverse_dispatch!(
    "3 channel planar `YCbCr420` to 3 channel planar RGB color conversion.\n\nEnqueues work on the selected stream.\nRow padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nChroma width and height must equal the full-resolution dimensions divided by 2 and 2, respectively.\nFull-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.\nThe three full-resolution destination color planes must have equal byte strides.\nChroma plane strides are passed independently.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples must be initialized and ready when read by the selected stream.\nKeep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.\nRetain image allocations through queued completion, including submission errors and captured graph replay.\nColor-component order, packed layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.",
Ycbcr420ToRgbJpegP3, ycbcr420_to_rgb_jpeg_p3, ycbcr420_to_rgb_jpeg_p3, [
    u8 => ycbcr420_to_rgb_jpeg_u8_p3
]
);

impl_jpeg_subsampled_planar_inverse_dispatch!(
    "3 channel planar `YCbCr422` to 3 channel planar RGB color conversion.\n\nEnqueues work on the selected stream.\nRow padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nChroma width and height must equal the full-resolution dimensions divided by 2 and 1, respectively.\nFull-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.\nThe three full-resolution destination color planes must have equal byte strides.\nChroma plane strides are passed independently.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples must be initialized and ready when read by the selected stream.\nKeep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.\nRetain image allocations through queued completion, including submission errors and captured graph replay.\nColor-component order, packed layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.",
Ycbcr422ToRgbJpegP3, ycbcr422_to_rgb_jpeg_p3, ycbcr422_to_rgb_jpeg_p3, [
    u8 => ycbcr422_to_rgb_jpeg_u8_p3
]
);

impl_jpeg_subsampled_planar_inverse_dispatch!(
    "3 channel planar `YCbCr411` to 3 channel planar RGB color conversion.\n\nEnqueues work on the selected stream.\nRow padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nChroma width and height must equal the full-resolution dimensions divided by 4 and 1, respectively.\nFull-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.\nThe three full-resolution destination color planes must have equal byte strides.\nChroma plane strides are passed independently.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples must be initialized and ready when read by the selected stream.\nKeep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.\nRetain image allocations through queued completion, including submission errors and captured graph replay.\nColor-component order, packed layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.",
Ycbcr411ToRgbJpegP3, ycbcr411_to_rgb_jpeg_p3, ycbcr411_to_rgb_jpeg_p3, [
    u8 => ycbcr411_to_rgb_jpeg_u8_p3
]
);

impl_jpeg_subsampled_planar_inverse_dispatch!(
    "3 channel planar `YCbCr420` to 3 channel planar BGR color conversion.\n\nEnqueues work on the selected stream.\nRow padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nChroma width and height must equal the full-resolution dimensions divided by 2 and 2, respectively.\nFull-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.\nThe three full-resolution destination color planes must have equal byte strides.\nChroma plane strides are passed independently.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples must be initialized and ready when read by the selected stream.\nKeep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.\nRetain image allocations through queued completion, including submission errors and captured graph replay.\nColor-component order, packed layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.",
Ycbcr420ToBgrJpegP3, ycbcr420_to_bgr_jpeg_p3, ycbcr420_to_bgr_jpeg_p3, [
    u8 => ycbcr420_to_bgr_jpeg_u8_p3
]
);

impl_jpeg_subsampled_planar_inverse_dispatch!(
    "3 channel planar `YCbCr422` to 3 channel planar BGR color conversion.\n\nEnqueues work on the selected stream.\nRow padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nChroma width and height must equal the full-resolution dimensions divided by 2 and 1, respectively.\nFull-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.\nThe three full-resolution destination color planes must have equal byte strides.\nChroma plane strides are passed independently.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples must be initialized and ready when read by the selected stream.\nKeep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.\nRetain image allocations through queued completion, including submission errors and captured graph replay.\nColor-component order, packed layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.",
Ycbcr422ToBgrJpegP3, ycbcr422_to_bgr_jpeg_p3, ycbcr422_to_bgr_jpeg_p3, [
    u8 => ycbcr422_to_bgr_jpeg_u8_p3
]
);

impl_jpeg_subsampled_planar_inverse_dispatch!(
    "3 channel planar `YCbCr411` to 3 channel planar BGR color conversion.\n\nEnqueues work on the selected stream.\nRow padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nChroma width and height must equal the full-resolution dimensions divided by 4 and 1, respectively.\nFull-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.\nThe three full-resolution destination color planes must have equal byte strides.\nChroma plane strides are passed independently.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples must be initialized and ready when read by the selected stream.\nKeep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.\nRetain image allocations through queued completion, including submission errors and captured graph replay.\nColor-component order, packed layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.",
Ycbcr411ToBgrJpegP3, ycbcr411_to_bgr_jpeg_p3, ycbcr411_to_bgr_jpeg_p3, [
    u8 => ycbcr411_to_bgr_jpeg_u8_p3
]
);

impl_jpeg_subsampled_packed_inverse_dispatch!(
    "3 channel planar YUV420 to 3 channel packed RGB color conversion.\n\nEnqueues work on the selected stream.\nRow padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nChroma width and height must equal the full-resolution dimensions divided by 2 and 2, respectively.\nFull-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.\nChroma plane strides are passed independently.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples must be initialized and ready when read by the selected stream.\nKeep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.\nRetain image allocations through queued completion, including submission errors and captured graph replay.\nColor-component order, packed layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.",
Yuv420ToRgbP3ToC3, yuv420_to_rgb_p3_to_c3, yuv420_to_rgb_p3_to_c3, C3, [
    u8 => yuv420_to_rgb_u8_p3_to_c3
]
);

impl_jpeg_subsampled_packed_inverse_dispatch!(
    "3 channel planar YUV422 to 3 channel packed RGB color conversion.\n\nEnqueues work on the selected stream.\nRow padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nChroma width and height must equal the full-resolution dimensions divided by 2 and 1, respectively.\nFull-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.\nChroma plane strides are passed independently.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples must be initialized and ready when read by the selected stream.\nKeep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.\nRetain image allocations through queued completion, including submission errors and captured graph replay.\nColor-component order, packed layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.",
Yuv422ToRgbP3ToC3, yuv422_to_rgb_p3_to_c3, yuv422_to_rgb_p3_to_c3, C3, [
    u8 => yuv422_to_rgb_u8_p3_to_c3
]
);

impl_jpeg_subsampled_packed_inverse_dispatch!(
    "3 channel planar YUV420 to 3 channel packed BGR color conversion.\n\nEnqueues work on the selected stream.\nRow padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nChroma width and height must equal the full-resolution dimensions divided by 2 and 2, respectively.\nFull-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.\nChroma plane strides are passed independently.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples must be initialized and ready when read by the selected stream.\nKeep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.\nRetain image allocations through queued completion, including submission errors and captured graph replay.\nColor-component order, packed layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.",
Yuv420ToBgrP3ToC3, yuv420_to_bgr_p3_to_c3, yuv420_to_bgr_p3_to_c3, C3, [
    u8 => yuv420_to_bgr_u8_p3_to_c3
]
);

impl_jpeg_subsampled_packed_inverse_dispatch!(
    "3 channel planar `YCbCr420` to 3 channel packed RGB color conversion.\n\nEnqueues work on the selected stream.\nRow padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nChroma width and height must equal the full-resolution dimensions divided by 2 and 2, respectively.\nFull-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.\nChroma plane strides are passed independently.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples must be initialized and ready when read by the selected stream.\nKeep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.\nRetain image allocations through queued completion, including submission errors and captured graph replay.\nColor-component order, packed layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.",
Ycbcr420ToRgbJpegP3ToC3, ycbcr420_to_rgb_jpeg_p3_to_c3, ycbcr420_to_rgb_jpeg_p3_to_c3, C3, [
    u8 => ycbcr420_to_rgb_jpeg_u8_p3_to_c3
]
);

impl_jpeg_subsampled_packed_inverse_dispatch!(
    "3 channel planar `YCbCr422` to 3 channel packed RGB color conversion.\n\nEnqueues work on the selected stream.\nRow padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nChroma width and height must equal the full-resolution dimensions divided by 2 and 1, respectively.\nFull-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.\nChroma plane strides are passed independently.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples must be initialized and ready when read by the selected stream.\nKeep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.\nRetain image allocations through queued completion, including submission errors and captured graph replay.\nColor-component order, packed layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.",
Ycbcr422ToRgbJpegP3ToC3, ycbcr422_to_rgb_jpeg_p3_to_c3, ycbcr422_to_rgb_jpeg_p3_to_c3, C3, [
    u8 => ycbcr422_to_rgb_jpeg_u8_p3_to_c3
]
);

impl_jpeg_subsampled_packed_inverse_dispatch!(
    "3 channel planar `YCbCr411` to 3 channel packed RGB color conversion.\n\nEnqueues work on the selected stream.\nRow padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nChroma width and height must equal the full-resolution dimensions divided by 4 and 1, respectively.\nFull-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.\nChroma plane strides are passed independently.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples must be initialized and ready when read by the selected stream.\nKeep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.\nRetain image allocations through queued completion, including submission errors and captured graph replay.\nColor-component order, packed layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.",
Ycbcr411ToRgbJpegP3ToC3, ycbcr411_to_rgb_jpeg_p3_to_c3, ycbcr411_to_rgb_jpeg_p3_to_c3, C3, [
    u8 => ycbcr411_to_rgb_jpeg_u8_p3_to_c3
]
);

impl_jpeg_subsampled_packed_inverse_dispatch!(
    "3 channel planar `YCbCr420` to 3 channel packed BGR color conversion.\n\nEnqueues work on the selected stream.\nRow padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nChroma width and height must equal the full-resolution dimensions divided by 2 and 2, respectively.\nFull-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.\nChroma plane strides are passed independently.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples must be initialized and ready when read by the selected stream.\nKeep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.\nRetain image allocations through queued completion, including submission errors and captured graph replay.\nColor-component order, packed layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.",
Ycbcr420ToBgrJpegP3ToC3, ycbcr420_to_bgr_jpeg_p3_to_c3, ycbcr420_to_bgr_jpeg_p3_to_c3, C3, [
    u8 => ycbcr420_to_bgr_jpeg_u8_p3_to_c3
]
);

impl_jpeg_subsampled_packed_inverse_dispatch!(
    "3 channel planar `YCbCr422` to 3 channel packed BGR color conversion.\n\nEnqueues work on the selected stream.\nRow padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nChroma width and height must equal the full-resolution dimensions divided by 2 and 1, respectively.\nFull-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.\nChroma plane strides are passed independently.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples must be initialized and ready when read by the selected stream.\nKeep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.\nRetain image allocations through queued completion, including submission errors and captured graph replay.\nColor-component order, packed layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.",
Ycbcr422ToBgrJpegP3ToC3, ycbcr422_to_bgr_jpeg_p3_to_c3, ycbcr422_to_bgr_jpeg_p3_to_c3, C3, [
    u8 => ycbcr422_to_bgr_jpeg_u8_p3_to_c3
]
);

impl_jpeg_subsampled_packed_inverse_dispatch!(
    "3 channel planar `YCbCr411` to 3 channel packed BGR color conversion.\n\nEnqueues work on the selected stream.\nRow padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nChroma width and height must equal the full-resolution dimensions divided by 4 and 1, respectively.\nFull-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.\nChroma plane strides are passed independently.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples must be initialized and ready when read by the selected stream.\nKeep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.\nRetain image allocations through queued completion, including submission errors and captured graph replay.\nColor-component order, packed layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.",
Ycbcr411ToBgrJpegP3ToC3, ycbcr411_to_bgr_jpeg_p3_to_c3, ycbcr411_to_bgr_jpeg_p3_to_c3, C3, [
    u8 => ycbcr411_to_bgr_jpeg_u8_p3_to_c3
]
);

impl_jpeg_subsampled_packed_inverse_dispatch!(
    "3 channel planar `YCbCr422` to 3 channel packed RGB color conversion.\n\nEnqueues work on the selected stream.\nRow padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nChroma width and height must equal the full-resolution dimensions divided by 2 and 1, respectively.\nFull-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.\nChroma plane strides are passed independently.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples must be initialized and ready when read by the selected stream.\nKeep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.\nRetain image allocations through queued completion, including submission errors and captured graph replay.\nColor-component order, packed layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.",
Ycbcr422ToRgbP3ToC3, ycbcr422_to_rgb_p3_to_c3, ycbcr422_to_rgb_p3_to_c3, C3, [
    u8 => ycbcr422_to_rgb_u8_p3_to_c3
]
);

impl_jpeg_subsampled_packed_inverse_dispatch!(
    "3 channel planar `YCbCr422` to 3 channel packed BGR color conversion.\n\nEnqueues work on the selected stream.\nRow padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nChroma width and height must equal the full-resolution dimensions divided by 2 and 1, respectively.\nFull-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.\nChroma plane strides are passed independently.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples must be initialized and ready when read by the selected stream.\nKeep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.\nRetain image allocations through queued completion, including submission errors and captured graph replay.\nColor-component order, packed layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.",
Ycbcr422ToBgrP3ToC3, ycbcr422_to_bgr_p3_to_c3, ycbcr422_to_bgr_p3_to_c3, C3, [
    u8 => ycbcr422_to_bgr_u8_p3_to_c3
]
);

impl_jpeg_subsampled_packed_inverse_dispatch!(
    "3 channel planar `YCbCr411` to 3 channel packed RGB color conversion.\n\nEnqueues work on the selected stream.\nRow padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nChroma width and height must equal the full-resolution dimensions divided by 4 and 1, respectively.\nFull-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.\nChroma plane strides are passed independently.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples must be initialized and ready when read by the selected stream.\nKeep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.\nRetain image allocations through queued completion, including submission errors and captured graph replay.\nColor-component order, packed layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.",
Ycbcr411ToRgbP3ToC3, ycbcr411_to_rgb_p3_to_c3, ycbcr411_to_rgb_p3_to_c3, C3, [
    u8 => ycbcr411_to_rgb_u8_p3_to_c3
]
);

impl_jpeg_subsampled_packed_inverse_dispatch!(
    "3 channel planar `YCbCr411` to 3 channel packed BGR color conversion.\n\nEnqueues work on the selected stream.\nRow padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nChroma width and height must equal the full-resolution dimensions divided by 4 and 1, respectively.\nFull-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.\nChroma plane strides are passed independently.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples must be initialized and ready when read by the selected stream.\nKeep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.\nRetain image allocations through queued completion, including submission errors and captured graph replay.\nColor-component order, packed layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.",
Ycbcr411ToBgrP3ToC3, ycbcr411_to_bgr_p3_to_c3, ycbcr411_to_bgr_p3_to_c3, C3, [
    u8 => ycbcr411_to_bgr_u8_p3_to_c3
]
);

impl_jpeg_subsampled_packed_inverse_dispatch!(
    "3 channel planar `YCbCr420` to 3 channel packed RGB color conversion.\n\nEnqueues work on the selected stream.\nRow padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nChroma width and height must equal the full-resolution dimensions divided by 2 and 2, respectively.\nFull-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.\nChroma plane strides are passed independently.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples must be initialized and ready when read by the selected stream.\nKeep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.\nRetain image allocations through queued completion, including submission errors and captured graph replay.\nColor-component order, packed layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.",
Ycbcr420ToRgbP3ToC3, ycbcr420_to_rgb_p3_to_c3, ycbcr420_to_rgb_p3_to_c3, C3, [
    u8 => ycbcr420_to_rgb_u8_p3_to_c3
]
);

impl_jpeg_subsampled_packed_inverse_dispatch!(
    "3 channel planar `YCbCr420` to 3 channel packed BGR color conversion.\n\nEnqueues work on the selected stream.\nRow padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nChroma width and height must equal the full-resolution dimensions divided by 2 and 2, respectively.\nFull-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.\nChroma plane strides are passed independently.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples must be initialized and ready when read by the selected stream.\nKeep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.\nRetain image allocations through queued completion, including submission errors and captured graph replay.\nColor-component order, packed layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.",
Ycbcr420ToBgrP3ToC3, ycbcr420_to_bgr_p3_to_c3, ycbcr420_to_bgr_p3_to_c3, C3, [
    u8 => ycbcr420_to_bgr_u8_p3_to_c3
]
);

impl_jpeg_subsampled_packed_inverse_dispatch!(
    "3 channel planar `YCbCr420` to 3 channel packed `BGR_709CSC` color conversion.\n\nEnqueues work on the selected stream.\nRow padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nChroma width and height must equal the full-resolution dimensions divided by 2 and 2, respectively.\nFull-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.\nChroma plane strides are passed independently.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples must be initialized and ready when read by the selected stream.\nKeep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.\nRetain image allocations through queued completion, including submission errors and captured graph replay.\nColor-component order, packed layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.",

    Ycbcr420ToBgr709CscP3ToC3,
    ycbcr420_to_bgr_709csc_p3_to_c3,
    ycbcr420_to_bgr_709csc_p3_to_c3,
    C3,
    [u8 => ycbcr420_to_bgr_709csc_u8_p3_to_c3]

);
