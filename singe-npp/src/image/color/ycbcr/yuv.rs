use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        color::{
            ycbcr::core::{
                ycbcr_to_bgr_batch_u8_c3, ycbcr_to_bgr_batch_u8_c3_advanced,
                ycbcr_to_bgr_batch_u8_p3_to_c3, ycbcr_to_bgr_batch_u8_p3_to_c3_advanced,
                ycbcr_to_rgb_batch_u8_c3, ycbcr_to_rgb_batch_u8_c3_advanced,
                ycbcr_to_rgb_batch_u8_p3_to_c3, ycbcr_to_rgb_batch_u8_p3_to_c3_advanced,
                ycbcr420_to_bgr_batch_u8_p3_to_c3, ycbcr420_to_bgr_batch_u8_p3_to_c3_advanced,
                ycbcr420_to_rgb_batch_u8_p3_to_c3, ycbcr420_to_rgb_batch_u8_p3_to_c3_advanced,
                ycbcr422_to_bgr_batch_u8_p3_to_c3, ycbcr422_to_bgr_batch_u8_p3_to_c3_advanced,
                ycbcr422_to_rgb_batch_u8_p3_to_c3, ycbcr422_to_rgb_batch_u8_p3_to_c3_advanced,
            },
            yuv::{
                rgb_to_yuv_u8_c3_to_p3, yuv_to_rgb_batch_u8_c3, yuv_to_rgb_batch_u8_c3_advanced,
                yuv_to_rgb_batch_u8_p3_to_c3, yuv_to_rgb_batch_u8_p3_to_c3_advanced,
                yuv_to_rgb_u8_p3_to_c3, yuv420_to_rgb_batch_u8_p3_to_c3,
                yuv420_to_rgb_batch_u8_p3_to_c3_advanced, yuv422_to_rgb_batch_u8_p3_to_c3,
                yuv422_to_rgb_batch_u8_p3_to_c3_advanced,
            },
        },
        color_descriptors::{
            color_batch_advanced_descriptors, color_batch_descriptors,
            color_planar_batch_advanced_descriptors, color_planar_batch_descriptors,
            subsampled_color_batch_advanced_descriptors, subsampled_color_batch_descriptors,
            validate_same_size,
        },
        view::{AC4, C1, C3, ImageOutputView, ImageView, PlanarImageOutputView, PlanarImageView},
    },
    try_ffi,
    types::DataTypeLike,
};

impl_color_convert_planar_to_planar!(
    "Converts RGB color values to YUV color values.",
    rgb_to_yuv_u8_p3,
    u8,
    3,
    nppiRGBToYUV_8u_P3R_Ctx
);

impl_color_convert_same_layout!(
    "Converts BGR color values to YUV color values.",
    bgr_to_yuv_u8_c3,
    u8,
    C3,
    nppiBGRToYUV_8u_C3R_Ctx
);

impl_color_convert_same_layout!(
    "Converts BGR color values to YUV color values. The destination alpha channel is not processed.",
    bgr_to_yuv_u8_ac4,
    u8,
    AC4,
    nppiBGRToYUV_8u_AC4R_Ctx
);

impl_generic_color_convert_same_layout!("Converts BGR color values to YUV color values.",
BgrToYuvC3, bgr_to_yuv, bgr_to_yuv_c3, C3, [
    u8 => bgr_to_yuv_u8_c3,
]);

impl_generic_color_convert_same_layout!("Converts BGR color values to YUV color values. The destination alpha channel is not processed.",
BgrToYuvAC4, bgr_to_yuv, bgr_to_yuv_ac4, AC4, [
    u8 => bgr_to_yuv_u8_ac4,
]);

impl_color_convert_planar_to_planar!(
    "Converts BGR color values to YUV color values.",
    bgr_to_yuv_u8_p3,
    u8,
    3,
    nppiBGRToYUV_8u_P3R_Ctx
);

impl_color_convert_planar_to_planar!(
    "Converts YUV color values to RGB color values.",
    yuv_to_rgb_u8_p3,
    u8,
    3,
    nppiYUVToRGB_8u_P3R_Ctx
);

impl_color_convert_same_layout!(
    "Converts YUV color values to BGR color values.",
    yuv_to_bgr_u8_c3,
    u8,
    C3,
    nppiYUVToBGR_8u_C3R_Ctx
);

impl_color_convert_same_layout!(
    "Converts YUV color values to BGR color values. The destination alpha channel is not processed.",
    yuv_to_bgr_u8_ac4,
    u8,
    AC4,
    nppiYUVToBGR_8u_AC4R_Ctx
);

impl_generic_color_convert_same_layout!("Converts YUV color values to BGR color values.",
YuvToBgrC3, yuv_to_bgr, yuv_to_bgr_c3, C3, [
    u8 => yuv_to_bgr_u8_c3,
]);

impl_generic_color_convert_same_layout!("Converts YUV color values to BGR color values. The destination alpha channel is not processed.",
YuvToBgrAC4, yuv_to_bgr, yuv_to_bgr_ac4, AC4, [
    u8 => yuv_to_bgr_u8_ac4,
]);

impl_color_convert_batch_same_layout!(
    "Converts YUV color values to BGR color values.",
    yuv_to_bgr_batch_u8_c3,
    u8,
    C3,
    nppiYUVToBGRBatch_8u_C3R_Ctx
);

impl_color_convert_batch_same_layout_advanced!(
    "Converts YUV color values to BGR color values.",
    yuv_to_bgr_batch_u8_c3_advanced,
    u8,
    C3,
    nppiYUVToBGRBatch_8u_C3R_Advanced_Ctx
);

macro_rules! impl_color_convert_batch_c3_dispatch {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Native element and plane-layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and plane-layout ABI and use a compatible conversion routine.
        /// They must uphold the method's geometry checks, access rules, and completion contracts.
        pub unsafe trait $trait: DataTypeLike {
            #[doc = $doc]
            ///
            /// Owns temporary GPU descriptors, rejects stream capture, and returns after successful stream completion.
            /// Ignored channels, pixels outside the processed ROI, and row padding may remain unwritten.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails or NPP rejects the conversion parameters.
            /// Participating images must belong to the selected CUDA context.
            /// Source and destination batch lengths must match, contain at least two entries, and fit in `i32`.
            /// Each destination must have the corresponding full-resolution source size.
            /// Descriptor allocation or upload, capture checks, and stream synchronization can also fail.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source samples must be initialized and ready when read by the selected stream.
            /// Keep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.
            /// Keep image allocations valid until this call completes its native work.
            /// Planes must satisfy the native common-size, common-byte-stride, and component-order requirements.
            /// Image layouts, component ranges, subsampling geometry, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn $method(
                stream_context: &StreamContext,
                sources: &[ImageView<'_, Self, C3>],
                destinations: &mut [ImageOutputView<'_, Self, C3>],
            ) -> Result<()>;
        }

        #[doc = $doc]
        ///
        /// Owns temporary GPU descriptors, rejects stream capture, and returns after successful stream completion.
        /// Ignored channels, pixels outside the processed ROI, and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or NPP rejects the conversion parameters.
        /// Participating images must belong to the selected CUDA context.
        /// Source and destination batch lengths must match, contain at least two entries, and fit in `i32`.
        /// Each destination must have the corresponding full-resolution source size.
        /// Descriptor allocation or upload, capture checks, and stream synchronization can also fail.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.
        /// Keep image allocations valid until this call completes its native work.
        /// Planes must satisfy the native common-size, common-byte-stride, and component-order requirements.
        /// Image layouts, component ranges, subsampling geometry, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            sources: &[ImageView<'_, T, C3>],
            destinations: &mut [ImageOutputView<'_, T, C3>],
        ) -> Result<()>
        where
            T: $trait,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, sources, destinations)
            }
}

        $(
            unsafe impl $trait for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    sources: &[ImageView<'_, Self, C3>],
                    destinations: &mut [ImageOutputView<'_, Self, C3>],
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, sources, destinations)
                    }
}
            }
        )*
    };
}

impl_color_convert_batch_c3_dispatch!(
    "Converts YUV color values to RGB color values.\n\nConverts the common ROI defined by the smallest source width and smallest source height in the batch.",
YuvToRgbBatchC3, yuv_to_rgb_batch, yuv_to_rgb_batch_c3, [
    u8 => yuv_to_rgb_batch_u8_c3,
]
);

impl_color_convert_batch_c3_dispatch!(
    "Converts YUV color values to RGB color values.\n\nConverts each image pair using its own size; the native maximum ROI is the component-wise maximum destination size.",

    YuvToRgbBatchAdvancedC3,
    yuv_to_rgb_batch_advanced,
    yuv_to_rgb_batch_c3_advanced,
    [u8 => yuv_to_rgb_batch_u8_c3_advanced]

);

impl_color_convert_batch_c3_dispatch!(
    "Converts YCbCr color values to RGB color values.\n\nConverts the common ROI defined by the smallest source width and smallest source height in the batch.",

    YcbcrToRgbBatchC3,
    ycbcr_to_rgb_batch,
    ycbcr_to_rgb_batch_c3,
    [u8 => ycbcr_to_rgb_batch_u8_c3]

);

impl_color_convert_batch_c3_dispatch!(
    "Converts YCbCr color values to RGB color values.\n\nConverts each image pair using its own size; the native maximum ROI is the component-wise maximum destination size.",

    YcbcrToRgbBatchAdvancedC3,
    ycbcr_to_rgb_batch_advanced,
    ycbcr_to_rgb_batch_c3_advanced,
    [u8 => ycbcr_to_rgb_batch_u8_c3_advanced]

);

impl_color_convert_batch_c3_dispatch!(
    "Converts YCbCr color values to BGR color values.\n\nConverts the common ROI defined by the smallest source width and smallest source height in the batch.",

    YcbcrToBgrBatchC3,
    ycbcr_to_bgr_batch,
    ycbcr_to_bgr_batch_c3,
    [u8 => ycbcr_to_bgr_batch_u8_c3]

);

impl_color_convert_batch_c3_dispatch!(
    "Converts YCbCr color values to BGR color values.\n\nConverts each image pair using its own size; the native maximum ROI is the component-wise maximum destination size.",

    YcbcrToBgrBatchAdvancedC3,
    ycbcr_to_bgr_batch_advanced,
    ycbcr_to_bgr_batch_c3_advanced,
    [u8 => ycbcr_to_bgr_batch_u8_c3_advanced]

);

impl_color_convert_batch_c3_dispatch!(
    "Converts YUV color values to BGR color values.\n\nConverts the common ROI defined by the smallest source width and smallest source height in the batch.",
YuvToBgrBatchC3, yuv_to_bgr_batch, yuv_to_bgr_batch_c3, [
    u8 => yuv_to_bgr_batch_u8_c3,
]
);

impl_color_convert_batch_c3_dispatch!(
    "Converts YUV color values to BGR color values.\n\nConverts each image pair using its own size; the native maximum ROI is the component-wise maximum destination size.",

    YuvToBgrBatchAdvancedC3,
    yuv_to_bgr_batch_advanced,
    yuv_to_bgr_batch_c3_advanced,
    [u8 => yuv_to_bgr_batch_u8_c3_advanced]

);

impl_color_convert_batch_planar_to_packed!(
    "Converts YUV color values to BGR color values.",
    yuv_to_bgr_batch_u8_p3_to_c3,
    u8,
    3,
    C3,
    nppiYUVToBGRBatch_8u_P3C3R_Ctx
);

impl_color_convert_batch_planar_to_packed_advanced!(
    "Converts YUV color values to BGR color values.",
    yuv_to_bgr_batch_u8_p3_to_c3_advanced,
    u8,
    3,
    C3,
    nppiYUVToBGRBatch_8u_P3C3R_Advanced_Ctx
);

macro_rules! impl_color_convert_batch_p3_to_c3_dispatch {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Native element and plane-layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and plane-layout ABI and use a compatible conversion routine.
        /// They must uphold the method's geometry checks, access rules, and completion contracts.
        pub unsafe trait $trait: DataTypeLike {
            #[doc = $doc]
            ///
            /// Owns temporary GPU descriptors, rejects stream capture, and returns after successful stream completion.
            /// Ignored channels, pixels outside the processed ROI, and row padding may remain unwritten.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails or NPP rejects the conversion parameters.
            /// Destination images must belong to the selected CUDA context.
            /// Source and destination batch lengths must match, contain at least two entries, and fit in `i32`.
            /// Each destination must have the corresponding full-resolution source size.
            /// Descriptor allocation or upload, capture checks, and stream synchronization can also fail.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source samples must be initialized and ready when read by the selected stream.
            /// Keep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.
            /// Every source plane must belong to the selected CUDA context; the provided implementations check destination contexts only.
            /// Keep image allocations valid until this call completes its native work.
            /// Planes must satisfy the native common-size, common-byte-stride, and component-order requirements.
            /// Image layouts, component ranges, subsampling geometry, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn $method(
                stream_context: &StreamContext,
                sources: &[PlanarImageView<'_, Self, 3>],
                destinations: &mut [ImageOutputView<'_, Self, C3>],
            ) -> Result<()>;
        }

        #[doc = $doc]
        ///
        /// Owns temporary GPU descriptors, rejects stream capture, and returns after successful stream completion.
        /// Ignored channels, pixels outside the processed ROI, and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or NPP rejects the conversion parameters.
        /// Destination images must belong to the selected CUDA context.
        /// Source and destination batch lengths must match, contain at least two entries, and fit in `i32`.
        /// Each destination must have the corresponding full-resolution source size.
        /// Descriptor allocation or upload, capture checks, and stream synchronization can also fail.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.
        /// Every source plane must belong to the selected CUDA context; the provided implementations check destination contexts only.
        /// Keep image allocations valid until this call completes its native work.
        /// Planes must satisfy the native common-size, common-byte-stride, and component-order requirements.
        /// Image layouts, component ranges, subsampling geometry, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            sources: &[PlanarImageView<'_, T, 3>],
            destinations: &mut [ImageOutputView<'_, T, C3>],
        ) -> Result<()>
        where
            T: $trait,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, sources, destinations)
            }
}

        $(
            unsafe impl $trait for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    sources: &[PlanarImageView<'_, Self, 3>],
                    destinations: &mut [ImageOutputView<'_, Self, C3>],
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, sources, destinations)
                    }
}
            }
        )*
    };
}

impl_color_convert_batch_p3_to_c3_dispatch!(
    "Converts YUV color values to RGB color values.\n\nConverts the common ROI defined by the smallest source width and smallest source height in the batch.",

    YuvToRgbBatchP3ToC3,
    yuv_to_rgb_batch_p3_to_c3,
    yuv_to_rgb_batch_p3_to_c3,
    [u8 => yuv_to_rgb_batch_u8_p3_to_c3]

);

impl_color_convert_batch_p3_to_c3_dispatch!(
    "Converts YUV color values to RGB color values.\n\nConverts each image pair using its own size; the native maximum ROI is the component-wise maximum destination size.",

    YuvToRgbBatchP3ToC3Advanced,
    yuv_to_rgb_batch_p3_to_c3_advanced,
    yuv_to_rgb_batch_p3_to_c3_advanced,
    [u8 => yuv_to_rgb_batch_u8_p3_to_c3_advanced]

);

impl_color_convert_batch_p3_to_c3_dispatch!(
    "Converts YCbCr color values to RGB color values.\n\nConverts the common ROI defined by the smallest source width and smallest source height in the batch.",

    YcbcrToRgbBatchP3ToC3,
    ycbcr_to_rgb_batch_p3_to_c3,
    ycbcr_to_rgb_batch_p3_to_c3,
    [u8 => ycbcr_to_rgb_batch_u8_p3_to_c3]

);

impl_color_convert_batch_p3_to_c3_dispatch!(
    "Converts YCbCr color values to RGB color values.\n\nConverts each image pair using its own size; the native maximum ROI is the component-wise maximum destination size.",

    YcbcrToRgbBatchP3ToC3Advanced,
    ycbcr_to_rgb_batch_p3_to_c3_advanced,
    ycbcr_to_rgb_batch_p3_to_c3_advanced,
    [u8 => ycbcr_to_rgb_batch_u8_p3_to_c3_advanced]

);

impl_color_convert_batch_p3_to_c3_dispatch!(
    "Converts YCbCr color values to BGR color values.\n\nConverts the common ROI defined by the smallest source width and smallest source height in the batch.",

    YcbcrToBgrBatchP3ToC3,
    ycbcr_to_bgr_batch_p3_to_c3,
    ycbcr_to_bgr_batch_p3_to_c3,
    [u8 => ycbcr_to_bgr_batch_u8_p3_to_c3]

);

impl_color_convert_batch_p3_to_c3_dispatch!(
    "Converts YCbCr color values to BGR color values.\n\nConverts each image pair using its own size; the native maximum ROI is the component-wise maximum destination size.",

    YcbcrToBgrBatchP3ToC3Advanced,
    ycbcr_to_bgr_batch_p3_to_c3_advanced,
    ycbcr_to_bgr_batch_p3_to_c3_advanced,
    [u8 => ycbcr_to_bgr_batch_u8_p3_to_c3_advanced]

);

impl_color_convert_batch_p3_to_c3_dispatch!(
    "Converts YUV color values to BGR color values.\n\nConverts the common ROI defined by the smallest source width and smallest source height in the batch.",

    YuvToBgrBatchP3ToC3,
    yuv_to_bgr_batch_p3_to_c3,
    yuv_to_bgr_batch_p3_to_c3,
    [u8 => yuv_to_bgr_batch_u8_p3_to_c3]

);

impl_color_convert_batch_p3_to_c3_dispatch!(
    "Converts YUV color values to BGR color values.\n\nConverts each image pair using its own size; the native maximum ROI is the component-wise maximum destination size.",

    YuvToBgrBatchP3ToC3Advanced,
    yuv_to_bgr_batch_p3_to_c3_advanced,
    yuv_to_bgr_batch_p3_to_c3_advanced,
    [u8 => yuv_to_bgr_batch_u8_p3_to_c3_advanced]

);

impl_subsampled_color_convert_batch_planar_to_packed!(
    "Converts YUV422 color values to BGR color values.",
    yuv422_to_bgr_batch_u8_p3_to_c3,
    u8,
    C3,
    nppiYUV422ToBGRBatch_8u_P3C3R_Ctx,
    2,
    1
);

impl_subsampled_color_convert_batch_planar_to_packed_advanced!(
    "Converts YUV422 color values to BGR color values.",
    yuv422_to_bgr_batch_u8_p3_to_c3_advanced,
    u8,
    C3,
    nppiYUV422ToBGRBatch_8u_P3C3R_Advanced_Ctx,
    2,
    1
);

impl_subsampled_color_convert_batch_planar_to_packed!(
    "Converts YUV420 color values to BGR color values.",
    yuv420_to_bgr_batch_u8_p3_to_c3,
    u8,
    C3,
    nppiYUV420ToBGRBatch_8u_P3C3R_Ctx,
    2,
    2
);

impl_subsampled_color_convert_batch_planar_to_packed_advanced!(
    "Converts YUV420 color values to BGR color values.",
    yuv420_to_bgr_batch_u8_p3_to_c3_advanced,
    u8,
    C3,
    nppiYUV420ToBGRBatch_8u_P3C3R_Advanced_Ctx,
    2,
    2
);

macro_rules! impl_subsampled_color_convert_batch_p3_to_c3_dispatch {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Native element and plane-layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and plane-layout ABI and use a compatible conversion routine.
        /// They must uphold the method's geometry checks, access rules, and completion contracts.
        pub unsafe trait $trait: DataTypeLike {
            #[doc = $doc]
            ///
            /// Owns temporary GPU descriptors, rejects stream capture, and returns after successful stream completion.
            /// Ignored channels, pixels outside the processed ROI, and row padding may remain unwritten.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails or NPP rejects the conversion parameters.
            /// Participating images must belong to the selected CUDA context.
            /// Source and destination batch lengths must match, contain at least two entries, and fit in `i32`.
            /// Each destination must have the corresponding full-resolution source size.
            /// Descriptor allocation or upload, capture checks, and stream synchronization can also fail.
            /// All three source-plane batches must have equal lengths.
            /// Both chroma planes must match the full-resolution dimensions divided by the conversion's horizontal and vertical subsampling factors.
            /// Full-resolution dimensions must be divisible by those factors.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source samples must be initialized and ready when read by the selected stream.
            /// Keep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.
            /// Keep image allocations valid until this call completes its native work.
            /// Image layouts, component ranges, subsampling geometry, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn $method(
                stream_context: &StreamContext,
                source_plane_0: &[ImageView<'_, Self, C1>],
                source_plane_1: &[ImageView<'_, Self, C1>],
                source_plane_2: &[ImageView<'_, Self, C1>],
                destinations: &mut [ImageOutputView<'_, Self, C3>],
            ) -> Result<()>;
        }

        #[doc = $doc]
        ///
        /// Owns temporary GPU descriptors, rejects stream capture, and returns after successful stream completion.
        /// Ignored channels, pixels outside the processed ROI, and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or NPP rejects the conversion parameters.
        /// Participating images must belong to the selected CUDA context.
        /// Source and destination batch lengths must match, contain at least two entries, and fit in `i32`.
        /// Each destination must have the corresponding full-resolution source size.
        /// Descriptor allocation or upload, capture checks, and stream synchronization can also fail.
        /// All three source-plane batches must have equal lengths.
        /// Both chroma planes must match the full-resolution dimensions divided by the conversion's horizontal and vertical subsampling factors.
        /// Full-resolution dimensions must be divisible by those factors.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.
        /// Keep image allocations valid until this call completes its native work.
        /// Image layouts, component ranges, subsampling geometry, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source_plane_0: &[ImageView<'_, T, C1>],
            source_plane_1: &[ImageView<'_, T, C1>],
            source_plane_2: &[ImageView<'_, T, C1>],
            destinations: &mut [ImageOutputView<'_, T, C3>],
        ) -> Result<()>
        where
            T: $trait,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(
                    stream_context,
                    source_plane_0,
                    source_plane_1,
                    source_plane_2,
                    destinations,
                )
            }
        }

        $(
            unsafe impl $trait for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source_plane_0: &[ImageView<'_, Self, C1>],
                    source_plane_1: &[ImageView<'_, Self, C1>],
                    source_plane_2: &[ImageView<'_, Self, C1>],
                    destinations: &mut [ImageOutputView<'_, Self, C3>],
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(
                            stream_context,
                            source_plane_0,
                            source_plane_1,
                            source_plane_2,
                            destinations,
                        )
                    }
                }
            }
        )*
    };
}

impl_subsampled_color_convert_batch_p3_to_c3_dispatch!(
    "Converts YUV422 color values to RGB color values.\n\nConverts the common ROI defined by the smallest source width and smallest source height in the batch.",

    Yuv422ToRgbBatchP3ToC3,
    yuv422_to_rgb_batch_p3_to_c3,
    yuv422_to_rgb_batch_p3_to_c3,
    [u8 => yuv422_to_rgb_batch_u8_p3_to_c3]

);

impl_subsampled_color_convert_batch_p3_to_c3_dispatch!(
    "Converts YUV422 color values to RGB color values.\n\nConverts each image pair using its own size; the native maximum ROI is the component-wise maximum destination size.",

    Yuv422ToRgbBatchP3ToC3Advanced,
    yuv422_to_rgb_batch_p3_to_c3_advanced,
    yuv422_to_rgb_batch_p3_to_c3_advanced,
    [u8 => yuv422_to_rgb_batch_u8_p3_to_c3_advanced]

);

impl_subsampled_color_convert_batch_p3_to_c3_dispatch!(
    "Converts YUV420 color values to RGB color values.\n\nConverts the common ROI defined by the smallest source width and smallest source height in the batch.",

    Yuv420ToRgbBatchP3ToC3,
    yuv420_to_rgb_batch_p3_to_c3,
    yuv420_to_rgb_batch_p3_to_c3,
    [u8 => yuv420_to_rgb_batch_u8_p3_to_c3]

);

impl_subsampled_color_convert_batch_p3_to_c3_dispatch!(
    "Converts YUV420 color values to RGB color values.\n\nConverts each image pair using its own size; the native maximum ROI is the component-wise maximum destination size.",

    Yuv420ToRgbBatchP3ToC3Advanced,
    yuv420_to_rgb_batch_p3_to_c3_advanced,
    yuv420_to_rgb_batch_p3_to_c3_advanced,
    [u8 => yuv420_to_rgb_batch_u8_p3_to_c3_advanced]

);

impl_subsampled_color_convert_batch_p3_to_c3_dispatch!(
    "Converts `YCbCr422` color values to RGB color values.\n\nConverts the common ROI defined by the smallest source width and smallest source height in the batch.",

    Ycbcr422ToRgbBatchP3ToC3,
    ycbcr422_to_rgb_batch_p3_to_c3,
    ycbcr422_to_rgb_batch_p3_to_c3,
    [u8 => ycbcr422_to_rgb_batch_u8_p3_to_c3]

);

impl_subsampled_color_convert_batch_p3_to_c3_dispatch!(
    "Converts `YCbCr422` color values to RGB color values.\n\nConverts each image pair using its own size; the native maximum ROI is the component-wise maximum destination size.",

    Ycbcr422ToRgbBatchP3ToC3Advanced,
    ycbcr422_to_rgb_batch_p3_to_c3_advanced,
    ycbcr422_to_rgb_batch_p3_to_c3_advanced,
    [u8 => ycbcr422_to_rgb_batch_u8_p3_to_c3_advanced]

);

impl_subsampled_color_convert_batch_p3_to_c3_dispatch!(
    "Converts `YCbCr420` color values to RGB color values.\n\nConverts the common ROI defined by the smallest source width and smallest source height in the batch.",

    Ycbcr420ToRgbBatchP3ToC3,
    ycbcr420_to_rgb_batch_p3_to_c3,
    ycbcr420_to_rgb_batch_p3_to_c3,
    [u8 => ycbcr420_to_rgb_batch_u8_p3_to_c3]

);

impl_subsampled_color_convert_batch_p3_to_c3_dispatch!(
    "Converts `YCbCr420` color values to RGB color values.\n\nConverts each image pair using its own size; the native maximum ROI is the component-wise maximum destination size.",

    Ycbcr420ToRgbBatchP3ToC3Advanced,
    ycbcr420_to_rgb_batch_p3_to_c3_advanced,
    ycbcr420_to_rgb_batch_p3_to_c3_advanced,
    [u8 => ycbcr420_to_rgb_batch_u8_p3_to_c3_advanced]

);

impl_subsampled_color_convert_batch_p3_to_c3_dispatch!(
    "Converts `YCbCr422` color values to BGR color values.\n\nConverts the common ROI defined by the smallest source width and smallest source height in the batch.",

    Ycbcr422ToBgrBatchP3ToC3,
    ycbcr422_to_bgr_batch_p3_to_c3,
    ycbcr422_to_bgr_batch_p3_to_c3,
    [u8 => ycbcr422_to_bgr_batch_u8_p3_to_c3]

);

impl_subsampled_color_convert_batch_p3_to_c3_dispatch!(
    "Converts `YCbCr422` color values to BGR color values.\n\nConverts each image pair using its own size; the native maximum ROI is the component-wise maximum destination size.",

    Ycbcr422ToBgrBatchP3ToC3Advanced,
    ycbcr422_to_bgr_batch_p3_to_c3_advanced,
    ycbcr422_to_bgr_batch_p3_to_c3_advanced,
    [u8 => ycbcr422_to_bgr_batch_u8_p3_to_c3_advanced]

);

impl_subsampled_color_convert_batch_p3_to_c3_dispatch!(
    "Converts `YCbCr420` color values to BGR color values.\n\nConverts the common ROI defined by the smallest source width and smallest source height in the batch.",

    Ycbcr420ToBgrBatchP3ToC3,
    ycbcr420_to_bgr_batch_p3_to_c3,
    ycbcr420_to_bgr_batch_p3_to_c3,
    [u8 => ycbcr420_to_bgr_batch_u8_p3_to_c3]

);

impl_subsampled_color_convert_batch_p3_to_c3_dispatch!(
    "Converts `YCbCr420` color values to BGR color values.\n\nConverts each image pair using its own size; the native maximum ROI is the component-wise maximum destination size.",

    Ycbcr420ToBgrBatchP3ToC3Advanced,
    ycbcr420_to_bgr_batch_p3_to_c3_advanced,
    ycbcr420_to_bgr_batch_p3_to_c3_advanced,
    [u8 => ycbcr420_to_bgr_batch_u8_p3_to_c3_advanced]

);

impl_subsampled_color_convert_batch_p3_to_c3_dispatch!(
    "Converts YUV422 color values to BGR color values.\n\nConverts the common ROI defined by the smallest source width and smallest source height in the batch.",

    Yuv422ToBgrBatchP3ToC3,
    yuv422_to_bgr_batch_p3_to_c3,
    yuv422_to_bgr_batch_p3_to_c3,
    [u8 => yuv422_to_bgr_batch_u8_p3_to_c3]

);

impl_subsampled_color_convert_batch_p3_to_c3_dispatch!(
    "Converts YUV422 color values to BGR color values.\n\nConverts each image pair using its own size; the native maximum ROI is the component-wise maximum destination size.",

    Yuv422ToBgrBatchP3ToC3Advanced,
    yuv422_to_bgr_batch_p3_to_c3_advanced,
    yuv422_to_bgr_batch_p3_to_c3_advanced,
    [u8 => yuv422_to_bgr_batch_u8_p3_to_c3_advanced]

);

impl_subsampled_color_convert_batch_p3_to_c3_dispatch!(
    "Converts YUV420 color values to BGR color values.\n\nConverts the common ROI defined by the smallest source width and smallest source height in the batch.",

    Yuv420ToBgrBatchP3ToC3,
    yuv420_to_bgr_batch_p3_to_c3,
    yuv420_to_bgr_batch_p3_to_c3,
    [u8 => yuv420_to_bgr_batch_u8_p3_to_c3]

);

impl_subsampled_color_convert_batch_p3_to_c3_dispatch!(
    "Converts YUV420 color values to BGR color values.\n\nConverts each image pair using its own size; the native maximum ROI is the component-wise maximum destination size.",

    Yuv420ToBgrBatchP3ToC3Advanced,
    yuv420_to_bgr_batch_p3_to_c3_advanced,
    yuv420_to_bgr_batch_p3_to_c3_advanced,
    [u8 => yuv420_to_bgr_batch_u8_p3_to_c3_advanced]

);

impl_color_convert_planar_to_planar!(
    "Converts YUV color values to BGR color values.",
    yuv_to_bgr_u8_p3,
    u8,
    3,
    nppiYUVToBGR_8u_P3R_Ctx
);

impl_color_convert_packed_to_planar!(
    "Converts BGR color values to YUV color values.",
    bgr_to_yuv_u8_c3_to_p3,
    u8,
    C3,
    3,
    nppiBGRToYUV_8u_C3P3R_Ctx
);

impl_color_convert_packed_to_planar!(
    "Converts BGR color values to YUV color values.",
    bgr_to_yuv_u8_ac4_to_p4,
    u8,
    AC4,
    4,
    nppiBGRToYUV_8u_AC4P4R_Ctx
);

impl_color_convert_planar_to_packed!(
    "Converts YUV color values to BGR color values.",
    yuv_to_bgr_u8_p3_to_c3,
    u8,
    3,
    C3,
    nppiYUVToBGR_8u_P3C3R_Ctx
);

macro_rules! impl_color_convert_packed_c3_to_planar_dispatch {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, $planes:literal, [$($ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Native element and plane-layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and plane-layout ABI and use a compatible conversion routine.
        /// They must uphold the method's geometry checks, access rules, and completion contracts.
        pub unsafe trait $trait: DataTypeLike {
            #[doc = $doc]
            ///
            /// Enqueues work on the selected stream.
            /// Ignored channels, pixels outside the processed ROI, and row padding may remain unwritten.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails or NPP rejects the conversion parameters.
            /// Participating images must belong to the selected CUDA context.
            /// Source and destination image sizes must match.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source samples must be initialized and ready when read by the selected stream.
            /// Keep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.
            /// Retain image allocations through queued completion, including submission errors and captured graph replay.
            /// Planes must satisfy the native common-size, common-byte-stride, and component-order requirements.
            /// Image layouts, component ranges, subsampling geometry, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C3>,
                destination: &mut PlanarImageOutputView<'_, Self, $planes>,
            ) -> Result<()>;
        }

        #[doc = $doc]
        ///
        /// Enqueues work on the selected stream.
        /// Ignored channels, pixels outside the processed ROI, and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or NPP rejects the conversion parameters.
        /// Participating images must belong to the selected CUDA context.
        /// Source and destination image sizes must match.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Planes must satisfy the native common-size, common-byte-stride, and component-order requirements.
        /// Image layouts, component ranges, subsampling geometry, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, C3>,
            destination: &mut PlanarImageOutputView<'_, T, $planes>,
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
                    source: &ImageView<'_, Self, C3>,
                    destination: &mut PlanarImageOutputView<'_, Self, $planes>,
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

macro_rules! impl_color_convert_planar_to_packed_c3_dispatch {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, $planes:literal, [$($ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Native element and plane-layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and plane-layout ABI and use a compatible conversion routine.
        /// They must uphold the method's geometry checks, access rules, and completion contracts.
        pub unsafe trait $trait: DataTypeLike {
            #[doc = $doc]
            ///
            /// Enqueues work on the selected stream.
            /// Ignored channels, pixels outside the processed ROI, and row padding may remain unwritten.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails or NPP rejects the conversion parameters.
            /// Participating images must belong to the selected CUDA context.
            /// Source and destination image sizes must match.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source samples must be initialized and ready when read by the selected stream.
            /// Keep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.
            /// Retain image allocations through queued completion, including submission errors and captured graph replay.
            /// Planes must satisfy the native common-size, common-byte-stride, and component-order requirements.
            /// Image layouts, component ranges, subsampling geometry, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &PlanarImageView<'_, Self, $planes>,
                destination: &mut ImageOutputView<'_, Self, C3>,
            ) -> Result<()>;
        }

        #[doc = $doc]
        ///
        /// Enqueues work on the selected stream.
        /// Ignored channels, pixels outside the processed ROI, and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or NPP rejects the conversion parameters.
        /// Participating images must belong to the selected CUDA context.
        /// Source and destination image sizes must match.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Planes must satisfy the native common-size, common-byte-stride, and component-order requirements.
        /// Image layouts, component ranges, subsampling geometry, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &PlanarImageView<'_, T, $planes>,
            destination: &mut ImageOutputView<'_, T, C3>,
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
                    source: &PlanarImageView<'_, Self, $planes>,
                    destination: &mut ImageOutputView<'_, Self, C3>,
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

macro_rules! impl_color_convert_planar_to_planar_dispatch {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, $planes:literal, [$($ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Native element and plane-layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and plane-layout ABI and use a compatible conversion routine.
        /// They must uphold the method's geometry checks, access rules, and completion contracts.
        pub unsafe trait $trait: DataTypeLike {
            #[doc = $doc]
            ///
            /// Enqueues work on the selected stream.
            /// Ignored channels, pixels outside the processed ROI, and row padding may remain unwritten.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails or NPP rejects the conversion parameters.
            /// Participating images must belong to the selected CUDA context.
            /// Source and destination image sizes must match.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source samples must be initialized and ready when read by the selected stream.
            /// Keep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.
            /// Retain image allocations through queued completion, including submission errors and captured graph replay.
            /// Planes must satisfy the native common-size, common-byte-stride, and component-order requirements.
            /// Image layouts, component ranges, subsampling geometry, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &PlanarImageView<'_, Self, $planes>,
                destination: &mut PlanarImageOutputView<'_, Self, $planes>,
            ) -> Result<()>;
        }

        #[doc = $doc]
        ///
        /// Enqueues work on the selected stream.
        /// Ignored channels, pixels outside the processed ROI, and row padding may remain unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or NPP rejects the conversion parameters.
        /// Participating images must belong to the selected CUDA context.
        /// Source and destination image sizes must match.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Planes must satisfy the native common-size, common-byte-stride, and component-order requirements.
        /// Image layouts, component ranges, subsampling geometry, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &PlanarImageView<'_, T, $planes>,
            destination: &mut PlanarImageOutputView<'_, T, $planes>,
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
                    source: &PlanarImageView<'_, Self, $planes>,
                    destination: &mut PlanarImageOutputView<'_, Self, $planes>,
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

impl_color_convert_packed_c3_to_planar_dispatch!(
    "Converts RGB color values to YUV color values.",

    RgbToYuvC3ToP3,
    rgb_to_yuv_c3_to_p3,
    rgb_to_yuv_c3_to_p3,
    3,
    [u8 => rgb_to_yuv_u8_c3_to_p3]

);

impl_color_convert_packed_c3_to_planar_dispatch!(
    "Converts BGR color values to YUV color values.",

    BgrToYuvC3ToP3,
    bgr_to_yuv_c3_to_p3,
    bgr_to_yuv_c3_to_p3,
    3,
    [u8 => bgr_to_yuv_u8_c3_to_p3]

);

impl_color_convert_planar_to_packed_c3_dispatch!(
    "Converts YUV color values to RGB color values.",

    YuvToRgbP3ToC3,
    yuv_to_rgb_p3_to_c3,
    yuv_to_rgb_p3_to_c3,
    3,
    [u8 => yuv_to_rgb_u8_p3_to_c3]

);

impl_color_convert_planar_to_packed_c3_dispatch!(
    "Converts YUV color values to BGR color values.",

    YuvToBgrP3ToC3,
    yuv_to_bgr_p3_to_c3,
    yuv_to_bgr_p3_to_c3,
    3,
    [u8 => yuv_to_bgr_u8_p3_to_c3]

);

impl_color_convert_planar_to_planar_dispatch!(
    "Converts RGB color values to YUV color values.",
RgbToYuvP3, rgb_to_yuv_p3, rgb_to_yuv_p3, 3, [
    u8 => rgb_to_yuv_u8_p3,
]
);

impl_color_convert_planar_to_planar_dispatch!(
    "Converts BGR color values to YUV color values.",
BgrToYuvP3, bgr_to_yuv_p3, bgr_to_yuv_p3, 3, [
    u8 => bgr_to_yuv_u8_p3,
]
);

impl_color_convert_planar_to_planar_dispatch!(
    "Converts YUV color values to RGB color values.",
YuvToRgbP3, yuv_to_rgb_p3, yuv_to_rgb_p3, 3, [
    u8 => yuv_to_rgb_u8_p3,
]
);

impl_color_convert_planar_to_planar_dispatch!(
    "Converts YUV color values to BGR color values.",
YuvToBgrP3, yuv_to_bgr_p3, yuv_to_bgr_p3, 3, [
    u8 => yuv_to_bgr_u8_p3,
]
);
