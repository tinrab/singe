use crate::{
    context::StreamContext,
    error::Result,
    image::{
        color::{
            ColorTwistMatrix,
            yuv::{
                nv12_to_bgr_709csc_u8_p2_to_c3, nv12_to_bgr_709hdtv_u8_p2_to_c3,
                nv12_to_bgr_u8_p2_to_c3, nv12_to_rgb_709csc_u8_p2_to_c3,
                nv12_to_rgb_709hdtv_u8_p2_to_c3, nv12_to_rgb_u8_color_twist_p2_to_c3,
                nv12_to_rgb_u8_p2_to_c3, nv12_to_rgb_u16_color_twist_p2_to_c3,
                nv21_to_bgr_u8_p2_to_c4, nv21_to_rgb_u8_p2_to_c4, ycbcr411_to_ycbcr422_u8_p3_to_c2,
                ycbcr411_to_ycrcb422_u8_p3_to_c2, ycbcr422_to_ycrcb422_u8_p3_to_c2,
                ycbcr422_u8_c2_to_p3, ycbcr422_u8_p3_to_c2,
            },
        },
        raw::color::{
            rgb_to_ycbcr422_u8_p3_to_c2, rgb_to_ycrcb422_u8_p3_to_c2, ycbcr422_to_rgb_u8_c2_to_p3,
            ycrcb422_to_rgb_u8_c2_to_p3,
        },
        view::{C1, C2, C3, C4, ImageOutputView, ImageView},
    },
    types::DataTypeLike,
};

macro_rules! impl_nv12_color_twist_packed_inverse_dispatch {
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
                source_uv: &ImageView<'_, Self, C2>,
                destination: &mut ImageOutputView<'_, Self, $layout>,
                twist: ColorTwistMatrix,
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source_y: &ImageView<'_, Self, C1>,
                    source_uv: &ImageView<'_, Self, C2>,
                    destination: &mut ImageOutputView<'_, Self, $layout>,
                    twist: ColorTwistMatrix,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source_y, source_uv, destination, twist)
                    }
}
            }
        )*

        #[doc = $doc]
        pub unsafe fn $function<T: $trait>(
            stream_context: &StreamContext,
            source_y: &ImageView<'_, T, C1>,
            source_uv: &ImageView<'_, T, C2>,
            destination: &mut ImageOutputView<'_, T, $layout>,
            twist: ColorTwistMatrix,
        ) -> Result<()> {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, source_y, source_uv, destination, twist)
            }
}
    };
}

macro_rules! impl_nv12_packed_inverse_dispatch {
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
                source_uv: &ImageView<'_, Self, C2>,
                destination: &mut ImageOutputView<'_, Self, $layout>,
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source_y: &ImageView<'_, Self, C1>,
                    source_uv: &ImageView<'_, Self, C2>,
                    destination: &mut ImageOutputView<'_, Self, $layout>,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source_y, source_uv, destination)
                    }
}
            }
        )*

        #[doc = $doc]
        pub unsafe fn $function<T: $trait>(
            stream_context: &StreamContext,
            source_y: &ImageView<'_, T, C1>,
            source_uv: &ImageView<'_, T, C2>,
            destination: &mut ImageOutputView<'_, T, $layout>,
        ) -> Result<()> {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, source_y, source_uv, destination)
            }
}
    };
}

macro_rules! impl_packed_to_planar_subsampled_dispatch {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, $source_layout:ty, [$($ty:ty => $direct:ident),* $(,)?]) => {
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
                source: &ImageView<'_, Self, $source_layout>,
                destination_y: &mut ImageOutputView<'_, Self, C1>,
                destination_cb: &mut ImageOutputView<'_, Self, C1>,
                destination_cr: &mut ImageOutputView<'_, Self, C1>,
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $source_layout>,
                    destination_y: &mut ImageOutputView<'_, Self, C1>,
                    destination_cb: &mut ImageOutputView<'_, Self, C1>,
                    destination_cr: &mut ImageOutputView<'_, Self, C1>,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, destination_y, destination_cb, destination_cr)
                    }
}
            }
        )*

        #[doc = $doc]
        pub unsafe fn $function<T: $trait>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $source_layout>,
            destination_y: &mut ImageOutputView<'_, T, C1>,
            destination_cb: &mut ImageOutputView<'_, T, C1>,
            destination_cr: &mut ImageOutputView<'_, T, C1>,
        ) -> Result<()> {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, source, destination_y, destination_cb, destination_cr)
            }
}
    };
}

macro_rules! impl_planar_to_packed_c2_subsampled_dispatch {
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
                destination: &mut ImageOutputView<'_, Self, C2>,
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source_y: &ImageView<'_, Self, C1>,
                    source_cb: &ImageView<'_, Self, C1>,
                    source_cr: &ImageView<'_, Self, C1>,
                    destination: &mut ImageOutputView<'_, Self, C2>,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source_y, source_cb, source_cr, destination)
                    }
}
            }
        )*

        #[doc = $doc]
        pub unsafe fn $function<T: $trait>(
            stream_context: &StreamContext,
            source_y: &ImageView<'_, T, C1>,
            source_cb: &ImageView<'_, T, C1>,
            source_cr: &ImageView<'_, T, C1>,
            destination: &mut ImageOutputView<'_, T, C2>,
        ) -> Result<()> {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, source_y, source_cb, source_cr, destination)
            }
}
    };
}

impl_nv12_color_twist_packed_inverse_dispatch!(
    "Two channel planar NV12 conversion to three channel packed RGB, using a color twist to compute the exact color space arithmetic.\nEnqueues work on the selected stream without waiting for completion.\nChroma dimensions count positions; each interleaved chroma position stores two components.\nRow padding remains unwritten.\nPasses `twist` as a host color-twist matrix; coefficient values are not validated.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any participating image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nThe full-resolution source and destination dimensions must match.\nFull-resolution width and height must be even; each chroma plane must have half the full-resolution width and height.\nThe source luma and interleaved-chroma planes must have equal byte strides.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples must be initialized and ready when read by the selected stream.\nKeep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.\nRetain image allocations through queued completion, including submission errors and captured graph replay.\nPlane order, packed component layout, byte strides, alignment, and allocation bounds must satisfy the native routine.",

    Nv12ToRgbColorTwistP2ToC3,
    nv12_to_rgb_color_twist_p2_to_c3,
    nv12_to_rgb_color_twist_p2_to_c3,
    C3,
    [
        u8 => nv12_to_rgb_u8_color_twist_p2_to_c3,
        u16 => nv12_to_rgb_u16_color_twist_p2_to_c3
    ]

);

impl_nv12_packed_inverse_dispatch!(
    "2 channel planar NV12 to 3 channel packed RGB color conversion.\nEnqueues work on the selected stream without waiting for completion.\nChroma dimensions count positions; each interleaved chroma position stores two components.\nRow padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any participating image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nThe full-resolution source and destination dimensions must match.\nFull-resolution width and height must be even; each chroma plane must have half the full-resolution width and height.\nThe source luma and interleaved-chroma planes must have equal byte strides.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples must be initialized and ready when read by the selected stream.\nKeep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.\nRetain image allocations through queued completion, including submission errors and captured graph replay.\nPlane order, packed component layout, byte strides, alignment, and allocation bounds must satisfy the native routine.",
Nv12ToRgbP2ToC3, nv12_to_rgb_p2_to_c3, nv12_to_rgb_p2_to_c3, C3, [
    u8 => nv12_to_rgb_u8_p2_to_c3
]
);

impl_nv12_packed_inverse_dispatch!(
    "2 channel planar NV12 to 3 channel packed RGB 709 HDTV full color conversion.\nEnqueues work on the selected stream without waiting for completion.\nChroma dimensions count positions; each interleaved chroma position stores two components.\nRow padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any participating image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nThe full-resolution source and destination dimensions must match.\nFull-resolution width and height must be even; each chroma plane must have half the full-resolution width and height.\nThe source luma and interleaved-chroma planes must have equal byte strides.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples must be initialized and ready when read by the selected stream.\nKeep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.\nRetain image allocations through queued completion, including submission errors and captured graph replay.\nPlane order, packed component layout, byte strides, alignment, and allocation bounds must satisfy the native routine.",
Nv12ToRgb709HdtvP2ToC3, nv12_to_rgb_709hdtv_p2_to_c3, nv12_to_rgb_709hdtv_p2_to_c3, C3, [
    u8 => nv12_to_rgb_709hdtv_u8_p2_to_c3
]
);

impl_nv12_packed_inverse_dispatch!(
    "2 channel planar NV12 to 3 channel packed RGB 709 CSC color conversion.\nEnqueues work on the selected stream without waiting for completion.\nChroma dimensions count positions; each interleaved chroma position stores two components.\nRow padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any participating image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nThe full-resolution source and destination dimensions must match.\nFull-resolution width and height must be even; each chroma plane must have half the full-resolution width and height.\nThe source luma and interleaved-chroma planes must have equal byte strides.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples must be initialized and ready when read by the selected stream.\nKeep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.\nRetain image allocations through queued completion, including submission errors and captured graph replay.\nPlane order, packed component layout, byte strides, alignment, and allocation bounds must satisfy the native routine.",
Nv12ToRgb709CscP2ToC3, nv12_to_rgb_709csc_p2_to_c3, nv12_to_rgb_709csc_p2_to_c3, C3, [
    u8 => nv12_to_rgb_709csc_u8_p2_to_c3
]
);

impl_nv12_packed_inverse_dispatch!(
    "2 channel planar NV12 to 3 channel packed BGR color conversion.\nEnqueues work on the selected stream without waiting for completion.\nChroma dimensions count positions; each interleaved chroma position stores two components.\nRow padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any participating image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nThe full-resolution source and destination dimensions must match.\nFull-resolution width and height must be even; each chroma plane must have half the full-resolution width and height.\nThe source luma and interleaved-chroma planes must have equal byte strides.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples must be initialized and ready when read by the selected stream.\nKeep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.\nRetain image allocations through queued completion, including submission errors and captured graph replay.\nPlane order, packed component layout, byte strides, alignment, and allocation bounds must satisfy the native routine.",
Nv12ToBgrP2ToC3, nv12_to_bgr_p2_to_c3, nv12_to_bgr_p2_to_c3, C3, [
    u8 => nv12_to_bgr_u8_p2_to_c3
]
);

impl_nv12_packed_inverse_dispatch!(
    "2 channel planar NV12 to 3 channel packed RGB 709 HDTV full color conversion.\nEnqueues work on the selected stream without waiting for completion.\nChroma dimensions count positions; each interleaved chroma position stores two components.\nRow padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any participating image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nThe full-resolution source and destination dimensions must match.\nFull-resolution width and height must be even; each chroma plane must have half the full-resolution width and height.\nThe source luma and interleaved-chroma planes must have equal byte strides.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples must be initialized and ready when read by the selected stream.\nKeep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.\nRetain image allocations through queued completion, including submission errors and captured graph replay.\nPlane order, packed component layout, byte strides, alignment, and allocation bounds must satisfy the native routine.",
Nv12ToBgr709HdtvP2ToC3, nv12_to_bgr_709hdtv_p2_to_c3, nv12_to_bgr_709hdtv_p2_to_c3, C3, [
    u8 => nv12_to_bgr_709hdtv_u8_p2_to_c3
]
);

impl_nv12_packed_inverse_dispatch!(
    "2 channel planar NV12 to 3 channel packed RGB 709 CSC color conversion.\nEnqueues work on the selected stream without waiting for completion.\nChroma dimensions count positions; each interleaved chroma position stores two components.\nRow padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any participating image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nThe full-resolution source and destination dimensions must match.\nFull-resolution width and height must be even; each chroma plane must have half the full-resolution width and height.\nThe source luma and interleaved-chroma planes must have equal byte strides.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples must be initialized and ready when read by the selected stream.\nKeep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.\nRetain image allocations through queued completion, including submission errors and captured graph replay.\nPlane order, packed component layout, byte strides, alignment, and allocation bounds must satisfy the native routine.",
Nv12ToBgr709CscP2ToC3, nv12_to_bgr_709csc_p2_to_c3, nv12_to_bgr_709csc_p2_to_c3, C3, [
    u8 => nv12_to_bgr_709csc_u8_p2_to_c3
]
);

impl_nv12_packed_inverse_dispatch!(
    "2 channel planar NV21 to 4 channel packed RGBA color conversion with constant alpha (0xFF).\nEnqueues work on the selected stream without waiting for completion.\nChroma dimensions count positions; each interleaved chroma position stores two components.\nRow padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any participating image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nThe full-resolution source and destination dimensions must match.\nFull-resolution width and height must be even; each chroma plane must have half the full-resolution width and height.\nThe source luma and interleaved-chroma planes must have equal byte strides.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples must be initialized and ready when read by the selected stream.\nKeep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.\nRetain image allocations through queued completion, including submission errors and captured graph replay.\nPlane order, packed component layout, byte strides, alignment, and allocation bounds must satisfy the native routine.",
Nv21ToRgbP2ToC4, nv21_to_rgb_p2_to_c4, nv21_to_rgb_p2_to_c4, C4, [
    u8 => nv21_to_rgb_u8_p2_to_c4
]
);

impl_nv12_packed_inverse_dispatch!(
    "2 channel planar NV21 to 4 channel packed BGRA color conversion with constant alpha (0xFF).\nEnqueues work on the selected stream without waiting for completion.\nChroma dimensions count positions; each interleaved chroma position stores two components.\nRow padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any participating image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nThe full-resolution source and destination dimensions must match.\nFull-resolution width and height must be even; each chroma plane must have half the full-resolution width and height.\nThe source luma and interleaved-chroma planes must have equal byte strides.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples must be initialized and ready when read by the selected stream.\nKeep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.\nRetain image allocations through queued completion, including submission errors and captured graph replay.\nPlane order, packed component layout, byte strides, alignment, and allocation bounds must satisfy the native routine.",
Nv21ToBgrP2ToC4, nv21_to_bgr_p2_to_c4, nv21_to_bgr_p2_to_c4, C4, [
    u8 => nv21_to_bgr_u8_p2_to_c4
]
);

impl_packed_to_planar_subsampled_dispatch!(
    "2 channel packed `YCbCr422` to 3 channel planar `YCbCr422` sampling format conversion.\n\nEnqueues work on the selected stream.\nChroma dimensions count chroma positions; interleaved chroma has two components per position.\nEach plane's byte stride is passed independently.\nRow padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nDestination chroma width and height must equal the full-resolution dimensions divided by 2 and 1, respectively.\nDestination full-resolution dimensions must be divisible by those subsampling factors.\nSeparate Cb and Cr planes must have equal sizes.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples must be initialized and ready when read by the selected stream.\nKeep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.\nRetain image allocations through queued completion, including submission errors and captured graph replay.\nPlane ordering, packed component layout, byte strides, alignment, and allocation bounds must satisfy the native routine.",

    Ycbcr422C2ToP3,
    ycbcr422_c2_to_p3,
    ycbcr422_c2_to_p3,
    C2,
    [u8 => ycbcr422_u8_c2_to_p3]

);

impl_planar_to_packed_c2_subsampled_dispatch!(
    "3 channel planar `YCbCr422` to 2 channel packed `YCbCr422` sampling format conversion.\n\nEnqueues work on the selected stream.\nChroma dimensions count chroma positions; interleaved chroma has two components per position.\nEach plane's byte stride is passed independently.\nRow padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nSource chroma width and height must equal the full-resolution dimensions divided by 2 and 1, respectively.\nSource full-resolution dimensions must be divisible by those subsampling factors.\nSeparate Cb and Cr planes must have equal sizes.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples must be initialized and ready when read by the selected stream.\nKeep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.\nRetain image allocations through queued completion, including submission errors and captured graph replay.\nPlane ordering, packed component layout, byte strides, alignment, and allocation bounds must satisfy the native routine.",

    Ycbcr422P3ToC2,
    ycbcr422_p3_to_c2,
    ycbcr422_p3_to_c2,
    [u8 => ycbcr422_u8_p3_to_c2]

);

impl_planar_to_packed_c2_subsampled_dispatch!(
    "3 channel planar `YCbCr422` to 2 channel packed `YCrCb422` sampling format conversion.\n\nEnqueues work on the selected stream.\nChroma dimensions count chroma positions; interleaved chroma has two components per position.\nEach plane's byte stride is passed independently.\nRow padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nSource chroma width and height must equal the full-resolution dimensions divided by 2 and 1, respectively.\nSource full-resolution dimensions must be divisible by those subsampling factors.\nSeparate Cb and Cr planes must have equal sizes.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples must be initialized and ready when read by the selected stream.\nKeep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.\nRetain image allocations through queued completion, including submission errors and captured graph replay.\nPlane ordering, packed component layout, byte strides, alignment, and allocation bounds must satisfy the native routine.",

    Ycbcr422ToYcrcb422P3ToC2,
    ycbcr422_to_ycrcb422_p3_to_c2,
    ycbcr422_to_ycrcb422_p3_to_c2,
    [u8 => ycbcr422_to_ycrcb422_u8_p3_to_c2]

);

impl_planar_to_packed_c2_subsampled_dispatch!(
    "3 channel planar `YCbCr411` to 2 channel packed `YCbCr422` sampling format conversion.\n\nEnqueues work on the selected stream.\nChroma dimensions count chroma positions; interleaved chroma has two components per position.\nEach plane's byte stride is passed independently.\nRow padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nSource chroma width and height must equal the full-resolution dimensions divided by 4 and 1, respectively.\nSource full-resolution dimensions must be divisible by those subsampling factors.\nSeparate Cb and Cr planes must have equal sizes.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples must be initialized and ready when read by the selected stream.\nKeep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.\nRetain image allocations through queued completion, including submission errors and captured graph replay.\nPlane ordering, packed component layout, byte strides, alignment, and allocation bounds must satisfy the native routine.",

    Ycbcr411ToYcbcr422P3ToC2,
    ycbcr411_to_ycbcr422_p3_to_c2,
    ycbcr411_to_ycbcr422_p3_to_c2,
    [u8 => ycbcr411_to_ycbcr422_u8_p3_to_c2]

);

impl_planar_to_packed_c2_subsampled_dispatch!(
    "3 channel planar `YCbCr411` to 2 channel packed `YCrCb422` sampling format conversion.\n\nEnqueues work on the selected stream.\nChroma dimensions count chroma positions; interleaved chroma has two components per position.\nEach plane's byte stride is passed independently.\nRow padding remains unwritten.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nSource chroma width and height must equal the full-resolution dimensions divided by 4 and 1, respectively.\nSource full-resolution dimensions must be divisible by those subsampling factors.\nSeparate Cb and Cr planes must have equal sizes.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples must be initialized and ready when read by the selected stream.\nKeep inputs read-only and each destination exclusively writable and disjoint from all inputs and other destinations during native use.\nRetain image allocations through queued completion, including submission errors and captured graph replay.\nPlane ordering, packed component layout, byte strides, alignment, and allocation bounds must satisfy the native routine.",

    Ycbcr411ToYcrcb422P3ToC2,
    ycbcr411_to_ycrcb422_p3_to_c2,
    ycbcr411_to_ycrcb422_p3_to_c2,
    [u8 => ycbcr411_to_ycrcb422_u8_p3_to_c2]

);

impl_planar_to_packed_c2_subsampled_dispatch!(
    "Converts the separate red, green, and blue source planes to packed `YCbCr422` pixels.\n\nEnqueues work on the selected stream.\nRow padding is not initialized.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image belongs to another CUDA context, or NPP rejects the conversion parameters.\nAll source and destination views must have equal image sizes, and the three source planes must have equal byte strides.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples must be initialized and ready when read by the selected stream.\nKeep source storage read-only and each destination exclusively writable and disjoint from all sources and other destinations during native use.\nRetain image allocations through queued completion, including submission errors and captured graph replay.\nImage layouts, component ranges, 4:2:2 packing geometry, byte strides, alignment, and allocation bounds must satisfy the native routine.",

    RgbToYcbcr422P3ToC2,
    rgb_to_ycbcr422_p3_to_c2,
    rgb_to_ycbcr422_p3_to_c2,
    [u8 => rgb_to_ycbcr422_u8_p3_to_c2]

);

impl_planar_to_packed_c2_subsampled_dispatch!(
    "Converts the separate red, green, and blue source planes to packed `YCrCb422` pixels.\n\nEnqueues work on the selected stream.\nRow padding is not initialized.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image belongs to another CUDA context, or NPP rejects the conversion parameters.\nAll source and destination views must have equal image sizes, and the three source planes must have equal byte strides.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples must be initialized and ready when read by the selected stream.\nKeep source storage read-only and each destination exclusively writable and disjoint from all sources and other destinations during native use.\nRetain image allocations through queued completion, including submission errors and captured graph replay.\nImage layouts, component ranges, 4:2:2 packing geometry, byte strides, alignment, and allocation bounds must satisfy the native routine.",

    RgbToYcrcb422P3ToC2,
    rgb_to_ycrcb422_p3_to_c2,
    rgb_to_ycrcb422_p3_to_c2,
    [u8 => rgb_to_ycrcb422_u8_p3_to_c2]

);

impl_packed_to_planar_subsampled_dispatch!(
    "Converts packed `YCbCr422` pixels to separate red, green, and blue destination planes.\n\nEnqueues work on the selected stream.\nRow padding is not initialized.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image belongs to another CUDA context, or NPP rejects the conversion parameters.\nAll source and destination views must have equal image sizes, and the three destination planes must have equal byte strides.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples must be initialized and ready when read by the selected stream.\nKeep source storage read-only and each destination exclusively writable and disjoint from all sources and other destinations during native use.\nRetain image allocations through queued completion, including submission errors and captured graph replay.\nImage layouts, component ranges, 4:2:2 packing geometry, byte strides, alignment, and allocation bounds must satisfy the native routine.",

    Ycbcr422ToRgbC2ToP3,
    ycbcr422_to_rgb_c2_to_p3,
    ycbcr422_to_rgb_c2_to_p3,
    C2,
    [u8 => ycbcr422_to_rgb_u8_c2_to_p3]

);

impl_packed_to_planar_subsampled_dispatch!(
    "Converts packed `YCrCb422` pixels to separate red, green, and blue destination planes.\n\nEnqueues work on the selected stream.\nRow padding is not initialized.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image belongs to another CUDA context, or NPP rejects the conversion parameters.\nAll source and destination views must have equal image sizes, and the three destination planes must have equal byte strides.\n\n# Safety\n\nFollow the [raw image execution contract](crate::image::raw).\nSource samples must be initialized and ready when read by the selected stream.\nKeep source storage read-only and each destination exclusively writable and disjoint from all sources and other destinations during native use.\nRetain image allocations through queued completion, including submission errors and captured graph replay.\nImage layouts, component ranges, 4:2:2 packing geometry, byte strides, alignment, and allocation bounds must satisfy the native routine.",

    Ycrcb422ToRgbC2ToP3,
    ycrcb422_to_rgb_c2_to_p3,
    ycrcb422_to_rgb_c2_to_p3,
    C2,
    [u8 => ycrcb422_to_rgb_u8_c2_to_p3]

);
