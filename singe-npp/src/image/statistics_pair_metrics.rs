use singe_cuda::{memory::DeviceMemory, view::DeviceOutputView};
use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        statistics_validation::{validate_metric_output, validate_same_size},
        view::{C1, C3, C4, ImageView},
    },
    try_ffi,
    types::{DataTypeLike, Size},
    utility::to_usize,
};

macro_rules! impl_image_pair_metric {
    (
        $doc:literal,
        $buffer_size_name:ident,
        $name:ident,
        $ty:ty,
        $layout:ty,
        $channels:expr,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        /// Returns temporary scratch bytes required per image pair for a metric ROI.
        #[doc = concat!("Calls [`sys::", stringify!($buffer_size_ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if NPP rejects the ROI or buffer-size query, or its returned size cannot be represented as `usize`.
        pub fn $buffer_size_name(stream_context: &StreamContext, roi: Size) -> Result<usize> {
            let mut bytes = 0;
            // SAFETY: the `buffer_size_ffi` entry point receives live handles owned by the wrappers and valid output locations.
            unsafe {
                try_ffi!(sys::$buffer_size_ffi(
                    roi.into(),
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "buffer size")
        }

        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Owns a temporary GPU scratch buffer for the metric.
        /// Rejects stream capture and returns after successful stream completion.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, an image or result allocation belongs to another CUDA context, or NPP rejects the metric parameters.
        /// Source images must have matching dimensions.
        /// The output length must equal the processed channel count.
        /// Buffer-size queries, scratch allocation, capture checks, and stream synchronization can also fail.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels must be initialized and ready when read by the selected stream.
        /// Keep sources read-only and result storage exclusively writable and disjoint from all source allocations during native use.
        /// Keep source and result allocations valid until this call completes its native work.
        /// For WMSSSIM, every processed ROI must be at least 16 by 16 pixels; this wrapper does not validate that requirement.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source_1: &ImageView<'_, $ty, $layout>,
            source_2: &ImageView<'_, $ty, $layout>,
            output: &mut DeviceOutputView<'_, f32>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source_1)?;
            stream_context.validate_region(source_2)?;
            stream_context.validate_region(output)?;
            stream_context.stream().ensure_not_capturing()?;

            validate_same_size(source_1.size(), source_2.size())?;
            validate_metric_output(output, 1, $channels)?;
            let required_bytes = $buffer_size_name(stream_context, source_1.size())?;
            let scratch =
                DeviceMemory::<u8>::create_uninit(stream_context.cuda_context(), required_bytes)?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
                try_ffi!(sys::$ffi(
                    source_1.as_ptr().cast(),
                    source_1.step(),
                    source_2.as_ptr().cast(),
                    source_2.step(),
                    source_1.size().into(),
                    output.as_mut_ptr().cast(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))
            })?;

            Ok(())
        }
    };
}

macro_rules! impl_generic_image_pair_metric {
    ($doc:literal, $trait_name:ident, $name:ident, $buffer_size_name:ident, $layout:ty, [$(($ty:ty, $direct:ident, $direct_buffer_size:ident)),+ $(,)?]) => {
        #[doc = concat!("Native image-element and layout dispatch for [`", stringify!($name), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and channel-layout ABI, metric semantics, and output representation.
        /// They must uphold the methods' validation and completion contracts, and workspace queries must not underestimate the matching operation's requirements.
        pub unsafe trait $trait_name<Layout>: DataTypeLike {
            /// Returns temporary scratch bytes required per image pair for a metric ROI.
            ///
            /// # Errors
            ///
            /// Returns an error if NPP rejects the ROI or buffer-size query, or its returned size cannot be represented as `usize`.
            fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize>;

            #[doc = $doc]
            ///
            /// Owns a temporary GPU scratch buffer for the metric.
            /// Rejects stream capture and returns after successful stream completion.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails, an image or result allocation belongs to another CUDA context, or NPP rejects the metric parameters.
            /// Source images must have matching dimensions.
            /// The output length must equal the processed channel count.
            /// Buffer-size queries, scratch allocation, capture checks, and stream synchronization can also fail.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source pixels must be initialized and ready when read by the selected stream.
            /// Keep sources read-only and result storage exclusively writable and disjoint from all source allocations during native use.
            /// Keep source and result allocations valid until this call completes its native work.
            /// For WMSSSIM, every processed ROI must be at least 16 by 16 pixels; this wrapper does not validate that requirement.
            /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn dispatch(
                stream_context: &StreamContext,
                source_1: &ImageView<'_, Self, Layout>,
                source_2: &ImageView<'_, Self, Layout>,
                output: &mut DeviceOutputView<'_, f32>,
            ) -> Result<()>
            where
                Self: Sized;
        }

        $(
            unsafe impl $trait_name<$layout> for $ty {
                fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize> {
                    $direct_buffer_size(stream_context, roi)
                }

                unsafe fn dispatch(
                    stream_context: &StreamContext,
                    source_1: &ImageView<'_, Self, $layout>,
                    source_2: &ImageView<'_, Self, $layout>,
                    output: &mut DeviceOutputView<'_, f32>,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source_1, source_2, output)
                    }
}
            }
        )+

        /// Returns temporary scratch bytes required per image pair for a metric ROI.
        ///
        /// # Errors
        ///
        /// Returns an error if NPP rejects the ROI or buffer-size query, or its returned size cannot be represented as `usize`.
        pub fn $buffer_size_name<T: $trait_name<$layout>>(
            stream_context: &StreamContext,
            roi: Size,
        ) -> Result<usize> {
            T::buffer_size(stream_context, roi)
        }

        #[doc = $doc]
        ///
        /// Owns a temporary GPU scratch buffer for the metric.
        /// Rejects stream capture and returns after successful stream completion.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, an image or result allocation belongs to another CUDA context, or NPP rejects the metric parameters.
        /// Source images must have matching dimensions.
        /// The output length must equal the processed channel count.
        /// Buffer-size queries, scratch allocation, capture checks, and stream synchronization can also fail.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels must be initialized and ready when read by the selected stream.
        /// Keep sources read-only and result storage exclusively writable and disjoint from all source allocations during native use.
        /// Keep source and result allocations valid until this call completes its native work.
        /// For WMSSSIM, every processed ROI must be at least 16 by 16 pixels; this wrapper does not validate that requirement.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name<T: $trait_name<$layout>>(
            stream_context: &StreamContext,
            source_1: &ImageView<'_, T, $layout>,
            source_2: &ImageView<'_, T, $layout>,
            output: &mut DeviceOutputView<'_, f32>,
        ) -> Result<()> {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::dispatch(stream_context, source_1, source_2, output)
            }
}
    };
}

impl_image_pair_metric!(
    "Computes mean squared error (MSE) for an image pair, producing one result per processed channel.\n\nUses the matching source image dimensions as the metric ROI.",
    mse_u8_c1_buffer_size,
    mse_u8_c1,
    u8,
    C1,
    1,
    nppiMSEGetBufferHostSize_8u_C1R_Ctx,
    nppiMSE_8u_C1R_Ctx
);

impl_image_pair_metric!(
    "Computes mean squared error (MSE) for an image pair, producing one result per processed channel.\n\nUses the matching source image dimensions as the metric ROI.",
    mse_u8_c3_buffer_size,
    mse_u8_c3,
    u8,
    C3,
    3,
    nppiMSEGetBufferHostSize_8u_C3R_Ctx,
    nppiMSE_8u_C3R_Ctx
);

impl_image_pair_metric!(
    "Computes peak signal-to-noise ratio (PSNR) for an image pair, producing one result per processed channel.\n\nUses the matching source image dimensions as the metric ROI.",
    psnr_u8_c1_buffer_size,
    psnr_u8_c1,
    u8,
    C1,
    1,
    nppiPSNRGetBufferHostSize_8u_C1R_Ctx,
    nppiPSNR_8u_C1R_Ctx
);

impl_image_pair_metric!(
    "Computes peak signal-to-noise ratio (PSNR) for an image pair, producing one result per processed channel.\n\nUses the matching source image dimensions as the metric ROI.",
    psnr_u8_c3_buffer_size,
    psnr_u8_c3,
    u8,
    C3,
    3,
    nppiPSNRGetBufferHostSize_8u_C3R_Ctx,
    nppiPSNR_8u_C3R_Ctx
);

impl_image_pair_metric!(
    "Computes structural similarity (SSIM) for an image pair, producing one result per processed channel.\n\nUses the matching source image dimensions as the metric ROI.",
    ssim_u8_c1_buffer_size,
    ssim_u8_c1,
    u8,
    C1,
    1,
    nppiSSIMGetBufferHostSize_8u_C1R_Ctx,
    nppiSSIM_8u_C1R_Ctx
);

impl_image_pair_metric!(
    "Computes structural similarity (SSIM) for an image pair, producing one result per processed channel.\n\nUses the matching source image dimensions as the metric ROI.",
    ssim_u8_c3_buffer_size,
    ssim_u8_c3,
    u8,
    C3,
    3,
    nppiSSIMGetBufferHostSize_8u_C3R_Ctx,
    nppiSSIM_8u_C3R_Ctx
);

impl_image_pair_metric!(
    "Computes the NPP MSSSIM metric for an image pair, producing one result per processed channel.\n\nUses the matching source image dimensions as the metric ROI. The native documentation recommends WMSSSIM instead of MSSSIM.",
    msssim_u8_c1_buffer_size,
    msssim_u8_c1,
    u8,
    C1,
    1,
    nppiMSSSIMGetBufferHostSize_8u_C1R_Ctx,
    nppiMSSSIM_8u_C1R_Ctx
);

impl_image_pair_metric!(
    "Computes weighted multiscale structural similarity (WMSSSIM) for an image pair, producing one result per processed channel.\n\nUses the matching source image dimensions as the metric ROI.",
    wmsssim_u8_c1_buffer_size,
    wmsssim_u8_c1,
    u8,
    C1,
    1,
    nppiWMSSSIMGetBufferHostSize_8u_C1R_Ctx,
    nppiWMSSSIM_8u_C1R_Ctx
);

impl_image_pair_metric!(
    "Computes weighted multiscale structural similarity (WMSSSIM) for an image pair, producing one result per processed channel.\n\nUses the matching source image dimensions as the metric ROI.",
    wmsssim_u8_c3_buffer_size,
    wmsssim_u8_c3,
    u8,
    C3,
    3,
    nppiWMSSSIMGetBufferHostSize_8u_C3R_Ctx,
    nppiWMSSSIM_8u_C3R_Ctx
);

impl_image_pair_metric!(
    "Computes weighted multiscale structural similarity (WMSSSIM) for an image pair, producing one result per processed channel.\n\nUses the matching source image dimensions as the metric ROI.",
    wmsssim_u8_c4_buffer_size,
    wmsssim_u8_c4,
    u8,
    C4,
    4,
    nppiWMSSSIMGetBufferHostSize_8u_C4R_Ctx,
    nppiWMSSSIM_8u_C4R_Ctx
);

impl_generic_image_pair_metric!(
    "Computes mean squared error (MSE) for an image pair, producing one result per processed channel.\n\nUses the matching source image dimensions as the metric ROI.",
    MseC1,
    mse_c1,
    mse_c1_buffer_size,
    C1,
    [(u8, mse_u8_c1, mse_u8_c1_buffer_size)]
);

impl_generic_image_pair_metric!(
    "Computes mean squared error (MSE) for an image pair, producing one result per processed channel.\n\nUses the matching source image dimensions as the metric ROI.",
    MseC3,
    mse_c3,
    mse_c3_buffer_size,
    C3,
    [(u8, mse_u8_c3, mse_u8_c3_buffer_size)]
);

impl_generic_image_pair_metric!(
    "Computes peak signal-to-noise ratio (PSNR) for an image pair, producing one result per processed channel.\n\nUses the matching source image dimensions as the metric ROI.",
    PsnrC1,
    psnr_c1,
    psnr_c1_buffer_size,
    C1,
    [(u8, psnr_u8_c1, psnr_u8_c1_buffer_size)]
);

impl_generic_image_pair_metric!(
    "Computes peak signal-to-noise ratio (PSNR) for an image pair, producing one result per processed channel.\n\nUses the matching source image dimensions as the metric ROI.",
    PsnrC3,
    psnr_c3,
    psnr_c3_buffer_size,
    C3,
    [(u8, psnr_u8_c3, psnr_u8_c3_buffer_size)]
);

impl_generic_image_pair_metric!(
    "Computes structural similarity (SSIM) for an image pair, producing one result per processed channel.\n\nUses the matching source image dimensions as the metric ROI.",
    SsimC1,
    ssim_c1,
    ssim_c1_buffer_size,
    C1,
    [(u8, ssim_u8_c1, ssim_u8_c1_buffer_size)]
);

impl_generic_image_pair_metric!(
    "Computes structural similarity (SSIM) for an image pair, producing one result per processed channel.\n\nUses the matching source image dimensions as the metric ROI.",
    SsimC3,
    ssim_c3,
    ssim_c3_buffer_size,
    C3,
    [(u8, ssim_u8_c3, ssim_u8_c3_buffer_size)]
);

impl_generic_image_pair_metric!(
    "Computes the NPP MSSSIM metric for an image pair, producing one result per processed channel.\n\nUses the matching source image dimensions as the metric ROI. The native documentation recommends WMSSSIM instead of MSSSIM.",
    MsssimC1,
    msssim_c1,
    msssim_c1_buffer_size,
    C1,
    [(u8, msssim_u8_c1, msssim_u8_c1_buffer_size)]
);

impl_generic_image_pair_metric!(
    "Computes weighted multiscale structural similarity (WMSSSIM) for an image pair, producing one result per processed channel.\n\nUses the matching source image dimensions as the metric ROI.",
    WmsssimC1,
    wmsssim_c1,
    wmsssim_c1_buffer_size,
    C1,
    [(u8, wmsssim_u8_c1, wmsssim_u8_c1_buffer_size)]
);

impl_generic_image_pair_metric!(
    "Computes weighted multiscale structural similarity (WMSSSIM) for an image pair, producing one result per processed channel.\n\nUses the matching source image dimensions as the metric ROI.",
    WmsssimC3,
    wmsssim_c3,
    wmsssim_c3_buffer_size,
    C3,
    [(u8, wmsssim_u8_c3, wmsssim_u8_c3_buffer_size)]
);

impl_generic_image_pair_metric!(
    "Computes weighted multiscale structural similarity (WMSSSIM) for an image pair, producing one result per processed channel.\n\nUses the matching source image dimensions as the metric ROI.",
    WmsssimC4,
    wmsssim_c4,
    wmsssim_c4_buffer_size,
    C4,
    [(u8, wmsssim_u8_c4, wmsssim_u8_c4_buffer_size)]
);
