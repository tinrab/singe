use singe_cuda::view::DeviceOutputView;
use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        statistics_validation::{
            pair_metric_batch_advanced_descriptors, pair_metric_batch_buffers,
            pair_metric_batch_descriptors, validate_metric_output,
        },
        view::{C1, C3, C4, ImageView},
    },
    try_ffi,
    types::{DataTypeLike, Size},
    utility::to_usize,
};

macro_rules! impl_image_pair_metric_batch {
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
        /// Owns temporary GPU image descriptors and one scratch buffer per image pair.
        /// Rejects stream capture and returns after successful stream completion.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, an image or result allocation belongs to another CUDA context, or NPP rejects the metric parameters.
        /// Source batch lengths must match, contain at least two image pairs, and fit in `i32`.
        /// Corresponding source images must have matching sizes.
        /// The result length must equal the checked product of batch length and processed channel count.
        /// Buffer-size queries, descriptor or scratch allocation and upload, native scratch-size conversion to `i32`, capture checks, and stream synchronization can also fail.
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
            source_0: &[ImageView<'_, $ty, $layout>],
            source_1: &[ImageView<'_, $ty, $layout>],
            results: &mut DeviceOutputView<'_, f32>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(results)?;
            for image in source_0.iter() {
                stream_context.validate_region(image)?;
            }
            for image in source_1.iter() {
                stream_context.validate_region(image)?;
            }
            stream_context.stream().ensure_not_capturing()?;

            let (batch_size, roi, descriptors_0, descriptors_1) =
                pair_metric_batch_descriptors(stream_context.cuda_context(), source_0, source_1)?;
            validate_metric_output(results, batch_size as usize, $channels)?;
            let required_bytes = $buffer_size_name(stream_context, roi)?;
            let (_scratch_buffers, mut scratch_descriptors) = pair_metric_batch_buffers(
                stream_context.cuda_context(),
                batch_size as usize,
                required_bytes,
            )?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
                try_ffi!(sys::$ffi(
                    descriptors_0.as_ptr().cast(),
                    descriptors_1.as_ptr().cast(),
                    batch_size,
                    roi.into(),
                    results.as_mut_ptr().cast(),
                    scratch_descriptors.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))
            })?;

            Ok(())
        }
    };
}

macro_rules! impl_image_pair_metric_batch_advanced {
    (
        $doc:literal,
        $buffer_size_name:ident,
        $name:ident,
        $ty:ty,
        $layout:ty,
        $channels:expr,
        $ffi:ident
    ) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Owns temporary GPU image descriptors and one scratch buffer per image pair.
        /// Rejects stream capture and returns after successful stream completion.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, an image or result allocation belongs to another CUDA context, or NPP rejects the metric parameters.
        /// Source batch lengths must match, contain at least two image pairs, and fit in `i32`.
        /// Corresponding source images must have matching sizes.
        /// The result length must equal the checked product of batch length and processed channel count.
        /// Buffer-size queries, descriptor or scratch allocation and upload, native scratch-size conversion to `i32`, capture checks, and stream synchronization can also fail.
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
            source_0: &[ImageView<'_, $ty, $layout>],
            source_1: &[ImageView<'_, $ty, $layout>],
            results: &mut DeviceOutputView<'_, f32>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(results)?;
            for image in source_0.iter() {
                stream_context.validate_region(image)?;
            }
            for image in source_1.iter() {
                stream_context.validate_region(image)?;
            }
            stream_context.stream().ensure_not_capturing()?;

            let (batch_size, max_roi, descriptors_0, descriptors_1) =
                pair_metric_batch_advanced_descriptors(
                    stream_context.cuda_context(),
                    source_0,
                    source_1,
                )?;
            validate_metric_output(results, batch_size as usize, $channels)?;
            let required_bytes = $buffer_size_name(stream_context, max_roi)?;
            let (_scratch_buffers, mut scratch_descriptors) = pair_metric_batch_buffers(
                stream_context.cuda_context(),
                batch_size as usize,
                required_bytes,
            )?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
                try_ffi!(sys::$ffi(
                    descriptors_0.as_ptr().cast(),
                    descriptors_1.as_ptr().cast(),
                    batch_size,
                    max_roi.into(),
                    results.as_mut_ptr().cast(),
                    scratch_descriptors.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))
            })?;

            Ok(())
        }
    };
}

macro_rules! impl_generic_image_pair_metric_batch {
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
            /// Owns temporary GPU image descriptors and one scratch buffer per image pair.
            /// Rejects stream capture and returns after successful stream completion.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails, an image or result allocation belongs to another CUDA context, or NPP rejects the metric parameters.
            /// Source batch lengths must match, contain at least two image pairs, and fit in `i32`.
            /// Corresponding source images must have matching sizes.
            /// The result length must equal the checked product of batch length and processed channel count.
            /// Buffer-size queries, descriptor or scratch allocation and upload, native scratch-size conversion to `i32`, capture checks, and stream synchronization can also fail.
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
                source_0: &[ImageView<'_, Self, Layout>],
                source_1: &[ImageView<'_, Self, Layout>],
                results: &mut DeviceOutputView<'_, f32>,
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
                    source_0: &[ImageView<'_, Self, $layout>],
                    source_1: &[ImageView<'_, Self, $layout>],
                    results: &mut DeviceOutputView<'_, f32>,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source_0, source_1, results)
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
        /// Owns temporary GPU image descriptors and one scratch buffer per image pair.
        /// Rejects stream capture and returns after successful stream completion.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, an image or result allocation belongs to another CUDA context, or NPP rejects the metric parameters.
        /// Source batch lengths must match, contain at least two image pairs, and fit in `i32`.
        /// Corresponding source images must have matching sizes.
        /// The result length must equal the checked product of batch length and processed channel count.
        /// Buffer-size queries, descriptor or scratch allocation and upload, native scratch-size conversion to `i32`, capture checks, and stream synchronization can also fail.
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
            source_0: &[ImageView<'_, T, $layout>],
            source_1: &[ImageView<'_, T, $layout>],
            results: &mut DeviceOutputView<'_, f32>,
        ) -> Result<()> {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::dispatch(stream_context, source_0, source_1, results)
            }
}
    };
}

impl_image_pair_metric_batch!(
    "Computes mean squared error (MSE) for a batch of image pairs, producing one result per pair and processed channel.\n\nUses the common ROI defined by the smallest image width and smallest image height across both batches.",
    mse_batch_u8_c1_buffer_size,
    mse_batch_u8_c1,
    u8,
    C1,
    1,
    nppiMSEBatchGetBufferHostSize_8u_C1R_Ctx,
    nppiMSEBatch_8u_C1R_Ctx
);

impl_image_pair_metric_batch!(
    "Computes mean squared error (MSE) for a batch of image pairs, producing one result per pair and processed channel.\n\nUses the common ROI defined by the smallest image width and smallest image height across both batches.",
    mse_batch_u8_c3_buffer_size,
    mse_batch_u8_c3,
    u8,
    C3,
    3,
    nppiMSEBatchGetBufferHostSize_8u_C3R_Ctx,
    nppiMSEBatch_8u_C3R_Ctx
);

impl_image_pair_metric_batch!(
    "Computes peak signal-to-noise ratio (PSNR) for a batch of image pairs, producing one result per pair and processed channel.\n\nUses the common ROI defined by the smallest image width and smallest image height across both batches.",
    psnr_batch_u8_c1_buffer_size,
    psnr_batch_u8_c1,
    u8,
    C1,
    1,
    nppiPSNRBatchGetBufferHostSize_8u_C1R_Ctx,
    nppiPSNRBatch_8u_C1R_Ctx
);

impl_image_pair_metric_batch!(
    "Computes peak signal-to-noise ratio (PSNR) for a batch of image pairs, producing one result per pair and processed channel.\n\nUses the common ROI defined by the smallest image width and smallest image height across both batches.",
    psnr_batch_u8_c3_buffer_size,
    psnr_batch_u8_c3,
    u8,
    C3,
    3,
    nppiPSNRBatchGetBufferHostSize_8u_C3R_Ctx,
    nppiPSNRBatch_8u_C3R_Ctx
);

impl_image_pair_metric_batch!(
    "Computes structural similarity (SSIM) for a batch of image pairs, producing one result per pair and processed channel.\n\nUses the common ROI defined by the smallest image width and smallest image height across both batches.",
    ssim_batch_u8_c1_buffer_size,
    ssim_batch_u8_c1,
    u8,
    C1,
    1,
    nppiSSIMBatchGetBufferHostSize_8u_C1R_Ctx,
    nppiSSIMBatch_8u_C1R_Ctx
);

impl_image_pair_metric_batch!(
    "Computes structural similarity (SSIM) for a batch of image pairs, producing one result per pair and processed channel.\n\nUses the common ROI defined by the smallest image width and smallest image height across both batches.",
    ssim_batch_u8_c3_buffer_size,
    ssim_batch_u8_c3,
    u8,
    C3,
    3,
    nppiSSIMBatchGetBufferHostSize_8u_C3R_Ctx,
    nppiSSIMBatch_8u_C3R_Ctx
);

impl_image_pair_metric_batch!(
    "Computes weighted multiscale structural similarity (WMSSSIM) for a batch of image pairs, producing one result per pair and processed channel.\n\nUses the common ROI defined by the smallest image width and smallest image height across both batches.",
    wmsssim_batch_u8_c1_buffer_size,
    wmsssim_batch_u8_c1,
    u8,
    C1,
    1,
    nppiWMSSSIMBatchGetBufferHostSize_8u_C1R_Ctx,
    nppiWMSSSIMBatch_8u_C1R_Ctx
);

impl_image_pair_metric_batch!(
    "Computes weighted multiscale structural similarity (WMSSSIM) for a batch of image pairs, producing one result per pair and processed channel.\n\nUses the common ROI defined by the smallest image width and smallest image height across both batches.",
    wmsssim_batch_u8_c3_buffer_size,
    wmsssim_batch_u8_c3,
    u8,
    C3,
    3,
    nppiWMSSSIMBatchGetBufferHostSize_8u_C3R_Ctx,
    nppiWMSSSIMBatch_8u_C3R_Ctx
);

impl_image_pair_metric_batch!(
    "Computes weighted multiscale structural similarity (WMSSSIM) for a batch of image pairs, producing one result per pair and processed channel.\n\nUses the common ROI defined by the smallest image width and smallest image height across both batches.",
    wmsssim_batch_u8_c4_buffer_size,
    wmsssim_batch_u8_c4,
    u8,
    C4,
    4,
    nppiWMSSSIMBatchGetBufferHostSize_8u_C4R_Ctx,
    nppiWMSSSIMBatch_8u_C4R_Ctx
);

impl_image_pair_metric_batch_advanced!(
    "Computes mean squared error (MSE) for a batch of image pairs, producing one result per pair and processed channel.\n\nUses each image pair's own ROI; the native maximum ROI takes the largest width and largest height across the batch.",
    mse_batch_u8_c1_buffer_size,
    mse_batch_u8_c1_advanced,
    u8,
    C1,
    1,
    nppiMSEBatch_8u_C1R_Advanced_Ctx
);

impl_image_pair_metric_batch_advanced!(
    "Computes mean squared error (MSE) for a batch of image pairs, producing one result per pair and processed channel.\n\nUses each image pair's own ROI; the native maximum ROI takes the largest width and largest height across the batch.",
    mse_batch_u8_c3_buffer_size,
    mse_batch_u8_c3_advanced,
    u8,
    C3,
    3,
    nppiMSEBatch_8u_C3R_Advanced_Ctx
);

impl_image_pair_metric_batch_advanced!(
    "Computes peak signal-to-noise ratio (PSNR) for a batch of image pairs, producing one result per pair and processed channel.\n\nUses each image pair's own ROI; the native maximum ROI takes the largest width and largest height across the batch.",
    psnr_batch_u8_c1_buffer_size,
    psnr_batch_u8_c1_advanced,
    u8,
    C1,
    1,
    nppiPSNRBatch_8u_C1R_Advanced_Ctx
);

impl_image_pair_metric_batch_advanced!(
    "Computes peak signal-to-noise ratio (PSNR) for a batch of image pairs, producing one result per pair and processed channel.\n\nUses each image pair's own ROI; the native maximum ROI takes the largest width and largest height across the batch.",
    psnr_batch_u8_c3_buffer_size,
    psnr_batch_u8_c3_advanced,
    u8,
    C3,
    3,
    nppiPSNRBatch_8u_C3R_Advanced_Ctx
);

impl_image_pair_metric_batch_advanced!(
    "Computes weighted multiscale structural similarity (WMSSSIM) for a batch of image pairs, producing one result per pair and processed channel.\n\nUses each image pair's own ROI; the native maximum ROI takes the largest width and largest height across the batch.",
    wmsssim_batch_u8_c1_buffer_size,
    wmsssim_batch_u8_c1_advanced,
    u8,
    C1,
    1,
    nppiWMSSSIMBatch_8u_C1R_Advanced_Ctx
);

impl_image_pair_metric_batch_advanced!(
    "Computes weighted multiscale structural similarity (WMSSSIM) for a batch of image pairs, producing one result per pair and processed channel.\n\nUses each image pair's own ROI; the native maximum ROI takes the largest width and largest height across the batch.",
    wmsssim_batch_u8_c3_buffer_size,
    wmsssim_batch_u8_c3_advanced,
    u8,
    C3,
    3,
    nppiWMSSSIMBatch_8u_C3R_Advanced_Ctx
);

impl_image_pair_metric_batch_advanced!(
    "Computes weighted multiscale structural similarity (WMSSSIM) for a batch of image pairs, producing one result per pair and processed channel.\n\nUses each image pair's own ROI; the native maximum ROI takes the largest width and largest height across the batch.",
    wmsssim_batch_u8_c4_buffer_size,
    wmsssim_batch_u8_c4_advanced,
    u8,
    C4,
    4,
    nppiWMSSSIMBatch_8u_C4R_Advanced_Ctx
);

impl_generic_image_pair_metric_batch!(
    "Computes mean squared error (MSE) for a batch of image pairs, producing one result per pair and processed channel.\n\nUses the common ROI defined by the smallest image width and smallest image height across both batches.",
    MseBatchC1,
    mse_batch_c1,
    mse_batch_c1_buffer_size,
    C1,
    [(u8, mse_batch_u8_c1, mse_batch_u8_c1_buffer_size)]
);

impl_generic_image_pair_metric_batch!(
    "Computes mean squared error (MSE) for a batch of image pairs, producing one result per pair and processed channel.\n\nUses the common ROI defined by the smallest image width and smallest image height across both batches.",
    MseBatchC3,
    mse_batch_c3,
    mse_batch_c3_buffer_size,
    C3,
    [(u8, mse_batch_u8_c3, mse_batch_u8_c3_buffer_size)]
);

impl_generic_image_pair_metric_batch!(
    "Computes peak signal-to-noise ratio (PSNR) for a batch of image pairs, producing one result per pair and processed channel.\n\nUses the common ROI defined by the smallest image width and smallest image height across both batches.",
    PsnrBatchC1,
    psnr_batch_c1,
    psnr_batch_c1_buffer_size,
    C1,
    [(u8, psnr_batch_u8_c1, psnr_batch_u8_c1_buffer_size)]
);

impl_generic_image_pair_metric_batch!(
    "Computes peak signal-to-noise ratio (PSNR) for a batch of image pairs, producing one result per pair and processed channel.\n\nUses the common ROI defined by the smallest image width and smallest image height across both batches.",
    PsnrBatchC3,
    psnr_batch_c3,
    psnr_batch_c3_buffer_size,
    C3,
    [(u8, psnr_batch_u8_c3, psnr_batch_u8_c3_buffer_size)]
);

impl_generic_image_pair_metric_batch!(
    "Computes structural similarity (SSIM) for a batch of image pairs, producing one result per pair and processed channel.\n\nUses the common ROI defined by the smallest image width and smallest image height across both batches.",
    SsimBatchC1,
    ssim_batch_c1,
    ssim_batch_c1_buffer_size,
    C1,
    [(u8, ssim_batch_u8_c1, ssim_batch_u8_c1_buffer_size)]
);

impl_generic_image_pair_metric_batch!(
    "Computes structural similarity (SSIM) for a batch of image pairs, producing one result per pair and processed channel.\n\nUses the common ROI defined by the smallest image width and smallest image height across both batches.",
    SsimBatchC3,
    ssim_batch_c3,
    ssim_batch_c3_buffer_size,
    C3,
    [(u8, ssim_batch_u8_c3, ssim_batch_u8_c3_buffer_size)]
);

impl_generic_image_pair_metric_batch!(
    "Computes weighted multiscale structural similarity (WMSSSIM) for a batch of image pairs, producing one result per pair and processed channel.\n\nUses the common ROI defined by the smallest image width and smallest image height across both batches.",
    WmsssimBatchC1,
    wmsssim_batch_c1,
    wmsssim_batch_c1_buffer_size,
    C1,
    [(u8, wmsssim_batch_u8_c1, wmsssim_batch_u8_c1_buffer_size)]
);

impl_generic_image_pair_metric_batch!(
    "Computes weighted multiscale structural similarity (WMSSSIM) for a batch of image pairs, producing one result per pair and processed channel.\n\nUses the common ROI defined by the smallest image width and smallest image height across both batches.",
    WmsssimBatchC3,
    wmsssim_batch_c3,
    wmsssim_batch_c3_buffer_size,
    C3,
    [(u8, wmsssim_batch_u8_c3, wmsssim_batch_u8_c3_buffer_size)]
);

impl_generic_image_pair_metric_batch!(
    "Computes weighted multiscale structural similarity (WMSSSIM) for a batch of image pairs, producing one result per pair and processed channel.\n\nUses the common ROI defined by the smallest image width and smallest image height across both batches.",
    WmsssimBatchC4,
    wmsssim_batch_c4,
    wmsssim_batch_c4_buffer_size,
    C4,
    [(u8, wmsssim_batch_u8_c4, wmsssim_batch_u8_c4_buffer_size)]
);

impl_generic_image_pair_metric_batch!(
    "Computes mean squared error (MSE) for a batch of image pairs, producing one result per pair and processed channel.\n\nUses each image pair's own ROI; the native maximum ROI takes the largest width and largest height across the batch.",
    MseBatchAdvancedC1,
    mse_batch_advanced_c1,
    mse_batch_advanced_c1_buffer_size,
    C1,
    [(u8, mse_batch_u8_c1_advanced, mse_batch_u8_c1_buffer_size)]
);

impl_generic_image_pair_metric_batch!(
    "Computes mean squared error (MSE) for a batch of image pairs, producing one result per pair and processed channel.\n\nUses each image pair's own ROI; the native maximum ROI takes the largest width and largest height across the batch.",
    MseBatchAdvancedC3,
    mse_batch_advanced_c3,
    mse_batch_advanced_c3_buffer_size,
    C3,
    [(u8, mse_batch_u8_c3_advanced, mse_batch_u8_c3_buffer_size)]
);

impl_generic_image_pair_metric_batch!(
    "Computes peak signal-to-noise ratio (PSNR) for a batch of image pairs, producing one result per pair and processed channel.\n\nUses each image pair's own ROI; the native maximum ROI takes the largest width and largest height across the batch.",
    PsnrBatchAdvancedC1,
    psnr_batch_advanced_c1,
    psnr_batch_advanced_c1_buffer_size,
    C1,
    [(u8, psnr_batch_u8_c1_advanced, psnr_batch_u8_c1_buffer_size)]
);

impl_generic_image_pair_metric_batch!(
    "Computes peak signal-to-noise ratio (PSNR) for a batch of image pairs, producing one result per pair and processed channel.\n\nUses each image pair's own ROI; the native maximum ROI takes the largest width and largest height across the batch.",
    PsnrBatchAdvancedC3,
    psnr_batch_advanced_c3,
    psnr_batch_advanced_c3_buffer_size,
    C3,
    [(u8, psnr_batch_u8_c3_advanced, psnr_batch_u8_c3_buffer_size)]
);

impl_generic_image_pair_metric_batch!(
    "Computes weighted multiscale structural similarity (WMSSSIM) for a batch of image pairs, producing one result per pair and processed channel.\n\nUses each image pair's own ROI; the native maximum ROI takes the largest width and largest height across the batch.",
    WmsssimBatchAdvancedC1,
    wmsssim_batch_advanced_c1,
    wmsssim_batch_advanced_c1_buffer_size,
    C1,
    [(
        u8,
        wmsssim_batch_u8_c1_advanced,
        wmsssim_batch_u8_c1_buffer_size
    )]
);

impl_generic_image_pair_metric_batch!(
    "Computes weighted multiscale structural similarity (WMSSSIM) for a batch of image pairs, producing one result per pair and processed channel.\n\nUses each image pair's own ROI; the native maximum ROI takes the largest width and largest height across the batch.",
    WmsssimBatchAdvancedC3,
    wmsssim_batch_advanced_c3,
    wmsssim_batch_advanced_c3_buffer_size,
    C3,
    [(
        u8,
        wmsssim_batch_u8_c3_advanced,
        wmsssim_batch_u8_c3_buffer_size
    )]
);

impl_generic_image_pair_metric_batch!(
    "Computes weighted multiscale structural similarity (WMSSSIM) for a batch of image pairs, producing one result per pair and processed channel.\n\nUses each image pair's own ROI; the native maximum ROI takes the largest width and largest height across the batch.",
    WmsssimBatchAdvancedC4,
    wmsssim_batch_advanced_c4,
    wmsssim_batch_advanced_c4_buffer_size,
    C4,
    [(
        u8,
        wmsssim_batch_u8_c4_advanced,
        wmsssim_batch_u8_c4_buffer_size
    )]
);
