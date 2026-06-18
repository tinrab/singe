use singe_cuda::memory::DeviceMemory;
use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, C3, C4, ImageView},
    try_ffi,
    types::{DataTypeLike, Size},
    utility::to_usize,
};

use super::statistics_validation::*;

macro_rules! impl_image_pair_metric_batch {
    (
        $buffer_size_name:ident,
        $name:ident,
        $ty:ty,
        $layout:ty,
        $channels:expr,
        $buffer_size_ffi:ident,
        $ffi:ident
    ) => {
        pub fn $buffer_size_name(stream_context: &StreamContext, roi: Size) -> Result<usize> {
            let mut bytes = 0;
            unsafe {
                try_ffi!(sys::$buffer_size_ffi(
                    roi.into(),
                    &raw mut bytes,
                    stream_context.as_raw(),
                ))?;
            }
            to_usize(bytes, "buffer size")
        }

        pub fn $name(
            stream_context: &StreamContext,
            source_0: &[ImageView<'_, $ty, $layout>],
            source_1: &[ImageView<'_, $ty, $layout>],
            results: &mut DeviceMemory<f32>,
        ) -> Result<()> {
            let (batch_size, roi, descriptors_0, descriptors_1) =
                pair_metric_batch_descriptors(source_0, source_1)?;
            validate_metric_output(results, batch_size as usize, $channels)?;
            let required_bytes = $buffer_size_name(stream_context, roi)?;
            let (_scratch_buffers, mut scratch_descriptors) =
                pair_metric_batch_buffers(batch_size as usize, required_bytes)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    descriptors_0.as_ptr().cast(),
                    descriptors_1.as_ptr().cast(),
                    batch_size,
                    roi.into(),
                    results.as_mut_ptr().cast(),
                    scratch_descriptors.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_image_pair_metric_batch_advanced {
    (
        $buffer_size_name:ident,
        $name:ident,
        $ty:ty,
        $layout:ty,
        $channels:expr,
        $ffi:ident
    ) => {
        pub fn $name(
            stream_context: &StreamContext,
            source_0: &[ImageView<'_, $ty, $layout>],
            source_1: &[ImageView<'_, $ty, $layout>],
            results: &mut DeviceMemory<f32>,
        ) -> Result<()> {
            let (batch_size, max_roi, descriptors_0, descriptors_1) =
                pair_metric_batch_advanced_descriptors(source_0, source_1)?;
            validate_metric_output(results, batch_size as usize, $channels)?;
            let required_bytes = $buffer_size_name(stream_context, max_roi)?;
            let (_scratch_buffers, mut scratch_descriptors) =
                pair_metric_batch_buffers(batch_size as usize, required_bytes)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    descriptors_0.as_ptr().cast(),
                    descriptors_1.as_ptr().cast(),
                    batch_size,
                    max_roi.into(),
                    results.as_mut_ptr().cast(),
                    scratch_descriptors.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_generic_image_pair_metric_batch {
    ($trait_name:ident, $name:ident, $buffer_size_name:ident, $layout:ty, [$(($ty:ty, $direct:ident, $direct_buffer_size:ident)),+ $(,)?]) => {
        pub trait $trait_name<Layout>: DataTypeLike {
            fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize>;

            fn dispatch(
                stream_context: &StreamContext,
                source_0: &[ImageView<'_, Self, Layout>],
                source_1: &[ImageView<'_, Self, Layout>],
                results: &mut DeviceMemory<f32>,
            ) -> Result<()>
            where
                Self: Sized;
        }

        $(
            impl $trait_name<$layout> for $ty {
                fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize> {
                    $direct_buffer_size(stream_context, roi)
                }

                fn dispatch(
                    stream_context: &StreamContext,
                    source_0: &[ImageView<'_, Self, $layout>],
                    source_1: &[ImageView<'_, Self, $layout>],
                    results: &mut DeviceMemory<f32>,
                ) -> Result<()> {
                    $direct(stream_context, source_0, source_1, results)
                }
            }
        )+

        pub fn $buffer_size_name<T: $trait_name<$layout>>(
            stream_context: &StreamContext,
            roi: Size,
        ) -> Result<usize> {
            T::buffer_size(stream_context, roi)
        }

        pub fn $name<T: $trait_name<$layout>>(
            stream_context: &StreamContext,
            source_0: &[ImageView<'_, T, $layout>],
            source_1: &[ImageView<'_, T, $layout>],
            results: &mut DeviceMemory<f32>,
        ) -> Result<()> {
            T::dispatch(stream_context, source_0, source_1, results)
        }
    };
}

impl_image_pair_metric_batch!(
    mse_batch_u8_c1_buffer_size,
    mse_batch_u8_c1,
    u8,
    C1,
    1,
    nppiMSEBatchGetBufferHostSize_8u_C1R_Ctx,
    nppiMSEBatch_8u_C1R_Ctx
);
impl_image_pair_metric_batch!(
    mse_batch_u8_c3_buffer_size,
    mse_batch_u8_c3,
    u8,
    C3,
    3,
    nppiMSEBatchGetBufferHostSize_8u_C3R_Ctx,
    nppiMSEBatch_8u_C3R_Ctx
);
impl_image_pair_metric_batch!(
    psnr_batch_u8_c1_buffer_size,
    psnr_batch_u8_c1,
    u8,
    C1,
    1,
    nppiPSNRBatchGetBufferHostSize_8u_C1R_Ctx,
    nppiPSNRBatch_8u_C1R_Ctx
);
impl_image_pair_metric_batch!(
    psnr_batch_u8_c3_buffer_size,
    psnr_batch_u8_c3,
    u8,
    C3,
    3,
    nppiPSNRBatchGetBufferHostSize_8u_C3R_Ctx,
    nppiPSNRBatch_8u_C3R_Ctx
);
impl_image_pair_metric_batch!(
    ssim_batch_u8_c1_buffer_size,
    ssim_batch_u8_c1,
    u8,
    C1,
    1,
    nppiSSIMBatchGetBufferHostSize_8u_C1R_Ctx,
    nppiSSIMBatch_8u_C1R_Ctx
);
impl_image_pair_metric_batch!(
    ssim_batch_u8_c3_buffer_size,
    ssim_batch_u8_c3,
    u8,
    C3,
    3,
    nppiSSIMBatchGetBufferHostSize_8u_C3R_Ctx,
    nppiSSIMBatch_8u_C3R_Ctx
);
impl_image_pair_metric_batch!(
    wmsssim_batch_u8_c1_buffer_size,
    wmsssim_batch_u8_c1,
    u8,
    C1,
    1,
    nppiWMSSSIMBatchGetBufferHostSize_8u_C1R_Ctx,
    nppiWMSSSIMBatch_8u_C1R_Ctx
);
impl_image_pair_metric_batch!(
    wmsssim_batch_u8_c3_buffer_size,
    wmsssim_batch_u8_c3,
    u8,
    C3,
    3,
    nppiWMSSSIMBatchGetBufferHostSize_8u_C3R_Ctx,
    nppiWMSSSIMBatch_8u_C3R_Ctx
);
impl_image_pair_metric_batch!(
    wmsssim_batch_u8_c4_buffer_size,
    wmsssim_batch_u8_c4,
    u8,
    C4,
    4,
    nppiWMSSSIMBatchGetBufferHostSize_8u_C4R_Ctx,
    nppiWMSSSIMBatch_8u_C4R_Ctx
);
impl_image_pair_metric_batch_advanced!(
    mse_batch_u8_c1_buffer_size,
    mse_batch_u8_c1_advanced,
    u8,
    C1,
    1,
    nppiMSEBatch_8u_C1R_Advanced_Ctx
);
impl_image_pair_metric_batch_advanced!(
    mse_batch_u8_c3_buffer_size,
    mse_batch_u8_c3_advanced,
    u8,
    C3,
    3,
    nppiMSEBatch_8u_C3R_Advanced_Ctx
);
impl_image_pair_metric_batch_advanced!(
    psnr_batch_u8_c1_buffer_size,
    psnr_batch_u8_c1_advanced,
    u8,
    C1,
    1,
    nppiPSNRBatch_8u_C1R_Advanced_Ctx
);
impl_image_pair_metric_batch_advanced!(
    psnr_batch_u8_c3_buffer_size,
    psnr_batch_u8_c3_advanced,
    u8,
    C3,
    3,
    nppiPSNRBatch_8u_C3R_Advanced_Ctx
);
impl_image_pair_metric_batch_advanced!(
    wmsssim_batch_u8_c1_buffer_size,
    wmsssim_batch_u8_c1_advanced,
    u8,
    C1,
    1,
    nppiWMSSSIMBatch_8u_C1R_Advanced_Ctx
);
impl_image_pair_metric_batch_advanced!(
    wmsssim_batch_u8_c3_buffer_size,
    wmsssim_batch_u8_c3_advanced,
    u8,
    C3,
    3,
    nppiWMSSSIMBatch_8u_C3R_Advanced_Ctx
);
impl_image_pair_metric_batch_advanced!(
    wmsssim_batch_u8_c4_buffer_size,
    wmsssim_batch_u8_c4_advanced,
    u8,
    C4,
    4,
    nppiWMSSSIMBatch_8u_C4R_Advanced_Ctx
);
impl_generic_image_pair_metric_batch!(
    MseBatchC1,
    mse_batch_c1,
    mse_batch_c1_buffer_size,
    C1,
    [(u8, mse_batch_u8_c1, mse_batch_u8_c1_buffer_size)]
);
impl_generic_image_pair_metric_batch!(
    MseBatchC3,
    mse_batch_c3,
    mse_batch_c3_buffer_size,
    C3,
    [(u8, mse_batch_u8_c3, mse_batch_u8_c3_buffer_size)]
);
impl_generic_image_pair_metric_batch!(
    PsnrBatchC1,
    psnr_batch_c1,
    psnr_batch_c1_buffer_size,
    C1,
    [(u8, psnr_batch_u8_c1, psnr_batch_u8_c1_buffer_size)]
);
impl_generic_image_pair_metric_batch!(
    PsnrBatchC3,
    psnr_batch_c3,
    psnr_batch_c3_buffer_size,
    C3,
    [(u8, psnr_batch_u8_c3, psnr_batch_u8_c3_buffer_size)]
);
impl_generic_image_pair_metric_batch!(
    SsimBatchC1,
    ssim_batch_c1,
    ssim_batch_c1_buffer_size,
    C1,
    [(u8, ssim_batch_u8_c1, ssim_batch_u8_c1_buffer_size)]
);
impl_generic_image_pair_metric_batch!(
    SsimBatchC3,
    ssim_batch_c3,
    ssim_batch_c3_buffer_size,
    C3,
    [(u8, ssim_batch_u8_c3, ssim_batch_u8_c3_buffer_size)]
);
impl_generic_image_pair_metric_batch!(
    WmsssimBatchC1,
    wmsssim_batch_c1,
    wmsssim_batch_c1_buffer_size,
    C1,
    [(u8, wmsssim_batch_u8_c1, wmsssim_batch_u8_c1_buffer_size)]
);
impl_generic_image_pair_metric_batch!(
    WmsssimBatchC3,
    wmsssim_batch_c3,
    wmsssim_batch_c3_buffer_size,
    C3,
    [(u8, wmsssim_batch_u8_c3, wmsssim_batch_u8_c3_buffer_size)]
);
impl_generic_image_pair_metric_batch!(
    WmsssimBatchC4,
    wmsssim_batch_c4,
    wmsssim_batch_c4_buffer_size,
    C4,
    [(u8, wmsssim_batch_u8_c4, wmsssim_batch_u8_c4_buffer_size)]
);
impl_generic_image_pair_metric_batch!(
    MseBatchAdvancedC1,
    mse_batch_advanced_c1,
    mse_batch_advanced_c1_buffer_size,
    C1,
    [(u8, mse_batch_u8_c1_advanced, mse_batch_u8_c1_buffer_size)]
);
impl_generic_image_pair_metric_batch!(
    MseBatchAdvancedC3,
    mse_batch_advanced_c3,
    mse_batch_advanced_c3_buffer_size,
    C3,
    [(u8, mse_batch_u8_c3_advanced, mse_batch_u8_c3_buffer_size)]
);
impl_generic_image_pair_metric_batch!(
    PsnrBatchAdvancedC1,
    psnr_batch_advanced_c1,
    psnr_batch_advanced_c1_buffer_size,
    C1,
    [(u8, psnr_batch_u8_c1_advanced, psnr_batch_u8_c1_buffer_size)]
);
impl_generic_image_pair_metric_batch!(
    PsnrBatchAdvancedC3,
    psnr_batch_advanced_c3,
    psnr_batch_advanced_c3_buffer_size,
    C3,
    [(u8, psnr_batch_u8_c3_advanced, psnr_batch_u8_c3_buffer_size)]
);
impl_generic_image_pair_metric_batch!(
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
