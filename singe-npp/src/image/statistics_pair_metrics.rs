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

macro_rules! impl_image_pair_metric {
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
            source_1: &ImageView<'_, $ty, $layout>,
            source_2: &ImageView<'_, $ty, $layout>,
            output: &mut DeviceMemory<f32>,
        ) -> Result<()> {
            validate_same_size(source_1.size(), source_2.size())?;
            validate_metric_output(output, 1, $channels)?;
            let required_bytes = $buffer_size_name(stream_context, source_1.size())?;
            let scratch = DeviceMemory::<u8>::create(required_bytes)?;

            unsafe {
                try_ffi!(sys::$ffi(
                    source_1.as_ptr().cast(),
                    source_1.step(),
                    source_2.as_ptr().cast(),
                    source_2.step(),
                    source_1.size().into(),
                    output.as_mut_ptr().cast(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_generic_image_pair_metric {
    ($trait_name:ident, $name:ident, $buffer_size_name:ident, $layout:ty, [$(($ty:ty, $direct:ident, $direct_buffer_size:ident)),+ $(,)?]) => {
        pub trait $trait_name<Layout>: DataTypeLike {
            fn buffer_size(stream_context: &StreamContext, roi: Size) -> Result<usize>;

            fn dispatch(
                stream_context: &StreamContext,
                source_1: &ImageView<'_, Self, Layout>,
                source_2: &ImageView<'_, Self, Layout>,
                output: &mut DeviceMemory<f32>,
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
                    source_1: &ImageView<'_, Self, $layout>,
                    source_2: &ImageView<'_, Self, $layout>,
                    output: &mut DeviceMemory<f32>,
                ) -> Result<()> {
                    $direct(stream_context, source_1, source_2, output)
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
            source_1: &ImageView<'_, T, $layout>,
            source_2: &ImageView<'_, T, $layout>,
            output: &mut DeviceMemory<f32>,
        ) -> Result<()> {
            T::dispatch(stream_context, source_1, source_2, output)
        }
    };
}

impl_image_pair_metric!(
    mse_u8_c1_buffer_size,
    mse_u8_c1,
    u8,
    C1,
    1,
    nppiMSEGetBufferHostSize_8u_C1R_Ctx,
    nppiMSE_8u_C1R_Ctx
);
impl_image_pair_metric!(
    mse_u8_c3_buffer_size,
    mse_u8_c3,
    u8,
    C3,
    3,
    nppiMSEGetBufferHostSize_8u_C3R_Ctx,
    nppiMSE_8u_C3R_Ctx
);
impl_image_pair_metric!(
    psnr_u8_c1_buffer_size,
    psnr_u8_c1,
    u8,
    C1,
    1,
    nppiPSNRGetBufferHostSize_8u_C1R_Ctx,
    nppiPSNR_8u_C1R_Ctx
);
impl_image_pair_metric!(
    psnr_u8_c3_buffer_size,
    psnr_u8_c3,
    u8,
    C3,
    3,
    nppiPSNRGetBufferHostSize_8u_C3R_Ctx,
    nppiPSNR_8u_C3R_Ctx
);
impl_image_pair_metric!(
    ssim_u8_c1_buffer_size,
    ssim_u8_c1,
    u8,
    C1,
    1,
    nppiSSIMGetBufferHostSize_8u_C1R_Ctx,
    nppiSSIM_8u_C1R_Ctx
);
impl_image_pair_metric!(
    ssim_u8_c3_buffer_size,
    ssim_u8_c3,
    u8,
    C3,
    3,
    nppiSSIMGetBufferHostSize_8u_C3R_Ctx,
    nppiSSIM_8u_C3R_Ctx
);
impl_image_pair_metric!(
    msssim_u8_c1_buffer_size,
    msssim_u8_c1,
    u8,
    C1,
    1,
    nppiMSSSIMGetBufferHostSize_8u_C1R_Ctx,
    nppiMSSSIM_8u_C1R_Ctx
);
impl_image_pair_metric!(
    wmsssim_u8_c1_buffer_size,
    wmsssim_u8_c1,
    u8,
    C1,
    1,
    nppiWMSSSIMGetBufferHostSize_8u_C1R_Ctx,
    nppiWMSSSIM_8u_C1R_Ctx
);
impl_image_pair_metric!(
    wmsssim_u8_c3_buffer_size,
    wmsssim_u8_c3,
    u8,
    C3,
    3,
    nppiWMSSSIMGetBufferHostSize_8u_C3R_Ctx,
    nppiWMSSSIM_8u_C3R_Ctx
);
impl_image_pair_metric!(
    wmsssim_u8_c4_buffer_size,
    wmsssim_u8_c4,
    u8,
    C4,
    4,
    nppiWMSSSIMGetBufferHostSize_8u_C4R_Ctx,
    nppiWMSSSIM_8u_C4R_Ctx
);
impl_generic_image_pair_metric!(
    MseC1,
    mse_c1,
    mse_c1_buffer_size,
    C1,
    [(u8, mse_u8_c1, mse_u8_c1_buffer_size)]
);
impl_generic_image_pair_metric!(
    MseC3,
    mse_c3,
    mse_c3_buffer_size,
    C3,
    [(u8, mse_u8_c3, mse_u8_c3_buffer_size)]
);
impl_generic_image_pair_metric!(
    PsnrC1,
    psnr_c1,
    psnr_c1_buffer_size,
    C1,
    [(u8, psnr_u8_c1, psnr_u8_c1_buffer_size)]
);
impl_generic_image_pair_metric!(
    PsnrC3,
    psnr_c3,
    psnr_c3_buffer_size,
    C3,
    [(u8, psnr_u8_c3, psnr_u8_c3_buffer_size)]
);
impl_generic_image_pair_metric!(
    SsimC1,
    ssim_c1,
    ssim_c1_buffer_size,
    C1,
    [(u8, ssim_u8_c1, ssim_u8_c1_buffer_size)]
);
impl_generic_image_pair_metric!(
    SsimC3,
    ssim_c3,
    ssim_c3_buffer_size,
    C3,
    [(u8, ssim_u8_c3, ssim_u8_c3_buffer_size)]
);
impl_generic_image_pair_metric!(
    MsssimC1,
    msssim_c1,
    msssim_c1_buffer_size,
    C1,
    [(u8, msssim_u8_c1, msssim_u8_c1_buffer_size)]
);
impl_generic_image_pair_metric!(
    WmsssimC1,
    wmsssim_c1,
    wmsssim_c1_buffer_size,
    C1,
    [(u8, wmsssim_u8_c1, wmsssim_u8_c1_buffer_size)]
);
impl_generic_image_pair_metric!(
    WmsssimC3,
    wmsssim_c3,
    wmsssim_c3_buffer_size,
    C3,
    [(u8, wmsssim_u8_c3, wmsssim_u8_c3_buffer_size)]
);
impl_generic_image_pair_metric!(
    WmsssimC4,
    wmsssim_c4,
    wmsssim_c4_buffer_size,
    C4,
    [(u8, wmsssim_u8_c4, wmsssim_u8_c4_buffer_size)]
);
