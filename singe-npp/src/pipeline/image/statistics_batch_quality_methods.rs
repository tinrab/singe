use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        statistics,
        view::{C1, C3, C4, ChannelLayout, ImageView},
    },
    types::Size,
    utility::{checked_len, to_i32},
};

use super::{
    ImagePipeline,
    statistics::{BatchQualityMetric, ImageStatistic},
};

macro_rules! impl_batch_quality_metric_image {
    ($layout:ty, $channels:literal, $($method:ident => $metric:path),+ $(,)?) => {
        impl<'a> ImagePipeline<'a, u8, $layout> {
            $(
                pub fn $method(
                    stream_context: &StreamContext,
                    source_0: &[ImageView<'_, u8, $layout>],
                    source_1: &[ImageView<'_, u8, $layout>],
                ) -> Result<ImageStatistic<f32>> {
                    batch_quality_metric::<$layout>(
                        stream_context,
                        source_0,
                        source_1,
                        $channels,
                        $metric,
                    )
                }
            )+
        }
    };
}

impl_batch_quality_metric_image!(
    C1,
    1,
    mean_squared_error_batch => statistics::mse_batch_u8_c1,
    mean_squared_error_batch_advanced => statistics::mse_batch_u8_c1_advanced,
    peak_signal_to_noise_ratio_batch => statistics::psnr_batch_u8_c1,
    peak_signal_to_noise_ratio_batch_advanced => statistics::psnr_batch_u8_c1_advanced,
    structural_similarity_batch => statistics::ssim_batch_u8_c1,
    weighted_multi_scale_structural_similarity_batch => statistics::wmsssim_batch_u8_c1,
    weighted_multi_scale_structural_similarity_batch_advanced => statistics::wmsssim_batch_u8_c1_advanced,
);
impl_batch_quality_metric_image!(
    C3,
    3,
    mean_squared_error_batch => statistics::mse_batch_u8_c3,
    mean_squared_error_batch_advanced => statistics::mse_batch_u8_c3_advanced,
    peak_signal_to_noise_ratio_batch => statistics::psnr_batch_u8_c3,
    peak_signal_to_noise_ratio_batch_advanced => statistics::psnr_batch_u8_c3_advanced,
    structural_similarity_batch => statistics::ssim_batch_u8_c3,
    weighted_multi_scale_structural_similarity_batch => statistics::wmsssim_batch_u8_c3,
    weighted_multi_scale_structural_similarity_batch_advanced => statistics::wmsssim_batch_u8_c3_advanced,
);
impl_batch_quality_metric_image!(
    C4,
    4,
    weighted_multi_scale_structural_similarity_batch => statistics::wmsssim_batch_u8_c4,
    weighted_multi_scale_structural_similarity_batch_advanced => statistics::wmsssim_batch_u8_c4_advanced,
);

fn batch_quality_metric<L>(
    stream_context: &StreamContext,
    source_0: &[ImageView<'_, u8, L>],
    source_1: &[ImageView<'_, u8, L>],
    channels: usize,
    metric: BatchQualityMetric<u8, L>,
) -> Result<ImageStatistic<f32>>
where
    L: ChannelLayout,
{
    let len = checked_len(
        Size::new(to_i32(source_0.len(), "source length")?, channels as i32),
        1,
    )?;
    let mut output = DeviceMemory::<f32>::create(len)?;
    metric(stream_context, source_0, source_1, &mut output)?;
    Ok(ImageStatistic::from_values(output))
}
