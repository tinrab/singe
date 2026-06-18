use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView},
};

use super::super::{
    ImagePipeline,
    statistics::{ImageStatistic, QualityMetricImage},
};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: QualityMetricImage<T, L>,
{
    pub fn mean_squared_error_into(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, L>,
        source_2: &ImageView<'_, T, L>,
        output: &mut DeviceMemory<f32>,
    ) -> Result<()> {
        <Self as QualityMetricImage<T, L>>::mean_squared_error(
            stream_context,
            source_1,
            source_2,
            output,
        )
    }

    pub fn peak_signal_to_noise_ratio_into(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, L>,
        source_2: &ImageView<'_, T, L>,
        output: &mut DeviceMemory<f32>,
    ) -> Result<()> {
        <Self as QualityMetricImage<T, L>>::peak_signal_to_noise_ratio(
            stream_context,
            source_1,
            source_2,
            output,
        )
    }

    pub fn mean_squared_error(self, other: &ImageView<'_, T, L>) -> Result<ImageStatistic<f32>> {
        self.quality_metric(
            other,
            <Self as QualityMetricImage<T, L>>::mean_squared_error,
        )
    }

    pub fn peak_signal_to_noise_ratio(
        self,
        other: &ImageView<'_, T, L>,
    ) -> Result<ImageStatistic<f32>> {
        self.quality_metric(
            other,
            <Self as QualityMetricImage<T, L>>::peak_signal_to_noise_ratio,
        )
    }

    fn quality_metric(
        self,
        other: &ImageView<'_, T, L>,
        metric: fn(
            &StreamContext,
            &ImageView<'_, T, L>,
            &ImageView<'_, T, L>,
            &mut DeviceMemory<f32>,
        ) -> Result<()>,
    ) -> Result<ImageStatistic<f32>> {
        let mut output =
            DeviceMemory::<f32>::create(<Self as QualityMetricImage<T, L>>::OUTPUT_CHANNELS)?;
        {
            let source = self.view()?;
            metric(self.stream_context, &source, other, &mut output)?;
        }
        Ok(ImageStatistic::from_values(output))
    }
}
