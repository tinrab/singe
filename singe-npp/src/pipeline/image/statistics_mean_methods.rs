use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView},
};

use super::{
    ImagePipeline,
    statistics::{
        ChannelMeanStandardDeviationImage, ImageMeanStandardDeviation, ImageStatistic,
        MeanStandardDeviationImage,
    },
};

#[path = "statistics_mean_masked_channel_methods.rs"]
mod masked_channel_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: MeanStandardDeviationImage<T, L>,
{
    pub fn mean_standard_deviation_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        mean: &mut DeviceMemory<f64>,
        standard_deviation: &mut DeviceMemory<f64>,
    ) -> Result<()> {
        <Self as MeanStandardDeviationImage<T, L>>::mean_standard_deviation(
            stream_context,
            source,
            mean,
            standard_deviation,
        )
    }

    pub fn mean_standard_deviation(self) -> Result<ImageMeanStandardDeviation<f64>> {
        let mut mean = DeviceMemory::<f64>::create(
            <Self as MeanStandardDeviationImage<T, L>>::OUTPUT_CHANNELS,
        )?;
        let mut standard_deviation = DeviceMemory::<f64>::create(
            <Self as MeanStandardDeviationImage<T, L>>::OUTPUT_CHANNELS,
        )?;
        {
            let source = self.view()?;
            <Self as MeanStandardDeviationImage<T, L>>::mean_standard_deviation(
                self.stream_context,
                &source,
                &mut mean,
                &mut standard_deviation,
            )?;
        }
        Ok(ImageMeanStandardDeviation {
            mean: ImageStatistic::from_values(mean),
            standard_deviation: ImageStatistic::from_values(standard_deviation),
        })
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: ChannelMeanStandardDeviationImage<T, L>,
{
    pub fn mean_standard_deviation_channel_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        channel: usize,
        mean: &mut DeviceMemory<f64>,
        standard_deviation: &mut DeviceMemory<f64>,
    ) -> Result<()> {
        <Self as ChannelMeanStandardDeviationImage<T, L>>::mean_standard_deviation_channel(
            stream_context,
            source,
            channel,
            mean,
            standard_deviation,
        )
    }

    pub fn mean_standard_deviation_channel(
        self,
        channel: usize,
    ) -> Result<ImageMeanStandardDeviation<f64>> {
        let mut mean = DeviceMemory::<f64>::create(1)?;
        let mut standard_deviation = DeviceMemory::<f64>::create(1)?;
        {
            let source = self.view()?;
            <Self as ChannelMeanStandardDeviationImage<T, L>>::mean_standard_deviation_channel(
                self.stream_context,
                &source,
                channel,
                &mut mean,
                &mut standard_deviation,
            )?;
        }
        Ok(ImageMeanStandardDeviation {
            mean: ImageStatistic::from_values(mean),
            standard_deviation: ImageStatistic::from_values(standard_deviation),
        })
    }
}
