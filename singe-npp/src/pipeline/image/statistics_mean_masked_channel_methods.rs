use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, MaskView},
};

use super::{
    super::statistics::{
        ImageMeanStandardDeviation, ImageStatistic, MaskedChannelMeanStandardDeviationImage,
    },
    ImagePipeline,
};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: MaskedChannelMeanStandardDeviationImage<T, L>,
{
    pub fn mean_standard_deviation_channel_masked_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        channel: usize,
        mean: &mut DeviceMemory<f64>,
        standard_deviation: &mut DeviceMemory<f64>,
    ) -> Result<()> {
        <Self as MaskedChannelMeanStandardDeviationImage<
            T,
            L,
        >>::mean_standard_deviation_channel_masked(
            stream_context,
            source,
            mask,
            channel,
            mean,
            standard_deviation,
        )
    }

    pub fn mean_standard_deviation_channel_masked(
        self,
        mask: &MaskView<'_>,
        channel: usize,
    ) -> Result<ImageMeanStandardDeviation<f64>> {
        let mut mean = DeviceMemory::<f64>::create(1)?;
        let mut standard_deviation = DeviceMemory::<f64>::create(1)?;
        {
            let source = self.view()?;
            <Self as MaskedChannelMeanStandardDeviationImage<
                T,
                L,
            >>::mean_standard_deviation_channel_masked(
                self.stream_context,
                &source,
                mask,
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
