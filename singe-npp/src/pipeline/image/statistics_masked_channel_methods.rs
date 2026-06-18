use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C3, ImageView, MaskView},
};

use super::{
    ImagePipeline,
    statistics::{ImageStatistic, MaskedChannelScalarStatisticImage},
};

#[path = "statistics_masked_channel_norm_methods.rs"]
mod norm_methods;

impl<'a, T> ImagePipeline<'a, T, C3>
where
    T: Copy,
    Self: MaskedChannelScalarStatisticImage<T, C3>,
{
    pub fn mean_channel_masked_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C3>,
        mask: &MaskView<'_>,
        channel: usize,
        output: &mut DeviceMemory<f64>,
    ) -> Result<()> {
        <Self as MaskedChannelScalarStatisticImage<T, C3>>::mean_channel_masked(
            stream_context,
            source,
            mask,
            channel,
            output,
        )
    }

    pub fn mean_channel_masked(
        self,
        mask: &MaskView<'_>,
        channel: usize,
    ) -> Result<ImageStatistic<f64>> {
        self.masked_channel_scalar_statistic(
            mask,
            channel,
            <Self as MaskedChannelScalarStatisticImage<T, C3>>::mean_channel_masked,
        )
    }

    fn masked_channel_scalar_statistic(
        self,
        mask: &MaskView<'_>,
        channel: usize,
        statistic: fn(
            &StreamContext,
            &ImageView<'_, T, C3>,
            &MaskView<'_>,
            usize,
            &mut DeviceMemory<f64>,
        ) -> Result<()>,
    ) -> Result<ImageStatistic<f64>> {
        let mut values = DeviceMemory::<f64>::create(1)?;
        {
            let source = self.view()?;
            statistic(self.stream_context, &source, mask, channel, &mut values)?;
        }
        Ok(ImageStatistic::from_values(values))
    }
}
