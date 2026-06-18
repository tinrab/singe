use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, MaskView},
};

use super::super::{
    ImagePipeline,
    statistics::{ImageStatistic, MaskedScalarStatisticImage},
};

#[path = "statistics_masked_scalar_norm_methods.rs"]
mod norm_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: MaskedScalarStatisticImage<T, L>,
{
    pub fn mean_masked_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        output: &mut DeviceMemory<f64>,
    ) -> Result<()> {
        <Self as MaskedScalarStatisticImage<T, L>>::mean_masked(
            stream_context,
            source,
            mask,
            output,
        )
    }

    pub fn mean_masked(self, mask: &MaskView<'_>) -> Result<ImageStatistic<f64>> {
        self.masked_scalar_statistic(
            mask,
            <Self as MaskedScalarStatisticImage<T, L>>::mean_masked,
        )
    }

    pub(super) fn masked_scalar_statistic(
        self,
        mask: &MaskView<'_>,
        statistic: fn(
            &StreamContext,
            &ImageView<'_, T, L>,
            &MaskView<'_>,
            &mut DeviceMemory<f64>,
        ) -> Result<()>,
    ) -> Result<ImageStatistic<f64>> {
        let mut values = DeviceMemory::<f64>::create(
            <Self as MaskedScalarStatisticImage<T, L>>::OUTPUT_CHANNELS,
        )?;
        {
            let source = self.view()?;
            statistic(self.stream_context, &source, mask, &mut values)?;
        }
        Ok(ImageStatistic::from_values(values))
    }
}
