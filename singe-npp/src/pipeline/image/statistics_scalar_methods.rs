use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView},
};

use super::{
    ImagePipeline,
    statistics::{ImageStatistic, ScalarStatisticImage},
};

#[path = "statistics_masked_scalar_methods.rs"]
mod masked_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: ScalarStatisticImage<T, L>,
{
    pub fn sum_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        output: &mut DeviceMemory<f64>,
    ) -> Result<()> {
        <Self as ScalarStatisticImage<T, L>>::sum(stream_context, source, output)
    }

    pub fn mean_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        output: &mut DeviceMemory<f64>,
    ) -> Result<()> {
        <Self as ScalarStatisticImage<T, L>>::mean(stream_context, source, output)
    }

    pub fn norm_inf_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        output: &mut DeviceMemory<f64>,
    ) -> Result<()> {
        <Self as ScalarStatisticImage<T, L>>::norm_inf(stream_context, source, output)
    }

    pub fn norm_l1_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        output: &mut DeviceMemory<f64>,
    ) -> Result<()> {
        <Self as ScalarStatisticImage<T, L>>::norm_l1(stream_context, source, output)
    }

    pub fn norm_l2_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        output: &mut DeviceMemory<f64>,
    ) -> Result<()> {
        <Self as ScalarStatisticImage<T, L>>::norm_l2(stream_context, source, output)
    }

    pub fn sum(self) -> Result<ImageStatistic<f64>> {
        self.scalar_statistic(<Self as ScalarStatisticImage<T, L>>::sum)
    }

    pub fn mean(self) -> Result<ImageStatistic<f64>> {
        self.scalar_statistic(<Self as ScalarStatisticImage<T, L>>::mean)
    }

    pub fn norm_inf(self) -> Result<ImageStatistic<f64>> {
        self.scalar_statistic(<Self as ScalarStatisticImage<T, L>>::norm_inf)
    }

    pub fn norm_l1(self) -> Result<ImageStatistic<f64>> {
        self.scalar_statistic(<Self as ScalarStatisticImage<T, L>>::norm_l1)
    }

    pub fn norm_l2(self) -> Result<ImageStatistic<f64>> {
        self.scalar_statistic(<Self as ScalarStatisticImage<T, L>>::norm_l2)
    }

    fn scalar_statistic(
        self,
        statistic: fn(&StreamContext, &ImageView<'_, T, L>, &mut DeviceMemory<f64>) -> Result<()>,
    ) -> Result<ImageStatistic<f64>> {
        let mut values =
            DeviceMemory::<f64>::create(<Self as ScalarStatisticImage<T, L>>::OUTPUT_CHANNELS)?;
        {
            let source = self.view()?;
            statistic(self.stream_context, &source, &mut values)?;
        }
        Ok(ImageStatistic::from_values(values))
    }
}
