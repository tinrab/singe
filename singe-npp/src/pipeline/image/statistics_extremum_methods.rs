use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView},
};

use super::{
    ImagePipeline,
    statistics::{ExtremumStatisticImage, ImageMinMax, ImageStatistic},
};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: ExtremumStatisticImage<T, L>,
{
    pub fn min_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        output: &mut DeviceMemory<T>,
    ) -> Result<()> {
        <Self as ExtremumStatisticImage<T, L>>::min(stream_context, source, output)
    }

    pub fn max_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        output: &mut DeviceMemory<T>,
    ) -> Result<()> {
        <Self as ExtremumStatisticImage<T, L>>::max(stream_context, source, output)
    }

    pub fn min_max_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        min: &mut DeviceMemory<T>,
        max: &mut DeviceMemory<T>,
    ) -> Result<()> {
        <Self as ExtremumStatisticImage<T, L>>::min_max(stream_context, source, min, max)
    }

    pub fn min(self) -> Result<ImageStatistic<T>> {
        self.extremum_statistic(<Self as ExtremumStatisticImage<T, L>>::min)
    }

    pub fn max(self) -> Result<ImageStatistic<T>> {
        self.extremum_statistic(<Self as ExtremumStatisticImage<T, L>>::max)
    }

    pub fn min_max(self) -> Result<ImageMinMax<T>> {
        let mut min =
            DeviceMemory::<T>::create(<Self as ExtremumStatisticImage<T, L>>::OUTPUT_CHANNELS)?;
        let mut max =
            DeviceMemory::<T>::create(<Self as ExtremumStatisticImage<T, L>>::OUTPUT_CHANNELS)?;
        {
            let source = self.view()?;
            <Self as ExtremumStatisticImage<T, L>>::min_max(
                self.stream_context,
                &source,
                &mut min,
                &mut max,
            )?;
        }
        Ok(ImageMinMax {
            min: ImageStatistic::from_values(min),
            max: ImageStatistic::from_values(max),
        })
    }

    fn extremum_statistic(
        self,
        statistic: fn(&StreamContext, &ImageView<'_, T, L>, &mut DeviceMemory<T>) -> Result<()>,
    ) -> Result<ImageStatistic<T>> {
        let mut values =
            DeviceMemory::<T>::create(<Self as ExtremumStatisticImage<T, L>>::OUTPUT_CHANNELS)?;
        {
            let source = self.view()?;
            statistic(self.stream_context, &source, &mut values)?;
        }
        Ok(ImageStatistic::from_values(values))
    }
}
