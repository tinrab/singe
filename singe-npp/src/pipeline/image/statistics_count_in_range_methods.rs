use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView},
};

use super::{
    super::statistics::{CountInRangeImage, ImageStatistic},
    ImagePipeline,
};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: CountInRangeImage<T, L>,
{
    pub fn count_in_range_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        counts: &mut DeviceMemory<i32>,
        lower_bound: <Self as CountInRangeImage<T, L>>::Bounds,
        upper_bound: <Self as CountInRangeImage<T, L>>::Bounds,
    ) -> Result<()> {
        <Self as CountInRangeImage<T, L>>::count_in_range(
            stream_context,
            source,
            counts,
            lower_bound,
            upper_bound,
        )
    }

    pub fn count_in_range(
        self,
        lower_bound: <Self as CountInRangeImage<T, L>>::Bounds,
        upper_bound: <Self as CountInRangeImage<T, L>>::Bounds,
    ) -> Result<ImageStatistic<i32>> {
        let mut counts =
            DeviceMemory::<i32>::create(<Self as CountInRangeImage<T, L>>::OUTPUT_CHANNELS)?;
        {
            let source = self.view()?;
            <Self as CountInRangeImage<T, L>>::count_in_range(
                self.stream_context,
                &source,
                &mut counts,
                lower_bound,
                upper_bound,
            )?;
        }
        Ok(ImageStatistic::from_values(counts))
    }
}
