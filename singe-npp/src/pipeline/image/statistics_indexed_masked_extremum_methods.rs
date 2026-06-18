use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, MaskView},
    types::Point,
};

use super::{
    ImagePipeline,
    statistics::{
        ImageIndexedMinMax, MaskedChannelPairIndexedExtremumStatisticImage,
        MaskedPairIndexedExtremumStatisticImage,
    },
    statistics_indexed_extremum_methods::image_indexed_min_max,
};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: MaskedChannelPairIndexedExtremumStatisticImage<T, L>,
{
    pub fn min_max_index_channel_masked_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        channel: usize,
        min: &mut DeviceMemory<T>,
        max: &mut DeviceMemory<T>,
        min_index: &mut DeviceMemory<Point>,
        max_index: &mut DeviceMemory<Point>,
    ) -> Result<()> {
        <Self as MaskedChannelPairIndexedExtremumStatisticImage<T, L>>::min_max_index_channel_masked(
            stream_context,
            source,
            mask,
            channel,
            min,
            max,
            min_index,
            max_index,
        )
    }

    pub fn min_max_index_channel_masked(
        self,
        mask: &MaskView<'_>,
        channel: usize,
    ) -> Result<ImageIndexedMinMax<T>> {
        let mut min = DeviceMemory::<T>::create(1)?;
        let mut max = DeviceMemory::<T>::create(1)?;
        let mut min_index = DeviceMemory::<Point>::create(1)?;
        let mut max_index = DeviceMemory::<Point>::create(1)?;
        {
            let source = self.view()?;
            <Self as MaskedChannelPairIndexedExtremumStatisticImage<
                T,
                L,
            >>::min_max_index_channel_masked(
                self.stream_context,
                &source,
                mask,
                channel,
                &mut min,
                &mut max,
                &mut min_index,
                &mut max_index,
            )?;
        }
        image_indexed_min_max(min, max, min_index, max_index)
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: MaskedPairIndexedExtremumStatisticImage<T, L>,
{
    pub fn min_max_index_masked_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        min: &mut DeviceMemory<T>,
        max: &mut DeviceMemory<T>,
        min_index: &mut DeviceMemory<Point>,
        max_index: &mut DeviceMemory<Point>,
    ) -> Result<()> {
        <Self as MaskedPairIndexedExtremumStatisticImage<T, L>>::min_max_index_masked(
            stream_context,
            source,
            mask,
            min,
            max,
            min_index,
            max_index,
        )
    }

    pub fn min_max_index_masked(self, mask: &MaskView<'_>) -> Result<ImageIndexedMinMax<T>> {
        let mut min = DeviceMemory::<T>::create(1)?;
        let mut max = DeviceMemory::<T>::create(1)?;
        let mut min_index = DeviceMemory::<Point>::create(1)?;
        let mut max_index = DeviceMemory::<Point>::create(1)?;
        {
            let source = self.view()?;
            <Self as MaskedPairIndexedExtremumStatisticImage<T, L>>::min_max_index_masked(
                self.stream_context,
                &source,
                mask,
                &mut min,
                &mut max,
                &mut min_index,
                &mut max_index,
            )?;
        }
        image_indexed_min_max(min, max, min_index, max_index)
    }
}
