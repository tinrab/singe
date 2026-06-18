use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView},
    types::Point,
};

use super::{
    super::{
        ImagePipeline,
        statistics::{
            ChannelPairIndexedExtremumStatisticImage, ImageIndexedMinMax,
            PairIndexedExtremumStatisticImage,
        },
    },
    image_indexed_min_max,
};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: PairIndexedExtremumStatisticImage<T, L>,
{
    pub fn min_max_index_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        min: &mut DeviceMemory<T>,
        max: &mut DeviceMemory<T>,
        min_index: &mut DeviceMemory<Point>,
        max_index: &mut DeviceMemory<Point>,
    ) -> Result<()> {
        <Self as PairIndexedExtremumStatisticImage<T, L>>::min_max_index(
            stream_context,
            source,
            min,
            max,
            min_index,
            max_index,
        )
    }

    pub fn min_max_index(self) -> Result<ImageIndexedMinMax<T>> {
        let mut min = DeviceMemory::<T>::create(1)?;
        let mut max = DeviceMemory::<T>::create(1)?;
        let mut min_index = DeviceMemory::<Point>::create(1)?;
        let mut max_index = DeviceMemory::<Point>::create(1)?;
        {
            let source = self.view()?;
            <Self as PairIndexedExtremumStatisticImage<T, L>>::min_max_index(
                self.stream_context,
                &source,
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
    Self: ChannelPairIndexedExtremumStatisticImage<T, L>,
{
    pub fn min_max_index_channel_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        channel: usize,
        min: &mut DeviceMemory<T>,
        max: &mut DeviceMemory<T>,
        min_index: &mut DeviceMemory<Point>,
        max_index: &mut DeviceMemory<Point>,
    ) -> Result<()> {
        <Self as ChannelPairIndexedExtremumStatisticImage<T, L>>::min_max_index_channel(
            stream_context,
            source,
            channel,
            min,
            max,
            min_index,
            max_index,
        )
    }

    pub fn min_max_index_channel(self, channel: usize) -> Result<ImageIndexedMinMax<T>> {
        let mut min = DeviceMemory::<T>::create(1)?;
        let mut max = DeviceMemory::<T>::create(1)?;
        let mut min_index = DeviceMemory::<Point>::create(1)?;
        let mut max_index = DeviceMemory::<Point>::create(1)?;
        {
            let source = self.view()?;
            <Self as ChannelPairIndexedExtremumStatisticImage<T, L>>::min_max_index_channel(
                self.stream_context,
                &source,
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
