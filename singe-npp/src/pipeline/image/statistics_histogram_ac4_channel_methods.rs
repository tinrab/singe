use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{AC4, ImageView},
};

use super::{
    ImagePipeline,
    statistics::{HistogramEvenChannelsImage, HistogramRangeChannelsImage, ImageHistograms},
    statistics_histogram_channel_methods::{
        histogram_buffers, histogram_range_buffers, image_histograms,
    },
};

impl<'a, T> ImagePipeline<'a, T, AC4>
where
    T: Copy,
    Self: HistogramEvenChannelsImage<T, AC4, 3>,
{
    pub fn histogram_even_channels_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, AC4>,
        histograms: [&mut DeviceMemory<i32>; 3],
        levels: [i32; 3],
        lower_levels: [i32; 3],
        upper_levels: [i32; 3],
    ) -> Result<()> {
        <Self as HistogramEvenChannelsImage<T, AC4, 3>>::histogram_even_channels(
            stream_context,
            source,
            histograms,
            levels,
            lower_levels,
            upper_levels,
        )
    }

    pub fn histogram_even_channels(
        self,
        levels: [i32; 3],
        lower_levels: [i32; 3],
        upper_levels: [i32; 3],
    ) -> Result<ImageHistograms> {
        let [mut h0, mut h1, mut h2] = histogram_buffers(levels)?;
        {
            let source = self.view()?;
            <Self as HistogramEvenChannelsImage<T, AC4, 3>>::histogram_even_channels(
                self.stream_context,
                &source,
                [&mut h0, &mut h1, &mut h2],
                levels,
                lower_levels,
                upper_levels,
            )?;
        }
        Ok(image_histograms(vec![h0, h1, h2]))
    }
}

impl<'a, T> ImagePipeline<'a, T, AC4>
where
    T: Copy,
{
    pub fn histogram_range_channels_into<Level>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, AC4>,
        histograms: [&mut DeviceMemory<i32>; 3],
        levels: [&[Level]; 3],
    ) -> Result<()>
    where
        Level: Copy,
        Self: HistogramRangeChannelsImage<T, AC4, Level, 3>,
    {
        <Self as HistogramRangeChannelsImage<T, AC4, Level, 3>>::histogram_range_channels(
            stream_context,
            source,
            histograms,
            levels,
        )
    }

    pub fn histogram_range_channels<Level>(self, levels: [&[Level]; 3]) -> Result<ImageHistograms>
    where
        Level: Copy,
        Self: HistogramRangeChannelsImage<T, AC4, Level, 3>,
    {
        let [mut h0, mut h1, mut h2] = histogram_range_buffers(levels)?;
        {
            let source = self.view()?;
            <Self as HistogramRangeChannelsImage<T, AC4, Level, 3>>::histogram_range_channels(
                self.stream_context,
                &source,
                [&mut h0, &mut h1, &mut h2],
                levels,
            )?;
        }
        Ok(image_histograms(vec![h0, h1, h2]))
    }
}
