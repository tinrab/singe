use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C4, ImageView},
};

use super::{
    HistogramEvenChannelsImage, HistogramRangeChannelsImage, ImageHistograms, ImagePipeline,
    histogram_buffers, histogram_range_buffers, image_histograms,
};

impl<'a, T> ImagePipeline<'a, T, C4>
where
    T: Copy,
    Self: HistogramEvenChannelsImage<T, C4, 4>,
{
    pub fn histogram_even_channels_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C4>,
        histograms: [&mut DeviceMemory<i32>; 4],
        levels: [i32; 4],
        lower_levels: [i32; 4],
        upper_levels: [i32; 4],
    ) -> Result<()> {
        <Self as HistogramEvenChannelsImage<T, C4, 4>>::histogram_even_channels(
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
        levels: [i32; 4],
        lower_levels: [i32; 4],
        upper_levels: [i32; 4],
    ) -> Result<ImageHistograms> {
        let [mut h0, mut h1, mut h2, mut h3] = histogram_buffers(levels)?;
        {
            let source = self.view()?;
            <Self as HistogramEvenChannelsImage<T, C4, 4>>::histogram_even_channels(
                self.stream_context,
                &source,
                [&mut h0, &mut h1, &mut h2, &mut h3],
                levels,
                lower_levels,
                upper_levels,
            )?;
        }
        Ok(image_histograms(vec![h0, h1, h2, h3]))
    }
}

impl<'a, T> ImagePipeline<'a, T, C4>
where
    T: Copy,
{
    pub fn histogram_range_channels_into<Level>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C4>,
        histograms: [&mut DeviceMemory<i32>; 4],
        levels: [&[Level]; 4],
    ) -> Result<()>
    where
        Level: Copy,
        Self: HistogramRangeChannelsImage<T, C4, Level, 4>,
    {
        <Self as HistogramRangeChannelsImage<T, C4, Level, 4>>::histogram_range_channels(
            stream_context,
            source,
            histograms,
            levels,
        )
    }

    pub fn histogram_range_channels<Level>(self, levels: [&[Level]; 4]) -> Result<ImageHistograms>
    where
        Level: Copy,
        Self: HistogramRangeChannelsImage<T, C4, Level, 4>,
    {
        let [mut h0, mut h1, mut h2, mut h3] = histogram_range_buffers(levels)?;
        {
            let source = self.view()?;
            <Self as HistogramRangeChannelsImage<T, C4, Level, 4>>::histogram_range_channels(
                self.stream_context,
                &source,
                [&mut h0, &mut h1, &mut h2, &mut h3],
                levels,
            )?;
        }
        Ok(image_histograms(vec![h0, h1, h2, h3]))
    }
}
