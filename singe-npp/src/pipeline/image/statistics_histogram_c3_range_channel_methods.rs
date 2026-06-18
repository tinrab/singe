use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C3, ImageView},
};

use super::{
    HistogramRangeChannelsImage, ImageHistograms, ImagePipeline, histogram_range_buffers,
    image_histograms,
};

impl<'a, T> ImagePipeline<'a, T, C3>
where
    T: Copy,
{
    pub fn histogram_range_channels_into<Level>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C3>,
        histograms: [&mut DeviceMemory<i32>; 3],
        levels: [&[Level]; 3],
    ) -> Result<()>
    where
        Level: Copy,
        Self: HistogramRangeChannelsImage<T, C3, Level, 3>,
    {
        <Self as HistogramRangeChannelsImage<T, C3, Level, 3>>::histogram_range_channels(
            stream_context,
            source,
            histograms,
            levels,
        )
    }

    pub fn histogram_range_channels<Level>(self, levels: [&[Level]; 3]) -> Result<ImageHistograms>
    where
        Level: Copy,
        Self: HistogramRangeChannelsImage<T, C3, Level, 3>,
    {
        let [mut h0, mut h1, mut h2] = histogram_range_buffers(levels)?;
        {
            let source = self.view()?;
            <Self as HistogramRangeChannelsImage<T, C3, Level, 3>>::histogram_range_channels(
                self.stream_context,
                &source,
                [&mut h0, &mut h1, &mut h2],
                levels,
            )?;
        }
        Ok(image_histograms(vec![h0, h1, h2]))
    }
}
