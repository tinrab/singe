use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C3, ImageView},
};

use super::{
    ImagePipeline,
    statistics::{
        HistogramEvenChannelsImage, HistogramRangeChannelsImage, ImageHistograms, ImageStatistic,
    },
};

#[path = "statistics_histogram_channel_helpers.rs"]
mod helpers;
pub(super) use helpers::{histogram_buffers, histogram_range_buffers, image_histograms};

#[path = "statistics_histogram_c3_range_channel_methods.rs"]
mod c3_range_channel_methods;
#[path = "statistics_histogram_c4_channel_methods.rs"]
mod c4_channel_methods;

impl<'a, T> ImagePipeline<'a, T, C3>
where
    T: Copy,
    Self: HistogramEvenChannelsImage<T, C3, 3>,
{
    pub fn histogram_even_channels_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C3>,
        histograms: [&mut DeviceMemory<i32>; 3],
        levels: [i32; 3],
        lower_levels: [i32; 3],
        upper_levels: [i32; 3],
    ) -> Result<()> {
        <Self as HistogramEvenChannelsImage<T, C3, 3>>::histogram_even_channels(
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
            <Self as HistogramEvenChannelsImage<T, C3, 3>>::histogram_even_channels(
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
