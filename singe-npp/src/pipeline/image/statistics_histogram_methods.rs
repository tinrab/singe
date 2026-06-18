use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::{Error, Result},
    image::view::{ChannelLayout, ImageView},
};

use super::{
    ImagePipeline,
    statistics::{HistogramEvenImage, HistogramRangeImage, ImageStatistic},
};

#[path = "statistics_count_in_range_methods.rs"]
mod count_in_range_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    pub fn histogram_range_into<Level>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        histogram: &mut DeviceMemory<i32>,
        levels: &[Level],
    ) -> Result<()>
    where
        Level: Copy,
        Self: HistogramRangeImage<T, L, Level>,
    {
        <Self as HistogramRangeImage<T, L, Level>>::histogram_range(
            stream_context,
            source,
            histogram,
            levels,
        )
    }

    pub fn histogram_range<Level>(self, levels: &[Level]) -> Result<ImageStatistic<i32>>
    where
        Level: Copy,
        Self: HistogramRangeImage<T, L, Level>,
    {
        let mut histogram = DeviceMemory::<i32>::create(histogram_range_bins(levels)?)?;
        {
            let source = self.view()?;
            <Self as HistogramRangeImage<T, L, Level>>::histogram_range(
                self.stream_context,
                &source,
                &mut histogram,
                levels,
            )?;
        }
        Ok(ImageStatistic::from_values(histogram))
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: HistogramEvenImage<T, L>,
{
    pub fn histogram_even_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        histogram: &mut DeviceMemory<i32>,
        levels: i32,
        lower_level: i32,
        upper_level: i32,
    ) -> Result<()> {
        <Self as HistogramEvenImage<T, L>>::histogram_even(
            stream_context,
            source,
            histogram,
            levels,
            lower_level,
            upper_level,
        )
    }

    pub fn histogram_even(
        self,
        levels: i32,
        lower_level: i32,
        upper_level: i32,
    ) -> Result<ImageStatistic<i32>> {
        if levels < 2 {
            return Err(Error::OutOfRange {
                name: "levels".into(),
            });
        }
        let bins = usize::try_from(levels - 1).map_err(|_| Error::OutOfRange {
            name: "levels".into(),
        })?;
        let mut histogram = DeviceMemory::<i32>::create(bins)?;
        {
            let source = self.view()?;
            <Self as HistogramEvenImage<T, L>>::histogram_even(
                self.stream_context,
                &source,
                &mut histogram,
                levels,
                lower_level,
                upper_level,
            )?;
        }
        Ok(ImageStatistic::from_values(histogram))
    }
}

fn histogram_range_bins<Level>(levels: &[Level]) -> Result<usize> {
    if levels.len() < 2 {
        return Err(Error::OutOfRange {
            name: "levels".into(),
        });
    }
    Ok(levels.len() - 1)
}
