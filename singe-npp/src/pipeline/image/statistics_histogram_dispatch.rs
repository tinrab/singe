use singe_cuda::memory::DeviceMemory;

use crate::{context::StreamContext, error::Result, image::view::ImageView};

pub trait HistogramEvenImage<T, L> {
    fn histogram_even(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        histogram: &mut DeviceMemory<i32>,
        levels: i32,
        lower_level: i32,
        upper_level: i32,
    ) -> Result<()>;
}

pub trait HistogramEvenChannelsImage<T, L, const C: usize> {
    fn histogram_even_channels(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        histograms: [&mut DeviceMemory<i32>; C],
        levels: [i32; C],
        lower_levels: [i32; C],
        upper_levels: [i32; C],
    ) -> Result<()>;
}

pub trait HistogramRangeImage<T, L, Level> {
    fn histogram_range(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        histogram: &mut DeviceMemory<i32>,
        levels: &[Level],
    ) -> Result<()>;
}

pub trait HistogramRangeChannelsImage<T, L, Level, const C: usize> {
    fn histogram_range_channels(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        histograms: [&mut DeviceMemory<i32>; C],
        levels: [&[Level]; C],
    ) -> Result<()>;
}
