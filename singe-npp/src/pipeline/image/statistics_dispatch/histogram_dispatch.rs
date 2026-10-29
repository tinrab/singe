use singe_cuda::view::DeviceOutput;

use crate::{context::StreamContext, error::Result, image::view::ImageView};

/// Produces histogram bins without reading their previous values.
///
/// # Safety
///
/// Successful submission must write every requested bin with valid counts,
/// ordered on the supplied stream before subsequent consumers.
pub unsafe trait HistogramEvenImage<T, L> {
    fn histogram_even(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        histogram: &mut (impl DeviceOutput<i32> + ?Sized),
        levels: i32,
        lower_level: i32,
        upper_level: i32,
    ) -> Result<()>;
}

/// Produces histogram bins without reading their previous values.
///
/// # Safety
///
/// Successful submission must write every requested bin with valid counts,
/// ordered on the supplied stream before subsequent consumers.
pub unsafe trait HistogramEvenChannelsImage<T, L, const C: usize> {
    fn histogram_even_channels(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        histograms: [&mut (impl DeviceOutput<i32> + ?Sized); C],
        levels: [i32; C],
        lower_levels: [i32; C],
        upper_levels: [i32; C],
    ) -> Result<()>;
}

/// Produces histogram bins without reading their previous values.
///
/// # Safety
///
/// Successful submission must write every requested bin with valid counts,
/// ordered on the supplied stream before subsequent consumers.
pub unsafe trait HistogramRangeImage<T, L, Level> {
    fn histogram_range(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        histogram: &mut (impl DeviceOutput<i32> + ?Sized),
        levels: &[Level],
    ) -> Result<()>;
}

/// Produces histogram bins without reading their previous values.
///
/// # Safety
///
/// Successful submission must write every requested bin with valid counts,
/// ordered on the supplied stream before subsequent consumers.
pub unsafe trait HistogramRangeChannelsImage<T, L, Level, const C: usize> {
    fn histogram_range_channels(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        histograms: [&mut (impl DeviceOutput<i32> + ?Sized); C],
        levels: [&[Level]; C],
    ) -> Result<()>;
}
