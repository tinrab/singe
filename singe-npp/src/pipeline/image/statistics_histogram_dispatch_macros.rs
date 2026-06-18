macro_rules! impl_histogram_even_image {
    ($ty:ty, $layout:ty, $histogram_even:path) => {
        impl<'a> HistogramEvenImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn histogram_even(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                histogram: &mut DeviceMemory<i32>,
                levels: i32,
                lower_level: i32,
                upper_level: i32,
            ) -> Result<()> {
                $histogram_even(
                    stream_context,
                    source,
                    histogram,
                    levels,
                    lower_level,
                    upper_level,
                )
            }
        }
    };
}

macro_rules! impl_histogram_even_channels_image {
    ($ty:ty, $layout:ty, $channels:literal, $histogram_even:path) => {
        impl<'a> HistogramEvenChannelsImage<$ty, $layout, $channels>
            for ImagePipeline<'a, $ty, $layout>
        {
            fn histogram_even_channels(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                histograms: [&mut DeviceMemory<i32>; $channels],
                levels: [i32; $channels],
                lower_levels: [i32; $channels],
                upper_levels: [i32; $channels],
            ) -> Result<()> {
                $histogram_even(
                    stream_context,
                    source,
                    histograms,
                    levels,
                    lower_levels,
                    upper_levels,
                )
            }
        }
    };
}

macro_rules! impl_histogram_range_image {
    ($ty:ty, $level_ty:ty, $layout:ty, $histogram_range:path) => {
        impl<'a> HistogramRangeImage<$ty, $layout, $level_ty> for ImagePipeline<'a, $ty, $layout> {
            fn histogram_range(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                histogram: &mut DeviceMemory<i32>,
                levels: &[$level_ty],
            ) -> Result<()> {
                $histogram_range(stream_context, source, histogram, levels)
            }
        }
    };
}

macro_rules! impl_histogram_range_channels_image {
    ($ty:ty, $level_ty:ty, $layout:ty, $channels:literal, $histogram_range:path) => {
        impl<'a> HistogramRangeChannelsImage<$ty, $layout, $level_ty, $channels>
            for ImagePipeline<'a, $ty, $layout>
        {
            fn histogram_range_channels(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                histograms: [&mut DeviceMemory<i32>; $channels],
                levels: [&[$level_ty]; $channels],
            ) -> Result<()> {
                $histogram_range(stream_context, source, histograms, levels)
            }
        }
    };
}
