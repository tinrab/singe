macro_rules! impl_indexed_extremum_statistic_image {
    ($ty:ty, $layout:ty, $min_index:path, $max_index:path) => {
        unsafe impl<'a> IndexedExtremumStatisticImage<$ty, $layout>
            for ImagePipeline<'a, $ty, $layout>
        {
            const OUTPUT_CHANNELS: usize = <$layout as ChannelLayout>::CHANNELS;

            fn min_index(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                output: &mut DeviceOutputView<'_, $ty>,
                index_x: &mut DeviceOutputView<'_, i32>,
                index_y: &mut DeviceOutputView<'_, i32>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(output)?;
                stream_context.prepare_region(index_x)?;
                stream_context.prepare_region(index_y)?;
                // SAFETY: the regions were prepared for this stream above, so earlier producers are ordered before this launch.
                unsafe { $min_index(stream_context, source, output, index_x, index_y) }
            }

            fn max_index(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                output: &mut DeviceOutputView<'_, $ty>,
                index_x: &mut DeviceOutputView<'_, i32>,
                index_y: &mut DeviceOutputView<'_, i32>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(output)?;
                stream_context.prepare_region(index_x)?;
                stream_context.prepare_region(index_y)?;
                // SAFETY: the regions were prepared for this stream above, so earlier producers are ordered before this launch.
                unsafe { $max_index(stream_context, source, output, index_x, index_y) }
            }
        }
    };
}

macro_rules! impl_pair_indexed_extremum_statistic_image {
    ($ty:ty, $layout:ty, $min_max_index:path) => {
        unsafe impl<'a> PairIndexedExtremumStatisticImage<$ty, $layout>
            for ImagePipeline<'a, $ty, $layout>
        {
            fn min_max_index(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                min: &mut DeviceOutputView<'_, $ty>,
                max: &mut DeviceOutputView<'_, $ty>,
                min_index: &mut DeviceOutputView<'_, Point>,
                max_index: &mut DeviceOutputView<'_, Point>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(min)?;
                stream_context.prepare_region(max)?;
                stream_context.prepare_region(min_index)?;
                stream_context.prepare_region(max_index)?;
                // SAFETY: the regions were prepared for this stream above, so earlier producers are ordered before this launch.
                unsafe { $min_max_index(stream_context, source, min, max, min_index, max_index) }
            }
        }
    };
}

macro_rules! impl_channel_pair_indexed_extremum_statistic_image {
    ($ty:ty, $layout:ty, $min_max_index:path) => {
        unsafe impl<'a> ChannelPairIndexedExtremumStatisticImage<$ty, $layout>
            for ImagePipeline<'a, $ty, $layout>
        {
            fn min_max_index_channel(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                channel: usize,
                min: &mut DeviceOutputView<'_, $ty>,
                max: &mut DeviceOutputView<'_, $ty>,
                min_index: &mut DeviceOutputView<'_, Point>,
                max_index: &mut DeviceOutputView<'_, Point>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(min)?;
                stream_context.prepare_region(max)?;
                stream_context.prepare_region(min_index)?;
                stream_context.prepare_region(max_index)?;
                // SAFETY: the regions were prepared for this stream above, so earlier producers are ordered before this launch.
                unsafe {
                    $min_max_index(
                        stream_context,
                        source,
                        channel,
                        min,
                        max,
                        min_index,
                        max_index,
                    )
                }
            }
        }
    };
}

macro_rules! impl_masked_pair_indexed_extremum_statistic_image {
    ($ty:ty, $min_max_index:path) => {
        unsafe impl<'a> MaskedPairIndexedExtremumStatisticImage<$ty, C1>
            for ImagePipeline<'a, $ty, C1>
        {
            fn min_max_index_masked(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, C1>,
                mask: &MaskView<'_>,
                min: &mut DeviceOutputView<'_, $ty>,
                max: &mut DeviceOutputView<'_, $ty>,
                min_index: &mut DeviceOutputView<'_, Point>,
                max_index: &mut DeviceOutputView<'_, Point>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(mask)?;
                stream_context.prepare_region(min)?;
                stream_context.prepare_region(max)?;
                stream_context.prepare_region(min_index)?;
                stream_context.prepare_region(max_index)?;
                // SAFETY: the regions were prepared for this stream above, so earlier producers are ordered before this launch.
                unsafe {
                    $min_max_index(stream_context, source, mask, min, max, min_index, max_index)
                }
            }
        }
    };
}

macro_rules! impl_masked_channel_pair_indexed_extremum_statistic_image {
    ($ty:ty, $min_max_index:path) => {
        unsafe impl<'a> MaskedChannelPairIndexedExtremumStatisticImage<$ty, C3>
            for ImagePipeline<'a, $ty, C3>
        {
            fn min_max_index_channel_masked(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, C3>,
                mask: &MaskView<'_>,
                channel: usize,
                min: &mut DeviceOutputView<'_, $ty>,
                max: &mut DeviceOutputView<'_, $ty>,
                min_index: &mut DeviceOutputView<'_, Point>,
                max_index: &mut DeviceOutputView<'_, Point>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(mask)?;
                stream_context.prepare_region(min)?;
                stream_context.prepare_region(max)?;
                stream_context.prepare_region(min_index)?;
                stream_context.prepare_region(max_index)?;
                // SAFETY: the regions were prepared for this stream above, so earlier producers are ordered before this launch.
                unsafe {
                    $min_max_index(
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
            }
        }
    };
}
