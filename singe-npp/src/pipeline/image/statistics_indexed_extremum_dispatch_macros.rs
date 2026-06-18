macro_rules! impl_indexed_extremum_statistic_image {
    ($ty:ty, $layout:ty, $min_index:path, $max_index:path) => {
        impl<'a> IndexedExtremumStatisticImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            const OUTPUT_CHANNELS: usize = <$layout as ChannelLayout>::CHANNELS;

            fn min_index(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                output: &mut DeviceMemory<$ty>,
                index_x: &mut DeviceMemory<i32>,
                index_y: &mut DeviceMemory<i32>,
            ) -> Result<()> {
                $min_index(stream_context, source, output, index_x, index_y)
            }

            fn max_index(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                output: &mut DeviceMemory<$ty>,
                index_x: &mut DeviceMemory<i32>,
                index_y: &mut DeviceMemory<i32>,
            ) -> Result<()> {
                $max_index(stream_context, source, output, index_x, index_y)
            }
        }
    };
}

macro_rules! impl_pair_indexed_extremum_statistic_image {
    ($ty:ty, $layout:ty, $min_max_index:path) => {
        impl<'a> PairIndexedExtremumStatisticImage<$ty, $layout>
            for ImagePipeline<'a, $ty, $layout>
        {
            fn min_max_index(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                min: &mut DeviceMemory<$ty>,
                max: &mut DeviceMemory<$ty>,
                min_index: &mut DeviceMemory<Point>,
                max_index: &mut DeviceMemory<Point>,
            ) -> Result<()> {
                $min_max_index(stream_context, source, min, max, min_index, max_index)
            }
        }
    };
}

macro_rules! impl_channel_pair_indexed_extremum_statistic_image {
    ($ty:ty, $layout:ty, $min_max_index:path) => {
        impl<'a> ChannelPairIndexedExtremumStatisticImage<$ty, $layout>
            for ImagePipeline<'a, $ty, $layout>
        {
            fn min_max_index_channel(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                channel: usize,
                min: &mut DeviceMemory<$ty>,
                max: &mut DeviceMemory<$ty>,
                min_index: &mut DeviceMemory<Point>,
                max_index: &mut DeviceMemory<Point>,
            ) -> Result<()> {
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
    };
}

macro_rules! impl_masked_pair_indexed_extremum_statistic_image {
    ($ty:ty, $min_max_index:path) => {
        impl<'a> MaskedPairIndexedExtremumStatisticImage<$ty, C1> for ImagePipeline<'a, $ty, C1> {
            fn min_max_index_masked(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, C1>,
                mask: &MaskView<'_>,
                min: &mut DeviceMemory<$ty>,
                max: &mut DeviceMemory<$ty>,
                min_index: &mut DeviceMemory<Point>,
                max_index: &mut DeviceMemory<Point>,
            ) -> Result<()> {
                $min_max_index(stream_context, source, mask, min, max, min_index, max_index)
            }
        }
    };
}

macro_rules! impl_masked_channel_pair_indexed_extremum_statistic_image {
    ($ty:ty, $min_max_index:path) => {
        impl<'a> MaskedChannelPairIndexedExtremumStatisticImage<$ty, C3>
            for ImagePipeline<'a, $ty, C3>
        {
            fn min_max_index_channel_masked(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, C3>,
                mask: &MaskView<'_>,
                channel: usize,
                min: &mut DeviceMemory<$ty>,
                max: &mut DeviceMemory<$ty>,
                min_index: &mut DeviceMemory<Point>,
                max_index: &mut DeviceMemory<Point>,
            ) -> Result<()> {
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
    };
}
