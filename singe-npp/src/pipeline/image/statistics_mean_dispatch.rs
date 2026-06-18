use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ImageView, MaskView},
};

pub trait MeanStandardDeviationImage<T, L> {
    const OUTPUT_CHANNELS: usize;

    fn mean_standard_deviation(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        mean: &mut DeviceMemory<f64>,
        standard_deviation: &mut DeviceMemory<f64>,
    ) -> Result<()>;
}

pub trait MaskedMeanStandardDeviationImage<T, L> {
    const OUTPUT_CHANNELS: usize;

    fn mean_standard_deviation_masked(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        mean: &mut DeviceMemory<f64>,
        standard_deviation: &mut DeviceMemory<f64>,
    ) -> Result<()>;
}

pub trait ChannelMeanStandardDeviationImage<T, L> {
    fn mean_standard_deviation_channel(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        channel: usize,
        mean: &mut DeviceMemory<f64>,
        standard_deviation: &mut DeviceMemory<f64>,
    ) -> Result<()>;
}

pub trait MaskedChannelMeanStandardDeviationImage<T, L> {
    fn mean_standard_deviation_channel_masked(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        channel: usize,
        mean: &mut DeviceMemory<f64>,
        standard_deviation: &mut DeviceMemory<f64>,
    ) -> Result<()>;
}

macro_rules! impl_mean_standard_deviation_image {
    ($ty:ty, $layout:ty, $mean_standard_deviation:path) => {
        impl<'a> MeanStandardDeviationImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            const OUTPUT_CHANNELS: usize = <$layout as ChannelLayout>::CHANNELS;

            fn mean_standard_deviation(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                mean: &mut DeviceMemory<f64>,
                standard_deviation: &mut DeviceMemory<f64>,
            ) -> Result<()> {
                $mean_standard_deviation(stream_context, source, mean, standard_deviation)
            }
        }
    };
}

macro_rules! impl_masked_mean_standard_deviation_image {
    ($ty:ty, $mean_standard_deviation:path) => {
        impl<'a> MaskedMeanStandardDeviationImage<$ty, C1> for ImagePipeline<'a, $ty, C1> {
            const OUTPUT_CHANNELS: usize = 1;

            fn mean_standard_deviation_masked(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, C1>,
                mask: &MaskView<'_>,
                mean: &mut DeviceMemory<f64>,
                standard_deviation: &mut DeviceMemory<f64>,
            ) -> Result<()> {
                $mean_standard_deviation(stream_context, source, mask, mean, standard_deviation)
            }
        }
    };
}

macro_rules! impl_channel_mean_standard_deviation_image {
    ($ty:ty, $layout:ty, $mean_standard_deviation:path) => {
        impl<'a> ChannelMeanStandardDeviationImage<$ty, $layout>
            for ImagePipeline<'a, $ty, $layout>
        {
            fn mean_standard_deviation_channel(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                channel: usize,
                mean: &mut DeviceMemory<f64>,
                standard_deviation: &mut DeviceMemory<f64>,
            ) -> Result<()> {
                $mean_standard_deviation(stream_context, source, channel, mean, standard_deviation)
            }
        }
    };
}

macro_rules! impl_masked_channel_mean_standard_deviation_image {
    ($ty:ty, $layout:ty, $mean_standard_deviation:path) => {
        impl<'a> MaskedChannelMeanStandardDeviationImage<$ty, $layout>
            for ImagePipeline<'a, $ty, $layout>
        {
            fn mean_standard_deviation_channel_masked(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                mask: &MaskView<'_>,
                channel: usize,
                mean: &mut DeviceMemory<f64>,
                standard_deviation: &mut DeviceMemory<f64>,
            ) -> Result<()> {
                $mean_standard_deviation(
                    stream_context,
                    source,
                    mask,
                    channel,
                    mean,
                    standard_deviation,
                )
            }
        }
    };
}
