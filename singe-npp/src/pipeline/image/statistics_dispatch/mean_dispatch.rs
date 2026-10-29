use singe_cuda::view::DeviceOutputView;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ImageView, MaskView},
};

/// # Safety
/// Complete borrowed native work on every path and uphold the declared operation coverage and ABI.
/// On success, initialize every element of each supplied output view.
pub unsafe trait MeanStandardDeviationImage<T, L> {
    const OUTPUT_CHANNELS: usize;

    fn mean_standard_deviation(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        mean: &mut DeviceOutputView<'_, f64>,
        standard_deviation: &mut DeviceOutputView<'_, f64>,
    ) -> Result<()>;
}

/// # Safety
/// Complete borrowed native work on every path and uphold the declared operation coverage and ABI.
/// On success, initialize every element of each supplied output view.
pub unsafe trait MaskedMeanStandardDeviationImage<T, L> {
    const OUTPUT_CHANNELS: usize;

    fn mean_standard_deviation_masked(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        mean: &mut DeviceOutputView<'_, f64>,
        standard_deviation: &mut DeviceOutputView<'_, f64>,
    ) -> Result<()>;
}

/// # Safety
/// Complete borrowed native work on every path and uphold the declared operation coverage and ABI.
/// On success, initialize every element of each supplied output view.
pub unsafe trait ChannelMeanStandardDeviationImage<T, L> {
    fn mean_standard_deviation_channel(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        channel: usize,
        mean: &mut DeviceOutputView<'_, f64>,
        standard_deviation: &mut DeviceOutputView<'_, f64>,
    ) -> Result<()>;
}

/// # Safety
/// Complete borrowed native work on every path and uphold the declared operation coverage and ABI.
/// On success, initialize every element of each supplied output view.
pub unsafe trait MaskedChannelMeanStandardDeviationImage<T, L> {
    fn mean_standard_deviation_channel_masked(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        channel: usize,
        mean: &mut DeviceOutputView<'_, f64>,
        standard_deviation: &mut DeviceOutputView<'_, f64>,
    ) -> Result<()>;
}

macro_rules! impl_mean_standard_deviation_image {
    ($ty:ty, $layout:ty, $mean_standard_deviation:path) => {
        unsafe impl<'a> MeanStandardDeviationImage<$ty, $layout>
            for ImagePipeline<'a, $ty, $layout>
        {
            const OUTPUT_CHANNELS: usize = <$layout as ChannelLayout>::CHANNELS;

            fn mean_standard_deviation(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                mean: &mut DeviceOutputView<'_, f64>,
                standard_deviation: &mut DeviceOutputView<'_, f64>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(mean)?;
                stream_context.prepare_region(standard_deviation)?;
                // SAFETY: the regions were prepared for this stream above, so earlier producers are ordered before this launch.
                unsafe {
                    $mean_standard_deviation(stream_context, source, mean, standard_deviation)
                }
            }
        }
    };
}

macro_rules! impl_masked_mean_standard_deviation_image {
    ($ty:ty, $mean_standard_deviation:path) => {
        unsafe impl<'a> MaskedMeanStandardDeviationImage<$ty, C1> for ImagePipeline<'a, $ty, C1> {
            const OUTPUT_CHANNELS: usize = 1;

            fn mean_standard_deviation_masked(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, C1>,
                mask: &MaskView<'_>,
                mean: &mut DeviceOutputView<'_, f64>,
                standard_deviation: &mut DeviceOutputView<'_, f64>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(mask)?;
                stream_context.prepare_region(mean)?;
                stream_context.prepare_region(standard_deviation)?;
                // SAFETY: the regions were prepared for this stream above, so earlier producers are ordered before this launch.
                unsafe {
                    $mean_standard_deviation(stream_context, source, mask, mean, standard_deviation)
                }
            }
        }
    };
}

macro_rules! impl_channel_mean_standard_deviation_image {
    ($ty:ty, $layout:ty, $mean_standard_deviation:path) => {
        unsafe impl<'a> ChannelMeanStandardDeviationImage<$ty, $layout>
            for ImagePipeline<'a, $ty, $layout>
        {
            fn mean_standard_deviation_channel(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                channel: usize,
                mean: &mut DeviceOutputView<'_, f64>,
                standard_deviation: &mut DeviceOutputView<'_, f64>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(mean)?;
                stream_context.prepare_region(standard_deviation)?;
                // SAFETY: the regions were prepared for this stream above, so earlier producers are ordered before this launch.
                unsafe {
                    $mean_standard_deviation(
                        stream_context,
                        source,
                        channel,
                        mean,
                        standard_deviation,
                    )
                }
            }
        }
    };
}

macro_rules! impl_masked_channel_mean_standard_deviation_image {
    ($ty:ty, $layout:ty, $mean_standard_deviation:path) => {
        unsafe impl<'a> MaskedChannelMeanStandardDeviationImage<$ty, $layout>
            for ImagePipeline<'a, $ty, $layout>
        {
            fn mean_standard_deviation_channel_masked(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                mask: &MaskView<'_>,
                channel: usize,
                mean: &mut DeviceOutputView<'_, f64>,
                standard_deviation: &mut DeviceOutputView<'_, f64>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(mask)?;
                stream_context.prepare_region(mean)?;
                stream_context.prepare_region(standard_deviation)?;
                // SAFETY: the regions were prepared for this stream above, so earlier producers are ordered before this launch.
                unsafe {
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
        }
    };
}
