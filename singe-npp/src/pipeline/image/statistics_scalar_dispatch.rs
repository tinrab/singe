use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ImageView, MaskView},
};

#[macro_use]
#[path = "statistics_masked_scalar_dispatch.rs"]
mod masked_scalar_dispatch;
#[macro_use]
#[path = "statistics_masked_channel_scalar_dispatch_macros.rs"]
mod masked_channel_macros;

pub use masked_scalar_dispatch::MaskedScalarStatisticImage;

pub trait ScalarStatisticImage<T, L> {
    const OUTPUT_CHANNELS: usize;

    fn sum(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        output: &mut DeviceMemory<f64>,
    ) -> Result<()>;

    fn mean(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        output: &mut DeviceMemory<f64>,
    ) -> Result<()>;

    fn norm_inf(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        output: &mut DeviceMemory<f64>,
    ) -> Result<()>;

    fn norm_l1(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        output: &mut DeviceMemory<f64>,
    ) -> Result<()>;

    fn norm_l2(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        output: &mut DeviceMemory<f64>,
    ) -> Result<()>;
}

pub trait MaskedChannelScalarStatisticImage<T, L> {
    fn mean_channel_masked(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        channel: usize,
        output: &mut DeviceMemory<f64>,
    ) -> Result<()>;

    fn norm_inf_channel_masked(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        channel: usize,
        output: &mut DeviceMemory<f64>,
    ) -> Result<()>;

    fn norm_l1_channel_masked(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        channel: usize,
        output: &mut DeviceMemory<f64>,
    ) -> Result<()>;

    fn norm_l2_channel_masked(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        channel: usize,
        output: &mut DeviceMemory<f64>,
    ) -> Result<()>;
}

macro_rules! impl_scalar_statistic_image {
    (
        $ty:ty,
        $layout:ty,
        $sum:path,
        $mean:path,
        $norm_inf:path,
        $norm_l1:path,
        $norm_l2:path
    ) => {
        impl<'a> ScalarStatisticImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            const OUTPUT_CHANNELS: usize = <$layout as ChannelLayout>::CHANNELS;

            fn sum(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                output: &mut DeviceMemory<f64>,
            ) -> Result<()> {
                $sum(stream_context, source, output)
            }

            fn mean(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                output: &mut DeviceMemory<f64>,
            ) -> Result<()> {
                $mean(stream_context, source, output)
            }

            fn norm_inf(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                output: &mut DeviceMemory<f64>,
            ) -> Result<()> {
                $norm_inf(stream_context, source, output)
            }

            fn norm_l1(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                output: &mut DeviceMemory<f64>,
            ) -> Result<()> {
                $norm_l1(stream_context, source, output)
            }

            fn norm_l2(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                output: &mut DeviceMemory<f64>,
            ) -> Result<()> {
                $norm_l2(stream_context, source, output)
            }
        }
    };
}
