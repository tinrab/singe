use singe_cuda::view::DeviceOutputView;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ImageView, MaskView},
};

#[macro_use]
mod masked_scalar_dispatch;
#[macro_use]
mod masked_channel_macros;

pub use masked_scalar_dispatch::MaskedScalarStatisticImage;

/// Dispatches completed image reductions into per-channel `f64` results.
///
/// # Safety
///
/// Implementations must preserve native pixel and layout representations and reduction
/// semantics. Report the exact output element count, initialize every output element on
/// success, and complete borrowed native work on every path. Uphold the completed native
/// dispatch contract of [`ImagePipeline`](crate::pipeline::ImagePipeline).
pub unsafe trait ScalarStatisticImage<T, L> {
    /// Exact number of `f64` results required by this dispatch implementation.
    const OUTPUT_CHANNELS: usize;

    /// Computes the sum of samples for each processed channel.
    /// The provided C1, C3, and C4 implementations return one result per channel in channel order.
    /// Writes `f64` results into `output`, whose length must exactly match the dispatch output count.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful native completion. The native reduction owns its temporary scratch.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, scratch-size queries or allocation, native reduction, or stream synchronization fail.
    /// The output length must equal the required number of channel results.
    fn sum(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        output: &mut DeviceOutputView<'_, f64>,
    ) -> Result<()>;

    /// Computes the arithmetic mean of samples for each processed channel.
    /// The provided C1, C3, and C4 implementations return one result per channel in channel order.
    /// Writes `f64` results into `output`, whose length must exactly match the dispatch output count.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful native completion. The native reduction owns its temporary scratch.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, scratch-size queries or allocation, native reduction, or stream synchronization fail.
    /// The output length must equal the required number of channel results.
    fn mean(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        output: &mut DeviceOutputView<'_, f64>,
    ) -> Result<()>;

    /// Computes the maximum absolute sample value for each processed channel.
    /// The provided C1, C3, and C4 implementations return one result per channel in channel order.
    /// Writes `f64` results into `output`, whose length must exactly match the dispatch output count.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful native completion. The native reduction owns its temporary scratch.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, scratch-size queries or allocation, native reduction, or stream synchronization fail.
    /// The output length must equal the required number of channel results.
    fn norm_inf(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        output: &mut DeviceOutputView<'_, f64>,
    ) -> Result<()>;

    /// Computes the sum of absolute sample values for each processed channel.
    /// The provided C1, C3, and C4 implementations return one result per channel in channel order.
    /// Writes `f64` results into `output`, whose length must exactly match the dispatch output count.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful native completion. The native reduction owns its temporary scratch.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, scratch-size queries or allocation, native reduction, or stream synchronization fail.
    /// The output length must equal the required number of channel results.
    fn norm_l1(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        output: &mut DeviceOutputView<'_, f64>,
    ) -> Result<()>;

    /// Computes the square root of the sum of squared sample values for each processed channel.
    /// The provided C1, C3, and C4 implementations return one result per channel in channel order.
    /// Writes `f64` results into `output`, whose length must exactly match the dispatch output count.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful native completion. The native reduction owns its temporary scratch.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, scratch-size queries or allocation, native reduction, or stream synchronization fail.
    /// The output length must equal the required number of channel results.
    fn norm_l2(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        output: &mut DeviceOutputView<'_, f64>,
    ) -> Result<()>;
}

/// # Safety
/// Complete borrowed native work on every path and uphold the declared operation coverage and ABI.
/// On success, initialize every element of each supplied output view.
pub unsafe trait MaskedChannelScalarStatisticImage<T, L> {
    fn mean_channel_masked(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        channel: usize,
        output: &mut DeviceOutputView<'_, f64>,
    ) -> Result<()>;

    fn norm_inf_channel_masked(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        channel: usize,
        output: &mut DeviceOutputView<'_, f64>,
    ) -> Result<()>;

    fn norm_l1_channel_masked(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        channel: usize,
        output: &mut DeviceOutputView<'_, f64>,
    ) -> Result<()>;

    fn norm_l2_channel_masked(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        channel: usize,
        output: &mut DeviceOutputView<'_, f64>,
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
        unsafe impl<'a> ScalarStatisticImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            const OUTPUT_CHANNELS: usize = <$layout as ChannelLayout>::CHANNELS;

            fn sum(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                output: &mut DeviceOutputView<'_, f64>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(output)?;
                // SAFETY: the regions were prepared for this stream above, so earlier producers are ordered before this launch.
                unsafe { $sum(stream_context, source, output) }
            }

            fn mean(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                output: &mut DeviceOutputView<'_, f64>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(output)?;
                // SAFETY: the regions were prepared for this stream above, so earlier producers are ordered before this launch.
                unsafe { $mean(stream_context, source, output) }
            }

            fn norm_inf(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                output: &mut DeviceOutputView<'_, f64>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(output)?;
                // SAFETY: the regions were prepared for this stream above, so earlier producers are ordered before this launch.
                unsafe { $norm_inf(stream_context, source, output) }
            }

            fn norm_l1(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                output: &mut DeviceOutputView<'_, f64>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(output)?;
                // SAFETY: the regions were prepared for this stream above, so earlier producers are ordered before this launch.
                unsafe { $norm_l1(stream_context, source, output) }
            }

            fn norm_l2(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                output: &mut DeviceOutputView<'_, f64>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(output)?;
                // SAFETY: the regions were prepared for this stream above, so earlier producers are ordered before this launch.
                unsafe { $norm_l2(stream_context, source, output) }
            }
        }
    };
}
