use singe_cuda::view::DeviceOutputView;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ImageView, MaskView},
};

/// Dispatches completed masked image reductions into per-channel `f64` results.
///
/// # Safety
///
/// Implementations must preserve native pixel and layout representations and reduction
/// semantics. Report the exact output element count, initialize every output element on
/// success, and complete borrowed native work on every path. Uphold the completed native
/// dispatch contract of [`ImagePipeline`](crate::pipeline::ImagePipeline).
pub unsafe trait MaskedScalarStatisticImage<T, L> {
    /// Exact number of `f64` results required by this dispatch implementation.
    const OUTPUT_CHANNELS: usize;

    /// Computes the arithmetic mean of samples for each processed channel.
    /// Includes only pixels whose corresponding mask sample is nonzero. The mask must have
    /// the same dimensions as the source. The provided masked implementations process C1 images.
    /// Writes `f64` results into `output`, whose length must exactly match the dispatch output count.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful native completion. The native reduction owns its temporary scratch.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, scratch-size queries or allocation, native reduction, or stream synchronization fail.
    /// The mask dimensions must match the source.
    /// The output length must equal the required number of channel results.
    fn mean_masked(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        output: &mut DeviceOutputView<'_, f64>,
    ) -> Result<()>;

    /// Computes the maximum absolute sample value for each processed channel.
    /// Includes only pixels whose corresponding mask sample is nonzero. The mask must have
    /// the same dimensions as the source. The provided masked implementations process C1 images.
    /// Writes `f64` results into `output`, whose length must exactly match the dispatch output count.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful native completion. The native reduction owns its temporary scratch.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, scratch-size queries or allocation, native reduction, or stream synchronization fail.
    /// The mask dimensions must match the source.
    /// The output length must equal the required number of channel results.
    fn norm_inf_masked(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        output: &mut DeviceOutputView<'_, f64>,
    ) -> Result<()>;

    /// Computes the sum of absolute sample values for each processed channel.
    /// Includes only pixels whose corresponding mask sample is nonzero. The mask must have
    /// the same dimensions as the source. The provided masked implementations process C1 images.
    /// Writes `f64` results into `output`, whose length must exactly match the dispatch output count.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful native completion. The native reduction owns its temporary scratch.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, scratch-size queries or allocation, native reduction, or stream synchronization fail.
    /// The mask dimensions must match the source.
    /// The output length must equal the required number of channel results.
    fn norm_l1_masked(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        output: &mut DeviceOutputView<'_, f64>,
    ) -> Result<()>;

    /// Computes the square root of the sum of squared sample values for each processed channel.
    /// Includes only pixels whose corresponding mask sample is nonzero. The mask must have
    /// the same dimensions as the source. The provided masked implementations process C1 images.
    /// Writes `f64` results into `output`, whose length must exactly match the dispatch output count.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful native completion. The native reduction owns its temporary scratch.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, scratch-size queries or allocation, native reduction, or stream synchronization fail.
    /// The mask dimensions must match the source.
    /// The output length must equal the required number of channel results.
    fn norm_l2_masked(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        output: &mut DeviceOutputView<'_, f64>,
    ) -> Result<()>;
}

macro_rules! impl_masked_scalar_statistic_image {
    ($ty:ty, $mean:path, $norm_inf:path, $norm_l1:path, $norm_l2:path) => {
        unsafe impl<'a> MaskedScalarStatisticImage<$ty, C1> for ImagePipeline<'a, $ty, C1> {
            const OUTPUT_CHANNELS: usize = 1;

            fn mean_masked(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, C1>,
                mask: &MaskView<'_>,
                output: &mut DeviceOutputView<'_, f64>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(mask)?;
                stream_context.prepare_region(output)?;
                // SAFETY: the regions were prepared for this stream above, so earlier producers are ordered before this launch.
                unsafe { $mean(stream_context, source, mask, output) }
            }

            fn norm_inf_masked(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, C1>,
                mask: &MaskView<'_>,
                output: &mut DeviceOutputView<'_, f64>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(mask)?;
                stream_context.prepare_region(output)?;
                // SAFETY: the regions were prepared for this stream above, so earlier producers are ordered before this launch.
                unsafe { $norm_inf(stream_context, source, mask, output) }
            }

            fn norm_l1_masked(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, C1>,
                mask: &MaskView<'_>,
                output: &mut DeviceOutputView<'_, f64>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(mask)?;
                stream_context.prepare_region(output)?;
                // SAFETY: the regions were prepared for this stream above, so earlier producers are ordered before this launch.
                unsafe { $norm_l1(stream_context, source, mask, output) }
            }

            fn norm_l2_masked(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, C1>,
                mask: &MaskView<'_>,
                output: &mut DeviceOutputView<'_, f64>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(mask)?;
                stream_context.prepare_region(output)?;
                // SAFETY: the regions were prepared for this stream above, so earlier producers are ordered before this launch.
                unsafe { $norm_l2(stream_context, source, mask, output) }
            }
        }
    };
}
