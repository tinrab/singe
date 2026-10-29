use singe_cuda::view::DeviceOutputView;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, MaskView},
    pipeline::image::{
        ImagePipeline,
        statistics::{ImageStatistic, MaskedScalarStatisticImage},
    },
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: MaskedScalarStatisticImage<T, L>,
{
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
    pub fn norm_inf_masked_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        output: &mut DeviceOutputView<'_, f64>,
    ) -> Result<()> {
        <Self as MaskedScalarStatisticImage<T, L>>::norm_inf_masked(
            stream_context,
            source,
            mask,
            output,
        )
    }

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
    pub fn norm_l1_masked_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        output: &mut DeviceOutputView<'_, f64>,
    ) -> Result<()> {
        <Self as MaskedScalarStatisticImage<T, L>>::norm_l1_masked(
            stream_context,
            source,
            mask,
            output,
        )
    }

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
    pub fn norm_l2_masked_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        output: &mut DeviceOutputView<'_, f64>,
    ) -> Result<()> {
        <Self as MaskedScalarStatisticImage<T, L>>::norm_l2_masked(
            stream_context,
            source,
            mask,
            output,
        )
    }

    /// Computes the maximum absolute sample value for each processed channel.
    /// Includes only pixels whose corresponding mask sample is nonzero. The mask must have
    /// the same dimensions as the source. The provided masked implementations process C1 images.
    /// Allocates owned device result storage and returns it as an `ImageStatistic<f64>`.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful native completion. The native reduction owns its temporary scratch.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, scratch-size queries or allocation, native reduction, or stream synchronization fail.
    /// The mask dimensions must match the source.
    /// Allocating result storage or constructing the source view can also fail.
    pub fn norm_inf_masked(self, mask: &MaskView<'_>) -> Result<ImageStatistic<f64>> {
        self.masked_scalar_statistic(
            mask,
            <Self as MaskedScalarStatisticImage<T, L>>::norm_inf_masked,
        )
    }

    /// Computes the sum of absolute sample values for each processed channel.
    /// Includes only pixels whose corresponding mask sample is nonzero. The mask must have
    /// the same dimensions as the source. The provided masked implementations process C1 images.
    /// Allocates owned device result storage and returns it as an `ImageStatistic<f64>`.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful native completion. The native reduction owns its temporary scratch.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, scratch-size queries or allocation, native reduction, or stream synchronization fail.
    /// The mask dimensions must match the source.
    /// Allocating result storage or constructing the source view can also fail.
    pub fn norm_l1_masked(self, mask: &MaskView<'_>) -> Result<ImageStatistic<f64>> {
        self.masked_scalar_statistic(
            mask,
            <Self as MaskedScalarStatisticImage<T, L>>::norm_l1_masked,
        )
    }

    /// Computes the square root of the sum of squared sample values for each processed channel.
    /// Includes only pixels whose corresponding mask sample is nonzero. The mask must have
    /// the same dimensions as the source. The provided masked implementations process C1 images.
    /// Allocates owned device result storage and returns it as an `ImageStatistic<f64>`.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful native completion. The native reduction owns its temporary scratch.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, scratch-size queries or allocation, native reduction, or stream synchronization fail.
    /// The mask dimensions must match the source.
    /// Allocating result storage or constructing the source view can also fail.
    pub fn norm_l2_masked(self, mask: &MaskView<'_>) -> Result<ImageStatistic<f64>> {
        self.masked_scalar_statistic(
            mask,
            <Self as MaskedScalarStatisticImage<T, L>>::norm_l2_masked,
        )
    }
}
