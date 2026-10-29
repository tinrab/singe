use singe_cuda::{memory::DeviceMemory, view::DeviceOutputView};

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, MaskView},
    pipeline::image::{
        ImagePipeline,
        statistics::{ImageStatistic, MaskedScalarStatisticImage},
    },
};

mod norm_methods;

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: MaskedScalarStatisticImage<T, L>,
{
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
    pub fn mean_masked_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        output: &mut DeviceOutputView<'_, f64>,
    ) -> Result<()> {
        <Self as MaskedScalarStatisticImage<T, L>>::mean_masked(
            stream_context,
            source,
            mask,
            output,
        )
    }

    /// Computes the arithmetic mean of samples for each processed channel.
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
    pub fn mean_masked(self, mask: &MaskView<'_>) -> Result<ImageStatistic<f64>> {
        self.masked_scalar_statistic(
            mask,
            <Self as MaskedScalarStatisticImage<T, L>>::mean_masked,
        )
    }

    pub(crate) fn masked_scalar_statistic(
        self,
        mask: &MaskView<'_>,
        statistic: MaskedScalarStatisticCallback<L, T>,
    ) -> Result<ImageStatistic<f64>> {
        self.ensure_eager()?;
        let mut values = DeviceMemory::<f64>::create_uninit(
            self.stream_context.cuda_context(),
            <Self as MaskedScalarStatisticImage<T, L>>::OUTPUT_CHANNELS,
        )?;
        {
            let source = self.view()?;
            statistic(
                self.stream_context,
                &source,
                mask,
                &mut DeviceOutputView::from_output(&mut values),
            )?;
        }
        // SAFETY: the completed operation above wrote every element.
        Ok(ImageStatistic::from_values(unsafe { values.assume_init() }))
    }
}

type MaskedScalarStatisticCallback<L, T> = fn(
    &StreamContext,
    &ImageView<'_, T, L>,
    &MaskView<'_>,
    &mut DeviceOutputView<'_, f64>,
) -> Result<()>;
