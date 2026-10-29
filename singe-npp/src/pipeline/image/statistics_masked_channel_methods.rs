use singe_cuda::{memory::DeviceMemory, view::DeviceOutputView};

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C3, ImageView, MaskView},
    pipeline::image::{
        ImagePipeline,
        statistics::{ImageStatistic, MaskedChannelScalarStatisticImage},
    },
};

mod norm_methods;

impl<T> ImagePipeline<'_, T, C3>
where
    T: Copy,
    Self: MaskedChannelScalarStatisticImage<T, C3>,
{
    /// Computes the native mean of one masked C3 channel.
    /// `channel` selects a zero-based component in `0..3`. Only pixels whose mask value
    /// is nonzero participate. Source and mask dimensions must match.
    /// The supplied output view must contain exactly one `f64` element.
    /// Prepares source, mask, and result regions, waits for producers on other streams, rejects
    /// capture, queries and allocates temporary device scratch, and returns after successful
    /// stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, scratch-size queries or allocation, native execution, or stream synchronization fail.
    /// `channel` must be less than three and all source/mask dimensions must match.
    /// The output view must contain exactly one element.
    pub fn mean_channel_masked_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C3>,
        mask: &MaskView<'_>,
        channel: usize,
        output: &mut DeviceOutputView<'_, f64>,
    ) -> Result<()> {
        <Self as MaskedChannelScalarStatisticImage<T, C3>>::mean_channel_masked(
            stream_context,
            source,
            mask,
            channel,
            output,
        )
    }

    /// Computes the native mean of one masked C3 channel.
    /// `channel` selects a zero-based component in `0..3`. Only pixels whose mask value
    /// is nonzero participate. Source and mask dimensions must match.
    /// Allocates owned device storage for one `f64` result.
    /// Prepares source, mask, and result regions, waits for producers on other streams, rejects
    /// capture, queries and allocates temporary device scratch, and returns after successful
    /// stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, scratch-size queries or allocation, native execution, or stream synchronization fail.
    /// `channel` must be less than three and all source/mask dimensions must match.
    /// Result allocation or source view construction can also fail.
    pub fn mean_channel_masked(
        self,
        mask: &MaskView<'_>,
        channel: usize,
    ) -> Result<ImageStatistic<f64>> {
        self.masked_channel_scalar_statistic(
            mask,
            channel,
            <Self as MaskedChannelScalarStatisticImage<T, C3>>::mean_channel_masked,
        )
    }

    fn masked_channel_scalar_statistic(
        self,
        mask: &MaskView<'_>,
        channel: usize,
        statistic: MaskedChannelScalarStatisticCallback<T>,
    ) -> Result<ImageStatistic<f64>> {
        self.ensure_eager()?;
        let mut values = DeviceMemory::<f64>::create_uninit(self.stream_context.cuda_context(), 1)?;
        {
            let source = self.view()?;
            statistic(
                self.stream_context,
                &source,
                mask,
                channel,
                &mut DeviceOutputView::from_output(&mut values),
            )?;
        }
        // SAFETY: the completed operation above wrote every element.
        Ok(ImageStatistic::from_values(unsafe { values.assume_init() }))
    }
}

type MaskedChannelScalarStatisticCallback<T> = fn(
    &StreamContext,
    &ImageView<'_, T, C3>,
    &MaskView<'_>,
    usize,
    &mut DeviceOutputView<'_, f64>,
) -> Result<()>;
