use singe_cuda::view::DeviceOutputView;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C3, ImageView, MaskView},
    pipeline::image::{
        ImagePipeline,
        statistics::{ImageStatistic, MaskedChannelScalarStatisticImage},
    },
};

impl<T> ImagePipeline<'_, T, C3>
where
    T: Copy,
    Self: MaskedChannelScalarStatisticImage<T, C3>,
{
    /// Computes the native L2 norm for one masked C3 channel.
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
    pub fn norm_l2_channel_masked_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C3>,
        mask: &MaskView<'_>,
        channel: usize,
        output: &mut DeviceOutputView<'_, f64>,
    ) -> Result<()> {
        <Self as MaskedChannelScalarStatisticImage<T, C3>>::norm_l2_channel_masked(
            stream_context,
            source,
            mask,
            channel,
            output,
        )
    }

    /// Computes the native L2 norm for one masked C3 channel.
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
    pub fn norm_l2_channel_masked(
        self,
        mask: &MaskView<'_>,
        channel: usize,
    ) -> Result<ImageStatistic<f64>> {
        self.masked_channel_scalar_norm(
            mask,
            channel,
            <Self as MaskedChannelScalarStatisticImage<T, C3>>::norm_l2_channel_masked,
        )
    }
}
