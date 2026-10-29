use singe_cuda::view::DeviceOutputView;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C3, ImageView, MaskView},
    pipeline::image::{ImagePipeline, statistics::MaskedChannelNormDiffImage},
};

impl<T> ImagePipeline<'_, T, C3>
where
    T: Copy,
    Self: MaskedChannelNormDiffImage<T, C3>,
{
    /// Computes the native infinity norm of image differences for one masked C3 channel.
    /// `channel` selects a zero-based component in `0..3`. Only pixels whose mask value
    /// is nonzero participate. Both source images and the mask must have matching dimensions.
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
    pub fn norm_diff_inf_channel_masked_into(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, C3>,
        source_2: &ImageView<'_, T, C3>,
        mask: &MaskView<'_>,
        channel: usize,
        output: &mut DeviceOutputView<'_, f64>,
    ) -> Result<()> {
        <Self as MaskedChannelNormDiffImage<T, C3>>::norm_diff_inf_channel_masked(
            stream_context,
            source_1,
            source_2,
            mask,
            channel,
            output,
        )
    }

    /// Computes the native L1 norm of image differences for one masked C3 channel.
    /// `channel` selects a zero-based component in `0..3`. Only pixels whose mask value
    /// is nonzero participate. Both source images and the mask must have matching dimensions.
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
    pub fn norm_diff_l1_channel_masked_into(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, C3>,
        source_2: &ImageView<'_, T, C3>,
        mask: &MaskView<'_>,
        channel: usize,
        output: &mut DeviceOutputView<'_, f64>,
    ) -> Result<()> {
        <Self as MaskedChannelNormDiffImage<T, C3>>::norm_diff_l1_channel_masked(
            stream_context,
            source_1,
            source_2,
            mask,
            channel,
            output,
        )
    }

    /// Computes the native L2 norm of image differences for one masked C3 channel.
    /// `channel` selects a zero-based component in `0..3`. Only pixels whose mask value
    /// is nonzero participate. Both source images and the mask must have matching dimensions.
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
    pub fn norm_diff_l2_channel_masked_into(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, C3>,
        source_2: &ImageView<'_, T, C3>,
        mask: &MaskView<'_>,
        channel: usize,
        output: &mut DeviceOutputView<'_, f64>,
    ) -> Result<()> {
        <Self as MaskedChannelNormDiffImage<T, C3>>::norm_diff_l2_channel_masked(
            stream_context,
            source_1,
            source_2,
            mask,
            channel,
            output,
        )
    }
}
