use singe_cuda::{memory::DeviceMemory, view::DeviceOutputView};

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C3, ImageView, MaskView},
    pipeline::image::{
        ImagePipeline,
        statistics::{ImageStatistic, MaskedChannelNormDiffImage},
    },
};

mod norm_diff_into_methods;
mod relative_norm_methods;
mod scalar_norm_methods;

impl<T> ImagePipeline<'_, T, C3>
where
    T: Copy,
    Self: MaskedChannelNormDiffImage<T, C3>,
{
    /// Computes the native infinity norm of image differences for one masked C3 channel.
    /// `channel` selects a zero-based component in `0..3`. Only pixels whose mask value
    /// is nonzero participate. Both source images and the mask must have matching dimensions.
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
    pub fn norm_diff_inf_channel_masked(
        self,
        other: &ImageView<'_, T, C3>,
        mask: &MaskView<'_>,
        channel: usize,
    ) -> Result<ImageStatistic<f64>> {
        self.masked_channel_norm_diff(
            other,
            mask,
            channel,
            <Self as MaskedChannelNormDiffImage<T, C3>>::norm_diff_inf_channel_masked,
        )
    }

    /// Computes the native L1 norm of image differences for one masked C3 channel.
    /// `channel` selects a zero-based component in `0..3`. Only pixels whose mask value
    /// is nonzero participate. Both source images and the mask must have matching dimensions.
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
    pub fn norm_diff_l1_channel_masked(
        self,
        other: &ImageView<'_, T, C3>,
        mask: &MaskView<'_>,
        channel: usize,
    ) -> Result<ImageStatistic<f64>> {
        self.masked_channel_norm_diff(
            other,
            mask,
            channel,
            <Self as MaskedChannelNormDiffImage<T, C3>>::norm_diff_l1_channel_masked,
        )
    }

    /// Computes the native L2 norm of image differences for one masked C3 channel.
    /// `channel` selects a zero-based component in `0..3`. Only pixels whose mask value
    /// is nonzero participate. Both source images and the mask must have matching dimensions.
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
    pub fn norm_diff_l2_channel_masked(
        self,
        other: &ImageView<'_, T, C3>,
        mask: &MaskView<'_>,
        channel: usize,
    ) -> Result<ImageStatistic<f64>> {
        self.masked_channel_norm_diff(
            other,
            mask,
            channel,
            <Self as MaskedChannelNormDiffImage<T, C3>>::norm_diff_l2_channel_masked,
        )
    }

    fn masked_channel_norm_diff(
        self,
        other: &ImageView<'_, T, C3>,
        mask: &MaskView<'_>,
        channel: usize,
        metric: MaskedChannelNormDiffCallback<T>,
    ) -> Result<ImageStatistic<f64>> {
        self.ensure_eager()?;
        let mut output = DeviceMemory::<f64>::create_uninit(self.stream_context.cuda_context(), 1)?;
        {
            let source = self.view()?;
            metric(
                self.stream_context,
                &source,
                other,
                mask,
                channel,
                &mut DeviceOutputView::from_output(&mut output),
            )?;
        }
        // SAFETY: the completed operation above wrote every element.
        Ok(ImageStatistic::from_values(unsafe { output.assume_init() }))
    }
}

type MaskedChannelNormDiffCallback<T> = fn(
    &StreamContext,
    &ImageView<'_, T, C3>,
    &ImageView<'_, T, C3>,
    &MaskView<'_>,
    usize,
    &mut DeviceOutputView<'_, f64>,
) -> Result<()>;
