use singe_cuda::{memory::DeviceMemory, view::DeviceOutputView};

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, MaskView},
    pipeline::image::{
        ImagePipeline,
        statistics::{ImageStatistic, MaskedNormDiffImage},
    },
};

mod l2_methods;

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: MaskedNormDiffImage<T, L>,
{
    /// Computes the native infinity norm of image differences over masked pixels.
    /// The provided implementation processes C1 and produces one `f64` result. Only
    /// pixels whose mask value is nonzero participate.
    /// The supplied output view must contain exactly one element.
    /// Prepares both source images, the mask, and output storage, waits for producers on other
    /// streams, rejects capture, queries and allocates temporary device scratch, and returns
    /// after successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, scratch-size queries or allocation, native execution, or stream synchronization fail.
    /// Source image dimensions and mask dimensions must match.
    /// The output view must have the exact result count described above.
    pub fn norm_diff_inf_masked_into(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, L>,
        source_2: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        output: &mut DeviceOutputView<'_, f64>,
    ) -> Result<()> {
        <Self as MaskedNormDiffImage<T, L>>::norm_diff_inf_masked(
            stream_context,
            source_1,
            source_2,
            mask,
            output,
        )
    }

    /// Computes the native L1 norm of image differences over masked pixels.
    /// The provided implementation processes C1 and produces one `f64` result. Only
    /// pixels whose mask value is nonzero participate.
    /// The supplied output view must contain exactly one element.
    /// Prepares both source images, the mask, and output storage, waits for producers on other
    /// streams, rejects capture, queries and allocates temporary device scratch, and returns
    /// after successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, scratch-size queries or allocation, native execution, or stream synchronization fail.
    /// Source image dimensions and mask dimensions must match.
    /// The output view must have the exact result count described above.
    pub fn norm_diff_l1_masked_into(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, L>,
        source_2: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        output: &mut DeviceOutputView<'_, f64>,
    ) -> Result<()> {
        <Self as MaskedNormDiffImage<T, L>>::norm_diff_l1_masked(
            stream_context,
            source_1,
            source_2,
            mask,
            output,
        )
    }

    /// Computes the native infinity norm of image differences over masked pixels.
    /// The provided implementation processes C1 and produces one `f64` result. Only
    /// pixels whose mask value is nonzero participate.
    /// Allocates owned device storage for the single result.
    /// Prepares both source images, the mask, and output storage, waits for producers on other
    /// streams, rejects capture, queries and allocates temporary device scratch, and returns
    /// after successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, scratch-size queries or allocation, native execution, or stream synchronization fail.
    /// Source image dimensions and mask dimensions must match.
    /// Result allocation or source view construction can also fail.
    pub fn norm_diff_inf_masked(
        self,
        other: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
    ) -> Result<ImageStatistic<f64>> {
        self.masked_norm_diff(
            other,
            mask,
            <Self as MaskedNormDiffImage<T, L>>::norm_diff_inf_masked,
        )
    }

    /// Computes the native L1 norm of image differences over masked pixels.
    /// The provided implementation processes C1 and produces one `f64` result. Only
    /// pixels whose mask value is nonzero participate.
    /// Allocates owned device storage for the single result.
    /// Prepares both source images, the mask, and output storage, waits for producers on other
    /// streams, rejects capture, queries and allocates temporary device scratch, and returns
    /// after successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, scratch-size queries or allocation, native execution, or stream synchronization fail.
    /// Source image dimensions and mask dimensions must match.
    /// Result allocation or source view construction can also fail.
    pub fn norm_diff_l1_masked(
        self,
        other: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
    ) -> Result<ImageStatistic<f64>> {
        self.masked_norm_diff(
            other,
            mask,
            <Self as MaskedNormDiffImage<T, L>>::norm_diff_l1_masked,
        )
    }

    pub(crate) fn masked_norm_diff(
        self,
        other: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        metric: MaskedNormDiffCallback<L, T>,
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
                &mut DeviceOutputView::from_output(&mut output),
            )?;
        }
        // SAFETY: the completed operation above wrote every element.
        Ok(ImageStatistic::from_values(unsafe { output.assume_init() }))
    }
}

type MaskedNormDiffCallback<L, T> = fn(
    &StreamContext,
    &ImageView<'_, T, L>,
    &ImageView<'_, T, L>,
    &MaskView<'_>,
    &mut DeviceOutputView<'_, f64>,
) -> Result<()>;
