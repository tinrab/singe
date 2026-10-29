use singe_cuda::{memory::DeviceMemory, view::DeviceOutputView};

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, MaskView},
    pipeline::image::{
        ImagePipeline,
        statistics::{ImageStatistic, MaskedNormRelativeImage},
    },
};

mod mean_standard_deviation_methods;

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: MaskedNormRelativeImage<T, L>,
{
    /// Computes the native relative infinity-norm error between the two image inputs
    /// over pixels selected by `mask`.
    /// Only nonzero mask samples select pixels. Image and mask dimensions must match.
    /// The provided implementations process single-channel images and produce one `f64`
    /// result.
    /// Each supplied output view must contain exactly one element.
    /// Prepares all participating images, masks, and outputs, waits for producers on other
    /// streams, rejects capture, and returns after successful native completion.
    /// The native reduction owns its temporary scratch.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Each output view must have length one.
    /// Image and mask dimensions must match. Scratch-size queries and allocation can also fail.
    pub fn norm_relative_inf_masked_into(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, L>,
        source_2: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        output: &mut DeviceOutputView<'_, f64>,
    ) -> Result<()> {
        <Self as MaskedNormRelativeImage<T, L>>::norm_relative_inf_masked(
            stream_context,
            source_1,
            source_2,
            mask,
            output,
        )
    }

    /// Computes the native relative L1-norm error between the two image inputs
    /// over pixels selected by `mask`.
    /// Only nonzero mask samples select pixels. Image and mask dimensions must match.
    /// The provided implementations process single-channel images and produce one `f64`
    /// result.
    /// Each supplied output view must contain exactly one element.
    /// Prepares all participating images, masks, and outputs, waits for producers on other
    /// streams, rejects capture, and returns after successful native completion.
    /// The native reduction owns its temporary scratch.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Each output view must have length one.
    /// Image and mask dimensions must match. Scratch-size queries and allocation can also fail.
    pub fn norm_relative_l1_masked_into(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, L>,
        source_2: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        output: &mut DeviceOutputView<'_, f64>,
    ) -> Result<()> {
        <Self as MaskedNormRelativeImage<T, L>>::norm_relative_l1_masked(
            stream_context,
            source_1,
            source_2,
            mask,
            output,
        )
    }

    /// Computes the native relative L2-norm error between the two image inputs
    /// over pixels selected by `mask`.
    /// Only nonzero mask samples select pixels. Image and mask dimensions must match.
    /// The provided implementations process single-channel images and produce one `f64`
    /// result.
    /// Each supplied output view must contain exactly one element.
    /// Prepares all participating images, masks, and outputs, waits for producers on other
    /// streams, rejects capture, and returns after successful native completion.
    /// The native reduction owns its temporary scratch.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Each output view must have length one.
    /// Image and mask dimensions must match. Scratch-size queries and allocation can also fail.
    pub fn norm_relative_l2_masked_into(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, L>,
        source_2: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        output: &mut DeviceOutputView<'_, f64>,
    ) -> Result<()> {
        <Self as MaskedNormRelativeImage<T, L>>::norm_relative_l2_masked(
            stream_context,
            source_1,
            source_2,
            mask,
            output,
        )
    }

    /// Computes the native relative infinity-norm error between the two image inputs
    /// over pixels selected by `mask`.
    /// Only nonzero mask samples select pixels. Image and mask dimensions must match.
    /// The provided implementations process single-channel images and produce one `f64`
    /// result.
    /// Allocates owned device result storage.
    /// Prepares all participating images, masks, and outputs, waits for producers on other
    /// streams, rejects capture, and returns after successful native completion.
    /// The native reduction owns its temporary scratch.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Result allocation and constructing the source view can also fail.
    /// Image and mask dimensions must match. Scratch-size queries and allocation can also fail.
    pub fn norm_relative_inf_masked(
        self,
        other: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
    ) -> Result<ImageStatistic<f64>> {
        self.masked_norm_relative(
            other,
            mask,
            <Self as MaskedNormRelativeImage<T, L>>::norm_relative_inf_masked,
        )
    }

    /// Computes the native relative L1-norm error between the two image inputs
    /// over pixels selected by `mask`.
    /// Only nonzero mask samples select pixels. Image and mask dimensions must match.
    /// The provided implementations process single-channel images and produce one `f64`
    /// result.
    /// Allocates owned device result storage.
    /// Prepares all participating images, masks, and outputs, waits for producers on other
    /// streams, rejects capture, and returns after successful native completion.
    /// The native reduction owns its temporary scratch.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Result allocation and constructing the source view can also fail.
    /// Image and mask dimensions must match. Scratch-size queries and allocation can also fail.
    pub fn norm_relative_l1_masked(
        self,
        other: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
    ) -> Result<ImageStatistic<f64>> {
        self.masked_norm_relative(
            other,
            mask,
            <Self as MaskedNormRelativeImage<T, L>>::norm_relative_l1_masked,
        )
    }

    /// Computes the native relative L2-norm error between the two image inputs
    /// over pixels selected by `mask`.
    /// Only nonzero mask samples select pixels. Image and mask dimensions must match.
    /// The provided implementations process single-channel images and produce one `f64`
    /// result.
    /// Allocates owned device result storage.
    /// Prepares all participating images, masks, and outputs, waits for producers on other
    /// streams, rejects capture, and returns after successful native completion.
    /// The native reduction owns its temporary scratch.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Result allocation and constructing the source view can also fail.
    /// Image and mask dimensions must match. Scratch-size queries and allocation can also fail.
    pub fn norm_relative_l2_masked(
        self,
        other: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
    ) -> Result<ImageStatistic<f64>> {
        self.masked_norm_relative(
            other,
            mask,
            <Self as MaskedNormRelativeImage<T, L>>::norm_relative_l2_masked,
        )
    }

    fn masked_norm_relative(
        self,
        other: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        metric: MaskedNormRelativeCallback<L, T>,
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

type MaskedNormRelativeCallback<L, T> = fn(
    &StreamContext,
    &ImageView<'_, T, L>,
    &ImageView<'_, T, L>,
    &MaskView<'_>,
    &mut DeviceOutputView<'_, f64>,
) -> Result<()>;
