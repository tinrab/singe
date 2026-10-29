use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageOutputView, ImageWindow},
    pipeline::image::{ImagePipeline, filtering::MaskedKernelFilterImage},
    types::MaskSize,
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: MaskedKernelFilterImage<T, L>,
{
    /// Applies the native high-pass filter over the valid source window.
    /// Uses the selected square `mask_size` with a centered anchor. The wrapper rejects
    /// one-dimensional masks; NPP may reject unsupported square masks.
    /// The source window must retain a complete parent-image halo for the selected
    /// centered square mask. Source-window and destination dimensions must match.
    /// AC4 destination alpha retains its previous initialization state; this call does not
    /// initialize alpha.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// The source window must match the destination size and retain the required halo.
    /// The mask must be square and supported by NPP.
    pub fn filter_high_pass_into(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        mask_size: MaskSize,
    ) -> Result<()> {
        <Self as MaskedKernelFilterImage<T, L>>::filter_high_pass_image(
            stream_context,
            source,
            destination,
            mask_size,
        )
    }

    /// Applies the native low-pass filter over the valid source window.
    /// Uses the selected square `mask_size` with a centered anchor. The wrapper rejects
    /// one-dimensional masks; NPP may reject unsupported square masks.
    /// The source window must retain a complete parent-image halo for the selected
    /// centered square mask. Source-window and destination dimensions must match.
    /// AC4 destination alpha retains its previous initialization state; this call does not
    /// initialize alpha.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// The source window must match the destination size and retain the required halo.
    /// The mask must be square and supported by NPP.
    pub fn filter_low_pass_into(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        mask_size: MaskSize,
    ) -> Result<()> {
        <Self as MaskedKernelFilterImage<T, L>>::filter_low_pass_image(
            stream_context,
            source,
            destination,
            mask_size,
        )
    }

    /// Applies the native Gaussian filter over the valid source window.
    /// Uses the selected square `mask_size` with a centered anchor. The wrapper rejects
    /// one-dimensional masks; NPP may reject unsupported square masks.
    /// The source window must retain a complete parent-image halo for the selected
    /// centered square mask. Source-window and destination dimensions must match.
    /// AC4 destination alpha retains its previous initialization state; this call does not
    /// initialize alpha.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// The source window must match the destination size and retain the required halo.
    /// The mask must be square and supported by NPP.
    pub fn filter_gauss_into(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        mask_size: MaskSize,
    ) -> Result<()> {
        <Self as MaskedKernelFilterImage<T, L>>::filter_gauss_image(
            stream_context,
            source,
            destination,
            mask_size,
        )
    }

    /// Applies the native Laplace filter over the valid source window.
    /// Uses the selected square `mask_size` with a centered anchor. The wrapper rejects
    /// one-dimensional masks; NPP may reject unsupported square masks.
    /// The source window must retain a complete parent-image halo for the selected
    /// centered square mask. Source-window and destination dimensions must match.
    /// AC4 destination alpha retains its previous initialization state; this call does not
    /// initialize alpha.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// The source window must match the destination size and retain the required halo.
    /// The mask must be square and supported by NPP.
    pub fn filter_laplace_into(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        mask_size: MaskSize,
    ) -> Result<()> {
        <Self as MaskedKernelFilterImage<T, L>>::filter_laplace_image(
            stream_context,
            source,
            destination,
            mask_size,
        )
    }
}
