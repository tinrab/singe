use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageOutputView, ImageWindow},
    pipeline::image::{ImagePipeline, filtering::HighLowGaussFilterImage},
    types::MaskSize,
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: HighLowGaussFilterImage<T, L>,
{
    /// Applies the native high-pass filter over the valid source window.
    /// Uses the selected square `mask_size` with its center as anchor. One-dimensional
    /// mask variants are rejected; NPP determines which square masks the routine supports.
    /// The source window must retain a complete parent-image halo for the centered square
    /// mask.
    /// AC4 destination alpha retains its previous initialization state; this call does not
    /// initialize alpha.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Source-window and destination dimensions must match, and the parent halo must cover
    /// the complete neighborhood.
    /// The mask must be square and supported by NPP.
    pub fn filter_high_pass_partial_into(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        mask_size: MaskSize,
    ) -> Result<()> {
        <Self as HighLowGaussFilterImage<T, L>>::filter_high_pass_image(
            stream_context,
            source,
            destination,
            mask_size,
        )
    }

    /// Applies the native low-pass filter over the valid source window.
    /// Uses the selected square `mask_size` with its center as anchor. One-dimensional
    /// mask variants are rejected; NPP determines which square masks the routine supports.
    /// The source window must retain a complete parent-image halo for the centered square
    /// mask.
    /// AC4 destination alpha retains its previous initialization state; this call does not
    /// initialize alpha.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Source-window and destination dimensions must match, and the parent halo must cover
    /// the complete neighborhood.
    /// The mask must be square and supported by NPP.
    pub fn filter_low_pass_partial_into(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        mask_size: MaskSize,
    ) -> Result<()> {
        <Self as HighLowGaussFilterImage<T, L>>::filter_low_pass_image(
            stream_context,
            source,
            destination,
            mask_size,
        )
    }

    /// Applies the native fixed-coefficient Gaussian filter over the valid source window.
    /// Uses the selected square `mask_size` with its center as anchor. One-dimensional
    /// mask variants are rejected; NPP determines which square masks the routine supports.
    /// The source window must retain a complete parent-image halo for the centered square
    /// mask.
    /// AC4 destination alpha retains its previous initialization state; this call does not
    /// initialize alpha.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Source-window and destination dimensions must match, and the parent halo must cover
    /// the complete neighborhood.
    /// The mask must be square and supported by NPP.
    pub fn filter_gauss_partial_into(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        mask_size: MaskSize,
    ) -> Result<()> {
        <Self as HighLowGaussFilterImage<T, L>>::filter_gauss_image(
            stream_context,
            source,
            destination,
            mask_size,
        )
    }
}
