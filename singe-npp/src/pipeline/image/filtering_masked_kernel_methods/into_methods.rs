use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageOutputView, ImageView},
    pipeline::image::{ImagePipeline, filtering::MaskedKernelBorderFilterImage},
    types::{BorderType, MaskSize, Point},
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: MaskedKernelBorderFilterImage<T, L>,
{
    /// Applies the native high-pass filter with border control.
    /// `mask_size` selects the native filter kernel. The wrapper forwards mask and border
    /// values to NPP, which may reject unsupported combinations.
    /// `source_offset` locates the output ROI in the full source. The destination dimensions
    /// select its extent.
    /// AC4 destination alpha retains its previous initialization state; this call does not
    /// initialize alpha.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// The source offset must select a source pixel, and the destination-sized ROI must fit
    /// inside the source without coordinate overflow.
    pub fn filter_high_pass_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()> {
        <Self as MaskedKernelBorderFilterImage<T, L>>::filter_high_pass_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            mask_size,
            border_type,
        )
    }

    /// Applies the native low-pass filter with border control.
    /// `mask_size` selects the native filter kernel. The wrapper forwards mask and border
    /// values to NPP, which may reject unsupported combinations.
    /// `source_offset` locates the output ROI in the full source. The destination dimensions
    /// select its extent.
    /// AC4 destination alpha retains its previous initialization state; this call does not
    /// initialize alpha.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// The source offset must select a source pixel, and the destination-sized ROI must fit
    /// inside the source without coordinate overflow.
    pub fn filter_low_pass_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()> {
        <Self as MaskedKernelBorderFilterImage<T, L>>::filter_low_pass_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            mask_size,
            border_type,
        )
    }

    /// Applies the native Gaussian filter with border control.
    /// `mask_size` selects the native filter kernel. The wrapper forwards mask and border
    /// values to NPP, which may reject unsupported combinations.
    /// `source_offset` locates the output ROI in the full source. The destination dimensions
    /// select its extent.
    /// AC4 destination alpha retains its previous initialization state; this call does not
    /// initialize alpha.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// The source offset must select a source pixel, and the destination-sized ROI must fit
    /// inside the source without coordinate overflow.
    pub fn filter_gauss_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()> {
        <Self as MaskedKernelBorderFilterImage<T, L>>::filter_gauss_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            mask_size,
            border_type,
        )
    }

    /// Applies the native Laplace filter with border control.
    /// `mask_size` selects the native filter kernel. The wrapper forwards mask and border
    /// values to NPP, which may reject unsupported combinations.
    /// `source_offset` locates the output ROI in the full source. The destination dimensions
    /// select its extent.
    /// AC4 destination alpha retains its previous initialization state; this call does not
    /// initialize alpha.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// The source offset must select a source pixel, and the destination-sized ROI must fit
    /// inside the source without coordinate overflow.
    pub fn filter_laplace_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()> {
        <Self as MaskedKernelBorderFilterImage<T, L>>::filter_laplace_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            mask_size,
            border_type,
        )
    }
}
