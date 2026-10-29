use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageOutputView, ImageView},
    pipeline::image::{ImagePipeline, morphology_traits::CompositeMorphologyBorderScratchImage},
    types::{BorderType, Point, Size},
    workspace::ScratchBuffer,
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    /// Computes the source minus its morphological opening.
    /// Nonzero mask entries select the samples participating in the native morphology passes.
    /// Passes `mask.as_ptr()` directly to NPP as its device mask address. The slice storage
    /// must also be GPU-readable in the selected CUDA context; this call does not upload it.
    /// `source_offset` locates the output ROI in the full source image.
    /// The native routines support `BorderType::Replicate`; other values are forwarded to NPP.
    /// The destination dimensions select the output ROI extent.
    /// AC4 destination alpha retains its previous initialization state; this call does not
    /// initialize alpha.
    /// Uses the supplied scratch buffer, which must be ready for exclusive use in the selected
    /// CUDA context. Checks its queried byte capacity, without checking its context or waiting
    /// for its producers.
    /// Prepares participating image regions, waits for their producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Mask dimensions must be positive, their checked product must equal `mask.len()`, and
    /// `anchor` must lie within the mask.
    /// The source offset must select a source pixel, checked ROI endpoints must fit inside
    /// the source, and output dimensions must be positive.
    /// The scratch-size query or capacity check can also fail.
    pub fn morph_top_hat_border_with_scratch_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        mask: &[u8],
        mask_size: Size,
        anchor: Point,
        scratch: &mut ScratchBuffer,
        border_type: BorderType,
    ) -> Result<()>
    where
        Self: CompositeMorphologyBorderScratchImage<T, L>,
    {
        <Self as CompositeMorphologyBorderScratchImage<T, L>>::morph_top_hat_border_image_with_scratch(
            stream_context,
            source,
            source_offset,
            destination,
            mask,
            mask_size,
            anchor,
            scratch,
            border_type,
        )
    }

    /// Computes the morphological closing minus the source.
    /// Nonzero mask entries select the samples participating in the native morphology passes.
    /// Passes `mask.as_ptr()` directly to NPP as its device mask address. The slice storage
    /// must also be GPU-readable in the selected CUDA context; this call does not upload it.
    /// `source_offset` locates the output ROI in the full source image.
    /// The native routines support `BorderType::Replicate`; other values are forwarded to NPP.
    /// The destination dimensions select the output ROI extent.
    /// AC4 destination alpha retains its previous initialization state; this call does not
    /// initialize alpha.
    /// Uses the supplied scratch buffer, which must be ready for exclusive use in the selected
    /// CUDA context. Checks its queried byte capacity, without checking its context or waiting
    /// for its producers.
    /// Prepares participating image regions, waits for their producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Mask dimensions must be positive, their checked product must equal `mask.len()`, and
    /// `anchor` must lie within the mask.
    /// The source offset must select a source pixel, checked ROI endpoints must fit inside
    /// the source, and output dimensions must be positive.
    /// The scratch-size query or capacity check can also fail.
    pub fn morph_black_hat_border_with_scratch_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        mask: &[u8],
        mask_size: Size,
        anchor: Point,
        scratch: &mut ScratchBuffer,
        border_type: BorderType,
    ) -> Result<()>
    where
        Self: CompositeMorphologyBorderScratchImage<T, L>,
    {
        <Self as CompositeMorphologyBorderScratchImage<T, L>>::morph_black_hat_border_image_with_scratch(
            stream_context,
            source,
            source_offset,
            destination,
            mask,
            mask_size,
            anchor,
            scratch,
            border_type,
        )
    }

    /// Computes morphological dilation minus erosion.
    /// Nonzero mask entries select the samples participating in the native morphology passes.
    /// Passes `mask.as_ptr()` directly to NPP as its device mask address. The slice storage
    /// must also be GPU-readable in the selected CUDA context; this call does not upload it.
    /// `source_offset` locates the output ROI in the full source image.
    /// The native routines support `BorderType::Replicate`; other values are forwarded to NPP.
    /// The destination dimensions select the output ROI extent.
    /// AC4 destination alpha retains its previous initialization state; this call does not
    /// initialize alpha.
    /// Uses the supplied scratch buffer, which must be ready for exclusive use in the selected
    /// CUDA context. Checks its queried byte capacity, without checking its context or waiting
    /// for its producers.
    /// Prepares participating image regions, waits for their producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Mask dimensions must be positive, their checked product must equal `mask.len()`, and
    /// `anchor` must lie within the mask.
    /// The source offset must select a source pixel, checked ROI endpoints must fit inside
    /// the source, and output dimensions must be positive.
    /// The scratch-size query or capacity check can also fail.
    pub fn morph_gradient_border_with_scratch_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        mask: &[u8],
        mask_size: Size,
        anchor: Point,
        scratch: &mut ScratchBuffer,
        border_type: BorderType,
    ) -> Result<()>
    where
        Self: CompositeMorphologyBorderScratchImage<T, L>,
    {
        <Self as CompositeMorphologyBorderScratchImage<T, L>>::morph_gradient_border_image_with_scratch(
            stream_context,
            source,
            source_offset,
            destination,
            mask,
            mask_size,
            anchor,
            scratch,
            border_type,
        )
    }
}
