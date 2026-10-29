use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageOutputView, ImageView, ImageWindow},
    pipeline::image::{
        ImagePipeline,
        morphology_traits::{MaskMorphologyBorderImage, MaskMorphologyImage},
    },
    types::{BorderType, Point, Size},
};

mod gray_border_into_methods;

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    /// Dilates each active channel using the nonzero entries of `mask`.
    /// Passes `mask.as_ptr()` directly to NPP as its device mask address. The slice storage
    /// must also be GPU-readable in the selected CUDA context; this call does not upload it.
    /// The source window must retain a complete parent-image halo for `mask_size` and `anchor`.
    /// AC4 destination alpha retains its previous initialization state; this call does not
    /// initialize alpha.
    /// Prepares participating image regions, waits for their producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Mask dimensions must be positive, their checked product must equal `mask.len()`, and
    /// `anchor` must lie within the mask.
    /// Source-window and destination dimensions must match, and the parent halo must cover
    /// the entire neighborhood.
    pub fn dilate_into(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        mask: &[u8],
        mask_size: Size,
        anchor: Point,
    ) -> Result<()>
    where
        Self: MaskMorphologyImage<T, L>,
    {
        <Self as MaskMorphologyImage<T, L>>::dilate_image(
            stream_context,
            source,
            destination,
            mask,
            mask_size,
            anchor,
        )
    }

    /// Erodes each active channel using the nonzero entries of `mask`.
    /// Passes `mask.as_ptr()` directly to NPP as its device mask address. The slice storage
    /// must also be GPU-readable in the selected CUDA context; this call does not upload it.
    /// The source window must retain a complete parent-image halo for `mask_size` and `anchor`.
    /// AC4 destination alpha retains its previous initialization state; this call does not
    /// initialize alpha.
    /// Prepares participating image regions, waits for their producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Mask dimensions must be positive, their checked product must equal `mask.len()`, and
    /// `anchor` must lie within the mask.
    /// Source-window and destination dimensions must match, and the parent halo must cover
    /// the entire neighborhood.
    pub fn erode_into(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        mask: &[u8],
        mask_size: Size,
        anchor: Point,
    ) -> Result<()>
    where
        Self: MaskMorphologyImage<T, L>,
    {
        <Self as MaskMorphologyImage<T, L>>::erode_image(
            stream_context,
            source,
            destination,
            mask,
            mask_size,
            anchor,
        )
    }

    /// Dilates each active channel using the nonzero entries of `mask`.
    /// Passes `mask.as_ptr()` directly to NPP as its device mask address. The slice storage
    /// must also be GPU-readable in the selected CUDA context; this call does not upload it.
    /// `source_offset` locates the output ROI in the full source image.
    /// The native routines support `BorderType::Replicate`; other values are forwarded to NPP.
    /// The destination dimensions select the output ROI extent.
    /// AC4 destination alpha retains its previous initialization state; this call does not
    /// initialize alpha.
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
    pub fn dilate_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        mask: &[u8],
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<()>
    where
        Self: MaskMorphologyBorderImage<T, L>,
    {
        <Self as MaskMorphologyBorderImage<T, L>>::dilate_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            mask,
            mask_size,
            anchor,
            border_type,
        )
    }

    /// Erodes each active channel using the nonzero entries of `mask`.
    /// Passes `mask.as_ptr()` directly to NPP as its device mask address. The slice storage
    /// must also be GPU-readable in the selected CUDA context; this call does not upload it.
    /// `source_offset` locates the output ROI in the full source image.
    /// The native routines support `BorderType::Replicate`; other values are forwarded to NPP.
    /// The destination dimensions select the output ROI extent.
    /// AC4 destination alpha retains its previous initialization state; this call does not
    /// initialize alpha.
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
    pub fn erode_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        mask: &[u8],
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<()>
    where
        Self: MaskMorphologyBorderImage<T, L>,
    {
        <Self as MaskMorphologyBorderImage<T, L>>::erode_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            mask,
            mask_size,
            anchor,
            border_type,
        )
    }
}
