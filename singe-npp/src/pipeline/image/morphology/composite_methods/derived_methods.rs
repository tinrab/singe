use crate::{
    error::Result,
    image::{SupportedImage, view::ChannelLayout},
    pipeline::image::{ImagePipeline, morphology_traits::CompositeMorphologyBorderImage},
    types::{BorderType, Point, Size},
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: CompositeMorphologyBorderImage<T, L>,
{
    /// Computes the source minus its morphological opening.
    /// Nonzero mask entries select the samples participating in the native morphology passes.
    /// Passes `mask.as_ptr()` directly to NPP as its device mask address. The slice storage
    /// must also be GPU-readable in the selected CUDA context; this call does not upload it.
    /// `source_offset` locates the output ROI in the full source image.
    /// The native routines support `BorderType::Replicate`; other values are forwarded to NPP.
    /// Allocates a new owned image spanning from `source_offset` to the lower-right source edge.
    /// Preserves AC4 alpha from the corresponding source window.
    /// Queries and allocates temporary device scratch for the native operation.
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
    /// Image allocation, view construction, or alpha preservation copies can also fail.
    /// The scratch-size query or scratch allocation can also fail.
    pub fn morph_top_hat_border(
        self,
        source_offset: Point,
        mask: &[u8],
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<Self> {
        self.composite_morphology_border(
            source_offset,
            mask,
            mask_size,
            anchor,
            border_type,
            <Self as CompositeMorphologyBorderImage<T, L>>::morph_top_hat_border_image,
        )
    }

    /// Computes the morphological closing minus the source.
    /// Nonzero mask entries select the samples participating in the native morphology passes.
    /// Passes `mask.as_ptr()` directly to NPP as its device mask address. The slice storage
    /// must also be GPU-readable in the selected CUDA context; this call does not upload it.
    /// `source_offset` locates the output ROI in the full source image.
    /// The native routines support `BorderType::Replicate`; other values are forwarded to NPP.
    /// Allocates a new owned image spanning from `source_offset` to the lower-right source edge.
    /// Preserves AC4 alpha from the corresponding source window.
    /// Queries and allocates temporary device scratch for the native operation.
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
    /// Image allocation, view construction, or alpha preservation copies can also fail.
    /// The scratch-size query or scratch allocation can also fail.
    pub fn morph_black_hat_border(
        self,
        source_offset: Point,
        mask: &[u8],
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<Self> {
        self.composite_morphology_border(
            source_offset,
            mask,
            mask_size,
            anchor,
            border_type,
            <Self as CompositeMorphologyBorderImage<T, L>>::morph_black_hat_border_image,
        )
    }

    /// Computes morphological dilation minus erosion.
    /// Nonzero mask entries select the samples participating in the native morphology passes.
    /// Passes `mask.as_ptr()` directly to NPP as its device mask address. The slice storage
    /// must also be GPU-readable in the selected CUDA context; this call does not upload it.
    /// `source_offset` locates the output ROI in the full source image.
    /// The native routines support `BorderType::Replicate`; other values are forwarded to NPP.
    /// Allocates a new owned image spanning from `source_offset` to the lower-right source edge.
    /// Preserves AC4 alpha from the corresponding source window.
    /// Queries and allocates temporary device scratch for the native operation.
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
    /// Image allocation, view construction, or alpha preservation copies can also fail.
    /// The scratch-size query or scratch allocation can also fail.
    pub fn morph_gradient_border(
        self,
        source_offset: Point,
        mask: &[u8],
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<Self> {
        self.composite_morphology_border(
            source_offset,
            mask,
            mask_size,
            anchor,
            border_type,
            <Self as CompositeMorphologyBorderImage<T, L>>::morph_gradient_border_image,
        )
    }
}
