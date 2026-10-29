use crate::{
    error::Result,
    image::{Image, SupportedImage, view::ChannelLayout},
    pipeline::image::{
        ImageBacking, ImagePipeline,
        morphology_traits::{CompositeMorphologyBorderImage, MaskMorphologyBorderOperation},
    },
    types::{BorderType, Point, Size},
};

mod derived_methods;

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: CompositeMorphologyBorderImage<T, L>,
{
    /// Performs morphological closing: dilation followed by erosion.
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
    pub fn morph_close_border(
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
            <Self as CompositeMorphologyBorderImage<T, L>>::morph_close_border_image,
        )
    }

    /// Performs morphological opening: erosion followed by dilation.
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
    pub fn morph_open_border(
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
            <Self as CompositeMorphologyBorderImage<T, L>>::morph_open_border_image,
        )
    }

    pub(crate) fn composite_morphology_border(
        self,
        source_offset: Point,
        mask: &[u8],
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
        operation: MaskMorphologyBorderOperation<T, L>,
    ) -> Result<Self> {
        self.ensure_eager()?;
        let source = self.view()?;
        let output_source = source.window_from(source_offset)?;
        let mut destination =
            Image::<T, L>::create_uninit(self.stream_context.cuda_context(), output_source.size())?;

        {
            let mut destination_view = destination.output()?;
            if L::IGNORES_ALPHA {
                destination_view.copy_from(&output_source.view())?;
            }
            operation(
                self.stream_context,
                &source,
                source_offset,
                &mut destination_view,
                mask,
                mask_size,
                anchor,
                border_type,
            )?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            // SAFETY: the completed operation above wrote every element.
            backing: ImageBacking::Owned(unsafe { destination.assume_init() }),
        })
    }
}
