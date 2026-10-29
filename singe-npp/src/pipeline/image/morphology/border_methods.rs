use crate::{
    error::Result,
    image::{Image, SupportedImage, view::ChannelLayout},
    pipeline::image::{ImageBacking, ImagePipeline, morphology_traits::MaskMorphologyBorderImage},
    types::{BorderType, Point, Size},
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: MaskMorphologyBorderImage<T, L>,
{
    /// Dilates each active channel using the nonzero entries of `mask`.
    /// Passes `mask.as_ptr()` directly to NPP as its device mask address. The slice storage
    /// must also be GPU-readable in the selected CUDA context; this call does not upload it.
    /// `source_offset` locates the output ROI in the full source image.
    /// The native routines support `BorderType::Replicate`; other values are forwarded to NPP.
    /// Allocates a new owned image spanning from `source_offset` to the lower-right source edge.
    /// Preserves AC4 alpha from the corresponding source window.
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
    pub fn dilate_border(
        self,
        source_offset: Point,
        mask: &[u8],
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<Self> {
        self.ensure_eager()?;
        let source = self.view()?;
        let output_window = source.window_from(source_offset)?;
        let mut destination =
            Image::<T, L>::create_uninit(self.stream_context.cuda_context(), output_window.size())?;

        {
            let mut destination_view = destination.output()?;
            if L::IGNORES_ALPHA {
                destination_view.copy_from(&output_window.view())?;
            }
            <Self as MaskMorphologyBorderImage<T, L>>::dilate_border_image(
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

    /// Erodes each active channel using the nonzero entries of `mask`.
    /// Passes `mask.as_ptr()` directly to NPP as its device mask address. The slice storage
    /// must also be GPU-readable in the selected CUDA context; this call does not upload it.
    /// `source_offset` locates the output ROI in the full source image.
    /// The native routines support `BorderType::Replicate`; other values are forwarded to NPP.
    /// Allocates a new owned image spanning from `source_offset` to the lower-right source edge.
    /// Preserves AC4 alpha from the corresponding source window.
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
    pub fn erode_border(
        self,
        source_offset: Point,
        mask: &[u8],
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<Self> {
        self.ensure_eager()?;
        let source = self.view()?;
        let output_window = source.window_from(source_offset)?;
        let mut destination =
            Image::<T, L>::create_uninit(self.stream_context.cuda_context(), output_window.size())?;

        {
            let mut destination_view = destination.output()?;
            if L::IGNORES_ALPHA {
                destination_view.copy_from(&output_window.view())?;
            }
            <Self as MaskMorphologyBorderImage<T, L>>::erode_border_image(
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
