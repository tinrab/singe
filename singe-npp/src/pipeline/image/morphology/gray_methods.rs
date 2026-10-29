use crate::{
    error::Result,
    image::{Image, SupportedImage, view::C1},
    pipeline::image::{
        ImageBacking,
        morphology::ImagePipeline,
        morphology_traits::{GrayMaskMorphologyBorderImage, GrayMaskMorphologyBorderOperation},
    },
    types::{BorderType, Point, Size},
};

impl<T> ImagePipeline<'_, T, C1>
where
    T: Copy + SupportedImage<C1>,
{
    /// Performs grayscale dilation with the supplied mask weights.
    /// Adds the corresponding signed mask value to each source sample before the native
    /// clamping and maximum selection. The provided implementations use `i32` masks for
    /// `u8` images and `f32` masks for `f32` images.
    /// Passes `mask.as_ptr()` directly to NPP as its device mask address. The slice storage
    /// must also be GPU-readable in the selected CUDA context; this call does not upload it.
    /// `source_offset` locates the output ROI in the full source image.
    /// The native routines support `BorderType::Replicate`; other values are forwarded to NPP.
    /// Allocates a new owned image spanning from `source_offset` to the lower-right source edge.
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
    /// Image allocation, view construction can also fail.
    pub fn gray_dilate_border<M>(
        self,
        source_offset: Point,
        mask: &[M],
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<Self>
    where
        M: Copy,
        Self: GrayMaskMorphologyBorderImage<T, M>,
    {
        self.gray_morphology_border(
            source_offset,
            mask,
            mask_size,
            anchor,
            border_type,
            <Self as GrayMaskMorphologyBorderImage<T, M>>::gray_dilate_border_image,
        )
    }

    /// Performs grayscale erosion with the supplied mask weights.
    /// Adds the corresponding signed mask value to each source sample before the native
    /// clamping and minimum selection. The provided implementations use `i32` masks for
    /// `u8` images and `f32` masks for `f32` images.
    /// Passes `mask.as_ptr()` directly to NPP as its device mask address. The slice storage
    /// must also be GPU-readable in the selected CUDA context; this call does not upload it.
    /// `source_offset` locates the output ROI in the full source image.
    /// The native routines support `BorderType::Replicate`; other values are forwarded to NPP.
    /// Allocates a new owned image spanning from `source_offset` to the lower-right source edge.
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
    /// Image allocation, view construction can also fail.
    pub fn gray_erode_border<M>(
        self,
        source_offset: Point,
        mask: &[M],
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<Self>
    where
        M: Copy,
        Self: GrayMaskMorphologyBorderImage<T, M>,
    {
        self.gray_morphology_border(
            source_offset,
            mask,
            mask_size,
            anchor,
            border_type,
            <Self as GrayMaskMorphologyBorderImage<T, M>>::gray_erode_border_image,
        )
    }

    fn gray_morphology_border<M>(
        self,
        source_offset: Point,
        mask: &[M],
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
        operation: GrayMaskMorphologyBorderOperation<T, M>,
    ) -> Result<Self>
    where
        M: Copy,
    {
        self.ensure_eager()?;
        let source = self.view()?;
        let output_size = source.window_from(source_offset)?.size();
        let mut destination =
            Image::<T, C1>::create_uninit(self.stream_context.cuda_context(), output_size)?;

        {
            let mut destination_view = destination.output()?;
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
