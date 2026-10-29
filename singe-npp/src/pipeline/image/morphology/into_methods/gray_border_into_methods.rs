use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, ImageOutputView, ImageView},
    pipeline::image::{ImagePipeline, morphology_traits::GrayMaskMorphologyBorderImage},
    types::{BorderType, Point, Size},
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
{
    /// Performs grayscale dilation with the supplied mask weights.
    /// Adds the corresponding signed mask value to each source sample before the native
    /// clamping and maximum selection. The provided implementations use `i32` masks for
    /// `u8` images and `f32` masks for `f32` images.
    /// Passes `mask.as_ptr()` directly to NPP as its device mask address. The slice storage
    /// must also be GPU-readable in the selected CUDA context; this call does not upload it.
    /// `source_offset` locates the output ROI in the full source image.
    /// The native routines support `BorderType::Replicate`; other values are forwarded to NPP.
    /// The destination dimensions select the output ROI extent.
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
    pub fn gray_dilate_border_into<M>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, C1>,
        mask: &[M],
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<()>
    where
        Self: GrayMaskMorphologyBorderImage<T, M>,
    {
        <Self as GrayMaskMorphologyBorderImage<T, M>>::gray_dilate_border_image(
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

    /// Performs grayscale erosion with the supplied mask weights.
    /// Adds the corresponding signed mask value to each source sample before the native
    /// clamping and minimum selection. The provided implementations use `i32` masks for
    /// `u8` images and `f32` masks for `f32` images.
    /// Passes `mask.as_ptr()` directly to NPP as its device mask address. The slice storage
    /// must also be GPU-readable in the selected CUDA context; this call does not upload it.
    /// `source_offset` locates the output ROI in the full source image.
    /// The native routines support `BorderType::Replicate`; other values are forwarded to NPP.
    /// The destination dimensions select the output ROI extent.
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
    pub fn gray_erode_border_into<M>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, C1>,
        mask: &[M],
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<()>
    where
        Self: GrayMaskMorphologyBorderImage<T, M>,
    {
        <Self as GrayMaskMorphologyBorderImage<T, M>>::gray_erode_border_image(
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
