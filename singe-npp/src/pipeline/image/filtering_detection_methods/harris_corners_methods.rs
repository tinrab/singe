use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{C1, ChannelLayout, ImageOutputView, ImageView},
    },
    pipeline::image::{
        ImageBacking,
        filtering_detection_methods::{HarrisCornersBorderFilterImage, ImagePipeline},
    },
    types::{BorderType, DifferentialKernel, MaskSize, Point},
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: HarrisCornersBorderFilterImage<T, L>,
{
    /// Computes Harris corner responses into a C1 `f32` image.
    ///
    /// The provided implementation accepts C1 `u8` input. `filter_type` selects Sobel or
    /// Scharr derivatives. `mask_size` selects the differential mask; `average_window_size`
    /// selects the averaging window. Both must be `MaskSize::Mask3x3` or `MaskSize::Mask5x5`.
    /// Only `BorderType::Replicate` is accepted. `k` is the Harris constant, commonly 0.04;
    /// `scale` scales the response. These two values are forwarded without validation.
    /// Allocates temporary device scratch and retains it through native completion.
    /// `source_offset` locates the output ROI in the full source image. The destination
    /// dimensions select its extent.
    /// Writes all logical output samples; row padding remains unwritten.
    /// Prepares participating regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native execution, or stream synchronization fail. The source offset
    /// must address a source pixel; output-sized ROI endpoints must not overflow or exceed
    /// the source image.
    /// Both masks must be 3-by-3 or 5-by-5, and the border type must be replicate.
    /// Scratch-size queries, checked size conversions, or scratch allocation can fail.
    pub fn filter_harris_corners_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, f32, C1>,
        filter_type: DifferentialKernel,
        mask_size: MaskSize,
        average_window_size: MaskSize,
        k: f32,
        scale: f32,
        border_type: BorderType,
    ) -> Result<()> {
        <Self as HarrisCornersBorderFilterImage<T, L>>::filter_harris_corners_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            filter_type,
            mask_size,
            average_window_size,
            k,
            scale,
            border_type,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    f32: SupportedImage<C1>,
    Self: HarrisCornersBorderFilterImage<T, L>,
{
    /// Computes Harris corner responses into a C1 `f32` image.
    ///
    /// The provided implementation accepts C1 `u8` input. `filter_type` selects Sobel or
    /// Scharr derivatives. `mask_size` selects the differential mask; `average_window_size`
    /// selects the averaging window. Both must be `MaskSize::Mask3x3` or `MaskSize::Mask5x5`.
    /// Only `BorderType::Replicate` is accepted. `k` is the Harris constant, commonly 0.04;
    /// `scale` scales the response. These two values are forwarded without validation.
    /// Allocates temporary device scratch and retains it through native completion.
    /// `source_offset` locates the output ROI in the full source image. The new owned
    /// output extends from that offset to the lower-right source edge.
    /// Writes all logical output samples; row padding remains unwritten.
    /// Prepares participating regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native execution, or stream synchronization fail. The source offset
    /// must address a source pixel; output-sized ROI endpoints must not overflow or exceed
    /// the source image.
    /// Both masks must be 3-by-3 or 5-by-5, and the border type must be replicate.
    /// Scratch-size queries, checked size conversions, or scratch allocation can fail.
    /// Output allocation or image view construction can also fail.
    pub fn filter_harris_corners_border(
        self,
        source_offset: Point,
        filter_type: DifferentialKernel,
        mask_size: MaskSize,
        average_window_size: MaskSize,
        k: f32,
        scale: f32,
        border_type: BorderType,
    ) -> Result<ImagePipeline<'a, f32, C1>> {
        self.ensure_eager()?;
        let source = self.view()?;
        let output_size = source.window_from(source_offset)?.size();
        let mut destination =
            Image::<f32, C1>::create_uninit(self.stream_context.cuda_context(), output_size)?;

        {
            let mut destination_view = destination.output()?;
            <Self as HarrisCornersBorderFilterImage<T, L>>::filter_harris_corners_border_image(
                self.stream_context,
                &source,
                source_offset,
                &mut destination_view,
                filter_type,
                mask_size,
                average_window_size,
                k,
                scale,
                border_type,
            )?;
        }

        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            // SAFETY: the completed operation above wrote every element.
            backing: ImageBacking::Owned(unsafe { destination.assume_init() }),
        })
    }
}
