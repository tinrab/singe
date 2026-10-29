use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{C1, ChannelLayout, ImageOutputView, ImageView},
    },
    pipeline::{
        ImageBacking,
        image::{
            ImagePipeline,
            filtering::{CannyBorderFilterImage, HarrisCornersBorderFilterImage},
        },
    },
    types::{BorderType, DifferentialKernel, ImageNormalization, MaskSize, Point},
};

mod harris_corners_methods;

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: CannyBorderFilterImage<T, L>,
{
    /// Detects Canny edges into a C1 `u8` image containing zero or `0xff`.
    ///
    /// The provided implementations accept C1 or C3 `u8` input. `filter_type` selects
    /// the Sobel or Scharr differential filter, `mask_size` selects its native mask, and
    /// `norm` selects the gradient magnitude calculation. `low_threshold` and
    /// `high_threshold` select hysteresis thresholds; both must be nonnegative and the
    /// low threshold must not exceed the high threshold. Allocates temporary device scratch
    /// and retains it through native completion. NPP may reject unsupported filter, mask,
    /// normalization, or border selections.
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
    /// Both thresholds must be nonnegative and ordered. Scratch-size queries, checked size
    /// conversions, or scratch allocation can fail.
    pub fn filter_canny_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, u8, C1>,
        filter_type: DifferentialKernel,
        mask_size: MaskSize,
        low_threshold: i16,
        high_threshold: i16,
        norm: ImageNormalization,
        border_type: BorderType,
    ) -> Result<()> {
        <Self as CannyBorderFilterImage<T, L>>::filter_canny_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            filter_type,
            mask_size,
            low_threshold,
            high_threshold,
            norm,
            border_type,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    u8: SupportedImage<C1>,
    Self: CannyBorderFilterImage<T, L>,
{
    /// Detects Canny edges into a C1 `u8` image containing zero or `0xff`.
    ///
    /// The provided implementations accept C1 or C3 `u8` input. `filter_type` selects
    /// the Sobel or Scharr differential filter, `mask_size` selects its native mask, and
    /// `norm` selects the gradient magnitude calculation. `low_threshold` and
    /// `high_threshold` select hysteresis thresholds; both must be nonnegative and the
    /// low threshold must not exceed the high threshold. Allocates temporary device scratch
    /// and retains it through native completion. NPP may reject unsupported filter, mask,
    /// normalization, or border selections.
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
    /// Both thresholds must be nonnegative and ordered. Scratch-size queries, checked size
    /// conversions, or scratch allocation can fail.
    /// Output allocation or image view construction can also fail.
    pub fn filter_canny_border(
        self,
        source_offset: Point,
        filter_type: DifferentialKernel,
        mask_size: MaskSize,
        low_threshold: i16,
        high_threshold: i16,
        norm: ImageNormalization,
        border_type: BorderType,
    ) -> Result<ImagePipeline<'a, u8, C1>> {
        self.ensure_eager()?;
        let source = self.view()?;
        let output_size = source.window_from(source_offset)?.size();
        let mut destination =
            Image::<u8, C1>::create_uninit(self.stream_context.cuda_context(), output_size)?;

        {
            let mut destination_view = destination.output()?;
            <Self as CannyBorderFilterImage<T, L>>::filter_canny_border_image(
                self.stream_context,
                &source,
                source_offset,
                &mut destination_view,
                filter_type,
                mask_size,
                low_threshold,
                high_threshold,
                norm,
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
