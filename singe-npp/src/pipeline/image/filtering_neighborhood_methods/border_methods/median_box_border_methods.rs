use crate::{
    error::Result,
    image::{Image, SupportedImage, view::ChannelLayout},
    pipeline::image::{
        filtering::{BoxBorderFilterImage, MedianBorderFilterImage},
        filtering_neighborhood_methods::{ImageBacking, border_methods::ImagePipeline},
    },
    types::{BorderType, Point, Size},
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: MedianBorderFilterImage<T, L>,
{
    /// Applies a median filter with source border control into a new owned image.
    ///
    /// `source_offset` locates the output ROI in the full source image. The output extends
    /// from that pixel to the lower-right source edge. `mask_size` gives the neighborhood
    /// dimensions and `anchor` locates the source pixel within it. For AC4, copies the
    /// source window before filtering to preserve alpha.
    /// Allocates temporary device scratch and retains it through native completion.
    /// Forwards `border_type` to NPP, which may reject unsupported values.
    /// Prepares source and destination regions, waits for producers on other streams,
    /// rejects capture, and returns after successful stream completion. Row padding remains
    /// unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native filtering, or stream synchronization fail.
    /// The source offset must select a source pixel; ROI endpoints must not overflow or
    /// exceed the source image. Mask dimensions must be positive and the anchor must lie
    /// inside the mask.
    /// Scratch-size queries, checked size conversions, or scratch allocation can fail.
    /// Output allocation or image view construction can also fail. Copying source alpha
    /// can fail.
    pub fn filter_median_border(
        self,
        source_offset: Point,
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
            <Self as MedianBorderFilterImage<T, L>>::filter_median_border_image(
                self.stream_context,
                &source,
                source_offset,
                &mut destination_view,
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

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: BoxBorderFilterImage<T, L>,
{
    /// Applies a rectangular averaging filter with source border control into a new owned image.
    ///
    /// `source_offset` locates the output ROI in the full source image. The output extends
    /// from that pixel to the lower-right source edge. `mask_size` gives the neighborhood
    /// dimensions and `anchor` locates the source pixel within it. For AC4, copies the
    /// source window before filtering to preserve alpha.
    /// Forwards `border_type` to NPP, which may reject unsupported values.
    /// Prepares source and destination regions, waits for producers on other streams,
    /// rejects capture, and returns after successful stream completion. Row padding remains
    /// unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native filtering, or stream synchronization fail.
    /// The source offset must select a source pixel; ROI endpoints must not overflow or
    /// exceed the source image. Mask dimensions must be positive and the anchor must lie
    /// inside the mask.
    /// Output allocation or image view construction can also fail. Copying source alpha
    /// can fail.
    pub fn filter_box_border(
        self,
        source_offset: Point,
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
            <Self as BoxBorderFilterImage<T, L>>::filter_box_border_image(
                self.stream_context,
                &source,
                source_offset,
                &mut destination_view,
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
