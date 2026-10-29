use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{ChannelLayout, ImageOutputView, ImageView},
    },
    pipeline::image::{ImageBacking, ImagePipeline, filtering::SumWindowBorderFilterImage},
    types::{BorderType, Point},
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: SumWindowBorderFilterImage<T, L>,
{
    /// Sums each channel over a vertical window into `f32` samples.
    /// `mask_size` is the number of source samples in the window; `anchor` is its zero-based
    /// origin offset along the filtered axis. The provided C1, C3, and C4 implementations
    /// write every channel, including the fourth channel in C4.
    /// `source_offset` locates the output ROI in the full source image. The destination
    /// dimensions select its extent.
    /// The wrapper forwards `border_type` to NPP, which may reject unsupported values.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// `mask_size` must be positive and `anchor` must be in `0..mask_size`.
    /// The source offset must select a source pixel, and the destination-sized ROI must fit
    /// inside the source without coordinate overflow.
    pub fn sum_window_column_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, f32, L>,
        mask_size: i32,
        anchor: i32,
        border_type: BorderType,
    ) -> Result<()> {
        <Self as SumWindowBorderFilterImage<T, L>>::sum_window_column_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            mask_size,
            anchor,
            border_type,
        )
    }

    /// Sums each channel over a horizontal window into `f32` samples.
    /// `mask_size` is the number of source samples in the window; `anchor` is its zero-based
    /// origin offset along the filtered axis. The provided C1, C3, and C4 implementations
    /// write every channel, including the fourth channel in C4.
    /// `source_offset` locates the output ROI in the full source image. The destination
    /// dimensions select its extent.
    /// The wrapper forwards `border_type` to NPP, which may reject unsupported values.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// `mask_size` must be positive and `anchor` must be in `0..mask_size`.
    /// The source offset must select a source pixel, and the destination-sized ROI must fit
    /// inside the source without coordinate overflow.
    pub fn sum_window_row_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, f32, L>,
        mask_size: i32,
        anchor: i32,
        border_type: BorderType,
    ) -> Result<()> {
        <Self as SumWindowBorderFilterImage<T, L>>::sum_window_row_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            mask_size,
            anchor,
            border_type,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    f32: SupportedImage<L>,
    Self: SumWindowBorderFilterImage<T, L>,
{
    /// Sums each channel over a vertical window into `f32` samples.
    /// `mask_size` is the number of source samples in the window; `anchor` is its zero-based
    /// origin offset along the filtered axis. The provided C1, C3, and C4 implementations
    /// write every channel, including the fourth channel in C4.
    /// `source_offset` locates the output ROI in the full source image. Allocates a new
    /// owned image spanning from that offset to the lower-right source edge.
    /// The wrapper forwards `border_type` to NPP, which may reject unsupported values.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// `mask_size` must be positive and `anchor` must be in `0..mask_size`.
    /// The source offset must select a source pixel, and the destination-sized ROI must fit
    /// inside the source without coordinate overflow.
    /// Image allocation or view construction can also fail.
    pub fn sum_window_column_border(
        self,
        source_offset: Point,
        mask_size: i32,
        anchor: i32,
        border_type: BorderType,
    ) -> Result<ImagePipeline<'a, f32, L>> {
        self.sum_window_border_filter(
            source_offset,
            mask_size,
            anchor,
            border_type,
            <Self as SumWindowBorderFilterImage<T, L>>::sum_window_column_border_image,
        )
    }

    /// Sums each channel over a horizontal window into `f32` samples.
    /// `mask_size` is the number of source samples in the window; `anchor` is its zero-based
    /// origin offset along the filtered axis. The provided C1, C3, and C4 implementations
    /// write every channel, including the fourth channel in C4.
    /// `source_offset` locates the output ROI in the full source image. Allocates a new
    /// owned image spanning from that offset to the lower-right source edge.
    /// The wrapper forwards `border_type` to NPP, which may reject unsupported values.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// `mask_size` must be positive and `anchor` must be in `0..mask_size`.
    /// The source offset must select a source pixel, and the destination-sized ROI must fit
    /// inside the source without coordinate overflow.
    /// Image allocation or view construction can also fail.
    pub fn sum_window_row_border(
        self,
        source_offset: Point,
        mask_size: i32,
        anchor: i32,
        border_type: BorderType,
    ) -> Result<ImagePipeline<'a, f32, L>> {
        self.sum_window_border_filter(
            source_offset,
            mask_size,
            anchor,
            border_type,
            <Self as SumWindowBorderFilterImage<T, L>>::sum_window_row_border_image,
        )
    }

    fn sum_window_border_filter(
        self,
        source_offset: Point,
        mask_size: i32,
        anchor: i32,
        border_type: BorderType,
        filter: SumWindowBorderFilterCallback<L, T>,
    ) -> Result<ImagePipeline<'a, f32, L>> {
        self.ensure_eager()?;
        let source = self.view()?;
        let output_size = source.window_from(source_offset)?.size();
        let mut destination =
            Image::<f32, L>::create_uninit(self.stream_context.cuda_context(), output_size)?;

        {
            let mut destination_view = destination.output()?;
            filter(
                self.stream_context,
                &source,
                source_offset,
                &mut destination_view,
                mask_size,
                anchor,
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

type SumWindowBorderFilterCallback<L, T> = fn(
    &StreamContext,
    &ImageView<'_, T, L>,
    Point,
    &mut ImageOutputView<'_, f32, L>,
    i32,
    i32,
    BorderType,
) -> Result<()>;
