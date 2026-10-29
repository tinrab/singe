use crate::{
    context::StreamContext,
    error::{Error, Result},
    image::{
        Image, SupportedImage,
        view::{ChannelLayout, ImageOutputView, ImageWindow},
    },
    pipeline::image::{
        ImageBacking, ImagePipeline,
        filtering::{FilterAxis, SumWindowFilterImage},
    },
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: SumWindowFilterImage<T, L>,
{
    /// Sums each channel over a vertical window into `f32` samples.
    /// `mask_size` is the number of source samples in the window; `anchor` is its zero-based
    /// origin offset along the filtered axis. The provided C1, C3, and C4 implementations
    /// write every channel, including the fourth channel in C4.
    /// The source window must retain the parent-image halo needed for the vertical window
    /// and anchor. Source-window and destination dimensions must match.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// `mask_size` must be positive and `anchor` must be in `0..mask_size`.
    /// The source window must match the destination size and retain the required halo.
    pub fn sum_window_column_into(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, f32, L>,
        mask_size: i32,
        anchor: i32,
    ) -> Result<()> {
        <Self as SumWindowFilterImage<T, L>>::sum_window_column_image(
            stream_context,
            source,
            destination,
            mask_size,
            anchor,
        )
    }

    /// Sums each channel over a horizontal window into `f32` samples.
    /// `mask_size` is the number of source samples in the window; `anchor` is its zero-based
    /// origin offset along the filtered axis. The provided C1, C3, and C4 implementations
    /// write every channel, including the fourth channel in C4.
    /// The source window must retain the parent-image halo needed for the horizontal window
    /// and anchor. Source-window and destination dimensions must match.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// `mask_size` must be positive and `anchor` must be in `0..mask_size`.
    /// The source window must match the destination size and retain the required halo.
    pub fn sum_window_row_into(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, f32, L>,
        mask_size: i32,
        anchor: i32,
    ) -> Result<()> {
        <Self as SumWindowFilterImage<T, L>>::sum_window_row_image(
            stream_context,
            source,
            destination,
            mask_size,
            anchor,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    f32: SupportedImage<L>,
    Self: SumWindowFilterImage<T, L>,
{
    /// Sums each channel over a vertical window into `f32` samples.
    /// `mask_size` is the number of source samples in the window; `anchor` is its zero-based
    /// origin offset along the filtered axis. The provided C1, C3, and C4 implementations
    /// write every channel, including the fourth channel in C4.
    /// Allocates a new owned image for the valid source window. Reduces height by
    /// `mask_size - 1` and preserves width.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// `mask_size` must be positive and `anchor` must be in `0..mask_size`.
    /// The source must contain the complete filter window and its valid output window.
    /// Image allocation or view construction can also fail.
    pub fn sum_window_column(
        self,
        mask_size: i32,
        anchor: i32,
    ) -> Result<ImagePipeline<'a, f32, L>> {
        self.sum_window_filter(
            FilterAxis::Column,
            mask_size,
            anchor,
            <Self as SumWindowFilterImage<T, L>>::sum_window_column_image,
        )
    }

    /// Sums each channel over a horizontal window into `f32` samples.
    /// `mask_size` is the number of source samples in the window; `anchor` is its zero-based
    /// origin offset along the filtered axis. The provided C1, C3, and C4 implementations
    /// write every channel, including the fourth channel in C4.
    /// Allocates a new owned image for the valid source window. Reduces width by
    /// `mask_size - 1` and preserves height.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// `mask_size` must be positive and `anchor` must be in `0..mask_size`.
    /// The source must contain the complete filter window and its valid output window.
    /// Image allocation or view construction can also fail.
    pub fn sum_window_row(self, mask_size: i32, anchor: i32) -> Result<ImagePipeline<'a, f32, L>> {
        self.sum_window_filter(
            FilterAxis::Row,
            mask_size,
            anchor,
            <Self as SumWindowFilterImage<T, L>>::sum_window_row_image,
        )
    }

    fn sum_window_filter(
        self,
        axis: FilterAxis,
        mask_size: i32,
        anchor: i32,
        filter: SumWindowFilterCallback<L, T>,
    ) -> Result<ImagePipeline<'a, f32, L>> {
        self.ensure_eager()?;
        let (mask, origin) = axis.geometry(
            usize::try_from(mask_size).map_err(|_| Error::out_of_range("filter taps"))?,
            anchor,
        )?;
        let source = self.view()?.valid_window(mask, origin)?;
        let mut destination =
            Image::<f32, L>::create_uninit(self.stream_context.cuda_context(), source.size())?;

        {
            let mut destination_view = destination.output()?;
            filter(
                self.stream_context,
                &source,
                &mut destination_view,
                mask_size,
                anchor,
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

type SumWindowFilterCallback<L, T> = fn(
    &StreamContext,
    &ImageWindow<'_, T, L>,
    &mut ImageOutputView<'_, f32, L>,
    i32,
    i32,
) -> Result<()>;
