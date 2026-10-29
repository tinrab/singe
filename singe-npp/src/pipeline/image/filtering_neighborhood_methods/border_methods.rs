use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{ChannelLayout, ImageOutputView, ImageView},
    },
    pipeline::image::{
        ImageBacking, filtering::NeighborhoodBorderFilterImage,
        filtering_neighborhood_methods::ImagePipeline,
    },
    types::{BorderType, Point, Size},
};

mod median_box_border_methods;

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: NeighborhoodBorderFilterImage<T, L>,
{
    /// Applies a neighborhood maximum filter with border control.
    /// Allocates a new owned image covering the source window from `source_offset` to the
    /// image's lower-right corner. Border handling uses `border_type`.
    /// For AC4, copies the corresponding source window first to preserve source alpha.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Mask dimensions must be positive and `anchor` must lie inside the mask.
    /// Selecting the source window, allocating output, constructing views, or preserving source
    /// alpha can also fail.
    pub fn filter_max_border(
        self,
        source_offset: Point,
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<Self> {
        self.neighborhood_border_filter(
            source_offset,
            mask_size,
            anchor,
            border_type,
            <Self as NeighborhoodBorderFilterImage<T, L>>::filter_max_border_image,
        )
    }

    /// Applies a neighborhood minimum filter with border control.
    /// Allocates a new owned image covering the source window from `source_offset` to the
    /// image's lower-right corner. Border handling uses `border_type`.
    /// For AC4, copies the corresponding source window first to preserve source alpha.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Mask dimensions must be positive and `anchor` must lie inside the mask.
    /// Selecting the source window, allocating output, constructing views, or preserving source
    /// alpha can also fail.
    pub fn filter_min_border(
        self,
        source_offset: Point,
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<Self> {
        self.neighborhood_border_filter(
            source_offset,
            mask_size,
            anchor,
            border_type,
            <Self as NeighborhoodBorderFilterImage<T, L>>::filter_min_border_image,
        )
    }

    fn neighborhood_border_filter(
        self,
        source_offset: Point,
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
        filter: NeighborhoodBorderFilterCallback<L, T>,
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

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            // SAFETY: the completed operation above wrote every element.
            backing: ImageBacking::Owned(unsafe { destination.assume_init() }),
        })
    }
}

type NeighborhoodBorderFilterCallback<L, T> = fn(
    &StreamContext,
    &ImageView<'_, T, L>,
    Point,
    &mut ImageOutputView<'_, T, L>,
    Size,
    Point,
    BorderType,
) -> Result<()>;
