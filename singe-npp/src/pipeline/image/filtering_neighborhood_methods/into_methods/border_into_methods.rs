use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageOutputView, ImageView},
    pipeline::image::filtering_neighborhood_methods::into_methods::{
        BoxBorderFilterImage, ImagePipeline, MedianBorderFilterImage,
    },
    types::{BorderType, Point, Size},
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: MedianBorderFilterImage<T, L>,
{
    /// Applies a neighborhood median filter with border control.
    /// Writes into the supplied `destination` without allocating output image storage.
    /// The destination-sized ROI starts at `source_offset` within the full source image.
    /// Border handling uses `border_type`.
    /// AC4 destination alpha retains its previous initialization state; completion does not
    /// initialize that channel.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Mask dimensions must be positive and `anchor` must lie inside the mask.
    /// Native scratch-size queries and temporary scratch allocation can also fail.
    /// The source offset must be valid, ROI endpoints must not overflow, and the destination-sized
    /// ROI must fit within the full source image.
    pub fn filter_median_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<()> {
        <Self as MedianBorderFilterImage<T, L>>::filter_median_border_image(
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

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: BoxBorderFilterImage<T, L>,
{
    /// Applies a box-average filter with border control.
    /// Writes into the supplied `destination` without allocating output image storage.
    /// The destination-sized ROI starts at `source_offset` within the full source image.
    /// Border handling uses `border_type`.
    /// AC4 destination alpha retains its previous initialization state; completion does not
    /// initialize that channel.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Mask dimensions must be positive and `anchor` must lie inside the mask.
    /// The source offset must be valid, ROI endpoints must not overflow, and the destination-sized
    /// ROI must fit within the full source image.
    pub fn filter_box_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<()> {
        <Self as BoxBorderFilterImage<T, L>>::filter_box_border_image(
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
