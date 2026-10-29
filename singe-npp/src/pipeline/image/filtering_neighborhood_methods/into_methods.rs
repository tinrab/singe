use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageOutputView, ImageView, ImageWindow},
    pipeline::image::{
        ImagePipeline,
        filtering::{
            BoxBorderFilterImage, MedianBorderFilterImage, MedianFilterImage,
            NeighborhoodBorderFilterImage, NeighborhoodFilterImage,
        },
    },
    types::{BorderType, Point, Size},
};

mod border_into_methods;

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: MedianFilterImage<T, L>,
{
    /// Applies a neighborhood median filter.
    /// Writes into the supplied `destination` without allocating output image storage.
    /// The source window must retain enough parent-image halo for the `mask_size` neighborhood at `anchor`.
    /// Its pixel dimensions must match `destination`.
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
    /// An insufficient source halo or mismatched source-window and destination dimensions
    /// also returns an error.
    pub fn filter_median_into(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        mask_size: Size,
        anchor: Point,
    ) -> Result<()> {
        <Self as MedianFilterImage<T, L>>::filter_median_image(
            stream_context,
            source,
            destination,
            mask_size,
            anchor,
        )
    }
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: NeighborhoodFilterImage<T, L>,
{
    /// Applies a neighborhood maximum filter.
    /// Writes into the supplied `destination` without allocating output image storage.
    /// The source window must retain enough parent-image halo for the `mask_size` neighborhood at `anchor`.
    /// Its pixel dimensions must match `destination`.
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
    /// An insufficient source halo or mismatched source-window and destination dimensions
    /// also returns an error.
    pub fn filter_max_into(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        mask_size: Size,
        anchor: Point,
    ) -> Result<()> {
        <Self as NeighborhoodFilterImage<T, L>>::filter_max_image(
            stream_context,
            source,
            destination,
            mask_size,
            anchor,
        )
    }

    /// Applies a neighborhood minimum filter.
    /// Writes into the supplied `destination` without allocating output image storage.
    /// The source window must retain enough parent-image halo for the `mask_size` neighborhood at `anchor`.
    /// Its pixel dimensions must match `destination`.
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
    /// An insufficient source halo or mismatched source-window and destination dimensions
    /// also returns an error.
    pub fn filter_min_into(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        mask_size: Size,
        anchor: Point,
    ) -> Result<()> {
        <Self as NeighborhoodFilterImage<T, L>>::filter_min_image(
            stream_context,
            source,
            destination,
            mask_size,
            anchor,
        )
    }
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: NeighborhoodBorderFilterImage<T, L>,
{
    /// Applies a neighborhood maximum filter with border control.
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
    pub fn filter_max_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<()> {
        <Self as NeighborhoodBorderFilterImage<T, L>>::filter_max_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            mask_size,
            anchor,
            border_type,
        )
    }

    /// Applies a neighborhood minimum filter with border control.
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
    pub fn filter_min_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<()> {
        <Self as NeighborhoodBorderFilterImage<T, L>>::filter_min_border_image(
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
