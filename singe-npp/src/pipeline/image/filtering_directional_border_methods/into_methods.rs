use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageOutputView, ImageView},
    pipeline::image::{ImagePipeline, filtering::EdgeDirectionalBorderFilterImage},
    types::{BorderType, Point},
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: EdgeDirectionalBorderFilterImage<T, L>,
{
    /// Applies the native horizontal Prewitt filter with border control.
    /// Writes into `destination` without allocating output storage.
    /// The destination-sized ROI begins at `source_offset` within the full source image.
    /// Native border handling uses `border_type`.
    /// For AC4, destination alpha retains its previous initialization state; completion does
    /// not initialize that channel.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native filtering, or stream synchronization fail.
    /// The source offset must be valid, ROI endpoints must not overflow, and the destination-sized
    /// ROI must fit within the full source image.
    pub fn filter_prewitt_horizontal_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        border_type: BorderType,
    ) -> Result<()> {
        <Self as EdgeDirectionalBorderFilterImage<T, L>>::filter_prewitt_horizontal_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            border_type,
        )
    }

    /// Applies the native vertical Prewitt filter with border control.
    /// Writes into `destination` without allocating output storage.
    /// The destination-sized ROI begins at `source_offset` within the full source image.
    /// Native border handling uses `border_type`.
    /// For AC4, destination alpha retains its previous initialization state; completion does
    /// not initialize that channel.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native filtering, or stream synchronization fail.
    /// The source offset must be valid, ROI endpoints must not overflow, and the destination-sized
    /// ROI must fit within the full source image.
    pub fn filter_prewitt_vertical_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        border_type: BorderType,
    ) -> Result<()> {
        <Self as EdgeDirectionalBorderFilterImage<T, L>>::filter_prewitt_vertical_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            border_type,
        )
    }

    /// Applies the native down Roberts filter with border control.
    /// Writes into `destination` without allocating output storage.
    /// The destination-sized ROI begins at `source_offset` within the full source image.
    /// Native border handling uses `border_type`.
    /// For AC4, destination alpha retains its previous initialization state; completion does
    /// not initialize that channel.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native filtering, or stream synchronization fail.
    /// The source offset must be valid, ROI endpoints must not overflow, and the destination-sized
    /// ROI must fit within the full source image.
    pub fn filter_roberts_down_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        border_type: BorderType,
    ) -> Result<()> {
        <Self as EdgeDirectionalBorderFilterImage<T, L>>::filter_roberts_down_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            border_type,
        )
    }

    /// Applies the native up Roberts filter with border control.
    /// Writes into `destination` without allocating output storage.
    /// The destination-sized ROI begins at `source_offset` within the full source image.
    /// Native border handling uses `border_type`.
    /// For AC4, destination alpha retains its previous initialization state; completion does
    /// not initialize that channel.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native filtering, or stream synchronization fail.
    /// The source offset must be valid, ROI endpoints must not overflow, and the destination-sized
    /// ROI must fit within the full source image.
    pub fn filter_roberts_up_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        border_type: BorderType,
    ) -> Result<()> {
        <Self as EdgeDirectionalBorderFilterImage<T, L>>::filter_roberts_up_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            border_type,
        )
    }

    /// Applies the native horizontal Sobel filter with border control.
    /// Writes into `destination` without allocating output storage.
    /// The destination-sized ROI begins at `source_offset` within the full source image.
    /// Native border handling uses `border_type`.
    /// For AC4, destination alpha retains its previous initialization state; completion does
    /// not initialize that channel.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native filtering, or stream synchronization fail.
    /// The source offset must be valid, ROI endpoints must not overflow, and the destination-sized
    /// ROI must fit within the full source image.
    pub fn filter_sobel_horizontal_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        border_type: BorderType,
    ) -> Result<()> {
        <Self as EdgeDirectionalBorderFilterImage<T, L>>::filter_sobel_horizontal_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            border_type,
        )
    }

    /// Applies the native vertical Sobel filter with border control.
    /// Writes into `destination` without allocating output storage.
    /// The destination-sized ROI begins at `source_offset` within the full source image.
    /// Native border handling uses `border_type`.
    /// For AC4, destination alpha retains its previous initialization state; completion does
    /// not initialize that channel.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native filtering, or stream synchronization fail.
    /// The source offset must be valid, ROI endpoints must not overflow, and the destination-sized
    /// ROI must fit within the full source image.
    pub fn filter_sobel_vertical_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        border_type: BorderType,
    ) -> Result<()> {
        <Self as EdgeDirectionalBorderFilterImage<T, L>>::filter_sobel_vertical_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            border_type,
        )
    }
}
