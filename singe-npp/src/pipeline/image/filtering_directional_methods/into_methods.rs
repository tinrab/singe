use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageOutputView, ImageWindow},
    pipeline::image::{ImagePipeline, filtering::EdgeDirectionalFilterImage},
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: EdgeDirectionalFilterImage<T, L>,
{
    /// Applies the native horizontal Prewitt filter.
    /// Writes into `destination` without allocating output storage.
    /// The source window and destination must have equal dimensions. The source must retain
    /// a centered 3-by-3 halo in its parent image, as checked by the dispatch implementation.
    /// For AC4, destination alpha retains its previous initialization state; completion does
    /// not initialize that channel.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native filtering, or stream synchronization fail.
    /// An insufficient source halo or mismatched source-window and destination dimensions
    /// also returns an error.
    pub fn filter_prewitt_horizontal_into(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()> {
        <Self as EdgeDirectionalFilterImage<T, L>>::filter_prewitt_horizontal_image(
            stream_context,
            source,
            destination,
        )
    }

    /// Applies the native vertical Prewitt filter.
    /// Writes into `destination` without allocating output storage.
    /// The source window and destination must have equal dimensions. The source must retain
    /// a centered 3-by-3 halo in its parent image, as checked by the dispatch implementation.
    /// For AC4, destination alpha retains its previous initialization state; completion does
    /// not initialize that channel.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native filtering, or stream synchronization fail.
    /// An insufficient source halo or mismatched source-window and destination dimensions
    /// also returns an error.
    pub fn filter_prewitt_vertical_into(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()> {
        <Self as EdgeDirectionalFilterImage<T, L>>::filter_prewitt_vertical_image(
            stream_context,
            source,
            destination,
        )
    }

    /// Applies the native down Roberts filter.
    /// Writes into `destination` without allocating output storage.
    /// The source window and destination must have equal dimensions. The source must retain
    /// a centered 3-by-3 halo in its parent image, as checked by the dispatch implementation.
    /// For AC4, destination alpha retains its previous initialization state; completion does
    /// not initialize that channel.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native filtering, or stream synchronization fail.
    /// An insufficient source halo or mismatched source-window and destination dimensions
    /// also returns an error.
    pub fn filter_roberts_down_into(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()> {
        <Self as EdgeDirectionalFilterImage<T, L>>::filter_roberts_down_image(
            stream_context,
            source,
            destination,
        )
    }

    /// Applies the native up Roberts filter.
    /// Writes into `destination` without allocating output storage.
    /// The source window and destination must have equal dimensions. The source must retain
    /// a centered 3-by-3 halo in its parent image, as checked by the dispatch implementation.
    /// For AC4, destination alpha retains its previous initialization state; completion does
    /// not initialize that channel.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native filtering, or stream synchronization fail.
    /// An insufficient source halo or mismatched source-window and destination dimensions
    /// also returns an error.
    pub fn filter_roberts_up_into(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()> {
        <Self as EdgeDirectionalFilterImage<T, L>>::filter_roberts_up_image(
            stream_context,
            source,
            destination,
        )
    }

    /// Applies the native horizontal Sobel filter.
    /// Writes into `destination` without allocating output storage.
    /// The source window and destination must have equal dimensions. The source must retain
    /// a centered 3-by-3 halo in its parent image, as checked by the dispatch implementation.
    /// For AC4, destination alpha retains its previous initialization state; completion does
    /// not initialize that channel.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native filtering, or stream synchronization fail.
    /// An insufficient source halo or mismatched source-window and destination dimensions
    /// also returns an error.
    pub fn filter_sobel_horizontal_into(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()> {
        <Self as EdgeDirectionalFilterImage<T, L>>::filter_sobel_horizontal_image(
            stream_context,
            source,
            destination,
        )
    }

    /// Applies the native vertical Sobel filter.
    /// Writes into `destination` without allocating output storage.
    /// The source window and destination must have equal dimensions. The source must retain
    /// a centered 3-by-3 halo in its parent image, as checked by the dispatch implementation.
    /// For AC4, destination alpha retains its previous initialization state; completion does
    /// not initialize that channel.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native filtering, or stream synchronization fail.
    /// An insufficient source halo or mismatched source-window and destination dimensions
    /// also returns an error.
    pub fn filter_sobel_vertical_into(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()> {
        <Self as EdgeDirectionalFilterImage<T, L>>::filter_sobel_vertical_image(
            stream_context,
            source,
            destination,
        )
    }
}
