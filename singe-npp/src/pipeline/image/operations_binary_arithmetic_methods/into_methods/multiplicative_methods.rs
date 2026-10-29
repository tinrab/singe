use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageOutputView, ImageView, ImageViewMut},
    pipeline::image::{ImagePipeline, operation_traits::BinaryArithmeticImage},
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: BinaryArithmeticImage<T, L>,
{
    /// Multiplies corresponding samples for each processed channel.
    /// Applies native integer scaling and saturation using `scale_factor`.
    /// The provided floating-point and floating-point complex implementations ignore `scale_factor`.
    /// Writes active channels in the supplied destination. AC4 destination alpha retains its
    /// previous initialization state; this call does not initialize alpha.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// All participating image dimensions must match.
    pub fn multiply_into(
        stream_context: &StreamContext,
        left: &ImageView<'_, T, L>,
        right: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as BinaryArithmeticImage<T, L>>::multiply_image(
            stream_context,
            left,
            right,
            destination,
            scale_factor,
        )
    }

    /// Multiplies corresponding samples in place.
    /// Applies native integer scaling and saturation using `scale_factor`.
    /// The provided floating-point and floating-point complex implementations ignore `scale_factor`.
    /// Updates the supplied image in place. AC4 alpha remains unchanged.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// All participating image dimensions must match.
    pub fn multiply_in_place(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as BinaryArithmeticImage<T, L>>::multiply_image_in_place(
            stream_context,
            source,
            source_destination,
            scale_factor,
        )
    }

    /// Divides `right` by `left` for each processed channel.
    /// Applies native integer scaling and saturation using `scale_factor`.
    /// The provided floating-point and floating-point complex implementations ignore `scale_factor`.
    /// Writes active channels in the supplied destination. AC4 destination alpha retains its
    /// previous initialization state; this call does not initialize alpha.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// All participating image dimensions must match.
    pub fn divide_into(
        stream_context: &StreamContext,
        left: &ImageView<'_, T, L>,
        right: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as BinaryArithmeticImage<T, L>>::divide_image(
            stream_context,
            left,
            right,
            destination,
            scale_factor,
        )
    }

    /// Divides `source_destination` by `source` in place.
    /// Applies native integer scaling and saturation using `scale_factor`.
    /// The provided floating-point and floating-point complex implementations ignore `scale_factor`.
    /// Updates the supplied image in place. AC4 alpha remains unchanged.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// All participating image dimensions must match.
    pub fn divide_in_place(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as BinaryArithmeticImage<T, L>>::divide_image_in_place(
            stream_context,
            source,
            source_destination,
            scale_factor,
        )
    }
}
