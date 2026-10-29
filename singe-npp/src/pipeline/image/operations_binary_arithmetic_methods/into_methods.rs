use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageOutputView, ImageView, ImageViewMut},
    pipeline::image::{ImagePipeline, operation_traits::BinaryArithmeticImage},
};

mod multiplicative_methods;

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: BinaryArithmeticImage<T, L>,
{
    /// Adds corresponding samples for each processed channel.
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
    pub fn add_into(
        stream_context: &StreamContext,
        left: &ImageView<'_, T, L>,
        right: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as BinaryArithmeticImage<T, L>>::add_image(
            stream_context,
            left,
            right,
            destination,
            scale_factor,
        )
    }

    /// Adds corresponding samples in place.
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
    pub fn add_in_place(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as BinaryArithmeticImage<T, L>>::add_image_in_place(
            stream_context,
            source,
            source_destination,
            scale_factor,
        )
    }

    /// Subtracts `left` from `right` for each processed channel.
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
    pub fn subtract_into(
        stream_context: &StreamContext,
        left: &ImageView<'_, T, L>,
        right: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as BinaryArithmeticImage<T, L>>::subtract_image(
            stream_context,
            left,
            right,
            destination,
            scale_factor,
        )
    }

    /// Subtracts `source` from `source_destination` in place.
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
    pub fn subtract_in_place(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as BinaryArithmeticImage<T, L>>::subtract_image_in_place(
            stream_context,
            source,
            source_destination,
            scale_factor,
        )
    }
}
