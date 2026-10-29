use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageOutputView, ImageView, ImageViewMut},
    pipeline::image::{ImagePipeline, operation_traits::LogicalBinaryImage},
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: LogicalBinaryImage<T, L>,
{
    /// Applies bitwise AND to corresponding active image samples of both inputs.
    /// Writes into `destination`. AC4 alpha retains its previous initialization state; this
    /// operation does not initialize that channel.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// All participating images must have matching pixel dimensions.
    pub fn logical_and_into(
        stream_context: &StreamContext,
        left: &ImageView<'_, T, L>,
        right: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()> {
        <Self as LogicalBinaryImage<T, L>>::logical_and_image(
            stream_context,
            left,
            right,
            destination,
        )
    }

    /// Applies bitwise AND to corresponding active image samples of both inputs.
    /// Updates `source_destination` in place. AC4 alpha remains unchanged.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// All participating images must have matching pixel dimensions.
    pub fn logical_and_in_place(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()> {
        <Self as LogicalBinaryImage<T, L>>::logical_and_image_in_place(
            stream_context,
            source,
            source_destination,
        )
    }

    /// Applies bitwise OR to corresponding active image samples of both inputs.
    /// Writes into `destination`. AC4 alpha retains its previous initialization state; this
    /// operation does not initialize that channel.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// All participating images must have matching pixel dimensions.
    pub fn logical_or_into(
        stream_context: &StreamContext,
        left: &ImageView<'_, T, L>,
        right: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()> {
        <Self as LogicalBinaryImage<T, L>>::logical_or_image(
            stream_context,
            left,
            right,
            destination,
        )
    }

    /// Applies bitwise OR to corresponding active image samples of both inputs.
    /// Updates `source_destination` in place. AC4 alpha remains unchanged.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// All participating images must have matching pixel dimensions.
    pub fn logical_or_in_place(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()> {
        <Self as LogicalBinaryImage<T, L>>::logical_or_image_in_place(
            stream_context,
            source,
            source_destination,
        )
    }

    /// Applies bitwise XOR to corresponding active image samples of both inputs.
    /// Writes into `destination`. AC4 alpha retains its previous initialization state; this
    /// operation does not initialize that channel.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// All participating images must have matching pixel dimensions.
    pub fn logical_xor_into(
        stream_context: &StreamContext,
        left: &ImageView<'_, T, L>,
        right: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()> {
        <Self as LogicalBinaryImage<T, L>>::logical_xor_image(
            stream_context,
            left,
            right,
            destination,
        )
    }

    /// Applies bitwise XOR to corresponding active image samples of both inputs.
    /// Updates `source_destination` in place. AC4 alpha remains unchanged.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// All participating images must have matching pixel dimensions.
    pub fn logical_xor_in_place(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()> {
        <Self as LogicalBinaryImage<T, L>>::logical_xor_image_in_place(
            stream_context,
            source,
            source_destination,
        )
    }
}
