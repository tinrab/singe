use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, ImageOutputView, ImageView, ImageViewMut},
    pipeline::image::{ImagePipeline, threshold_dispatch::FixedThresholdImage},
};

impl<T> ImagePipeline<'_, T, C1>
where
    T: Copy,
    Self: FixedThresholdImage<T>,
{
    /// Replaces samples strictly above their threshold with the threshold.
    /// Samples equal to the threshold or on its other side retain their source value.
    /// Uses scalar parameters for C1 images.
    /// Writes into the equally sized destination. Writes every logical output sample.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native threshold execution, or stream synchronization fail.
    /// All participating image dimensions must match.
    pub fn threshold_greater_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        destination: &mut ImageOutputView<'_, T, C1>,
        threshold: T,
    ) -> Result<()> {
        <Self as FixedThresholdImage<T>>::threshold_greater_image(
            stream_context,
            source,
            destination,
            threshold,
        )
    }

    /// Replaces samples strictly above their threshold with the threshold.
    /// Samples equal to the threshold or on its other side retain their source value.
    /// Uses scalar parameters for C1 images.
    /// Updates the supplied image in place.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native threshold execution, or stream synchronization fail.
    pub fn threshold_greater_in_place(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, T, C1>,
        threshold: T,
    ) -> Result<()> {
        <Self as FixedThresholdImage<T>>::threshold_greater_image_in_place(
            stream_context,
            image,
            threshold,
        )
    }

    /// Replaces samples strictly below their threshold with the threshold.
    /// Samples equal to the threshold or on its other side retain their source value.
    /// Uses scalar parameters for C1 images.
    /// Writes into the equally sized destination. Writes every logical output sample.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native threshold execution, or stream synchronization fail.
    /// All participating image dimensions must match.
    pub fn threshold_less_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        destination: &mut ImageOutputView<'_, T, C1>,
        threshold: T,
    ) -> Result<()> {
        <Self as FixedThresholdImage<T>>::threshold_less_image(
            stream_context,
            source,
            destination,
            threshold,
        )
    }

    /// Replaces samples strictly below their threshold with the threshold.
    /// Samples equal to the threshold or on its other side retain their source value.
    /// Uses scalar parameters for C1 images.
    /// Updates the supplied image in place.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native threshold execution, or stream synchronization fail.
    pub fn threshold_less_in_place(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, T, C1>,
        threshold: T,
    ) -> Result<()> {
        <Self as FixedThresholdImage<T>>::threshold_less_image_in_place(
            stream_context,
            image,
            threshold,
        )
    }
}
