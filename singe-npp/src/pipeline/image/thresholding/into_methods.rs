use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, ImageOutputView, ImageView, ImageViewMut},
    pipeline::image::{
        ImagePipeline,
        threshold_dispatch::{ThresholdImage, ValueThresholdImage},
    },
    types::ComparisonOperation,
};

mod fixed_into_methods;
mod fixed_value_into_methods;

impl<T> ImagePipeline<'_, T, C1>
where
    T: Copy,
    Self: ThresholdImage<T>,
{
    /// Replaces samples satisfying the selected native comparison with the corresponding threshold.
    /// Compares each source sample with its threshold. NPP supports `ComparisonOperation::Less`
    /// and `ComparisonOperation::Greater`; the wrapper forwards other selectors without
    /// validation. Samples not satisfying the comparison retain their source value.
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
    /// NPP may reject unsupported comparison or threshold parameters.
    pub fn threshold_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        destination: &mut ImageOutputView<'_, T, C1>,
        threshold: T,
        operation: ComparisonOperation,
    ) -> Result<()> {
        <Self as ThresholdImage<T>>::threshold_image(
            stream_context,
            source,
            destination,
            threshold,
            operation,
        )
    }

    /// Replaces samples satisfying the selected native comparison with the corresponding threshold.
    /// Compares each source sample with its threshold. NPP supports `ComparisonOperation::Less`
    /// and `ComparisonOperation::Greater`; the wrapper forwards other selectors without
    /// validation. Samples not satisfying the comparison retain their source value.
    /// Uses scalar parameters for C1 images.
    /// Updates the supplied image in place.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native threshold execution, or stream synchronization fail.
    /// NPP may reject unsupported comparison or threshold parameters.
    pub fn threshold_in_place(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, T, C1>,
        threshold: T,
        operation: ComparisonOperation,
    ) -> Result<()> {
        <Self as ThresholdImage<T>>::threshold_image_in_place(
            stream_context,
            image,
            threshold,
            operation,
        )
    }
}

impl<T> ImagePipeline<'_, T, C1>
where
    T: Copy,
    Self: ValueThresholdImage<T>,
{
    /// Replaces samples satisfying the selected native comparison with the corresponding replacement value.
    /// Compares each source sample with its threshold. NPP supports `ComparisonOperation::Less`
    /// and `ComparisonOperation::Greater`; the wrapper forwards other selectors without
    /// validation. Samples not satisfying the comparison retain their source value.
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
    /// NPP may reject unsupported comparison or threshold parameters.
    pub fn threshold_value_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        destination: &mut ImageOutputView<'_, T, C1>,
        threshold: T,
        value: T,
        operation: ComparisonOperation,
    ) -> Result<()> {
        <Self as ValueThresholdImage<T>>::threshold_value_image(
            stream_context,
            source,
            destination,
            threshold,
            value,
            operation,
        )
    }

    /// Replaces samples satisfying the selected native comparison with the corresponding replacement value.
    /// Compares each source sample with its threshold. NPP supports `ComparisonOperation::Less`
    /// and `ComparisonOperation::Greater`; the wrapper forwards other selectors without
    /// validation. Samples not satisfying the comparison retain their source value.
    /// Uses scalar parameters for C1 images.
    /// Updates the supplied image in place.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native threshold execution, or stream synchronization fail.
    /// NPP may reject unsupported comparison or threshold parameters.
    pub fn threshold_value_in_place(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, T, C1>,
        threshold: T,
        value: T,
        operation: ComparisonOperation,
    ) -> Result<()> {
        <Self as ValueThresholdImage<T>>::threshold_value_image_in_place(
            stream_context,
            image,
            threshold,
            value,
            operation,
        )
    }
}
