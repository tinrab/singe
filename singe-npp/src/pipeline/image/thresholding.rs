use crate::{
    error::Result,
    image::{Image, SupportedImage, view::C1},
    pipeline::image::{
        ImageBacking, ImagePipeline,
        threshold_dispatch::{ThresholdImage, ValueThresholdImage},
    },
    types::ComparisonOperation,
};

mod fixed_methods;
mod fixed_value_methods;
mod into_methods;

impl<T> ImagePipeline<'_, T, C1>
where
    T: Copy + SupportedImage<C1>,
    Self: ThresholdImage<T>,
{
    /// Replaces samples satisfying the selected native comparison with the corresponding threshold.
    /// Compares each source sample with its threshold. NPP supports `ComparisonOperation::Less`
    /// and `ComparisonOperation::Greater`; the wrapper forwards other selectors without
    /// validation. Samples not satisfying the comparison retain their source value.
    /// Uses scalar parameters for C1 images.
    /// Reuses owned backing; borrowed input produces a new owned image.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native threshold execution, or stream synchronization fail.
    /// NPP may reject unsupported comparison or threshold parameters.
    /// Image allocation, view construction can also fail.
    pub fn threshold(mut self, threshold: T, operation: ComparisonOperation) -> Result<Self> {
        self.ensure_eager()?;

        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                <Self as ThresholdImage<T>>::threshold_image_in_place(
                    self.stream_context,
                    &mut image_view,
                    threshold,
                    operation,
                )?;
            }
            ImageBacking::Borrowed(source) => {
                let mut destination = Image::<T, C1>::create_uninit(
                    self.stream_context.cuda_context(),
                    source.size(),
                )?;
                let mut destination_view = destination.output()?;
                <Self as ThresholdImage<T>>::threshold_image(
                    self.stream_context,
                    source,
                    &mut destination_view,
                    threshold,
                    operation,
                )?;
                // SAFETY: the completed operation above wrote every element.
                self.backing = ImageBacking::Owned(unsafe { destination.assume_init() });
            }
        }

        Ok(self)
    }
}

impl<T> ImagePipeline<'_, T, C1>
where
    T: Copy + SupportedImage<C1>,
    Self: ValueThresholdImage<T>,
{
    /// Replaces samples satisfying the selected native comparison with the corresponding replacement value.
    /// Compares each source sample with its threshold. NPP supports `ComparisonOperation::Less`
    /// and `ComparisonOperation::Greater`; the wrapper forwards other selectors without
    /// validation. Samples not satisfying the comparison retain their source value.
    /// Uses scalar parameters for C1 images.
    /// Reuses owned backing; borrowed input produces a new owned image.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native threshold execution, or stream synchronization fail.
    /// NPP may reject unsupported comparison or threshold parameters.
    /// Image allocation, view construction can also fail.
    pub fn threshold_value(
        mut self,
        threshold: T,
        value: T,
        operation: ComparisonOperation,
    ) -> Result<Self> {
        self.ensure_eager()?;

        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                <Self as ValueThresholdImage<T>>::threshold_value_image_in_place(
                    self.stream_context,
                    &mut image_view,
                    threshold,
                    value,
                    operation,
                )?;
            }
            ImageBacking::Borrowed(source) => {
                let mut destination = Image::<T, C1>::create_uninit(
                    self.stream_context.cuda_context(),
                    source.size(),
                )?;
                let mut destination_view = destination.output()?;
                <Self as ValueThresholdImage<T>>::threshold_value_image(
                    self.stream_context,
                    source,
                    &mut destination_view,
                    threshold,
                    value,
                    operation,
                )?;
                // SAFETY: the completed operation above wrote every element.
                self.backing = ImageBacking::Owned(unsafe { destination.assume_init() });
            }
        }

        Ok(self)
    }
}
