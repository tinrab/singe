use crate::{
    error::Result,
    image::{Image, SupportedImage, view::C1},
    pipeline::image::{ImageBacking, ImagePipeline, threshold_dispatch::FixedValueThresholdImage},
};

impl<T> ImagePipeline<'_, T, C1>
where
    T: Copy + SupportedImage<C1>,
    Self: FixedValueThresholdImage<T>,
{
    /// Replaces samples strictly above their threshold with the corresponding replacement value.
    /// Samples equal to the threshold or on its other side retain their source value.
    /// Uses scalar parameters for C1 images.
    /// Reuses owned backing; borrowed input produces a new owned image.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native threshold execution, or stream synchronization fail.
    /// Image allocation, view construction can also fail.
    pub fn threshold_greater_value(mut self, threshold: T, value: T) -> Result<Self> {
        self.ensure_eager()?;

        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                <Self as FixedValueThresholdImage<T>>::threshold_greater_value_image_in_place(
                    self.stream_context,
                    &mut image_view,
                    threshold,
                    value,
                )?;
            }
            ImageBacking::Borrowed(source) => {
                let mut destination = Image::<T, C1>::create_uninit(
                    self.stream_context.cuda_context(),
                    source.size(),
                )?;
                let mut destination_view = destination.output()?;
                <Self as FixedValueThresholdImage<T>>::threshold_greater_value_image(
                    self.stream_context,
                    source,
                    &mut destination_view,
                    threshold,
                    value,
                )?;
                // SAFETY: the completed operation above wrote every element.
                self.backing = ImageBacking::Owned(unsafe { destination.assume_init() });
            }
        }

        Ok(self)
    }

    /// Replaces samples strictly below their threshold with the corresponding replacement value.
    /// Samples equal to the threshold or on its other side retain their source value.
    /// Uses scalar parameters for C1 images.
    /// Reuses owned backing; borrowed input produces a new owned image.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native threshold execution, or stream synchronization fail.
    /// Image allocation, view construction can also fail.
    pub fn threshold_less_value(mut self, threshold: T, value: T) -> Result<Self> {
        self.ensure_eager()?;

        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                <Self as FixedValueThresholdImage<T>>::threshold_less_value_image_in_place(
                    self.stream_context,
                    &mut image_view,
                    threshold,
                    value,
                )?;
            }
            ImageBacking::Borrowed(source) => {
                let mut destination = Image::<T, C1>::create_uninit(
                    self.stream_context.cuda_context(),
                    source.size(),
                )?;
                let mut destination_view = destination.output()?;
                <Self as FixedValueThresholdImage<T>>::threshold_less_value_image(
                    self.stream_context,
                    source,
                    &mut destination_view,
                    threshold,
                    value,
                )?;
                // SAFETY: the completed operation above wrote every element.
                self.backing = ImageBacking::Owned(unsafe { destination.assume_init() });
            }
        }

        Ok(self)
    }
}
