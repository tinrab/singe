use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{C1, ImageOutputView, ImageView, ImageViewMut},
    },
    pipeline::image::{
        ImageBacking, ImagePipeline, threshold_dispatch::LessGreaterValueThresholdImage,
    },
};

impl<T> ImagePipeline<'_, T, C1>
where
    T: Copy,
    Self: LessGreaterValueThresholdImage<T>,
{
    /// Replaces samples strictly outside the lower and upper thresholds with their corresponding values.
    /// Uses the lower replacement for samples below the lower threshold, or otherwise
    /// the upper replacement for samples above the upper threshold. Other samples retain
    /// the source value. Threshold ordering is forwarded to NPP without validation.
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
    pub fn threshold_less_greater_value_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        destination: &mut ImageOutputView<'_, T, C1>,
        lower_threshold: T,
        lower_value: T,
        upper_threshold: T,
        upper_value: T,
    ) -> Result<()> {
        <Self as LessGreaterValueThresholdImage<T>>::threshold_less_greater_value_image(
            stream_context,
            source,
            destination,
            lower_threshold,
            lower_value,
            upper_threshold,
            upper_value,
        )
    }

    /// Replaces samples strictly outside the lower and upper thresholds with their corresponding values.
    /// Uses the lower replacement for samples below the lower threshold, or otherwise
    /// the upper replacement for samples above the upper threshold. Other samples retain
    /// the source value. Threshold ordering is forwarded to NPP without validation.
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
    pub fn threshold_less_greater_value_in_place(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, T, C1>,
        lower_threshold: T,
        lower_value: T,
        upper_threshold: T,
        upper_value: T,
    ) -> Result<()> {
        <Self as LessGreaterValueThresholdImage<T>>::threshold_less_greater_value_image_in_place(
            stream_context,
            image,
            lower_threshold,
            lower_value,
            upper_threshold,
            upper_value,
        )
    }
}

impl<T> ImagePipeline<'_, T, C1>
where
    T: Copy + SupportedImage<C1>,
    Self: LessGreaterValueThresholdImage<T>,
{
    /// Replaces samples strictly outside the lower and upper thresholds with their corresponding values.
    /// Uses the lower replacement for samples below the lower threshold, or otherwise
    /// the upper replacement for samples above the upper threshold. Other samples retain
    /// the source value. Threshold ordering is forwarded to NPP without validation.
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
    pub fn threshold_less_greater_value(
        mut self,
        lower_threshold: T,
        lower_value: T,
        upper_threshold: T,
        upper_value: T,
    ) -> Result<Self> {
        self.ensure_eager()?;

        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                <Self as LessGreaterValueThresholdImage<
                    T,
                >>::threshold_less_greater_value_image_in_place(
                    self.stream_context,
                    &mut image_view,
                    lower_threshold,
                    lower_value,
                    upper_threshold,
                    upper_value,
                )?;
            }
            ImageBacking::Borrowed(source) => {
                let mut destination = Image::<T, C1>::create_uninit(
                    self.stream_context.cuda_context(),
                    source.size(),
                )?;
                let mut destination_view = destination.output()?;
                <Self as LessGreaterValueThresholdImage<T>>::threshold_less_greater_value_image(
                    self.stream_context,
                    source,
                    &mut destination_view,
                    lower_threshold,
                    lower_value,
                    upper_threshold,
                    upper_value,
                )?;
                // SAFETY: the completed operation above wrote every element.
                self.backing = ImageBacking::Owned(unsafe { destination.assume_init() });
            }
        }

        Ok(self)
    }
}
