use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{ChannelLayout, ImageOutputView, ImageView, ImageViewMut},
    },
    pipeline::image::operations_binary_arithmetic_methods::special_methods::{
        ImageBacking, ImagePipeline, MultiplyScaleImage, UnscaledBinaryImageInPlaceOperation,
        UnscaledBinaryImageOperation,
    },
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: MultiplyScaleImage<T, L>,
{
    /// Multiplies corresponding samples for each processed channel.
    /// Uses NPP's fixed scaling by the maximum value of the unsigned sample bit width.
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
    pub fn multiply_scale_into(
        stream_context: &StreamContext,
        left: &ImageView<'_, T, L>,
        right: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()> {
        <Self as MultiplyScaleImage<T, L>>::multiply_scale_image(
            stream_context,
            left,
            right,
            destination,
        )
    }

    /// Multiplies corresponding samples in place.
    /// Uses NPP's fixed scaling by the maximum value of the unsigned sample bit width.
    /// Updates the supplied image in place. AC4 alpha remains unchanged.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// All participating image dimensions must match.
    pub fn multiply_scale_in_place(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()> {
        <Self as MultiplyScaleImage<T, L>>::multiply_scale_image_in_place(
            stream_context,
            source,
            source_destination,
        )
    }
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: MultiplyScaleImage<T, L>,
{
    /// Multiplies corresponding samples for each processed channel.
    /// Uses NPP's fixed scaling by the maximum value of the unsigned sample bit width.
    /// Reuses owned backing; borrowed input produces a new owned image.
    /// Preserves alpha from the pipeline image for AC4.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// All participating image dimensions must match.
    /// Allocation, image view construction, or alpha preservation copies can also fail.
    pub fn multiply_scale(self, other: &ImageView<'_, T, L>) -> Result<Self> {
        self.binary_arithmetic_unscaled(
            other,
            <Self as MultiplyScaleImage<T, L>>::multiply_scale_image,
            <Self as MultiplyScaleImage<T, L>>::multiply_scale_image_in_place,
        )
    }

    fn binary_arithmetic_unscaled(
        mut self,
        other: &ImageView<'_, T, L>,
        operation: UnscaledBinaryImageOperation<T, L>,
        operation_in_place: UnscaledBinaryImageInPlaceOperation<T, L>,
    ) -> Result<Self> {
        self.ensure_eager()?;

        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                operation_in_place(self.stream_context, other, &mut image_view)?;
            }
            ImageBacking::Borrowed(source) => {
                let mut destination = Image::<T, L>::create_uninit(
                    self.stream_context.cuda_context(),
                    source.size(),
                )?;
                let mut destination_view = destination.output()?;
                if L::IGNORES_ALPHA {
                    destination_view.copy_from(source)?;
                }
                operation(self.stream_context, source, other, &mut destination_view)?;
                // SAFETY: the completed operation above wrote every element.
                self.backing = ImageBacking::Owned(unsafe { destination.assume_init() });
            }
        }

        Ok(self)
    }
}
