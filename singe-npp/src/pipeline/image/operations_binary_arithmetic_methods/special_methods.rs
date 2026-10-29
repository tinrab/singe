use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{ChannelLayout, ImageOutputView, ImageView, ImageViewMut},
    },
    pipeline::image::{
        ImageBacking, ImagePipeline,
        operation_shapes::{
            ScaledSubtractImageInPlaceOperation, ScaledSubtractImageOperation,
            UnscaledBinaryImageInPlaceOperation, UnscaledBinaryImageOperation,
        },
        operation_traits::{MultiplyScaleImage, ScaledSubtractImage},
    },
};

mod multiply_scale_methods;
mod unscaled_methods;

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: ScaledSubtractImage<T, L>,
{
    /// Subtracts `left` from `right` for each processed channel.
    /// Applies native integer scaling and saturation using `scale_factor`.
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
    pub fn subtract_scaled_into(
        stream_context: &StreamContext,
        left: &ImageView<'_, T, L>,
        right: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as ScaledSubtractImage<T, L>>::subtract_scaled_image(
            stream_context,
            left,
            right,
            destination,
            scale_factor,
        )
    }

    /// Subtracts `source` from `source_destination` in place.
    /// Applies native integer scaling and saturation using `scale_factor`.
    /// Updates the supplied image in place. AC4 alpha remains unchanged.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// All participating image dimensions must match.
    pub fn subtract_scaled_in_place(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as ScaledSubtractImage<T, L>>::subtract_scaled_image_in_place(
            stream_context,
            source,
            source_destination,
            scale_factor,
        )
    }
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: ScaledSubtractImage<T, L>,
{
    /// Subtracts samples using the native operand order for the current backing.
    /// With owned backing, updates each pipeline sample as `self - other`. With borrowed
    /// backing, allocates output and computes `other - self`. Operand order therefore depends
    /// on the backing held by the pipeline.
    /// Applies native integer scaling and saturation using `scale_factor`.
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
    pub fn subtract_scaled(self, other: &ImageView<'_, T, L>, scale_factor: i32) -> Result<Self> {
        self.binary_arithmetic_scaled_subtract(
            other,
            scale_factor,
            <Self as ScaledSubtractImage<T, L>>::subtract_scaled_image,
            <Self as ScaledSubtractImage<T, L>>::subtract_scaled_image_in_place,
        )
    }

    fn binary_arithmetic_scaled_subtract(
        mut self,
        other: &ImageView<'_, T, L>,
        scale_factor: i32,
        operation: ScaledSubtractImageOperation<T, L>,
        operation_in_place: ScaledSubtractImageInPlaceOperation<T, L>,
    ) -> Result<Self> {
        self.ensure_eager()?;

        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                operation_in_place(self.stream_context, other, &mut image_view, scale_factor)?;
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
                operation(
                    self.stream_context,
                    source,
                    other,
                    &mut destination_view,
                    scale_factor,
                )?;
                // SAFETY: the completed operation above wrote every element.
                self.backing = ImageBacking::Owned(unsafe { destination.assume_init() });
            }
        }

        Ok(self)
    }
}
