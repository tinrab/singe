use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{ChannelLayout, ImageOutputView, ImageView},
    },
    pipeline::image::{
        ImageBacking, ImagePipeline, operation_shapes::OutOfPlaceBinaryImageOperation,
        operation_traits::UnscaledBinaryArithmeticImage,
    },
};

mod multiplicative_methods;

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: UnscaledBinaryArithmeticImage<T, L>,
{
    /// Adds corresponding samples for each processed channel.
    /// Uses the native unscaled operation; no integer scale factor is applied.
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
    pub fn add_unscaled_into(
        stream_context: &StreamContext,
        left: &ImageView<'_, T, L>,
        right: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()> {
        <Self as UnscaledBinaryArithmeticImage<T, L>>::add_unscaled_image(
            stream_context,
            left,
            right,
            destination,
        )
    }

    /// Subtracts `left` from `right` for each processed channel.
    /// Uses the native unscaled operation; no integer scale factor is applied.
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
    pub fn subtract_unscaled_into(
        stream_context: &StreamContext,
        left: &ImageView<'_, T, L>,
        right: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()> {
        <Self as UnscaledBinaryArithmeticImage<T, L>>::subtract_unscaled_image(
            stream_context,
            left,
            right,
            destination,
        )
    }
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: UnscaledBinaryArithmeticImage<T, L>,
{
    /// Adds corresponding samples for each processed channel.
    /// Uses the native unscaled operation; no integer scale factor is applied.
    /// Always allocates a new owned image, including when the input already owns storage.
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
    pub fn add_unscaled(self, other: &ImageView<'_, T, L>) -> Result<Self> {
        self.binary_arithmetic_out_of_place(
            other,
            <Self as UnscaledBinaryArithmeticImage<T, L>>::add_unscaled_image,
        )
    }

    /// Subtracts the pipeline samples from `other` for each processed channel.
    /// Uses the native unscaled operation; no integer scale factor is applied.
    /// Always allocates a new owned image, including when the input already owns storage.
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
    pub fn subtract_unscaled(self, other: &ImageView<'_, T, L>) -> Result<Self> {
        self.binary_arithmetic_out_of_place(
            other,
            <Self as UnscaledBinaryArithmeticImage<T, L>>::subtract_unscaled_image,
        )
    }

    pub(crate) fn binary_arithmetic_out_of_place(
        mut self,
        other: &ImageView<'_, T, L>,
        operation: OutOfPlaceBinaryImageOperation<T, L>,
    ) -> Result<Self> {
        self.ensure_eager()?;

        let size = self.size();
        let mut destination =
            Image::<T, L>::create_uninit(self.stream_context.cuda_context(), size)?;
        let source = self.view()?;
        let mut destination_view = destination.output()?;
        if L::IGNORES_ALPHA {
            destination_view.copy_from(&source)?;
        }
        operation(self.stream_context, &source, other, &mut destination_view)?;
        // SAFETY: the completed operation above wrote every element.
        self.backing = ImageBacking::Owned(unsafe { destination.assume_init() });
        Ok(self)
    }
}
