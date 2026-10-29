use crate::{
    context::StreamContext,
    error::Result,
    image::{
        SupportedImage,
        view::{ChannelLayout, ImageOutputView, ImageView},
    },
    pipeline::image::{ImagePipeline, operation_traits::UnscaledBinaryArithmeticImage},
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: UnscaledBinaryArithmeticImage<T, L>,
{
    /// Multiplies corresponding samples for each processed channel.
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
    pub fn multiply_unscaled_into(
        stream_context: &StreamContext,
        left: &ImageView<'_, T, L>,
        right: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()> {
        <Self as UnscaledBinaryArithmeticImage<T, L>>::multiply_unscaled_image(
            stream_context,
            left,
            right,
            destination,
        )
    }

    /// Divides `right` by `left` for each processed channel.
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
    pub fn divide_unscaled_into(
        stream_context: &StreamContext,
        left: &ImageView<'_, T, L>,
        right: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()> {
        <Self as UnscaledBinaryArithmeticImage<T, L>>::divide_unscaled_image(
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
    /// Multiplies corresponding samples for each processed channel.
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
    pub fn multiply_unscaled(self, other: &ImageView<'_, T, L>) -> Result<Self> {
        self.binary_arithmetic_out_of_place(
            other,
            <Self as UnscaledBinaryArithmeticImage<T, L>>::multiply_unscaled_image,
        )
    }

    /// Divides `other` by the pipeline samples for each processed channel.
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
    pub fn divide_unscaled(self, other: &ImageView<'_, T, L>) -> Result<Self> {
        self.binary_arithmetic_out_of_place(
            other,
            <Self as UnscaledBinaryArithmeticImage<T, L>>::divide_unscaled_image,
        )
    }
}
