use crate::{
    context::StreamContext,
    error::Result,
    image::{
        SupportedImage,
        view::{ChannelLayout, ImageOutputView, ImageView, ImageViewMut},
    },
    pipeline::image::{ImagePipeline, operation_traits::UnaryTranscendentalImage},
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: UnaryTranscendentalImage<T, L>,
{
    /// Computes the exponential of each processed channel value.
    ///
    /// For integer images, `scale_factor` controls native integer scaling and saturation.
    /// For `f32` images, the provided implementations ignore `scale_factor` and use the
    /// native floating-point operation. Source values are forwarded without domain checks.
    /// All participating images must have matching pixel dimensions. For AC4, writes only
    /// the color channels and leaves destination alpha untouched.
    /// Row padding remains unwritten. Prepares participating regions, waits for producers
    /// on other streams, rejects capture, and returns after successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native execution, or stream synchronization fail.
    /// Participating image dimensions must match.
    pub fn exponential_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as UnaryTranscendentalImage<T, L>>::exponential_image(
            stream_context,
            source,
            destination,
            scale_factor,
        )
    }

    /// Computes the exponential of each processed channel value.
    ///
    /// For integer images, `scale_factor` controls native integer scaling and saturation.
    /// For `f32` images, the provided implementations ignore `scale_factor` and use the
    /// native floating-point operation. Source values are forwarded without domain checks.
    /// Updates the image in place and preserves AC4 alpha.
    /// Row padding remains unwritten. Prepares participating regions, waits for producers
    /// on other streams, rejects capture, and returns after successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native execution, or stream synchronization fail.
    pub fn exponential_in_place(
        stream_context: &StreamContext,
        source_destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as UnaryTranscendentalImage<T, L>>::exponential_image_in_place(
            stream_context,
            source_destination,
            scale_factor,
        )
    }
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: UnaryTranscendentalImage<T, L>,
{
    /// Computes the exponential of each processed channel value.
    ///
    /// For integer images, `scale_factor` controls native integer scaling and saturation.
    /// For `f32` images, the provided implementations ignore `scale_factor` and use the
    /// native floating-point operation. Source values are forwarded without domain checks.
    /// Reuses an owned pipeline image in place; for borrowed input, allocates a new
    /// source-sized image.
    /// For AC4, preserves source alpha, copying the source before out-of-place execution
    /// when necessary.
    /// Row padding remains unwritten. Prepares participating regions, waits for producers
    /// on other streams, rejects capture, and returns after successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native execution, or stream synchronization fail.
    /// Image view construction, output allocation, or copying source alpha can also fail.
    pub fn exponential(self, scale_factor: i32) -> Result<Self> {
        self.unary_transcendental(
            scale_factor,
            <Self as UnaryTranscendentalImage<T, L>>::exponential_image,
            <Self as UnaryTranscendentalImage<T, L>>::exponential_image_in_place,
        )
    }
}
