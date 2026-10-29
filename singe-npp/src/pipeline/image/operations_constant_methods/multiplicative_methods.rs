use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageOutputView, ImageView, ImageViewMut},
    pipeline::image::{ImagePipeline, operation_traits::ConstantArithmeticImage},
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: ConstantArithmeticImage<T, L>,
    <Self as ConstantArithmeticImage<T, L>>::Constant: Copy,
{
    /// Multiplies each processed sample by the corresponding constant.
    /// The provided implementations use a scalar for C1, three values for C3 and AC4, and
    /// four values for C4. Half-precision images use `f32` constants. Integer implementations
    /// apply native scaling and saturation using `scale_factor`; floating-point and floating-point
    /// complex implementations ignore `scale_factor`.
    /// Writes active channels in the equally sized destination. AC4 alpha retains its
    /// previous initialization state; this call does not initialize alpha.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// All participating image dimensions must match.
    pub fn multiply_constant_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        constant: <Self as ConstantArithmeticImage<T, L>>::Constant,
        destination: &mut ImageOutputView<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as ConstantArithmeticImage<T, L>>::multiply_constant_image(
            stream_context,
            source,
            constant,
            destination,
            scale_factor,
        )
    }

    /// Multiplies each processed sample by the corresponding constant.
    /// The provided implementations use a scalar for C1, three values for C3 and AC4, and
    /// four values for C4. Half-precision images use `f32` constants. Integer implementations
    /// apply native scaling and saturation using `scale_factor`; floating-point and floating-point
    /// complex implementations ignore `scale_factor`.
    /// Updates the supplied image in place; AC4 alpha remains unchanged.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    pub fn multiply_constant_in_place(
        stream_context: &StreamContext,
        constant: <Self as ConstantArithmeticImage<T, L>>::Constant,
        source_destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as ConstantArithmeticImage<T, L>>::multiply_constant_image_in_place(
            stream_context,
            constant,
            source_destination,
            scale_factor,
        )
    }

    /// Divides each processed sample by the corresponding constant.
    /// The provided implementations use a scalar for C1, three values for C3 and AC4, and
    /// four values for C4. Half-precision images use `f32` constants. Integer implementations
    /// apply native scaling and saturation using `scale_factor`; floating-point and floating-point
    /// complex implementations ignore `scale_factor`.
    /// Writes active channels in the equally sized destination. AC4 alpha retains its
    /// previous initialization state; this call does not initialize alpha.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// All participating image dimensions must match.
    pub fn divide_constant_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        constant: <Self as ConstantArithmeticImage<T, L>>::Constant,
        destination: &mut ImageOutputView<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as ConstantArithmeticImage<T, L>>::divide_constant_image(
            stream_context,
            source,
            constant,
            destination,
            scale_factor,
        )
    }

    /// Divides each processed sample by the corresponding constant.
    /// The provided implementations use a scalar for C1, three values for C3 and AC4, and
    /// four values for C4. Half-precision images use `f32` constants. Integer implementations
    /// apply native scaling and saturation using `scale_factor`; floating-point and floating-point
    /// complex implementations ignore `scale_factor`.
    /// Updates the supplied image in place; AC4 alpha remains unchanged.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    pub fn divide_constant_in_place(
        stream_context: &StreamContext,
        constant: <Self as ConstantArithmeticImage<T, L>>::Constant,
        source_destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as ConstantArithmeticImage<T, L>>::divide_constant_image_in_place(
            stream_context,
            constant,
            source_destination,
            scale_factor,
        )
    }
}
