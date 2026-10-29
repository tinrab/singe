use crate::{
    error::Result,
    image::{Image, SupportedImage, view::ChannelLayout},
    pipeline::image::{
        ImageBacking, ImagePipeline,
        operation_shapes::{ConstantScaleInPlaceOperation, ConstantScaleOperation},
        operation_traits::ConstantArithmeticImage,
    },
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: ConstantArithmeticImage<T, L>,
    <Self as ConstantArithmeticImage<T, L>>::Constant: Copy,
{
    /// Adds the corresponding constant to each processed sample.
    /// The provided implementations use a scalar for C1, three values for C3 and AC4, and
    /// four values for C4. Half-precision images use `f32` constants. Integer implementations
    /// apply native scaling and saturation using `scale_factor`; floating-point and floating-point
    /// complex implementations ignore `scale_factor`.
    /// Reuses owned backing; borrowed input produces a new owned image. Preserves source
    /// alpha for AC4.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// All participating image dimensions must match.
    /// Image allocation, view construction, or alpha preservation copies can also fail.
    pub fn add_constant(
        self,
        constant: <Self as ConstantArithmeticImage<T, L>>::Constant,
        scale_factor: i32,
    ) -> Result<Self> {
        self.constant_arithmetic(
            constant,
            scale_factor,
            <Self as ConstantArithmeticImage<T, L>>::add_constant_image,
            <Self as ConstantArithmeticImage<T, L>>::add_constant_image_in_place,
        )
    }

    /// Subtracts the corresponding constant from each processed sample.
    /// The provided implementations use a scalar for C1, three values for C3 and AC4, and
    /// four values for C4. Half-precision images use `f32` constants. Integer implementations
    /// apply native scaling and saturation using `scale_factor`; floating-point and floating-point
    /// complex implementations ignore `scale_factor`.
    /// Reuses owned backing; borrowed input produces a new owned image. Preserves source
    /// alpha for AC4.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// All participating image dimensions must match.
    /// Image allocation, view construction, or alpha preservation copies can also fail.
    pub fn subtract_constant(
        self,
        constant: <Self as ConstantArithmeticImage<T, L>>::Constant,
        scale_factor: i32,
    ) -> Result<Self> {
        self.constant_arithmetic(
            constant,
            scale_factor,
            <Self as ConstantArithmeticImage<T, L>>::subtract_constant_image,
            <Self as ConstantArithmeticImage<T, L>>::subtract_constant_image_in_place,
        )
    }

    /// Multiplies each processed sample by the corresponding constant.
    /// The provided implementations use a scalar for C1, three values for C3 and AC4, and
    /// four values for C4. Half-precision images use `f32` constants. Integer implementations
    /// apply native scaling and saturation using `scale_factor`; floating-point and floating-point
    /// complex implementations ignore `scale_factor`.
    /// Reuses owned backing; borrowed input produces a new owned image. Preserves source
    /// alpha for AC4.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// All participating image dimensions must match.
    /// Image allocation, view construction, or alpha preservation copies can also fail.
    pub fn multiply_constant(
        self,
        constant: <Self as ConstantArithmeticImage<T, L>>::Constant,
        scale_factor: i32,
    ) -> Result<Self> {
        self.constant_arithmetic(
            constant,
            scale_factor,
            <Self as ConstantArithmeticImage<T, L>>::multiply_constant_image,
            <Self as ConstantArithmeticImage<T, L>>::multiply_constant_image_in_place,
        )
    }

    /// Divides each processed sample by the corresponding constant.
    /// The provided implementations use a scalar for C1, three values for C3 and AC4, and
    /// four values for C4. Half-precision images use `f32` constants. Integer implementations
    /// apply native scaling and saturation using `scale_factor`; floating-point and floating-point
    /// complex implementations ignore `scale_factor`.
    /// Reuses owned backing; borrowed input produces a new owned image. Preserves source
    /// alpha for AC4.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// All participating image dimensions must match.
    /// Image allocation, view construction, or alpha preservation copies can also fail.
    pub fn divide_constant(
        self,
        constant: <Self as ConstantArithmeticImage<T, L>>::Constant,
        scale_factor: i32,
    ) -> Result<Self> {
        self.constant_arithmetic(
            constant,
            scale_factor,
            <Self as ConstantArithmeticImage<T, L>>::divide_constant_image,
            <Self as ConstantArithmeticImage<T, L>>::divide_constant_image_in_place,
        )
    }

    fn constant_arithmetic(
        mut self,
        constant: <Self as ConstantArithmeticImage<T, L>>::Constant,
        scale_factor: i32,
        operation: ConstantScaleOperation<T, L, <Self as ConstantArithmeticImage<T, L>>::Constant>,
        operation_in_place: ConstantScaleInPlaceOperation<
            T,
            L,
            <Self as ConstantArithmeticImage<T, L>>::Constant,
        >,
    ) -> Result<Self> {
        self.ensure_eager()?;
        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                operation_in_place(self.stream_context, constant, &mut image_view, scale_factor)?;
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
                if L::IGNORES_ALPHA {
                    destination_view.copy_from(source)?;
                }
                operation(
                    self.stream_context,
                    source,
                    constant,
                    &mut destination_view,
                    scale_factor,
                )?;
                // SAFETY: completed dispatch wrote every active channel; any ignored alpha was copied from the source.
                self.backing = ImageBacking::Owned(unsafe { destination.assume_init() });
            }
        }

        Ok(self)
    }
}
