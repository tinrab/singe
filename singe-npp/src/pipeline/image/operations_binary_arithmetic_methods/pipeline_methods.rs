use crate::{
    error::Result,
    image::{
        Image, SupportedImage,
        view::{ChannelLayout, ImageView},
    },
    pipeline::image::{
        ImageBacking, ImagePipeline,
        operation_shapes::{ScaledBinaryImageInPlaceOperation, ScaledBinaryImageOperation},
        operation_traits::BinaryArithmeticImage,
    },
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: BinaryArithmeticImage<T, L>,
{
    /// Adds corresponding samples for each processed channel.
    /// Applies native integer scaling and saturation using `scale_factor`.
    /// The provided floating-point and floating-point complex implementations ignore `scale_factor`.
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
    pub fn add(self, other: &ImageView<'_, T, L>, scale_factor: i32) -> Result<Self> {
        self.binary_arithmetic(
            other,
            scale_factor,
            <Self as BinaryArithmeticImage<T, L>>::add_image,
            <Self as BinaryArithmeticImage<T, L>>::add_image_in_place,
        )
    }

    /// Subtracts samples using the native operand order for the current backing.
    /// With owned backing, updates each pipeline sample as `self - other`. With borrowed
    /// backing, allocates output and computes `other - self`. Operand order therefore depends
    /// on the backing held by the pipeline.
    /// Applies native integer scaling and saturation using `scale_factor`.
    /// The provided floating-point and floating-point complex implementations ignore `scale_factor`.
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
    pub fn subtract(self, other: &ImageView<'_, T, L>, scale_factor: i32) -> Result<Self> {
        self.binary_arithmetic(
            other,
            scale_factor,
            <Self as BinaryArithmeticImage<T, L>>::subtract_image,
            <Self as BinaryArithmeticImage<T, L>>::subtract_image_in_place,
        )
    }

    /// Multiplies corresponding samples for each processed channel.
    /// Applies native integer scaling and saturation using `scale_factor`.
    /// The provided floating-point and floating-point complex implementations ignore `scale_factor`.
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
    pub fn multiply(self, other: &ImageView<'_, T, L>, scale_factor: i32) -> Result<Self> {
        self.binary_arithmetic(
            other,
            scale_factor,
            <Self as BinaryArithmeticImage<T, L>>::multiply_image,
            <Self as BinaryArithmeticImage<T, L>>::multiply_image_in_place,
        )
    }

    /// Divides samples using the native operand order for the current backing.
    /// With owned backing, updates each pipeline sample as `self / other`. With borrowed
    /// backing, allocates output and computes `other / self`. Operand order therefore depends
    /// on the backing held by the pipeline.
    /// Applies native integer scaling and saturation using `scale_factor`.
    /// The provided floating-point and floating-point complex implementations ignore `scale_factor`.
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
    pub fn divide(self, other: &ImageView<'_, T, L>, scale_factor: i32) -> Result<Self> {
        self.binary_arithmetic(
            other,
            scale_factor,
            <Self as BinaryArithmeticImage<T, L>>::divide_image,
            <Self as BinaryArithmeticImage<T, L>>::divide_image_in_place,
        )
    }

    fn binary_arithmetic(
        mut self,
        other: &ImageView<'_, T, L>,
        scale_factor: i32,
        operation: ScaledBinaryImageOperation<T, L>,
        operation_in_place: ScaledBinaryImageInPlaceOperation<T, L>,
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
                // SAFETY: completed dispatch wrote every active channel; any ignored alpha was copied from the source.
                self.backing = ImageBacking::Owned(unsafe { destination.assume_init() });
            }
        }

        Ok(self)
    }
}
