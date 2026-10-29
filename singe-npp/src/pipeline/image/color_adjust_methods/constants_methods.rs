use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{C4, ImageOutputView, ImageView, ImageViewMut},
    },
    pipeline::image::{
        ColorTwistConstants4, ColorTwistMatrix4, ImageBacking, ImagePipeline,
        color::ColorTwistWithConstantsImage,
    },
};

mod batch_methods;

impl<T> ImagePipeline<'_, T, C4>
where
    T: Copy,
    Self: ColorTwistWithConstantsImage<T>,
{
    /// Applies a four-by-four color-twist matrix and additive output vector.
    ///
    /// For each C4 output pixel, computes the native matrix transform followed by the
    /// four `constants`. The provided implementations accept `u8`, `f16`, or `f32` samples.
    /// Coefficients and constants are host `f32` arrays passed to NPP without validation.
    /// Source and destination dimensions must match.
    /// Prepares participating regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native color transformation, or synchronization fail.
    /// Source and destination dimensions must match.
    pub fn color_twist_with_constants_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C4>,
        destination: &mut ImageOutputView<'_, T, C4>,
        twist: ColorTwistMatrix4,
        constants: ColorTwistConstants4,
    ) -> Result<()> {
        <Self as ColorTwistWithConstantsImage<T>>::color_twist_with_constants_image(
            stream_context,
            source,
            destination,
            twist,
            constants,
        )
    }

    /// Applies a four-by-four color-twist matrix and additive output vector.
    ///
    /// For each C4 output pixel, computes the native matrix transform followed by the
    /// four `constants`. The provided implementations accept `u8`, `f16`, or `f32` samples.
    /// Coefficients and constants are host `f32` arrays passed to NPP without validation.
    /// Updates all four channels of the supplied initialized image in place.
    /// Prepares participating regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native color transformation, or synchronization fail.
    pub fn color_twist_with_constants_in_place(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, T, C4>,
        twist: ColorTwistMatrix4,
        constants: ColorTwistConstants4,
    ) -> Result<()> {
        <Self as ColorTwistWithConstantsImage<T>>::color_twist_with_constants_image_in_place(
            stream_context,
            image,
            twist,
            constants,
        )
    }
}

impl<T> ImagePipeline<'_, T, C4>
where
    T: Copy + SupportedImage<C4>,
    Self: ColorTwistWithConstantsImage<T>,
{
    /// Applies a four-by-four color-twist matrix and additive output vector.
    ///
    /// For each C4 output pixel, computes the native matrix transform followed by the
    /// four `constants`. The provided implementations accept `u8`, `f16`, or `f32` samples.
    /// Coefficients and constants are host `f32` arrays passed to NPP without validation.
    /// Reuses an owned image in place; for borrowed input, allocates a new source-sized
    /// image and writes all four destination channels.
    /// Prepares participating regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native color transformation, or synchronization fail.
    /// Output allocation or image view construction can also fail.
    pub fn color_twist_with_constants(
        mut self,
        twist: ColorTwistMatrix4,
        constants: ColorTwistConstants4,
    ) -> Result<Self> {
        self.ensure_eager()?;

        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                <Self as ColorTwistWithConstantsImage<T>>::color_twist_with_constants_image_in_place(
                    self.stream_context,
                    &mut image_view,
                    twist,
                    constants,
                )?;
            }
            ImageBacking::Borrowed(source) => {
                let mut destination = Image::<T, C4>::create_uninit(
                    self.stream_context.cuda_context(),
                    source.size(),
                )?;
                let mut destination_view = destination.output()?;
                <Self as ColorTwistWithConstantsImage<T>>::color_twist_with_constants_image(
                    self.stream_context,
                    source,
                    &mut destination_view,
                    twist,
                    constants,
                )?;
                // SAFETY: the completed operation above wrote every element.
                self.backing = ImageBacking::Owned(unsafe { destination.assume_init() });
            }
        }

        Ok(self)
    }
}
