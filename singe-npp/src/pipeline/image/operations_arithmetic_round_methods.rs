use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{ChannelLayout, ImageOutputView, ImageView, ImageViewMut},
    },
    pipeline::image::{
        ImageBacking, ImagePipeline,
        operation_shapes::{RoundedBinaryImageInPlaceOperation, RoundedBinaryImageOperation},
        operation_traits::{DivideRoundImage, NaturalLogarithmImage},
    },
    types::RoundMode,
};

mod natural_logarithm_methods;

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: DivideRoundImage<T, L>,
{
    /// Divides `right` by `left` per processed channel.
    ///
    /// `round_mode` selects native rounding; `scale_factor` controls native integer scaling
    /// and saturation. The provided implementations accept `u8`, `u16`, or `i16` samples.
    /// Rounding selections and sample values are forwarded to NPP without validation.
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
    pub fn divide_round_into(
        stream_context: &StreamContext,
        left: &ImageView<'_, T, L>,
        right: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        scale_factor: i32,
        round_mode: RoundMode,
    ) -> Result<()> {
        <Self as DivideRoundImage<T, L>>::divide_round_image(
            stream_context,
            left,
            right,
            destination,
            scale_factor,
            round_mode,
        )
    }

    /// Divides `source_destination` by `source` per processed channel.
    ///
    /// `round_mode` selects native rounding; `scale_factor` controls native integer scaling
    /// and saturation. The provided implementations accept `u8`, `u16`, or `i16` samples.
    /// Rounding selections and sample values are forwarded to NPP without validation.
    /// Updates the image in place and preserves AC4 alpha. Source and destination dimensions must match.
    /// Row padding remains unwritten. Prepares participating regions, waits for producers
    /// on other streams, rejects capture, and returns after successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native execution, or stream synchronization fail.
    /// Participating image dimensions must match.
    pub fn divide_round_in_place(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
        round_mode: RoundMode,
    ) -> Result<()> {
        <Self as DivideRoundImage<T, L>>::divide_round_image_in_place(
            stream_context,
            source,
            source_destination,
            scale_factor,
            round_mode,
        )
    }
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: DivideRoundImage<T, L>,
{
    /// Divides image samples using native integer scaling and rounding per processed channel.
    ///
    /// For an owned pipeline, reuses its image and computes pipeline samples divided by
    /// `other`. For a borrowed pipeline, allocates a new image and computes `other` divided
    /// by pipeline samples. This follows the native operand order of each dispatch path.
    /// `round_mode` selects native rounding; `scale_factor` controls native integer scaling
    /// and saturation. The provided implementations accept `u8`, `u16`, or `i16` samples.
    /// Rounding selections and sample values are forwarded to NPP without validation.
    /// For AC4, preserves source alpha, copying the source before out-of-place execution
    /// when necessary.
    /// Row padding remains unwritten. Prepares participating regions, waits for producers
    /// on other streams, rejects capture, and returns after successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native execution, or stream synchronization fail.
    /// Participating image dimensions must match.
    /// Image view construction, output allocation, or copying source alpha can also fail.
    pub fn divide_round(
        self,
        other: &ImageView<'_, T, L>,
        scale_factor: i32,
        round_mode: RoundMode,
    ) -> Result<Self> {
        self.binary_arithmetic_round(
            other,
            scale_factor,
            round_mode,
            <Self as DivideRoundImage<T, L>>::divide_round_image,
            <Self as DivideRoundImage<T, L>>::divide_round_image_in_place,
        )
    }

    fn binary_arithmetic_round(
        mut self,
        other: &ImageView<'_, T, L>,
        scale_factor: i32,
        round_mode: RoundMode,
        operation: RoundedBinaryImageOperation<T, L>,
        operation_in_place: RoundedBinaryImageInPlaceOperation<T, L>,
    ) -> Result<Self> {
        self.ensure_eager()?;

        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                operation_in_place(
                    self.stream_context,
                    other,
                    &mut image_view,
                    scale_factor,
                    round_mode,
                )?;
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
                    round_mode,
                )?;
                // SAFETY: the completed operation above wrote every element.
                self.backing = ImageBacking::Owned(unsafe { destination.assume_init() });
            }
        }

        Ok(self)
    }
}
