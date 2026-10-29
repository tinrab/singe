pub use self::dispatch::{
    MaskedAccumulateProductImage, MaskedAccumulateSquareImage, MaskedAccumulateWeightedImage,
};
use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{C1, ImageView, ImageViewMut, MaskView},
    },
    pipeline::image::masked::{ImageBacking, ImagePipeline},
};

mod dispatch;
mod weighted_methods;

impl ImagePipeline<'_, f32, C1> {
    /// Adds the square of each source sample to the C1 `f32` accumulator where the mask is nonzero.
    ///
    /// The provided implementations accept C1 `u8`, `u16`, or `f32` sources. All source,
    /// mask, and accumulator dimensions must match. Zero mask pixels preserve the previous
    /// accumulator values.
    /// Updates the supplied initialized accumulator in place.
    /// Prepares participating regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains
    /// unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native execution, or synchronization fail. Participating image and
    /// mask dimensions must match.
    pub fn accumulate_square_masked_into<T>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        mask: &MaskView<'_>,
        source_destination: &mut ImageViewMut<'_, f32, C1>,
    ) -> Result<()>
    where
        T: Copy,
        Self: MaskedAccumulateSquareImage<T>,
    {
        <Self as MaskedAccumulateSquareImage<T>>::accumulate_square_masked_image(
            stream_context,
            source,
            mask,
            source_destination,
        )
    }
}

impl ImagePipeline<'_, f32, C1>
where
    f32: SupportedImage<C1>,
{
    /// Adds the square of each source sample to the C1 `f32` accumulator where the mask is nonzero.
    ///
    /// The provided implementations accept C1 `u8`, `u16`, or `f32` sources. All source,
    /// mask, and accumulator dimensions must match. Zero mask pixels preserve the previous
    /// accumulator values.
    /// Reuses an owned pipeline accumulator in place; for borrowed input, copies its
    /// initialized values into a new owned image before the update.
    /// Prepares participating regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains
    /// unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native execution, or synchronization fail. Participating image and
    /// mask dimensions must match.
    /// Output allocation, copying borrowed input, or image view construction can also fail.
    pub fn accumulate_square_masked_from<T>(
        self,
        source: &ImageView<'_, T, C1>,
        mask: &MaskView<'_>,
    ) -> Result<Self>
    where
        T: Copy,
        Self: MaskedAccumulateSquareImage<T>,
    {
        self.accumulate_masked_with(|stream_context, destination| {
            <Self as MaskedAccumulateSquareImage<T>>::accumulate_square_masked_image(
                stream_context,
                source,
                mask,
                destination,
            )
        })
    }

    pub(crate) fn accumulate_masked_with(
        mut self,
        operation: impl FnOnce(&StreamContext, &mut ImageViewMut<'_, f32, C1>) -> Result<()>,
    ) -> Result<Self> {
        self.ensure_eager()?;
        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                operation(self.stream_context, &mut image_view)?;
            }
            ImageBacking::Borrowed(source_destination) => {
                let mut destination = Image::<f32, C1>::from_view(source_destination)?;
                let mut destination_view = destination.view_mut()?;
                operation(self.stream_context, &mut destination_view)?;
                self.backing = ImageBacking::Owned(destination);
            }
        }

        Ok(self)
    }
}
