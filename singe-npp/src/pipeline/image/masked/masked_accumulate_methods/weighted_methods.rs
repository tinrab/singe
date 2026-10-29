use crate::{
    context::StreamContext,
    error::Result,
    image::{
        SupportedImage,
        view::{C1, ImageView, ImageViewMut, MaskView},
    },
    pipeline::image::masked::masked_accumulate_methods::{
        ImagePipeline, MaskedAccumulateWeightedImage,
    },
};

impl ImagePipeline<'_, f32, C1> {
    /// Applies native alpha-weighted source accumulation to the C1 `f32` accumulator where the mask is nonzero.
    ///
    /// The provided implementations accept C1 `u8`, `u16`, or `f32` sources. All source,
    /// mask, and accumulator dimensions must match. Zero mask pixels preserve the previous
    /// accumulator values.
    /// `alpha` is the source weight, expected by NPP in `0.0..=1.0`; the wrapper forwards
    /// it without validating its range.
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
    pub fn accumulate_weighted_masked_into<T>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        mask: &MaskView<'_>,
        source_destination: &mut ImageViewMut<'_, f32, C1>,
        alpha: f32,
    ) -> Result<()>
    where
        T: Copy,
        Self: MaskedAccumulateWeightedImage<T>,
    {
        <Self as MaskedAccumulateWeightedImage<T>>::accumulate_weighted_masked_image(
            stream_context,
            source,
            mask,
            source_destination,
            alpha,
        )
    }
}

impl ImagePipeline<'_, f32, C1>
where
    f32: SupportedImage<C1>,
{
    /// Applies native alpha-weighted source accumulation to the C1 `f32` accumulator where the mask is nonzero.
    ///
    /// The provided implementations accept C1 `u8`, `u16`, or `f32` sources. All source,
    /// mask, and accumulator dimensions must match. Zero mask pixels preserve the previous
    /// accumulator values.
    /// `alpha` is the source weight, expected by NPP in `0.0..=1.0`; the wrapper forwards
    /// it without validating its range.
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
    pub fn accumulate_weighted_masked_from<T>(
        self,
        source: &ImageView<'_, T, C1>,
        mask: &MaskView<'_>,
        alpha: f32,
    ) -> Result<Self>
    where
        T: Copy,
        Self: MaskedAccumulateWeightedImage<T>,
    {
        self.accumulate_masked_with(|stream_context, destination| {
            <Self as MaskedAccumulateWeightedImage<T>>::accumulate_weighted_masked_image(
                stream_context,
                source,
                mask,
                destination,
                alpha,
            )
        })
    }
}
