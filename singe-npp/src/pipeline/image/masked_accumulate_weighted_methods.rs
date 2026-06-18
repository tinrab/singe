use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, ImageView, ImageViewMut, MaskView},
    pipeline::{ImageAllocator, Workspace},
};

use super::{CopyImage, ImagePipeline, MaskedAccumulateWeightedImage};

impl<'a> ImagePipeline<'a, f32, C1> {
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

impl<'a> ImagePipeline<'a, f32, C1>
where
    Workspace: ImageAllocator<f32, C1>,
    Self: CopyImage<f32, C1>,
{
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
