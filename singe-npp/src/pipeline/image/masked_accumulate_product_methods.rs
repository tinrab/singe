use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, ImageView, ImageViewMut, MaskView},
    pipeline::{ImageAllocator, Workspace},
};

use super::{CopyImage, ImagePipeline, masked_accumulate_methods::MaskedAccumulateProductImage};

impl<'a> ImagePipeline<'a, f32, C1> {
    pub fn accumulate_product_masked_into<T>(
        stream_context: &StreamContext,
        source1: &ImageView<'_, T, C1>,
        source2: &ImageView<'_, T, C1>,
        mask: &MaskView<'_>,
        source_destination: &mut ImageViewMut<'_, f32, C1>,
    ) -> Result<()>
    where
        T: Copy,
        Self: MaskedAccumulateProductImage<T>,
    {
        <Self as MaskedAccumulateProductImage<T>>::accumulate_product_masked_image(
            stream_context,
            source1,
            source2,
            mask,
            source_destination,
        )
    }
}

impl<'a> ImagePipeline<'a, f32, C1>
where
    Workspace: ImageAllocator<f32, C1>,
    Self: CopyImage<f32, C1>,
{
    pub fn accumulate_product_masked_from<T>(
        self,
        source1: &ImageView<'_, T, C1>,
        source2: &ImageView<'_, T, C1>,
        mask: &MaskView<'_>,
    ) -> Result<Self>
    where
        T: Copy,
        Self: MaskedAccumulateProductImage<T>,
    {
        self.accumulate_masked_with(|stream_context, destination| {
            <Self as MaskedAccumulateProductImage<T>>::accumulate_product_masked_image(
                stream_context,
                source1,
                source2,
                mask,
                destination,
            )
        })
    }
}
