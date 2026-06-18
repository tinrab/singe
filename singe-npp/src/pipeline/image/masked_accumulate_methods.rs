use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, ImageView, ImageViewMut, MaskView},
    pipeline::{ImageAllocator, Workspace},
};

use super::{CopyImage, ImageBacking, ImagePipeline};

pub use self::dispatch::{
    MaskedAccumulateProductImage, MaskedAccumulateSquareImage, MaskedAccumulateWeightedImage,
};

#[path = "masked_accumulate_dispatch.rs"]
mod dispatch;
#[path = "masked_accumulate_weighted_methods.rs"]
mod weighted_methods;

impl<'a> ImagePipeline<'a, f32, C1> {
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

impl<'a> ImagePipeline<'a, f32, C1>
where
    Workspace: ImageAllocator<f32, C1>,
    Self: CopyImage<f32, C1>,
{
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

    pub(super) fn accumulate_masked_with(
        mut self,
        operation: impl FnOnce(&StreamContext, &mut ImageViewMut<'_, f32, C1>) -> Result<()>,
    ) -> Result<Self> {
        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                operation(self.stream_context, &mut image_view)?;
            }
            ImageBacking::Borrowed(source_destination) => {
                let mut destination = self.workspace.image::<f32, C1>(source_destination.size())?;
                let mut destination_view = destination.view_mut()?;
                <Self as CopyImage<f32, C1>>::copy(
                    self.stream_context,
                    source_destination,
                    &mut destination_view,
                )?;
                operation(self.stream_context, &mut destination_view)?;
                self.backing = ImageBacking::Owned(destination);
            }
        }

        Ok(self)
    }
}
