use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
};

use super::{
    ImageBacking, ImagePipeline, MultiplyScaleImage, UnscaledBinaryImageInPlaceOperation,
    UnscaledBinaryImageOperation,
};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: MultiplyScaleImage<T, L>,
{
    pub fn multiply_scale_into(
        stream_context: &StreamContext,
        left: &ImageView<'_, T, L>,
        right: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()> {
        <Self as MultiplyScaleImage<T, L>>::multiply_scale_image(
            stream_context,
            left,
            right,
            destination,
        )
    }

    pub fn multiply_scale_in_place(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()> {
        <Self as MultiplyScaleImage<T, L>>::multiply_scale_image_in_place(
            stream_context,
            source,
            source_destination,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: MultiplyScaleImage<T, L>,
{
    pub fn multiply_scale(self, other: &ImageView<'_, T, L>) -> Result<Self> {
        self.binary_arithmetic_unscaled(
            other,
            <Self as MultiplyScaleImage<T, L>>::multiply_scale_image,
            <Self as MultiplyScaleImage<T, L>>::multiply_scale_image_in_place,
        )
    }

    fn binary_arithmetic_unscaled(
        mut self,
        other: &ImageView<'_, T, L>,
        operation: UnscaledBinaryImageOperation<T, L>,
        operation_in_place: UnscaledBinaryImageInPlaceOperation<T, L>,
    ) -> Result<Self> {
        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                operation_in_place(self.stream_context, other, &mut image_view)?;
            }
            ImageBacking::Borrowed(source) => {
                let mut destination = self.workspace.image::<T, L>(source.size())?;
                let mut destination_view = destination.view_mut()?;
                operation(self.stream_context, source, other, &mut destination_view)?;
                self.backing = ImageBacking::Owned(destination);
            }
        }

        Ok(self)
    }
}
