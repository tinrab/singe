use crate::{
    error::Result,
    image::view::{ChannelLayout, ImageView},
    pipeline::{ImageAllocator, Workspace},
};

use super::{ImageBacking, ImagePipeline, operation_shapes::*, operation_traits::*};

#[path = "operations_logical_into_methods.rs"]
mod into_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: LogicalBinaryImage<T, L>,
{
    pub fn logical_and(self, other: &ImageView<'_, T, L>) -> Result<Self> {
        self.logical_binary(
            other,
            <Self as LogicalBinaryImage<T, L>>::logical_and_image,
            <Self as LogicalBinaryImage<T, L>>::logical_and_image_in_place,
        )
    }

    pub fn logical_or(self, other: &ImageView<'_, T, L>) -> Result<Self> {
        self.logical_binary(
            other,
            <Self as LogicalBinaryImage<T, L>>::logical_or_image,
            <Self as LogicalBinaryImage<T, L>>::logical_or_image_in_place,
        )
    }

    pub fn logical_xor(self, other: &ImageView<'_, T, L>) -> Result<Self> {
        self.logical_binary(
            other,
            <Self as LogicalBinaryImage<T, L>>::logical_xor_image,
            <Self as LogicalBinaryImage<T, L>>::logical_xor_image_in_place,
        )
    }

    fn logical_binary(
        mut self,
        other: &ImageView<'_, T, L>,
        operation: BinaryImageOperation<T, L>,
        operation_in_place: BinaryImageInPlaceOperation<T, L>,
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
