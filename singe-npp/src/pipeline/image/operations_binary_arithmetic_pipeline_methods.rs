use crate::{
    error::Result,
    image::view::{ChannelLayout, ImageView},
    pipeline::{ImageAllocator, Workspace},
};

use super::super::{
    ImageBacking, ImagePipeline, operation_shapes::*, operation_traits::BinaryArithmeticImage,
};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: BinaryArithmeticImage<T, L>,
{
    pub fn add(self, other: &ImageView<'_, T, L>, scale_factor: i32) -> Result<Self> {
        self.binary_arithmetic(
            other,
            scale_factor,
            <Self as BinaryArithmeticImage<T, L>>::add_image,
            <Self as BinaryArithmeticImage<T, L>>::add_image_in_place,
        )
    }

    pub fn subtract(self, other: &ImageView<'_, T, L>, scale_factor: i32) -> Result<Self> {
        self.binary_arithmetic(
            other,
            scale_factor,
            <Self as BinaryArithmeticImage<T, L>>::subtract_image,
            <Self as BinaryArithmeticImage<T, L>>::subtract_image_in_place,
        )
    }

    pub fn multiply(self, other: &ImageView<'_, T, L>, scale_factor: i32) -> Result<Self> {
        self.binary_arithmetic(
            other,
            scale_factor,
            <Self as BinaryArithmeticImage<T, L>>::multiply_image,
            <Self as BinaryArithmeticImage<T, L>>::multiply_image_in_place,
        )
    }

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
        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                operation_in_place(self.stream_context, other, &mut image_view, scale_factor)?;
            }
            ImageBacking::Borrowed(source) => {
                let mut destination = self.workspace.image::<T, L>(source.size())?;
                let mut destination_view = destination.view_mut()?;
                operation(
                    self.stream_context,
                    source,
                    other,
                    &mut destination_view,
                    scale_factor,
                )?;
                self.backing = ImageBacking::Owned(destination);
            }
        }

        Ok(self)
    }
}
