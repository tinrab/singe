use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
};

use super::{ImageBacking, ImagePipeline, operation_shapes::*, operation_traits::*};

#[path = "operations_absolute_difference_constant_methods.rs"]
mod absolute_difference_constant;
#[path = "operations_device_constant_arithmetic_methods.rs"]
mod device_constant;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: MultiplyConstantScaleImage<T, L>,
{
    pub fn multiply_constant_scale_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        constant: <Self as MultiplyConstantScaleImage<T, L>>::Constant,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()> {
        <Self as MultiplyConstantScaleImage<T, L>>::multiply_constant_scale_image(
            stream_context,
            source,
            constant,
            destination,
        )
    }

    pub fn multiply_constant_scale_in_place(
        stream_context: &StreamContext,
        constant: <Self as MultiplyConstantScaleImage<T, L>>::Constant,
        source_destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()> {
        <Self as MultiplyConstantScaleImage<T, L>>::multiply_constant_scale_image_in_place(
            stream_context,
            constant,
            source_destination,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: MultiplyConstantScaleImage<T, L>,
    <Self as MultiplyConstantScaleImage<T, L>>::Constant: Copy,
{
    pub fn multiply_constant_scale(
        self,
        constant: <Self as MultiplyConstantScaleImage<T, L>>::Constant,
    ) -> Result<Self> {
        self.multiply_constant_scale_operation(
            constant,
            <Self as MultiplyConstantScaleImage<T, L>>::multiply_constant_scale_image,
            <Self as MultiplyConstantScaleImage<T, L>>::multiply_constant_scale_image_in_place,
        )
    }

    fn multiply_constant_scale_operation(
        mut self,
        constant: <Self as MultiplyConstantScaleImage<T, L>>::Constant,
        operation: ConstantOperation<T, L, <Self as MultiplyConstantScaleImage<T, L>>::Constant>,
        operation_in_place: ConstantInPlaceOperation<
            T,
            L,
            <Self as MultiplyConstantScaleImage<T, L>>::Constant,
        >,
    ) -> Result<Self> {
        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                operation_in_place(self.stream_context, constant, &mut image_view)?;
            }
            ImageBacking::Borrowed(source) => {
                let mut destination = self.workspace.image::<T, L>(source.size())?;
                let mut destination_view = destination.view_mut()?;
                operation(self.stream_context, source, constant, &mut destination_view)?;
                self.backing = ImageBacking::Owned(destination);
            }
        }

        Ok(self)
    }
}
