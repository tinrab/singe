use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
};

use super::{ImageBacking, ImagePipeline, operation_shapes::*, operation_traits::*};

#[path = "operations_logical_xor_constant_methods.rs"]
mod xor_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: LogicalConstantImage<T, L>,
    <Self as LogicalConstantImage<T, L>>::Constant: Copy,
{
    pub fn logical_and_constant_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        constant: <Self as LogicalConstantImage<T, L>>::Constant,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()> {
        <Self as LogicalConstantImage<T, L>>::logical_and_constant_image(
            stream_context,
            source,
            constant,
            destination,
        )
    }

    pub fn logical_and_constant_in_place(
        stream_context: &StreamContext,
        constant: <Self as LogicalConstantImage<T, L>>::Constant,
        source_destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()> {
        <Self as LogicalConstantImage<T, L>>::logical_and_constant_image_in_place(
            stream_context,
            constant,
            source_destination,
        )
    }

    pub fn logical_or_constant_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        constant: <Self as LogicalConstantImage<T, L>>::Constant,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()> {
        <Self as LogicalConstantImage<T, L>>::logical_or_constant_image(
            stream_context,
            source,
            constant,
            destination,
        )
    }

    pub fn logical_or_constant_in_place(
        stream_context: &StreamContext,
        constant: <Self as LogicalConstantImage<T, L>>::Constant,
        source_destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()> {
        <Self as LogicalConstantImage<T, L>>::logical_or_constant_image_in_place(
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
    Self: LogicalConstantImage<T, L>,
    <Self as LogicalConstantImage<T, L>>::Constant: Copy,
{
    pub fn logical_and_constant(
        self,
        constant: <Self as LogicalConstantImage<T, L>>::Constant,
    ) -> Result<Self> {
        self.logical_constant(
            constant,
            <Self as LogicalConstantImage<T, L>>::logical_and_constant_image,
            <Self as LogicalConstantImage<T, L>>::logical_and_constant_image_in_place,
        )
    }

    pub fn logical_or_constant(
        self,
        constant: <Self as LogicalConstantImage<T, L>>::Constant,
    ) -> Result<Self> {
        self.logical_constant(
            constant,
            <Self as LogicalConstantImage<T, L>>::logical_or_constant_image,
            <Self as LogicalConstantImage<T, L>>::logical_or_constant_image_in_place,
        )
    }

    pub(super) fn logical_constant(
        mut self,
        constant: <Self as LogicalConstantImage<T, L>>::Constant,
        operation: ConstantOperation<T, L, <Self as LogicalConstantImage<T, L>>::Constant>,
        operation_in_place: ConstantInPlaceOperation<
            T,
            L,
            <Self as LogicalConstantImage<T, L>>::Constant,
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
