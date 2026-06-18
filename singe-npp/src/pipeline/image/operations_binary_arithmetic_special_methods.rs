use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
};

use super::super::{ImageBacking, ImagePipeline, operation_shapes::*, operation_traits::*};

#[path = "operations_multiply_scale_methods.rs"]
mod multiply_scale_methods;
#[path = "operations_unscaled_binary_arithmetic_methods.rs"]
mod unscaled_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: ScaledSubtractImage<T, L>,
{
    pub fn subtract_scaled_into(
        stream_context: &StreamContext,
        left: &ImageView<'_, T, L>,
        right: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as ScaledSubtractImage<T, L>>::subtract_scaled_image(
            stream_context,
            left,
            right,
            destination,
            scale_factor,
        )
    }

    pub fn subtract_scaled_in_place(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as ScaledSubtractImage<T, L>>::subtract_scaled_image_in_place(
            stream_context,
            source,
            source_destination,
            scale_factor,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: ScaledSubtractImage<T, L>,
{
    pub fn subtract_scaled(self, other: &ImageView<'_, T, L>, scale_factor: i32) -> Result<Self> {
        self.binary_arithmetic_scaled_subtract(
            other,
            scale_factor,
            <Self as ScaledSubtractImage<T, L>>::subtract_scaled_image,
            <Self as ScaledSubtractImage<T, L>>::subtract_scaled_image_in_place,
        )
    }

    fn binary_arithmetic_scaled_subtract(
        mut self,
        other: &ImageView<'_, T, L>,
        scale_factor: i32,
        operation: ScaledSubtractImageOperation<T, L>,
        operation_in_place: ScaledSubtractImageInPlaceOperation<T, L>,
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
