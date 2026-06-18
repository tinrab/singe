use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
};

use super::super::super::{
    ImageBacking, ImagePipeline, operation_shapes::OutOfPlaceBinaryImageOperation,
    operation_traits::UnscaledBinaryArithmeticImage,
};

#[path = "operations_unscaled_multiplicative_methods.rs"]
mod multiplicative_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: UnscaledBinaryArithmeticImage<T, L>,
{
    pub fn add_unscaled_into(
        stream_context: &StreamContext,
        left: &ImageView<'_, T, L>,
        right: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()> {
        <Self as UnscaledBinaryArithmeticImage<T, L>>::add_unscaled_image(
            stream_context,
            left,
            right,
            destination,
        )
    }

    pub fn subtract_unscaled_into(
        stream_context: &StreamContext,
        left: &ImageView<'_, T, L>,
        right: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()> {
        <Self as UnscaledBinaryArithmeticImage<T, L>>::subtract_unscaled_image(
            stream_context,
            left,
            right,
            destination,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: UnscaledBinaryArithmeticImage<T, L>,
{
    pub fn add_unscaled(self, other: &ImageView<'_, T, L>) -> Result<Self> {
        self.binary_arithmetic_out_of_place(
            other,
            <Self as UnscaledBinaryArithmeticImage<T, L>>::add_unscaled_image,
        )
    }

    pub fn subtract_unscaled(self, other: &ImageView<'_, T, L>) -> Result<Self> {
        self.binary_arithmetic_out_of_place(
            other,
            <Self as UnscaledBinaryArithmeticImage<T, L>>::subtract_unscaled_image,
        )
    }

    pub(super) fn binary_arithmetic_out_of_place(
        mut self,
        other: &ImageView<'_, T, L>,
        operation: OutOfPlaceBinaryImageOperation<T, L>,
    ) -> Result<Self> {
        let size = self.size();
        let mut destination = self.workspace.image::<T, L>(size)?;
        let mut destination_view = destination.view_mut()?;
        let source = self.view()?;
        operation(self.stream_context, &source, other, &mut destination_view)?;
        self.backing = ImageBacking::Owned(destination);
        Ok(self)
    }
}
