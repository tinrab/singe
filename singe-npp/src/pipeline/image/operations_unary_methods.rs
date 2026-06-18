use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
};

use super::{ImageBacking, ImagePipeline, operation_traits::*};

#[path = "operations_unary_power_methods.rs"]
mod unary_power_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: AbsoluteImage<T, L>,
{
    pub fn absolute_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()> {
        <Self as AbsoluteImage<T, L>>::absolute_image(stream_context, source, destination)
    }

    pub fn absolute_in_place(
        stream_context: &StreamContext,
        source_destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()> {
        <Self as AbsoluteImage<T, L>>::absolute_image_in_place(stream_context, source_destination)
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: AbsoluteImage<T, L>,
{
    pub fn absolute(mut self) -> Result<Self> {
        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                <Self as AbsoluteImage<T, L>>::absolute_image_in_place(
                    self.stream_context,
                    &mut image_view,
                )?;
            }
            ImageBacking::Borrowed(source) => {
                let mut destination = self.workspace.image::<T, L>(source.size())?;
                let mut destination_view = destination.view_mut()?;
                <Self as AbsoluteImage<T, L>>::absolute_image(
                    self.stream_context,
                    source,
                    &mut destination_view,
                )?;
                self.backing = ImageBacking::Owned(destination);
            }
        }

        Ok(self)
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: AbsoluteDifferenceImage<T, L>,
{
    pub fn absolute_difference_into(
        stream_context: &StreamContext,
        left: &ImageView<'_, T, L>,
        right: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()> {
        <Self as AbsoluteDifferenceImage<T, L>>::absolute_difference_image(
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
    Self: AbsoluteDifferenceImage<T, L>,
{
    pub fn absolute_difference(self, other: &ImageView<'_, T, L>) -> Result<Self> {
        let mut destination = self.workspace.image::<T, L>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as AbsoluteDifferenceImage<T, L>>::absolute_difference_image(
                self.stream_context,
                &source,
                other,
                &mut destination_view,
            )?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
