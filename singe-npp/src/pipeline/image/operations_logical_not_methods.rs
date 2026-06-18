use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
};

use super::super::{ImageBacking, ImagePipeline, operation_traits::LogicalNotImage};

impl<'a, L> ImagePipeline<'a, u8, L>
where
    L: ChannelLayout,
    Self: LogicalNotImage<u8, L>,
{
    pub fn logical_not_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, L>,
        destination: &mut ImageViewMut<'_, u8, L>,
    ) -> Result<()> {
        <Self as LogicalNotImage<u8, L>>::logical_not_image(stream_context, source, destination)
    }

    pub fn logical_not_in_place(
        stream_context: &StreamContext,
        source_destination: &mut ImageViewMut<'_, u8, L>,
    ) -> Result<()> {
        <Self as LogicalNotImage<u8, L>>::logical_not_image_in_place(
            stream_context,
            source_destination,
        )
    }
}

impl<'a, L> ImagePipeline<'a, u8, L>
where
    L: ChannelLayout,
    Workspace: ImageAllocator<u8, L>,
    Self: LogicalNotImage<u8, L>,
{
    pub fn logical_not(mut self) -> Result<Self> {
        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                <Self as LogicalNotImage<u8, L>>::logical_not_image_in_place(
                    self.stream_context,
                    &mut image_view,
                )?;
            }
            ImageBacking::Borrowed(source) => {
                let mut destination = self.workspace.image::<u8, L>(source.size())?;
                let mut destination_view = destination.view_mut()?;
                <Self as LogicalNotImage<u8, L>>::logical_not_image(
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
