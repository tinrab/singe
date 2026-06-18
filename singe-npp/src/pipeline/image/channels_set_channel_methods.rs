use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
};

use super::{CopyImage, ImageBacking, ImagePipeline, SetChannelImage};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: SetChannelImage<T, L>,
{
    pub fn set_channel_into(
        stream_context: &StreamContext,
        destination: &mut ImageViewMut<'_, T, L>,
        value: T,
        channel: usize,
    ) -> Result<()> {
        <Self as SetChannelImage<T, L>>::set_channel_image(
            stream_context,
            value,
            destination,
            channel,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: CopyImage<T, L> + SetChannelImage<T, L>,
{
    pub fn set_channel(mut self, value: T, channel: usize) -> Result<Self> {
        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                <Self as SetChannelImage<T, L>>::set_channel_image(
                    self.stream_context,
                    value,
                    &mut image_view,
                    channel,
                )?;
            }
            ImageBacking::Borrowed(source) => {
                let mut destination = self.workspace.image::<T, L>(source.size())?;
                let mut destination_view = destination.view_mut()?;
                <Self as CopyImage<T, L>>::copy(
                    self.stream_context,
                    source,
                    &mut destination_view,
                )?;
                <Self as SetChannelImage<T, L>>::set_channel_image(
                    self.stream_context,
                    value,
                    &mut destination_view,
                    channel,
                )?;
                self.backing = ImageBacking::Owned(destination);
            }
        }

        Ok(self)
    }
}
