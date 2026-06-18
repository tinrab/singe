use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
};

use super::{CopyChannelImage, CopyImage, ImageBacking, ImagePipeline};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: CopyChannelImage<T, L>,
{
    pub fn copy_channel_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_channel: usize,
        destination: &mut ImageViewMut<'_, T, L>,
        destination_channel: usize,
    ) -> Result<()> {
        <Self as CopyChannelImage<T, L>>::copy_channel_image(
            stream_context,
            source,
            source_channel,
            destination,
            destination_channel,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: CopyImage<T, L> + CopyChannelImage<T, L>,
{
    pub fn copy_channel(self, source_channel: usize, destination_channel: usize) -> Result<Self> {
        let mut destination = self.workspace.image::<T, L>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as CopyImage<T, L>>::copy(self.stream_context, &source, &mut destination_view)?;
            <Self as CopyChannelImage<T, L>>::copy_channel_image(
                self.stream_context,
                &source,
                source_channel,
                &mut destination_view,
                destination_channel,
            )?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
