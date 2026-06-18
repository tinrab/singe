use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
};

use super::{CopyImage, ImageBacking, ImagePipeline};

#[path = "channels_ac4_swap_methods.rs"]
mod ac4_swap_methods;
#[path = "channels_dispatch.rs"]
mod channels_dispatch;
#[path = "channels_copy_channel_methods.rs"]
mod copy_channel_methods;
#[path = "channels_set_channel_methods.rs"]
mod set_channel_methods;
#[path = "channels_traits.rs"]
mod traits;

pub use ac4_swap_methods::SwapChannelsAc4Image;
pub use traits::{CopyChannelImage, ExtractChannelImage, InsertChannelImage, SetChannelImage};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: ExtractChannelImage<T, L>,
{
    pub fn extract_channel_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, C1>,
        channel: usize,
    ) -> Result<()> {
        <Self as ExtractChannelImage<T, L>>::extract_channel_image(
            stream_context,
            source,
            destination,
            channel,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, C1>,
    Self: ExtractChannelImage<T, L>,
{
    pub fn extract_channel(self, channel: usize) -> Result<ImagePipeline<'a, T, C1>> {
        let mut destination = self.workspace.image::<T, C1>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as ExtractChannelImage<T, L>>::extract_channel_image(
                self.stream_context,
                &source,
                &mut destination_view,
                channel,
            )?;
        }

        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: InsertChannelImage<T, L>,
{
    pub fn insert_channel_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        destination: &mut ImageViewMut<'_, T, L>,
        channel: usize,
    ) -> Result<()> {
        <Self as InsertChannelImage<T, L>>::insert_channel_image(
            stream_context,
            source,
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
    Self: CopyImage<T, L> + InsertChannelImage<T, L>,
{
    pub fn insert_channel(mut self, source: &ImageView<'_, T, C1>, channel: usize) -> Result<Self> {
        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                <Self as InsertChannelImage<T, L>>::insert_channel_image(
                    self.stream_context,
                    source,
                    &mut image_view,
                    channel,
                )?;
            }
            ImageBacking::Borrowed(current_source) => {
                let mut destination = self.workspace.image::<T, L>(current_source.size())?;
                let mut destination_view = destination.view_mut()?;
                <Self as CopyImage<T, L>>::copy(
                    self.stream_context,
                    current_source,
                    &mut destination_view,
                )?;
                <Self as InsertChannelImage<T, L>>::insert_channel_image(
                    self.stream_context,
                    source,
                    &mut destination_view,
                    channel,
                )?;
                self.backing = ImageBacking::Owned(destination);
            }
        }

        Ok(self)
    }
}
