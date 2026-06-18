use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
};

use super::{ImageBacking, ImagePipeline};

pub trait SwapChannelsImage<T, L, const CHANNELS: usize> {
    fn swap_channels_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        order: [i32; CHANNELS],
    ) -> Result<()>;

    fn swap_channels_image_in_place(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, T, L>,
        order: [i32; CHANNELS],
    ) -> Result<()>;
}

macro_rules! impl_swap_channels_image {
    ($ty:ty, $layout:ty, $channels:literal, $swap:path, $swap_in_place:path) => {
        impl<'a> SwapChannelsImage<$ty, $layout, $channels> for ImagePipeline<'a, $ty, $layout> {
            fn swap_channels_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                order: [i32; $channels],
            ) -> Result<()> {
                $swap(stream_context, source, destination, order)
            }

            fn swap_channels_image_in_place(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, $ty, $layout>,
                order: [i32; $channels],
            ) -> Result<()> {
                $swap_in_place(stream_context, image, order)
            }
        }
    };
}

#[path = "channel_swap_dispatch_impls.rs"]
mod dispatch_impls;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    pub fn swap_channels_into<const CHANNELS: usize>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        order: [i32; CHANNELS],
    ) -> Result<()>
    where
        Self: SwapChannelsImage<T, L, CHANNELS>,
    {
        <Self as SwapChannelsImage<T, L, CHANNELS>>::swap_channels_image(
            stream_context,
            source,
            destination,
            order,
        )
    }

    pub fn swap_channels_in_place<const CHANNELS: usize>(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, T, L>,
        order: [i32; CHANNELS],
    ) -> Result<()>
    where
        Self: SwapChannelsImage<T, L, CHANNELS>,
    {
        <Self as SwapChannelsImage<T, L, CHANNELS>>::swap_channels_image_in_place(
            stream_context,
            image,
            order,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
{
    pub fn swap_channels<const CHANNELS: usize>(mut self, order: [i32; CHANNELS]) -> Result<Self>
    where
        Self: SwapChannelsImage<T, L, CHANNELS>,
    {
        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                <Self as SwapChannelsImage<T, L, CHANNELS>>::swap_channels_image_in_place(
                    self.stream_context,
                    &mut image_view,
                    order,
                )?;
            }
            ImageBacking::Borrowed(source) => {
                let mut destination = self.workspace.image::<T, L>(source.size())?;
                let mut destination_view = destination.view_mut()?;
                <Self as SwapChannelsImage<T, L, CHANNELS>>::swap_channels_image(
                    self.stream_context,
                    source,
                    &mut destination_view,
                    order,
                )?;
                self.backing = ImageBacking::Owned(destination);
            }
        }

        Ok(self)
    }
}
