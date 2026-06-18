use crate::{
    context::StreamContext,
    error::Result,
    image::view::{AC4, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
};

use super::super::{ImageBacking, ImagePipeline};

pub trait SwapChannelsAc4Image<T> {
    fn swap_channels_ac4_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, AC4>,
        destination: &mut ImageViewMut<'_, T, AC4>,
        order: [i32; 3],
    ) -> Result<()>;
}

impl<'a, T> ImagePipeline<'a, T, AC4>
where
    T: Copy,
    Self: SwapChannelsAc4Image<T>,
{
    pub fn swap_channels_ac4_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, AC4>,
        destination: &mut ImageViewMut<'_, T, AC4>,
        order: [i32; 3],
    ) -> Result<()> {
        <Self as SwapChannelsAc4Image<T>>::swap_channels_ac4_image(
            stream_context,
            source,
            destination,
            order,
        )
    }
}

impl<'a, T> ImagePipeline<'a, T, AC4>
where
    T: Copy,
    Workspace: ImageAllocator<T, AC4>,
    Self: SwapChannelsAc4Image<T>,
{
    pub fn swap_channels_ac4(self, order: [i32; 3]) -> Result<Self> {
        let mut destination = self.workspace.image::<T, AC4>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as SwapChannelsAc4Image<T>>::swap_channels_ac4_image(
                self.stream_context,
                &source,
                &mut destination_view,
                order,
            )?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
